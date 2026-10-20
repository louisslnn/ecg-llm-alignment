#!/usr/bin/env python
"""Free-generation evaluation of a trained resampler against the frozen 14B student.

Loads a resampler checkpoint, splices its latents in front of the student prompt
exactly as training does, and lets the frozen student generate the answer block on
its own (greedy, stop at ``<END>``). Nothing is teacher-forced: this measures what
the model writes when only the ECG latents and the question are given.

Two numbers come out of every generation:

* the **parsed conclusion** -- the Yes/No read out of the generated text with the
  same regex that parsed the teacher's targets (``src.teacher.parse``), so the
  prediction is extracted the way the supervision was. This is the hard decision
  scored by F1 / precision / recall.
* the **probability** -- at the token position where the model emitted that Yes/No,
  pool the logits over every Yes token id and every No token id with logsumexp, then
  softmax the two pooled values. That is the reference's calibration readout, and it
  is what AUPRC / AUROC are computed from.

A generation that has no parseable ``Conclusion: Yes/No`` is a parse failure: it is
counted and reported, never silently dropped. Metrics are computed over the parsed
generations only, so always read ``unparsed`` alongside them.

Reporting, per superclass and overall, plus NORM and the pooled abnormal
superclasses as separate subsets: F1, precision, recall, AUPRC, AUROC and the
**positive rate**, because AUPRC has no fixed floor -- a 0.30 AUPRC is strong at a
0.10 positive rate and worthless at 0.30. The ``lift`` column is AUPRC / positive
rate for exactly that reading.

Two controls decide whether any of this is about the waveform, and both leave the
prompts and labels exactly as they are:

* ``--shuffle-embeddings`` permutes the cached ECG embeddings across records within
  the split (one donor per record, so all five of a record's questions get the same
  wrong ECG): every question is now asked about someone else's heart. The
  permutation is a single cycle over a seeded shuffle, so it has no fixed points,
  and the seed is logged and written into every output record.
* ``--zero-latents`` runs the resampler and then throws its output away, splicing a
  zero tensor of the same shape in its place. The prefix is still 64 positions wide
  and still attended to; it simply carries nothing. Where shuffling asks "does it
  use *this* ECG?", zeroing asks "does it use the prefix at all?".

Metrics that survive either control were never coming from the signal -- they are
the label prior plus whatever age and sex leak through the question.

Every pass also prints, for its first batch, the L2 norm of the spliced latents next
to the mean norm of the token embeddings they sit against. Differing latents are not
enough: attention logits are dot products, so a prefix an order of magnitude shorter
than the tokens draws almost no attention weight and the frozen LLM reads past it.
That ratio says whether the prefix can compete for attention at all, and it is the
first thing to check when the controls come back indistinguishable.

``--teacher-forced-loss`` is the other half of the picture and generates nothing.
It builds the same prompts, appends the real teacher target, and computes the
next-token loss under training's masking (prompt, padding and latents at -100, loss
on the target only), reporting the mean over the evaluated examples. It answers a
question free generation cannot: whether the model assigns probability to the right
continuation, even when its own greedy decode wanders off. It is directly comparable
to the train/val loss the training loop prints, and it accepts both controls.

Every generation, its parsed conclusion, its probability and its label go to jsonl,
so ``--from-jsonl`` recomputes the whole metrics table without touching a GPU.

    # full test split
    python scripts/evaluate.py --model-dir $MODEL_DIR --ckpt $CKPT_DIR/best.pt

    # 500 examples, quick look: a seeded draw, 100 per superclass, half Yes and half
    # No (--stratify, --sample-seed), never the head of the split
    python scripts/evaluate.py --model-dir $MODEL_DIR --ckpt $CKPT_DIR/best.pt --limit 500

    # the controls (each writes its own default filename, so neither can overwrite
    # the run it is meant to be compared against)
    python scripts/evaluate.py --model-dir $MODEL_DIR --ckpt $CKPT_DIR/best.pt \
        --shuffle-embeddings
    python scripts/evaluate.py --model-dir $MODEL_DIR --ckpt $CKPT_DIR/best.pt \
        --zero-latents

    # teacher-forced loss on 200 examples, no generation
    python scripts/evaluate.py --model-dir $MODEL_DIR --ckpt $CKPT_DIR/best.pt \
        --teacher-forced-loss --limit 200

    # metrics only, no model load
    python scripts/evaluate.py --from-jsonl $EVAL_DIR/test.jsonl

    # all of the above in one job, one model load (see scripts/eval_controls.py)
    python scripts/eval_controls.py --model-dir $MODEL_DIR --ckpt $CKPT_DIR/best.pt

Generation is the cost here: ~200-400 new tokens per example on a 14B sharded over
two A100s. Budget accordingly, use --limit for anything interactive, and --resume to
continue a run that hit its time limit (already-generated examples are skipped).

Compute nodes have no internet: the student and tokenizer load from --model-dir with
local_files_only=True.
"""

import argparse
import json
import math
import os
import sys
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for probe_memory

from src.data.dataset import (  # noqa: E402
    DEFAULT_PREFIX_TEXT,
    TASK_STREAM_MODES,
    DatasetConfig,
    _format_prompt,
    build_datasets,
    make_collate_fn,
    question_stream_tensors,
    task_stream_ids,
)
from src.model.resampler import PerceiverResamplerConfig, Resampler  # noqa: E402
from src.teacher import parse as teacher_parse  # noqa: E402
from src.teacher.prompt import SUPERCLASSES  # noqa: E402
from probe_memory import load_student  # noqa: E402  -- the same student loader as training
from decision_token import MARKER, decision_index, tokenise_target  # noqa: E402

# The target format every block ends with (src/teacher/prompt.py MODE_B_TEMPLATE).
END_MARKER = "<END>"

# Max target length over the train split is 395 tokens, so 512 leaves headroom for a
# model that is still learning to close its block without letting a degenerate
# generation run for minutes.
DEFAULT_MAX_NEW_TOKENS = 512

ABNORMAL = [sc for sc in SUPERCLASSES if sc != "NORM"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=None, help="local student checkpoint dir")
    ap.add_argument("--ckpt", default=None,
                    help="resampler checkpoint (.pt file, or a dir holding best.pt)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate a seeded draw of N examples (--stratify, "
                         "--sample-seed) rather than the whole split. Never the first "
                         "N: the split is ordered by record and its head is not a "
                         "sample (fold 10's is heavily normal)")
    ap.add_argument("--sample-seed", type=int, default=42,
                    help="seed for the --limit draw; separate from --seed, which "
                         "seeds the embedding permutation")
    ap.add_argument("--stratify", choices=STRATIFY_MODES, default=DEFAULT_STRATIFY,
                    help="how --limit draws: 'superclass-label' (default) takes N/5 "
                         "per superclass, half Yes and half No; 'superclass' N/5 per "
                         "superclass, uniform within; 'none' uniform over the split. "
                         "The realised positive rate per superclass is printed beside "
                         "the split's")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="generation batch size (prompts are left-padded)")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--max-prompt-length", type=int, default=1024,
                    help="hard cap on prompt tokens; student prompts are ~60")
    ap.add_argument("--out", default=None,
                    help="jsonl of generations (default results/eval_<split>[_shuffled].jsonl)")
    ap.add_argument("--metrics-out", default=None,
                    help="metrics json (default: --out with a .metrics.json suffix)")
    ap.add_argument("--resume", action="store_true",
                    help="skip (ecg_id, superclass) pairs already present in --out "
                         "and append; for continuing a run that ran out of time")
    ap.add_argument("--from-jsonl", default=None,
                    help="recompute metrics from an existing jsonl and exit; no GPU, "
                         "no model load")
    # teacher-forced loss instead of generation
    ap.add_argument("--teacher-forced-loss", action="store_true",
                    help="skip generation: append the real target to the prompt and "
                         "report the mean next-token loss under training's masking")
    ap.add_argument("--max-length", type=int, default=2048,
                    help="token cap for prompt+target in --teacher-forced-loss "
                         "(training's MAX_TEXT_LEN)")
    ap.add_argument("--task-stream", choices=TASK_STREAM_MODES, default=None,
                    help="what the RESAMPLER's task-text stream reads. Default: the "
                         "mode the checkpoint recorded (train.py's --task-stream), or "
                         "'prompt' for a checkpoint that predates the record. "
                         "'prompt' is the real prompt tokens; 'prompt-and-target' "
                         "leaves the target in, reproducing run 1 -- in --teacher-"
                         "forced-loss only: generation has no target, so there it is "
                         "the prompt. 'question' is the question paraphrase alone, "
                         "tokenised separately; 'none' is no text. Neither changes "
                         "what the LLM receives or the loss positions")
    ap.add_argument("--strip-think", action=argparse.BooleanOptionalAction,
                    default=DatasetConfig.strip_think,
                    help="strip the template's trailing open <think> block from the "
                         "prompt. MUST match what training used (train.py's own "
                         "--strip-think). ON by default; pass --no-strip-think for a "
                         "checkpoint trained with the block, as runs 1 and 2 were")
    ap.add_argument("--allow-strip-think-mismatch", action="store_true",
                    help="run even though --strip-think disagrees with the value the "
                         "checkpoint was trained with. Without this the script "
                         "refuses; with it, it warns and marks the output, because "
                         "the student is then scored on a prompt it never trained on")
    # the controls
    ap.add_argument("--shuffle-embeddings", action="store_true",
                    help="permute ECG embeddings across records within the split "
                         "(prompts and labels untouched): the does-it-use-THIS-ECG control")
    ap.add_argument("--zero-latents", action="store_true",
                    help="run the resampler, then splice a zero tensor of the same "
                         "shape in place of its output: the does-it-use-the-prefix control")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the embedding permutation; logged and recorded")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--print-first", type=int, default=3,
                    help="print the first N generations in full, for a sanity read")
    ap.add_argument("--log-every", type=int, default=20, help="progress line every N batches")
    # data path overrides (default to DatasetConfig)
    ap.add_argument("--emb-cache", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--teacher", default=None)
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Metrics. Hand-rolled on numpy: the Alliance venv is built --no-index from the  #
# wheelhouse (torch/transformers/numpy/pandas/scipy), and sklearn is not in it.  #
# --------------------------------------------------------------------------- #


def _prf1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    """Positive-class P/R/F1. Recall (and so F1) is n/a when the group has no positives."""
    tp = float(np.sum(y_pred & y_true))
    fp = float(np.sum(y_pred & ~y_true))
    fn = float(np.sum(~y_pred & y_true))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # predicted nothing positive
    recall = tp / (tp + fn) if (tp + fn) > 0 else None     # nothing positive to find
    if recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties averaged (the rank statistic AUROC needs)."""
    order = np.argsort(x, kind="mergesort")
    ordered = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(y_true: np.ndarray, score: np.ndarray) -> Optional[float]:
    """Rank-based AUROC (Mann-Whitney U), ties averaged. None if one class is empty."""
    n_pos = int(np.sum(y_true))
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(score)
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(y_true: np.ndarray, score: np.ndarray) -> Optional[float]:
    """Average precision: sum over distinct thresholds of (dRecall * precision).

    Tied scores are collapsed into one threshold, so a model that emits the same
    probability for many examples cannot be credited with ordering inside the tie.
    """
    n_pos = int(np.sum(y_true))
    if n_pos == 0 or len(y_true) == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order].astype(np.float64)
    s = score[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]   # last index of each tie run
    tp, fp = tp[last], fp[last]
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1e-12)
    prev_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev_recall) * precision))


def metrics_for(records: Sequence[dict]) -> Dict[str, object]:
    """Every reported number for one group of records (one superclass, or a pool)."""
    n = len(records)
    parsed = [r for r in records if r.get("parsed_conclusion") is not None]
    scored = [r for r in parsed if r.get("p_yes") is not None]
    out: Dict[str, object] = {
        "n": n,
        "unparsed": n - len(parsed),
        "unparsed_rate": (n - len(parsed)) / n if n else None,
        "no_probability": len(parsed) - len(scored),
        "positive_rate": None,
        "precision": None, "recall": None, "f1": None,
        "auprc": None, "auroc": None, "auprc_lift": None,
        "tp": 0, "fp": 0, "fn": 0,
    }
    if not parsed:
        return out

    y_true = np.array([bool(r["label"]) for r in parsed])
    y_pred = np.array([bool(r["parsed_conclusion"]) for r in parsed])
    pos_rate = float(np.mean(y_true))
    out["positive_rate"] = pos_rate
    out["predicted_positive_rate"] = float(np.mean(y_pred))
    out.update(_prf1(y_true, y_pred))

    if scored:
        y_s = np.array([bool(r["label"]) for r in scored])
        p = np.array([float(r["p_yes"]) for r in scored])
        ap = auprc(y_s, p)
        out["auprc"] = ap
        out["auroc"] = auroc(y_s, p)
        # AUPRC's floor is the positive rate of the scored subset, so quote the lift
        # against that rate rather than against the parsed-subset rate.
        scored_pos_rate = float(np.mean(y_s))
        out["scored_positive_rate"] = scored_pos_rate
        out["auprc_lift"] = (ap / scored_pos_rate) if (ap is not None and scored_pos_rate > 0) else None
    return out


# The METRICS table's rows, in print order. Every metrics table groups by these, so
# the generation and fast-mode tables line up row for row.
METRIC_GROUPS = (["overall"] + list(SUPERCLASSES)
                 + ["NORM (normal subset)", "abnormal subset"])


def metric_groups(records: Sequence[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {"overall": list(records)}
    for sc in SUPERCLASSES:
        groups[sc] = [r for r in records if r["superclass"] == sc]
    # The two subsets asked for separately: the normality question on its own, and
    # the four abnormal questions pooled.
    groups["NORM (normal subset)"] = [r for r in records if r["superclass"] == "NORM"]
    groups["abnormal subset"] = [r for r in records if r["superclass"] in ABNORMAL]
    return groups


def compute_all_metrics(records: Sequence[dict]) -> Dict[str, Dict[str, object]]:
    return {name: metrics_for(rs) for name, rs in metric_groups(records).items()}


# Fast mode's fixed operating point: Yes when p_yes >= this.
FAST_THRESHOLD = 0.5


def best_f1_threshold(y_true: np.ndarray, score: np.ndarray):
    """``(threshold, prf1)``: the cut ``score >= threshold`` with the highest F1.

    Every distinct score is a candidate, ties kept together as in :func:`auprc`; the
    first maximum wins, which is the highest such threshold. Chosen on the very
    examples it is then scored on, so it is an in-sample upper bound on F1 rather
    than an operating point. ``(None, None)`` when there are no positives.
    """
    if not len(y_true) or not y_true.any():
        return None, None
    order = np.argsort(-score, kind="mergesort")
    s, y = score[order], y_true[order].astype(np.float64)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]   # last index of each tie run
    tp = np.cumsum(y)[last]
    pred = last + 1.0
    f1 = 2 * tp / (pred + y.sum())
    best = int(np.argmax(f1))
    threshold = float(s[last[best]])
    return threshold, _prf1(y_true, score >= threshold)


def fast_metrics_for(records: Sequence[dict]) -> Dict[str, object]:
    """Every fast-mode number for one group: the METRICS columns, from p_yes alone.

    No parse step, so every example is scored. F1, precision and recall come twice:
    at FAST_THRESHOLD, and at the group's own best-F1 threshold (in-sample, see
    :func:`best_f1_threshold`).
    """
    n = len(records)
    out: Dict[str, object] = {
        "n": n, "positive_rate": None, "mean_p_yes": None,
        "at_threshold": None, "best_f1": None,
        "auprc": None, "auprc_lift": None, "auroc": None,
        "n_with_loss": 0, "mean_loss": None,
    }
    if not n:
        return out
    y = np.array([bool(r["label"]) for r in records])
    p = np.array([float(r["p_yes"]) for r in records])
    pos_rate = float(y.mean())
    losses = [r["loss"] for r in records if r.get("loss") is not None]
    out.update({
        "positive_rate": pos_rate,
        "mean_p_yes": float(p.mean()),
        "at_threshold": {"threshold": FAST_THRESHOLD,
                         "predicted_positive_rate": float((p >= FAST_THRESHOLD).mean()),
                         **_prf1(y, p >= FAST_THRESHOLD)},
        "auroc": auroc(y, p),
        "n_with_loss": len(losses),
        "mean_loss": float(np.mean(losses)) if losses else None,
    })
    threshold, best = best_f1_threshold(y, p)
    if best is not None:
        out["best_f1"] = {"threshold": threshold, **best}
    ap = auprc(y, p)
    out["auprc"] = ap
    out["auprc_lift"] = ap / pos_rate if ap is not None and pos_rate > 0 else None
    return out


def compute_fast_metrics(records: Sequence[dict]) -> Dict[str, Dict[str, object]]:
    return {name: fast_metrics_for(rs) for name, rs in metric_groups(records).items()}


def _fmt(v, width=6, prec=3):
    if v is None:
        return f"{'n/a':>{width}}"
    return f"{v:>{width}.{prec}f}"


def print_metrics(table: Dict[str, Dict[str, object]]) -> None:
    order = METRIC_GROUPS
    name_w = max(len(n) for n in order)
    header = (f"{'group':<{name_w}}  {'n':>6} {'unpars':>6} {'pos_rt':>6} "
              f"{'F1':>6} {'prec':>6} {'rec':>6} {'AUPRC':>6} {'lift':>6} {'AUROC':>6}")
    print("\n" + "=" * len(header))
    print("METRICS  (parsed generations only; 'unpars' = generations with no "
          "readable Conclusion)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name in order:
        m = table[name]
        if name == "NORM (normal subset)":
            print("-" * len(header))
        print(f"{name:<{name_w}}  {m['n']:>6} {m['unparsed']:>6} "
              f"{_fmt(m['positive_rate'])} {_fmt(m['f1'])} {_fmt(m['precision'])} "
              f"{_fmt(m['recall'])} {_fmt(m['auprc'])} {_fmt(m['auprc_lift'], prec=2)} "
              f"{_fmt(m['auroc'])}")
    print("-" * len(header))
    print("AUPRC has no fixed floor: read it against pos_rt (lift = AUPRC / positive "
          "rate of the scored subset; lift 1.0 is chance).")


def print_fast_metrics(table: Dict[str, Dict[str, object]], title: str = "") -> None:
    """Fast mode's table: the METRICS rows and columns, F1 at two thresholds."""
    name_w = max(len(n) for n in METRIC_GROUPS)
    header = (f"{'group':<{name_w}}  {'n':>6} {'pos_rt':>6} {'p_yes':>6} | "
              f"{'F1@.5':>6} {'prec':>6} {'rec':>6} | "
              f"{'F1*':>6} {'thr*':>6} {'prec*':>6} {'rec*':>6} | "
              f"{'AUPRC':>6} {'lift':>6} {'AUROC':>6}")
    print("\n" + "=" * len(header))
    print("FAST-MODE METRICS  (p_yes after the prompt and \"Conclusion:\" alone; "
          "every example scored)" + (f"  -- {title}" if title else ""))
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name in METRIC_GROUPS:
        m = table[name]
        if name == "NORM (normal subset)":
            print("-" * len(header))
        at = m["at_threshold"] or {}
        best = m["best_f1"] or {}
        print(f"{name:<{name_w}}  {m['n']:>6} {_fmt(m['positive_rate'])} "
              f"{_fmt(m['mean_p_yes'])} | {_fmt(at.get('f1'))} "
              f"{_fmt(at.get('precision'))} {_fmt(at.get('recall'))} | "
              f"{_fmt(best.get('f1'))} {_fmt(best.get('threshold'))} "
              f"{_fmt(best.get('precision'))} {_fmt(best.get('recall'))} | "
              f"{_fmt(m['auprc'])} {_fmt(m['auprc_lift'], prec=2)} {_fmt(m['auroc'])}")
    print("-" * len(header))
    print(f"@.5: Yes when p_yes >= {FAST_THRESHOLD:g}, the same cut for every row.")
    print("*:   each row's own best-F1 threshold (thr*), chosen ON THESE EXAMPLES -- an "
          "in-sample upper\n     bound on F1, not an operating point to quote.")
    print("AUPRC against pos_rt as above (lift 1.0 is chance). On a balanced draw "
          "pos_rt is ~0.5 by\nconstruction, so precision and AUPRC are at that "
          "prevalence, not the split's.")


# --------------------------------------------------------------------------- #
# Which examples. A --limit is a draw, never the head of the split: the split is  #
# ordered by record, and its first examples are not a sample of anything.       #
# --------------------------------------------------------------------------- #

STRATIFY_MODES = ("none", "superclass", "superclass-label")
DEFAULT_STRATIFY = "superclass-label"


def _quotas(n: int, k: int) -> List[int]:
    """``n`` split over ``k`` cells as evenly as integers allow, extras first."""
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def sample_indices(dataset, n: Optional[int], seed: int,
                   stratify: str = DEFAULT_STRATIFY):
    """``(indices, info)``: a seeded draw of ``n`` examples, in dataset order.

    ``n`` None, or at least the split's size, is the whole split and draws nothing.
    Otherwise, by ``stratify``:

        none              a uniform draw over the split
        superclass        ``n`` spread evenly over the five superclasses, uniform
                          within each
        superclass-label  the same, and within each superclass half Yes and half No

    Each cell (a superclass, or a superclass and label) is permuted once with one
    ``np.random.default_rng(seed)`` in a fixed order, and its quota is taken from the
    front, so the draw depends on the seed and the quotas alone and a smaller draw
    with the same seed sits inside a larger one wherever no cell ran short. A cell
    that has fewer examples than its quota gives all it has, the rest comes from its
    sibling label where there is one, and the shortfall goes into ``info``. The
    result is sorted, so every condition handed these indices covers the same
    examples in the same order.
    """
    if stratify not in STRATIFY_MODES:
        raise ValueError(f"unknown stratify {stratify!r}; have {STRATIFY_MODES}")
    examples = base_of(dataset).examples
    total = len(examples)
    info: Dict[str, Any] = {"stratify": stratify, "sample_seed": seed,
                            "n_requested": n, "shortfalls": []}
    if n is None or n >= total:
        info.update(stratify="whole split", n_drawn=total)
        return list(range(total)), info
    rng = np.random.default_rng(seed)
    if stratify == "none":
        picked = rng.choice(total, size=n, replace=False).tolist()
    else:
        picked = []
        for sc, quota in zip(SUPERCLASSES, _quotas(n, len(SUPERCLASSES))):
            in_class = [i for i, e in enumerate(examples) if e.superclass == sc]
            if stratify == "superclass":
                cell = rng.permutation(in_class).tolist()
                if len(cell) < quota:
                    info["shortfalls"].append(f"{sc}: {len(cell)} of {quota}")
                picked += cell[:quota]
                continue
            cells = {label: rng.permutation(
                [i for i in in_class if bool(examples[i].label) == label]).tolist()
                for label in (True, False)}
            want = dict(zip((True, False), _quotas(quota, 2)))
            for label in (True, False):
                short = want[label] - len(cells[label])
                if short > 0:          # what one label lacks, the other gives
                    info["shortfalls"].append(
                        f"{sc} label={label}: {len(cells[label])} of {want[label]}")
                    want[label] -= short
                    want[not label] += short
            if len(in_class) < quota:
                info["shortfalls"].append(f"{sc}: {len(in_class)} of {quota}")
            for label in (True, False):
                picked += cells[label][:want[label]]
    picked = sorted(int(i) for i in picked)
    info["n_drawn"] = len(picked)
    return picked, info


def sample_rates(dataset, indices: Sequence[int]) -> Dict[str, Dict[str, Any]]:
    """Per superclass: the draw's size and positive rate, beside the split's own."""
    examples = base_of(dataset).examples
    drawn = [examples[i] for i in indices]
    out = {}
    for sc in list(SUPERCLASSES) + ["overall"]:
        def rate(exs):
            exs = [e for e in exs if sc == "overall" or e.superclass == sc]
            return len(exs), (float(np.mean([bool(e.label) for e in exs])) if exs else None)
        n, r = rate(drawn)
        n_split, r_split = rate(examples)
        out[sc] = {"n": n, "positive_rate": r, "n_split": n_split,
                   "split_positive_rate": r_split}
    return out


def print_sample(dataset, indices: Sequence[int], info: Dict[str, Any], what: str):
    """The draw, said in the log: how it was made, and its positive rate per
    superclass beside the split's, so a skewed draw is visible. Returns ``info``
    with the rates added, for the summary JSON."""
    rates = sample_rates(dataset, indices)
    how = (info["stratify"] if info["stratify"] == "whole split" else
           f"--stratify {info['stratify']}, --sample-seed {info['sample_seed']}")
    print(f"\n{what}: {len(indices):,} examples ({how})")
    print(f"  {'superclass':<10} {'n':>6} {'pos rate':>9} {'split rate':>11} "
          f"{'split n':>8}")
    for sc, r in rates.items():
        print(f"  {sc:<10} {r['n']:>6} {_fmt(r['positive_rate'], width=9)} "
              f"{_fmt(r['split_positive_rate'], width=11)} {r['n_split']:>8}")
    for s in info["shortfalls"]:
        print(f"  SHORT: {s} -- the draw is not balanced there")
    return {**info, "rates": rates}


# --------------------------------------------------------------------------- #
# Yes/No readout                                                              #
# --------------------------------------------------------------------------- #


def yes_no_token_ids(tokenizer):
    """Single-token ids spelling Yes / No, with and without a leading space.

    The conclusion line is "Conclusion: Yes, ...", so the emitted token usually
    carries the leading space; casing and spacing variants are pooled with logsumexp
    rather than picked between, per the reference.
    """
    def ids_for(words):
        found = []
        for w in words:
            for variant in (w, " " + w):
                enc = tokenizer.encode(variant, add_special_tokens=False)
                if len(enc) == 1:
                    found.append(enc[0])
        return sorted(set(found))

    yes_ids = ids_for(["Yes", "yes", "YES"])
    no_ids = ids_for(["No", "no", "NO"])
    if not yes_ids or not no_ids:
        raise RuntimeError("tokenizer yields no single-token Yes/No spellings")
    if set(yes_ids) & set(no_ids):
        raise RuntimeError("Yes and No token ids overlap")
    return yes_ids, no_ids


class ConclusionLogitCapture:
    """A logits processor that keeps only the Yes/No columns, on CPU.

    Holding the full (batch, vocab) scores for every one of ~512 steps would cost
    gigabytes of GPU for nothing; we need 10-odd columns. Returns the scores
    untouched, so greedy decoding is unaffected.
    """

    def __init__(self, cand_ids: Sequence[int], device):
        self.cand = torch.tensor(list(cand_ids), device=device)
        self.steps: List[torch.Tensor] = []

    def __call__(self, input_ids, scores):
        self.steps.append(scores[:, self.cand].detach().float().cpu())
        return scores

    def reset(self):
        self.steps.clear()


def locate_conclusion(tokenizer, token_ids: Sequence[int]):
    """Find the generated token that spells the Conclusion's Yes/No.

    Returns ``(step_index, token_text, status)``. The text is rebuilt token by token
    so every character maps back to the step that produced it; the Conclusion is then
    found with the teacher parser's own regex, and the token containing the first
    character of its Yes/No group is the position whose logits carry the answer.

    status is "clean" when that token is exactly Yes/No, "impure" when Yes/No is
    glued to following text in one token (the pooled logits are then over a merged
    token and mean a little less -- counted separately), "none" when there is no
    readable conclusion.
    """
    pieces = [tokenizer.decode([int(t)], skip_special_tokens=True) for t in token_ids]
    starts, pos = [], 0
    for piece in pieces:
        starts.append(pos)
        pos += len(piece)
    text = "".join(pieces)

    m = teacher_parse._CONCLUSION_RE.search(text)   # the regex that parsed the targets
    if not m:
        return None, None, "none"
    idx = bisect_right(starts, m.start(1)) - 1
    if idx < 0 or idx >= len(pieces):
        return None, None, "none"

    lead = pieces[idx].lstrip().lower()
    if lead.startswith("yes"):
        rest = lead[3:]
    elif lead.startswith("no"):
        rest = lead[2:]
    else:
        return None, pieces[idx], "none"
    status = "clean" if not rest.strip() else "impure"
    return idx, pieces[idx], status


def probability_from_logits(row: torch.Tensor, yes_pos: Sequence[int],
                            no_pos: Sequence[int]) -> float:
    """P(Yes) = softmax(logsumexp(Yes logits), logsumexp(No logits))[0].

    Pooling with logsumexp before the softmax is what the reference does: it treats
    "Yes"/" Yes"/"yes" as one event rather than scoring a single arbitrary spelling.
    """
    yes = torch.logsumexp(row[list(yes_pos)], dim=0)
    no = torch.logsumexp(row[list(no_pos)], dim=0)
    return float(torch.softmax(torch.stack([yes, no]), dim=0)[0])


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #


class ShuffledEmbeddings(torch.utils.data.Dataset):
    """The control: every record is served another record's ECG embedding.

    The permutation is over records, not examples, so a record's five questions all
    receive the same donor -- the model gets one consistent wrong heart per record,
    not five. Built as a single cycle over a seeded shuffle, which guarantees no
    record keeps its own embedding.

    ``ecg_ids`` restricts the permutation to those records (default: every record in
    ``base``), for a caller that evaluates a slice and needs every donor drawn from
    inside it. An example of any other record is then a KeyError rather than a
    silent pass-through of its own embedding.
    """

    def __init__(self, base, seed: int, ecg_ids: Optional[Sequence[int]] = None):
        self.base = base
        ids = sorted({e.ecg_id for e in base.examples} if ecg_ids is None
                     else {int(i) for i in ecg_ids})
        if len(ids) < 2:
            raise ValueError("need at least 2 records to permute embeddings")
        order = [int(i) for i in np.random.default_rng(seed).permutation(ids)]
        self.mapping = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
        self.seed = seed
        self.n_fixed_points = sum(1 for k, v in self.mapping.items() if k == v)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        donor = self.mapping[int(item["ecg_id"])]
        item["ecg_embed"] = torch.from_numpy(self.base.cache.get(donor))
        item["donor_ecg_id"] = donor
        return item


def make_eval_collate(tokenizer, max_prompt_length: int, strip_think: bool = True):
    """Prompt-only collate, LEFT-padded for batched generation.

    Training right-pads because the target follows the prompt; generation continues
    from the last position, so the padding has to go on the left instead. The prompt
    text itself is built by the dataset module's own helpers, so eval cannot drift
    from what training tokenised.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(batch):
        ecg_embed = torch.stack([b["ecg_embed"] for b in batch], dim=0)
        prompts, ids_list = [], []
        for b in batch:
            user_content = f"{DEFAULT_PREFIX_TEXT}\n\n{b['task_prompt']}"
            prompt_str = _format_prompt(tokenizer, user_content, strip_think)
            prompts.append(prompt_str)
            ids_list.append(
                tokenizer(prompt_str, add_special_tokens=False).input_ids[:max_prompt_length]
            )

        max_len = max(len(x) for x in ids_list)
        B = len(batch)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, ids in enumerate(ids_list):
            n = len(ids)
            input_ids[i, max_len - n:] = torch.tensor(ids, dtype=torch.long)  # left pad
            attention_mask[i, max_len - n:] = 1

        # The question stream, as the training collate carries it; whether the
        # resampler reads it is task_stream_ids' decision.
        task_input_ids, task_attention_mask = question_stream_tensors(
            tokenizer, [b["question"] for b in batch], pad_id)
        return {
            "ecg_embed": ecg_embed,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "task_input_ids": task_input_ids,
            "task_attention_mask": task_attention_mask,
            "prompt": prompts,
            "ecg_id": [int(b["ecg_id"]) for b in batch],
            "donor_ecg_id": [int(b.get("donor_ecg_id", b["ecg_id"])) for b in batch],
            "superclass": [b["superclass"] for b in batch],
            "strat_fold": [int(b["strat_fold"]) for b in batch],
            "label": [bool(b["label"]) for b in batch],
        }

    return collate


# --------------------------------------------------------------------------- #
# Generation                                                                  #
# --------------------------------------------------------------------------- #


def resolve_ckpt_path(ckpt_path: str) -> str:
    """A checkpoint file as given, or best.pt inside a checkpoint directory."""
    return os.path.join(ckpt_path, "best.pt") if os.path.isdir(ckpt_path) else ckpt_path


def checkpoint_strip_think(ckpt_path: str):
    """``(value, recorded)``: the strip_think setting a checkpoint was trained with.

    train.py writes it into the checkpoint's meta. A checkpoint with no such entry
    predates the flag -- run 1's does -- and was therefore trained with the
    template's open <think> block in the prompt: that is False, NOT whatever
    DatasetConfig defaults to today, and ``recorded`` is False so the caller can say
    the value was inferred.

    Read with mmap so the multi-GB tensors are not pulled off disk just to look at a
    dict; this runs before the student load, where failing costs seconds.
    """
    path = resolve_ckpt_path(ckpt_path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError, ValueError):    # older torch, or a legacy-format file
        payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload.get("meta") if isinstance(payload, dict) else None
    if isinstance(meta, dict) and meta.get("strip_think") is not None:
        return bool(meta["strip_think"]), True
    return False, False


def checkpoint_task_stream(ckpt_path: str) -> Optional[str]:
    """The --task-stream a checkpoint was trained with, or None if it has no record.

    Recorded in the meta since the flag existed (124b2a2), so None means a
    checkpoint older than that, whose resampler read the whole of input_ids --
    prompt-and-target -- in training. What None should default to is the caller's
    decision: a resume and an evaluation answer it differently.
    """
    path = resolve_ckpt_path(ckpt_path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError, ValueError):    # older torch, or a legacy-format file
        payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload.get("meta") if isinstance(payload, dict) else None
    return meta.get("task_stream") if isinstance(meta, dict) else None


def resolve_task_stream(ckpt_path: str, requested: Optional[str],
                        flag: str = "--task-stream") -> str:
    """The task stream an evaluation runs with, and where it came from, printed.

    An explicit --task-stream wins, so a deliberate comparison (run 1's
    prompt-and-target against prompt, say) stays possible; a disagreement with the
    checkpoint is printed, not refused. Unset, it is the checkpoint's own mode: a
    resampler trained on the question alone is never silently handed the full
    prompt. A checkpoint with no record gets 'prompt', the default every evaluation
    before the record used. ``flag`` is the caller's name for the option, for the
    log line (probe_latents.py calls it --task-text).
    """
    trained = checkpoint_task_stream(ckpt_path)
    if requested is None:
        mode = trained or "prompt"
        source = ("recorded in the checkpoint" if trained else
                  "the checkpoint has no record, so the pre-record default")
    else:
        mode, source = requested, flag
    print(f"resampler task stream: {mode} ({source})", flush=True)
    if trained and mode != trained:
        print(f"  NOTE: the checkpoint trained with --task-stream {trained}; this run "
              f"feeds its resampler {mode}", flush=True)
    return mode


def require_strip_think_match(ckpt_path: str, strip_think: bool,
                              allow_mismatch: bool = False,
                              escape_flag: Optional[str] = "--allow-strip-think-mismatch"):
    """Refuse to pair a checkpoint with a prompt format it was not trained on.

    ``strip_think`` decides whether the prompt ends in an open <think> block. If
    this run's setting differs from the checkpoint's, every number produced is about
    a prompt the student never saw -- a train/inference mismatch that looks, in the
    output, exactly like a model that learned nothing. So it is fatal, and the
    message names both values. ``allow_mismatch`` turns the refusal into a printed
    warning; the returned dict goes into the run's output so the result file carries
    the mismatch as well as the log.

    ``escape_flag`` is the flag the refusal points to; None for a caller that offers
    no way through (resuming training).
    """
    trained, recorded = checkpoint_strip_think(ckpt_path)
    source = ("recorded in its meta" if recorded else
              "NOT recorded: the checkpoint predates the flag, so it is treated as "
              "False -- trained with the template's open <think> block")
    path = resolve_ckpt_path(ckpt_path)
    print(f"checkpoint strip_think={trained} ({source}); this run strip_think="
          f"{bool(strip_think)}", flush=True)
    result = {"checkpoint_strip_think": trained,
              "checkpoint_strip_think_recorded": recorded,
              "strip_think_mismatch": trained != bool(strip_think)}
    if not result["strip_think_mismatch"]:
        return result

    matching = "--strip-think" if trained else "--no-strip-think"
    what = (f"strip_think MISMATCH: this run has strip_think={bool(strip_think)}, but "
            f"the checkpoint {path} was trained with strip_think={trained} ({source}). "
            "The prompt would differ from the one the resampler was trained against.")
    if allow_mismatch:
        bar = "!" * 78
        print(f"\n{bar}\nWARNING: {what}\nContinuing because {escape_flag} was "
              "passed. Every number below is measured on a prompt this checkpoint never "
              f"trained on.\n{bar}\n", flush=True)
        return result
    sys.exit(f"{what}\nPass {matching} to match the checkpoint"
             + (f", or {escape_flag} to run anyway." if escape_flag else "."))


def load_resampler(ckpt_path: str, dev) -> Resampler:
    """Load the resampler weights from a training checkpoint (or a bare state dict)."""
    ckpt_path = resolve_ckpt_path(ckpt_path)
    # our own checkpoint (trusted); it carries numpy/python RNG state that the
    # weights_only=True default (torch>=2.6) refuses to unpickle.
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["resampler"] if isinstance(payload, dict) and "resampler" in payload else payload
    # The checkpoint is the authority on its own architecture: a run trained before
    # these norms existed (or with either flag off) has no output_norm.* / input_norm.*
    # keys, and building them anyway would fail the strict load.
    has_output_norm = any(k.startswith("output_norm.") for k in state)
    has_input_norm = any(k.startswith("input_norm.") for k in state)
    resampler = Resampler(PerceiverResamplerConfig.ecg(
        final_output_norm=has_output_norm, input_layer_norm=has_input_norm))
    resampler.load_state_dict(state, strict=True)   # strict: a silently partial load
    resampler = resampler.to(dev).eval()            # would evaluate a half-random model
    for p in resampler.parameters():
        p.requires_grad_(False)
    if isinstance(payload, dict) and "global_step" in payload:
        print(f"resampler from {ckpt_path}: epoch {payload.get('epoch')} "
              f"step {payload.get('global_step')} "
              f"best_val {payload.get('best_val_loss')}")
    else:
        print(f"resampler from {ckpt_path} (bare state dict)")
    if has_output_norm:
        gain = float(resampler.output_norm.weight.mean())
        print(f"  final output LayerNorm present; mean gain {gain:.6f} "
              f"(implies a target latent norm of "
              f"{gain * math.sqrt(resampler.config.embed_dim):.3f})")
    else:
        print("  no final output LayerNorm in this checkpoint (reference behaviour); "
              "watch the prefix-scale ratio below")
    if not has_input_norm:
        print("  no input LayerNorm in this checkpoint: bio_projection sees the raw "
              "~350-norm ECG cache (reference behaviour)")
    return resampler


def build_generation_config(max_new_tokens: int, eos_id: int, pad_id: int):
    """Decoding is specified here, not inherited from the checkpoint.

    A model's shipped generation_config can carry sampling defaults and -- the one
    that actually bites -- a repetition_penalty, which HF applies even when
    do_sample is False. That would both bend the greedy decode and rescale the very
    Yes/No logits this script reads the probability from, since the penalty hits
    tokens that already appeared. Everything that could rewrite a logit is pinned to
    its identity value, so the captured scores are the model's raw logits.
    """
    from transformers import GenerationConfig

    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,             # greedy
        num_beams=1,
        stop_strings=[END_MARKER],   # the block terminator the targets end with
        repetition_penalty=1.0,      # identity
        encoder_repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        renormalize_logits=False,
        length_penalty=1.0,
        min_new_tokens=0,
        use_cache=True,              # generation without a KV cache is unusable
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )


def prefix_scale_report(latents, tok_embeds, attn, label="prompt"):
    """Print the size of the spliced prefix next to the tokens it sits against.

    Whether the latents differ from each other is not enough: they also have to be
    LOUD enough to matter. Attention logits are query-key dot products, so a prefix
    whose vectors are an order of magnitude shorter than the token embeddings
    contributes correspondingly smaller logits, its softmax weight goes to ~0, and
    the frozen LLM reads straight past it. A run can then pass every "the tensors
    are different" check and still be text-only in effect.

    Norms are per position (one L2 norm per latent, one per token), which is the
    comparison that means something, and taken over real tokens only -- padding
    would drag the token mean toward zero and flatter the ratio. Returns the numbers
    as well as printing them.
    """
    lat = latents.detach().float().norm(dim=-1).flatten()
    tok_norms = tok_embeds.detach().float().norm(dim=-1)
    real = tok_norms[attn.bool()]
    lat_mean = float(lat.mean())
    tok_mean = float(real.mean()) if real.numel() else float("nan")
    ratio = lat_mean / tok_mean if tok_mean else float("nan")

    tok_min = float(real.min()) if real.numel() else float("nan")
    tok_max = float(real.max()) if real.numel() else float("nan")
    w = 26
    print(f"\nprefix scale, first batch ({tuple(latents.shape)} latents vs "
          f"{int(attn.sum())} real {label} tokens):")
    print(f"  {'latent norm (per position)':<{w}} mean {lat_mean:9.3f}  "
          f"min {float(lat.min()):9.3f}  max {float(lat.max()):9.3f}")
    print(f"  {label + ' token norm':<{w}} mean {tok_mean:9.3f}  "
          f"min {tok_min:9.3f}  max {tok_max:9.3f}")
    print(f"  {'ratio latents / tokens':<{w}} {ratio:9.3g}")
    if ratio != ratio:                      # nan: nothing real to compare against
        print("  -> no real tokens in this batch; ratio undefined")
    elif lat_mean == 0.0:
        print("  -> the prefix is exactly zero (the --zero-latents control)")
    elif ratio < 0.1:
        print(f"  -> WARNING: the prefix is ~{1/ratio:.0f}x SHORTER than the tokens it "
              "sits against.\n     Attention weight on it will be near zero: the "
              "frozen LLM effectively ignores\n     the ECG however different the "
              "latents are from one another.")
    elif ratio > 10:
        print(f"  -> WARNING: the prefix is ~{ratio:.0f}x LONGER than the tokens it "
              "sits against and\n     may swamp the question rather than condition it.")
    else:
        print("  -> comparable scale: the prefix can compete for attention")
    return {"latent_norm_mean": lat_mean, "latent_norm_min": float(lat.min()),
            "latent_norm_max": float(lat.max()), "token_norm_mean": tok_mean,
            "ratio": ratio}


def splice_prefix(model, resampler, ecg, input_ids, attn, dev, zero_latents=False,
                  report_label=None, task_ids=None, task_mask=None):
    """Token embeddings with the resampler's latents spliced in front.

    The splice is the training splice (scripts/probe_memory.py splice_forward):
    latents first, then the token embeddings, with the prefix always attendable.
    Shared by generation and the teacher-forced loss so the two cannot drift.

    ``zero_latents`` runs the resampler and then discards its output, splicing zeros
    of the same shape. Zeroing AFTER the forward pass rather than skipping it keeps
    the prefix the same width and the tensor path identical, so the only thing that
    changes is whether the prefix carries information.

    ``report_label`` prints the prefix-vs-token norm comparison for this batch; the
    callers pass it on the first batch only.

    ``task_ids`` / ``task_mask`` are the resampler's task-text stream and what of it
    may be read (True = attend), as :func:`~src.data.dataset.task_stream_ids` builds
    them for the run's --task-stream. None falls back to ``input_ids`` / ``attn``,
    every real token the LLM receives. The splice itself is unaffected: the LLM
    receives ``input_ids`` whatever the resampler read.
    """
    if task_ids is None:
        task_ids = input_ids
    if task_mask is None:
        task_mask = attn.bool()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tok_embeds = model.get_input_embeddings()(input_ids)
        latents = resampler(ecg, task_embeddings=model.get_input_embeddings()(task_ids),
                            task_mask=task_mask.bool())
    if zero_latents:
        latents = torch.zeros_like(latents)
    # torch.cat needs one dtype; autocast can hand back fp32 from a LayerNorm-final
    # path, and the student runs in bf16 regardless.
    latents = latents.to(tok_embeds.dtype)

    if report_label:
        prefix_scale_report(latents, tok_embeds, attn, label=report_label)

    B, Lq, _ = latents.shape
    full_embeds = torch.cat([latents, tok_embeds], dim=1)
    full_attn = torch.cat(
        [torch.ones(B, Lq, dtype=attn.dtype, device=dev), attn], dim=1
    )
    return full_embeds, full_attn, Lq


@torch.no_grad()
def generate_batch(model, resampler, tokenizer, batch, dev, processors, capture,
                   gen_config, zero_latents=False, report_label=None,
                   task_stream: str = "prompt"):
    """One batch: splice latents in front of the prompt, then greedy-decode to <END>.

    Left padding means the pads sit between the latents and the real prompt; position
    ids are derived from the attention mask, so the real tokens still see the latents
    at the positions training put them. The resampler reads ``task_stream``; this
    collate has no labels, so the two prompt modes are both the whole prompt here.
    """
    task_ids, task_mask = task_stream_ids(batch, task_stream)
    full_embeds, full_attn, _ = splice_prefix(
        model, resampler, batch["ecg_embed"].to(dev), batch["input_ids"].to(dev),
        batch["attention_mask"].to(dev), dev, zero_latents=zero_latents,
        report_label=report_label, task_ids=task_ids.to(dev),
        task_mask=task_mask.to(dev),
    )

    capture.reset()
    out = model.generate(
        inputs_embeds=full_embeds,
        attention_mask=full_attn,
        generation_config=gen_config,
        tokenizer=tokenizer,         # StopStringCriteria needs it to match <END>
        logits_processor=processors,
    )
    # Generating from inputs_embeds returns ONLY the new tokens (no prompt echo).
    return out


@torch.no_grad()
def teacher_forced_batch(model, resampler, batch, dev, zero_latents=False,
                         report_label=None, task_stream: str = "prompt"):
    """Per-example next-token loss on the real target.

    Returns ``[(loss, n_tokens, position_losses)]``, where ``position_losses`` is the
    loss at each target position in order -- index 0 is the target's first token.
    ``loss`` is their mean, so the two are always consistent.

    The batch comes from training's own collate (``src.data.dataset.make_collate_fn``),
    so the prompt is built exactly as generation builds it, the real target and EOS
    follow it, and prompt plus padding are already -100 in ``labels``. The latent
    positions are masked to -100 here as well, leaving the loss on the target alone --
    the identical arrangement scripts/probe_memory.py trains on.

    The loss is reduced per example rather than per batch: a batch mean would weight
    long targets more heavily, and the number reported is a mean over examples.
    """
    labels = batch["labels"].to(dev)
    attn = batch["attention_mask"].to(dev)
    # Only the RESAMPLER's view changes. input_ids below is the whole sequence and
    # full_labels below still supervises every target position, so the teacher
    # forcing is byte-for-byte what it was.
    task_ids, task_mask = task_stream_ids(batch, task_stream)
    full_embeds, full_attn, Lq = splice_prefix(
        model, resampler, batch["ecg_embed"].to(dev), batch["input_ids"].to(dev),
        attn, dev, zero_latents=zero_latents, report_label=report_label,
        task_ids=task_ids.to(dev), task_mask=task_mask.to(dev),
    )
    B = labels.shape[0]
    full_labels = torch.cat(
        [torch.full((B, Lq), -100, dtype=labels.dtype, device=dev), labels], dim=1
    )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(inputs_embeds=full_embeds, attention_mask=full_attn, use_cache=False)

    # position t's logits predict token t+1
    logits = out.logits[:, :-1]
    targets = full_labels[:, 1:]
    per_example = []
    for i in range(B):
        keep = targets[i] != -100
        n = int(keep.sum())
        if n == 0:                       # a target truncated away entirely by --max-length
            per_example.append((None, 0, []))
            continue
        # Upcast only the target positions: the full (B, L, 152k) tensor in fp32
        # would be gigabytes for a number that depends on a few hundred rows.
        per_pos = F.cross_entropy(logits[i][keep].float(), targets[i][keep],
                                  reduction="none")
        # 4 dp keeps the jsonl a few hundred KB at 200 examples instead of megabytes;
        # a bucket mean over thousands of positions is unaffected at that precision.
        per_example.append((float(per_pos.mean()), n,
                            [round(float(x), 4) for x in per_pos]))
    return per_example


def trim_generated(ids: Sequence[int], eos_id: int, pad_id: int) -> List[int]:
    """Cut at the first EOS/pad: sequences that stopped early are padded to batch length."""
    kept = []
    for t in ids:
        t = int(t)
        if t == eos_id or t == pad_id:
            break
        kept.append(t)
    return kept


# --------------------------------------------------------------------------- #
# Run                                                                         #
# --------------------------------------------------------------------------- #


def condition_suffix(shuffle_embeddings: bool, zero_latents: bool,
                     teacher_forced: bool = False) -> str:
    """Filename tag for one condition, so no two conditions share an output file."""
    parts = []
    if shuffle_embeddings:
        parts.append("_shuffled")
    if zero_latents:
        parts.append("_zeroed")
    if teacher_forced:
        parts.append("_teacher_forced")
    return "".join(parts)


def default_out_path(args) -> str:
    suffix = condition_suffix(args.shuffle_embeddings, args.zero_latents,
                              args.teacher_forced_loss)
    return os.path.join(REPO_ROOT, "results", f"eval_{args.split}{suffix}.jsonl")


def read_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def preflight(args, data_cfg):
    """Check every input path BEFORE loading the 14B (slow)."""
    checks = [
        ("model dir", args.model_dir),
        ("model config", os.path.join(args.model_dir, "config.json")),
        ("embedding index", os.path.join(data_cfg.emb_cache_dir, "index.json")),
        ("manifest", data_cfg.manifest_path),
        ("teacher jsonl", data_cfg.teacher_path),
    ]
    ckpt = os.path.join(args.ckpt, "best.pt") if os.path.isdir(args.ckpt) else args.ckpt
    checks.append(("resampler checkpoint", ckpt))
    missing = [f"{name}: {path}" for name, path in checks if not os.path.exists(path)]
    if missing:
        sys.exit("preflight failed, missing input(s):\n  " + "\n  ".join(missing))
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    if not os.access(out_dir, os.W_OK):
        sys.exit(f"preflight failed: output dir not writable: {out_dir}")
    print("preflight OK; all inputs present, output dir writable")


def check_resume(args):
    """What is already in --out, and is it the same experiment?

    One file, one experiment. Appending a shuffled run onto a real one (or a
    different permutation seed, or another split) would quietly mix the control into
    the result it exists to be compared against, and the mixture cannot be undone
    from the metrics afterwards. Runs before anything slow is loaded.
    """
    if not (args.resume and os.path.exists(args.out)):
        return set(), "w"
    prior = read_jsonl(args.out)
    for field, want in (("shuffled_embeddings", bool(args.shuffle_embeddings)),
                        ("zero_latents", bool(args.zero_latents)),
                        ("seed", args.seed), ("split", args.split)):
        have = {r.get(field) for r in prior}
        if have and have != {want}:
            sys.exit(f"--resume refused: {args.out} holds {field}={sorted(have)} "
                     f"but this run has {field}={want}; write to a different --out")
    done = {(r["ecg_id"], r["superclass"]) for r in prior}
    print(f"--resume: {len(done):,} example(s) already in {args.out}, skipping them")
    return done, "a"


def summarise_run(records: List[dict], n_impure: int, elapsed: float) -> None:
    n = len(records)
    unparsed = sum(1 for r in records if r["parsed_conclusion"] is None)
    no_prob = sum(1 for r in records if r["p_yes"] is None)
    stopped = sum(1 for r in records if r["stopped_at_end"])
    lens = [r["n_generated_tokens"] for r in records] or [0]
    print(f"\ngenerated {n} examples in {elapsed/60:.1f} min "
          f"({n/max(elapsed,1e-9):.2f} ex/s)")
    print(f"  generated tokens: mean {np.mean(lens):.0f}  p50 {np.percentile(lens,50):.0f}  "
          f"p95 {np.percentile(lens,95):.0f}  max {int(np.max(lens))}")
    print(f"  closed with {END_MARKER}: {stopped}/{n} ({100*stopped/max(n,1):.1f}%)")
    print(f"  PARSE FAILURES (no readable Conclusion): {unparsed}/{n} "
          f"({100*unparsed/max(n,1):.1f}%)")
    print(f"  no probability read: {no_prob}/{n}; "
          f"Yes/No token merged with following text: {n_impure}")




# Where in the target the loss falls. The opening tokens are the block's fixed
# scaffolding ("Evidence:" and the first words after it), which the student can
# predict from the prompt alone; if the ECG is doing anything, it has to show up
# later, so a flat profile across these buckets is itself a finding.
POSITION_BUCKETS = ((0, 10, "0-9"), (10, 25, "10-24"), (25, 50, "25-49"),
                    (50, 100, "50-99"), (100, None, "100+"))


def summarise_positions(records: List[dict]) -> List[Dict[str, Any]]:
    """Pooled mean loss per target-position bucket.

    Pooled over (example, position) pairs rather than averaging per-example means:
    a bucket a long target barely reaches should not count as much as one every
    target fills. ``n_examples`` says how many targets reach each bucket at all.
    """
    sums = [0.0] * len(POSITION_BUCKETS)
    counts = [0] * len(POSITION_BUCKETS)
    examples = [0] * len(POSITION_BUCKETS)
    for r in records:
        pos = r.get("position_losses")
        if not pos:                      # older jsonl, or a target truncated away
            continue
        arr = np.asarray(pos, dtype=np.float64)
        for k, (lo, hi, _) in enumerate(POSITION_BUCKETS):
            seg = arr[lo:hi] if hi is not None else arr[lo:]
            if seg.size:
                sums[k] += float(seg.sum())
                counts[k] += int(seg.size)
                examples[k] += 1
    return [
        {"positions": label, "n_positions": counts[k], "n_examples": examples[k],
         "mean_loss": (sums[k] / counts[k]) if counts[k] else None}
        for k, (_, _, label) in enumerate(POSITION_BUCKETS)
    ]


def summarise_teacher_forced(records: List[dict]) -> Dict[str, Any]:
    """Mean loss over examples, plus the token-weighted mean and a per-superclass cut."""
    scored = [r for r in records if r["loss"] is not None]
    losses = np.array([r["loss"] for r in scored], dtype=np.float64)
    ntok = np.array([r["n_target_tokens"] for r in scored], dtype=np.float64)
    summary: Dict[str, Any] = {
        "n": len(records),
        "n_scored": len(scored),
        "mean_loss": float(losses.mean()) if len(scored) else None,
        # the batch-mean a training loop reports weights long targets more heavily;
        # both are given so the number can be lined up with either convention
        "token_weighted_mean_loss": (float((losses * ntok).sum() / ntok.sum())
                                     if len(scored) and ntok.sum() > 0 else None),
        "median_loss": float(np.median(losses)) if len(scored) else None,
        "mean_target_tokens": float(ntok.mean()) if len(scored) else None,
        "per_position": summarise_positions(scored),
        "per_superclass": {},
    }
    for sc in SUPERCLASSES:
        vals = [r["loss"] for r in scored if r["superclass"] == sc]
        summary["per_superclass"][sc] = {
            "n": len(vals),
            "mean_loss": float(np.mean(vals)) if vals else None,
        }
    return summary


def print_teacher_forced(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 56)
    print("TEACHER-FORCED LOSS  (next-token loss on the real target)")
    print("=" * 56)
    print(f"  examples scored          {summary['n_scored']:,} of {summary['n']:,}")
    print(f"  MEAN LOSS over examples  {_fmt(summary['mean_loss'], width=7, prec=4)}")
    print(f"  token-weighted mean      {_fmt(summary['token_weighted_mean_loss'], width=7, prec=4)}")
    print(f"  median                   {_fmt(summary['median_loss'], width=7, prec=4)}")
    mtt = summary["mean_target_tokens"]
    print(f"  mean target tokens       {mtt:.0f}" if mtt else "  mean target tokens       n/a")

    rows = summary.get("per_position") or []
    if any(r["n_positions"] for r in rows):
        print("\n  loss by position within the target (pooled over examples)")
        print(f"    {'positions':<10} {'mean loss':>9} {'tokens':>9} {'examples':>9}")
        for r in rows:
            print(f"    {r['positions']:<10} {_fmt(r['mean_loss'], width=9, prec=4)} "
                  f"{r['n_positions']:>9,} {r['n_examples']:>9,}")
        print("    (the first tokens are the block's fixed scaffolding, which the "
              "prompt\n     alone predicts; the ECG can only show up further in)")
    print("  per superclass:")
    for sc, m in summary["per_superclass"].items():
        print(f"    {sc:<5} n {m['n']:>6}  loss {_fmt(m['mean_loss'], width=7, prec=4)}")


# --------------------------------------------------------------------------- #
# The passes. Each takes a loaded context, so a runner can do several of them  #
# on one model load (see scripts/eval_controls.py).                           #
# --------------------------------------------------------------------------- #


@dataclass
class EvalContext:
    """Everything loaded once: the frozen student, the resampler, the readout ids."""

    model: Any
    resampler: Any
    tokenizer: Any
    dev: Any
    eos_id: int
    pad_id: int
    yes_pos: List[int]
    no_pos: List[int]
    capture: Any
    processors: Any
    # What require_strip_think_match found; empty for a context built from_loaded,
    # where the resampler was trained in this process under this config.
    strip_think_check: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, model_dir: str, ckpt: str, tokenizer=None, *, strip_think: bool,
             allow_strip_think_mismatch: bool = False) -> "EvalContext":
        """Load the student and a resampler checkpoint.

        ``strip_think`` is the setting this run builds its prompts with, and is
        required: the checkpoint is checked against it BEFORE the student loads, so
        no script can evaluate a checkpoint without saying which prompt format it is
        using, and a disagreement costs seconds rather than a model load.
        """
        from transformers import AutoTokenizer

        check = require_strip_think_match(ckpt, strip_think, allow_strip_think_mismatch)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

        print("loading frozen student (bf16, device_map=auto) ...")
        model = load_student(model_dir)
        model.config.use_cache = True  # load_student turns it off for the backward pass
        dev = next(model.get_input_embeddings().parameters()).device
        ctx = cls.from_loaded(model, load_resampler(ckpt, dev), tokenizer)
        ctx.strip_think_check = check
        return ctx

    @classmethod
    def from_loaded(cls, model, resampler, tokenizer) -> "EvalContext":
        """A context over a student and resampler already in memory.

        For a caller that has just trained the resampler and wants these passes on it
        without a checkpoint round trip (the overfit gate, scripts/overfit_gate.py).
        Nothing is loaded or modified here.
        """
        from transformers import LogitsProcessorList

        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
        dev = next(model.get_input_embeddings().parameters()).device

        yes_ids, no_ids = yes_no_token_ids(tokenizer)
        cand_ids = yes_ids + no_ids
        print(f"Yes token ids {yes_ids} -> {[tokenizer.decode([i]) for i in yes_ids]}")
        print(f"No  token ids {no_ids} -> {[tokenizer.decode([i]) for i in no_ids]}")
        capture = ConclusionLogitCapture(cand_ids, dev)
        return cls(
            model=model, resampler=resampler, tokenizer=tokenizer, dev=dev,
            eos_id=eos_id, pad_id=pad_id,
            yes_pos=list(range(len(yes_ids))),
            no_pos=list(range(len(yes_ids), len(cand_ids))),
            capture=capture, processors=LogitsProcessorList([capture]),
        )


def load_split(data_cfg: DatasetConfig, split: str):
    """The split, built once. Wrapping it for a control is cheap; this is not."""
    dataset = build_datasets(data_cfg)[split]
    print(f"split {split}: {len(dataset):,} examples "
          f"({len({e.ecg_id for e in dataset.examples}):,} records)")
    print(f"prompt: strip_think={data_cfg.strip_think} (must match the checkpoint's "
          "training run)")
    return dataset


def apply_shuffle(dataset, seed: int):
    """Wrap a split in the embedding permutation, announcing what it did."""
    shuffled = ShuffledEmbeddings(dataset, seed)
    print("*** SHUFFLE-EMBEDDINGS CONTROL ACTIVE ***")
    print(f"    ECG embeddings permuted across {len(shuffled.mapping):,} records "
          f"within the split, SEED {seed}, "
          f"{shuffled.n_fixed_points} record(s) kept their own embedding")
    print("    prompts and labels are untouched: every question is asked about "
          "another record's ECG")
    return shuffled


def base_of(dataset):
    return dataset.base if isinstance(dataset, ShuffledEmbeddings) else dataset


def run_generation(ctx: EvalContext, dataset, indices: Sequence[int], out_path: str, *,
                   split: str, seed: int, shuffled: bool, zero_latents: bool,
                   max_new_tokens: int, batch_size: int, num_workers: int = 4,
                   max_prompt_length: int = 1024, print_first: int = 0,
                   log_every: int = 20, mode: str = "w", task_stream: str = "prompt"):
    """Generate over ``indices``, writing one jsonl line per example as it lands.

    Returns ``(records, n_impure, elapsed)``. Written so a caller can run it several
    times on one loaded context, once per condition. ``task_stream`` is what the
    resampler reads; it has to be the mode the checkpoint trained with.
    """
    from torch.utils.data import DataLoader, Subset

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=make_eval_collate(ctx.tokenizer, max_prompt_length,
                                     strip_think=base_of(dataset).config.strip_think),
    )
    gen_config = build_generation_config(max_new_tokens, ctx.eos_id, ctx.pad_id)
    print(f"greedy decoding, max_new_tokens={max_new_tokens}, stop at {END_MARKER}, "
          f"batch {batch_size}; decoding config set explicitly "
          f"(repetition_penalty {gen_config.repetition_penalty}, "
          f"do_sample {gen_config.do_sample}); resampler task stream {task_stream}"
          + ("  [ZERO-LATENTS CONTROL: the prefix carries nothing]" if zero_latents else ""))

    mismatch = bool(ctx.strip_think_check.get("strip_think_mismatch", False))
    records: List[dict] = []
    n_impure = 0
    n_printed = 0
    t0 = time.time()
    with open(out_path, mode, encoding="utf-8") as fout:
        for bi, batch in enumerate(loader):
            sequences = generate_batch(ctx.model, ctx.resampler, ctx.tokenizer, batch,
                                       ctx.dev, ctx.processors, ctx.capture, gen_config,
                                       zero_latents=zero_latents,
                                       # the prefix-scale check, once per pass
                                       report_label="prompt" if bi == 0 else None,
                                       task_stream=task_stream)
            steps = ctx.capture.steps   # steps[t][b] = logits over cand_ids at step t
            for b in range(sequences.shape[0]):
                gen_ids = trim_generated(sequences[b].tolist(), ctx.eos_id, ctx.pad_id)
                text = ctx.tokenizer.decode(gen_ids, skip_special_tokens=True)
                step, token_text, status = locate_conclusion(ctx.tokenizer, gen_ids)

                p_yes = None
                if step is not None and step < len(steps):
                    p_yes = probability_from_logits(steps[step][b], ctx.yes_pos, ctx.no_pos)
                if status == "impure":
                    n_impure += 1

                rec = {
                    "ecg_id": batch["ecg_id"][b],
                    "superclass": batch["superclass"][b],
                    "strat_fold": batch["strat_fold"][b],
                    "split": split,
                    "label": batch["label"][b],
                    # the same regex that parsed the teacher's targets
                    "parsed_conclusion": teacher_parse._extract_conclusion(text),
                    "p_yes": p_yes,
                    "conclusion_token": token_text,
                    "conclusion_token_status": status,
                    "conclusion_step": step,
                    "n_generated_tokens": len(gen_ids),
                    "stopped_at_end": END_MARKER.lower() in text.lower(),
                    "generation": text,
                    "prompt": batch["prompt"][b],
                    "shuffled_embeddings": bool(shuffled),
                    "zero_latents": bool(zero_latents),
                    # True only under --allow-strip-think-mismatch
                    "strip_think_mismatch": mismatch,
                    "donor_ecg_id": batch["donor_ecg_id"][b] if shuffled else None,
                    "task_stream": task_stream,
                    "seed": seed,
                }
                records.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if n_printed < print_first:
                    n_printed += 1
                    print(f"\n--- sample {n_printed}: ecg {rec['ecg_id']} "
                          f"{rec['superclass']} label={rec['label']} "
                          f"parsed={rec['parsed_conclusion']} "
                          f"p_yes={rec['p_yes'] if rec['p_yes'] is None else round(rec['p_yes'], 4)} ---")
                    print(text)
                    print("--- end sample ---", flush=True)

            fout.flush()
            if log_every and (bi + 1) % log_every == 0:
                done = len(records)
                rate = done / (time.time() - t0)
                left = (len(indices) - done) / max(rate, 1e-9)
                print(f"[{done:>6}/{len(indices)}] {rate:.2f} ex/s  "
                      f"eta {left/60:.1f} min", flush=True)

    elapsed = time.time() - t0
    summarise_run(records, n_impure, elapsed)
    print(f"\nwrote {len(records):,} generations -> {out_path}")
    return records, n_impure, elapsed


def run_teacher_forced(ctx: EvalContext, dataset, indices: Sequence[int], *,
                       split: str, seed: int, shuffled: bool, zero_latents: bool,
                       batch_size: int, num_workers: int = 4, max_length: int = 2048,
                       out_path: Optional[str] = None, log_every: int = 20,
                       task_stream: str = "prompt"):
    """Next-token loss on the real targets. No generation.

    Uses training's collate, so prompt construction, target, EOS and -100 masking are
    the ones the loss was trained under; the only difference is torch.no_grad.
    """
    from torch.utils.data import DataLoader, Subset

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=make_collate_fn(ctx.tokenizer, max_length=max_length,
                                   strip_think=base_of(dataset).config.strip_think),
    )
    print(f"teacher-forced loss over {len(indices):,} examples, batch {batch_size}, "
          f"max_length {max_length}, resampler task stream {task_stream}"
          + ("  [ZERO-LATENTS CONTROL]" if zero_latents else ""))

    mismatch = bool(ctx.strip_think_check.get("strip_think_mismatch", False))
    # Training's collate does not carry donor_ecg_id through, so the donor is read
    # from the wrapper's mapping -- the dict __getitem__ serves the embedding from.
    donors = dataset.mapping if isinstance(dataset, ShuffledEmbeddings) else None
    records: List[dict] = []
    t0 = time.time()
    for bi, batch in enumerate(loader):
        per_example = teacher_forced_batch(ctx.model, ctx.resampler, batch, ctx.dev,
                                           zero_latents=zero_latents,
                                           task_stream=task_stream,
                                           # the prefix-scale check, once per pass
                                           report_label=("prompt+target" if bi == 0
                                                         else None))
        for b, (loss, n_tok, pos_losses) in enumerate(per_example):
            records.append({
                "ecg_id": batch["ecg_id"][b],
                "superclass": batch["superclass"][b],
                "strat_fold": batch["strat_fold"][b],
                "split": split,
                "label": bool(batch["head_label"][b]),
                "loss": loss,
                "n_target_tokens": n_tok,
                # loss at each target position in order; mean == "loss" above
                "position_losses": pos_losses,
                "shuffled_embeddings": bool(shuffled),
                "zero_latents": bool(zero_latents),
                "strip_think_mismatch": mismatch,   # only under the escape hatch
                # None unless the dataset really is the shuffled view, whatever
                # ``shuffled`` says: the flag is a label, the wrapper is the shuffle
                "donor_ecg_id": donors[int(batch["ecg_id"][b])] if donors else None,
                "seed": seed,
            })
        if log_every and (bi + 1) % log_every == 0:
            rate = len(records) / (time.time() - t0)
            print(f"[{len(records):>6}/{len(indices)}] {rate:.2f} ex/s", flush=True)

    elapsed = time.time() - t0
    print(f"\nteacher-forced pass over {len(records):,} examples in {elapsed/60:.1f} min")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote per-example losses -> {out_path}")
    return records, elapsed


# --------------------------------------------------------------------------- #
# The no-target-context decision score ("fast mode")                           #
# --------------------------------------------------------------------------- #


def no_context_text(tokenizer, target):
    """``(context, decision_id, why)``: "Conclusion:" as the target spells it, and the
    token that follows it there.

    ``context`` is the target from the marker up to the decision token's first
    character -- "Conclusion:" itself, whatever the target's spacing. It is checked
    against the full target: tokenised on its own it must be exactly the tokens the
    target has right before the decision token, so a probe asks for the same token
    after the same tokens as teacher forcing and differs only in what came before
    them. ``why`` names the problem when it cannot be built, and is None otherwise.
    """
    ids, offsets = tokenise_target(tokenizer, target)
    k, why = decision_index(target, offsets)
    if why:
        return None, None, why
    context = target[target.index(MARKER):offsets[k][0]]
    own = tokenizer(context, add_special_tokens=False).input_ids
    if not own or ids[k - len(own):k] != own:
        return None, None, (f"{context!r} tokenises on its own as {own}, but the target "
                            f"has {ids[max(0, k - len(own)):k]} before its decision token")
    return context, ids[k], None


@torch.no_grad()
def no_context_records(model, resampler, tokenizer, dev, dataset, indices, *,
                       batch_size: int, max_length: int, task_stream: str, seed: int,
                       shuffled: bool, zero_latents: bool, strict: bool = True,
                       log_every: int = 0, what: str = "no-context decision score"):
    """The decision token after the prompt and the literal "Conclusion:" alone.

    The one implementation of the no-target-context score: the overfit gate's
    condition 4 and its --loss-on decision stop rule (scripts/overfit_gate.py) and
    the control sweep's fast mode (scripts/eval_controls.py) all call this, so they
    cannot drift apart.

    The student reads the ECG prefix, the prompt, then MARKER -- no Evidence, no
    Reasoning, no target text. Each example goes through training's own collate
    with its target REPLACED by MARKER, so the prompt, the template, strip_think, the
    question stream and the -100 masking are training's (an EOS follows, which no
    earlier position can see). The task stream goes through task_stream_ids: under
    ``prompt`` the MARKER tokens are target-side and the resampler does not read
    them, as at inference. The position read is the last MARKER token, whose logits
    predict the answer: p_yes is the logsumexp-pooled Yes against No over
    :func:`yes_no_token_ids`, the readout generation uses.

    The target is used for one thing only, the token the loss is scored against: the
    one after "Conclusion:" in the target, located by decision_token and checked
    by :func:`no_context_text`, and required to follow exactly MARKER as given here.
    When it cannot be, ``strict`` (the gate) is fatal, naming the examples; otherwise
    (an evaluation) the example keeps its p_yes, its ``loss`` is None and
    ``target_problem`` says why. A collate that did not lay the input out as built
    is fatal either way.

    Returns the records in ``indices`` order; writes nothing.
    """
    collate = make_collate_fn(tokenizer, max_length=max_length,
                              strip_think=base_of(dataset).config.strip_think)
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    cand = torch.tensor(yes_ids + no_ids, device=dev)
    yes_pos = list(range(len(yes_ids)))
    no_pos = list(range(len(yes_ids), len(yes_ids) + len(no_ids)))
    n_ctx = len(tokenizer(MARKER, add_special_tokens=False).input_ids)
    donors = dataset.mapping if isinstance(dataset, ShuffledEmbeddings) else None
    records, problems, fatal = [], [], []
    t0 = time.time()
    for bi, start in enumerate(range(0, len(indices), batch_size)):
        items = [dataset[i] for i in indices[start:start + batch_size]]
        decision = []
        for it in items:
            context, decision_id, why = no_context_text(tokenizer, it["target"])
            if why is None and context != MARKER:
                why = f"the target spells the marker {context!r}, not {MARKER!r}"
            if why:
                problems.append(f"  {it['ecg_id']:>7} {it['superclass']:<6} {why}")
            decision.append((None, why) if why else (decision_id, None))
        if strict and problems:
            continue        # fatal below; no forward for a pass that cannot finish
        batch = collate([{**it, "target": MARKER} for it in items])
        task_ids, task_mask = task_stream_ids(batch, task_stream)
        full_embeds, full_attn, Lq = splice_prefix(
            model, resampler, batch["ecg_embed"].to(dev),
            batch["input_ids"].to(dev), batch["attention_mask"].to(dev), dev,
            zero_latents=zero_latents, task_ids=task_ids.to(dev),
            task_mask=task_mask.to(dev))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inputs_embeds=full_embeds, attention_mask=full_attn,
                           use_cache=False).logits
        for b, (it, (decision_id, why)) in enumerate(zip(items, decision)):
            span = (batch["labels"][b] != -100).nonzero().flatten()
            if len(span) != n_ctx + 1:            # MARKER, then the EOS
                fatal.append(f"  {it['ecg_id']:>7} {it['superclass']:<6} the collate "
                             f"laid out {len(span)} target tokens, not {n_ctx} + EOS")
                continue
            row = logits[b, Lq + int(span[n_ctx - 1])].float()
            records.append({
                "ecg_id": int(it["ecg_id"]), "superclass": it["superclass"],
                "strat_fold": int(it["strat_fold"]),
                "label": bool(it["label"]),
                "loss": None if decision_id is None else float(F.cross_entropy(
                    row[None], torch.tensor([decision_id], device=row.device))),
                "p_yes": probability_from_logits(row[cand], yes_pos, no_pos),
                "decision_token": (None if decision_id is None
                                   else tokenizer.decode([decision_id])),
                "target_problem": why,
                "context": MARKER,
                "shuffled_embeddings": bool(shuffled), "zero_latents": bool(zero_latents),
                "donor_ecg_id": donors[int(it["ecg_id"])] if donors else None,
                "task_stream": task_stream,
                "seed": seed,
            })
        if log_every and (bi + 1) % log_every == 0:
            rate = len(records) / max(time.time() - t0, 1e-9)
            left = (len(indices) - len(records)) / max(rate, 1e-9)
            print(f"[{len(records):>6}/{len(indices)}] {rate:.1f} ex/s  "
                  f"eta {left / 60:.1f} min", flush=True)
    if fatal or (strict and problems):
        bad = fatal + (problems if strict else [])
        sys.exit(f"{what} cannot be evaluated for {len(bad)} of {len(indices)} "
                 "examples:\n" + "\n".join(bad))
    if problems:
        print(f"{what}: {len(problems)} of {len(indices)} targets have no locatable "
              "decision token; those examples keep p_yes, with loss None:\n"
              + "\n".join(problems[:10]) + ("\n  ..." if len(problems) > 10 else ""))
    return records


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def build_data_config(args) -> DatasetConfig:
    cfg_kwargs = {"strip_think": bool(args.strip_think)}
    if args.emb_cache:
        cfg_kwargs["emb_cache_dir"] = args.emb_cache
    if args.manifest:
        cfg_kwargs["manifest_path"] = args.manifest
    if args.teacher:
        cfg_kwargs["teacher_path"] = args.teacher
    return DatasetConfig(**cfg_kwargs)


def main():
    args = parse_args()

    # ---- metrics-only path: no GPU, no model ------------------------------
    if args.from_jsonl:
        records = read_jsonl(args.from_jsonl)
        print(f"loaded {len(records):,} records from {args.from_jsonl}")
        n_mismatch = sum(1 for r in records if r.get("strip_think_mismatch"))
        if n_mismatch:
            print(f"WARNING: {n_mismatch:,} of these records were generated under "
                  "--allow-strip-think-mismatch: the prompt was not the one the "
                  "checkpoint trained on, and the metrics below inherit that.")
        table = compute_all_metrics(records)
        print_metrics(table)
        out = args.metrics_out or (os.path.splitext(args.from_jsonl)[0] + ".metrics.json")
        with open(out, "w") as f:
            json.dump({"source": args.from_jsonl, "metrics": table}, f, indent=2)
        print(f"\nwrote {out}")
        return

    if not args.model_dir or not args.ckpt:
        sys.exit("--model-dir and --ckpt are required (or use --from-jsonl)")
    args.out = args.out or default_out_path(args)
    metrics_out = args.metrics_out or (os.path.splitext(args.out)[0] + ".metrics.json")

    data_cfg = build_data_config(args)
    preflight(args, data_cfg)
    args.task_stream = resolve_task_stream(args.ckpt, args.task_stream)
    done_keys, mode = ((set(), "w") if args.teacher_forced_loss else check_resume(args))
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible; generation needs GPUs")
    print(f"visible GPUs: {torch.cuda.device_count()}")
    torch.manual_seed(args.seed)

    # ---- data --------------------------------------------------------------
    dataset = load_split(data_cfg, args.split)
    if args.shuffle_embeddings:
        dataset = apply_shuffle(dataset, args.seed)
    if not args.teacher_forced_loss and args.task_stream == "prompt-and-target":
        print("note: --task-stream prompt-and-target reads the prompt during "
              "generation; there is no target at inference to include")
    if args.zero_latents:
        print("*** ZERO-LATENTS CONTROL ACTIVE ***")
        print("    the resampler still runs; its output is replaced by zeros of the "
              "same shape before the splice, so the 64-position prefix is present "
              "and attendable but carries nothing")

    indices, sample = sample_indices(dataset, args.limit, args.sample_seed, args.stratify)
    sample = print_sample(dataset, indices, sample, "evaluating")

    # ---- teacher-forced loss: no generation --------------------------------
    if args.teacher_forced_loss:
        ctx = EvalContext.load(
            args.model_dir, args.ckpt, strip_think=data_cfg.strip_think,
            allow_strip_think_mismatch=args.allow_strip_think_mismatch)
        records, elapsed = run_teacher_forced(
            ctx, dataset, indices, split=args.split, seed=args.seed,
            shuffled=args.shuffle_embeddings, zero_latents=args.zero_latents,
            batch_size=args.batch_size, num_workers=args.num_workers,
            max_length=args.max_length, out_path=args.out, log_every=args.log_every,
            task_stream=args.task_stream,
        )
        summary = summarise_teacher_forced(records)
        print_teacher_forced(summary)
        with open(metrics_out, "w") as f:
            json.dump({
                "mode": "teacher_forced_loss",
                "split": args.split, "checkpoint": args.ckpt, "model_dir": args.model_dir,
                "shuffle_embeddings": bool(args.shuffle_embeddings),
                "zero_latents": bool(args.zero_latents),
                "strip_think": bool(args.strip_think), **ctx.strip_think_check,
                "task_stream": args.task_stream,
                "seed": args.seed, "limit": args.limit, "sampling": sample,
                "batch_size": args.batch_size,
                "max_length": args.max_length, "losses": args.out,
                "elapsed_seconds": round(elapsed, 1), "teacher_forced": summary,
            }, f, indent=2)
        print(f"wrote {metrics_out}")
        return

    # ---- generation --------------------------------------------------------
    meta = base_of(dataset)
    if done_keys:
        indices = [
            i for i in indices
            if (meta.examples[i].ecg_id, meta.examples[i].superclass) not in done_keys
        ]
        print(f"{len(indices):,} example(s) left to generate")
    if not indices:
        print("nothing to generate; computing metrics from the existing jsonl")
        records = read_jsonl(args.out)
        table = compute_all_metrics(records)
        print_metrics(table)
        with open(metrics_out, "w") as f:
            json.dump({"source": args.out, "n_records": len(records),
                       "metrics": table}, f, indent=2)
        print(f"wrote {metrics_out}")
        return

    ctx = EvalContext.load(
        args.model_dir, args.ckpt, strip_think=data_cfg.strip_think,
        allow_strip_think_mismatch=args.allow_strip_think_mismatch)
    records, n_impure, elapsed = run_generation(
        ctx, dataset, indices, args.out, split=args.split, seed=args.seed,
        shuffled=args.shuffle_embeddings, zero_latents=args.zero_latents,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        num_workers=args.num_workers, max_prompt_length=args.max_prompt_length,
        print_first=args.print_first, log_every=args.log_every, mode=mode,
        task_stream=args.task_stream,
    )

    # ---- metrics -----------------------------------------------------------
    all_records = read_jsonl(args.out) if args.resume else records
    table = compute_all_metrics(all_records)
    print_metrics(table)

    payload = {
        "mode": "generation",
        "split": args.split,
        "checkpoint": args.ckpt,
        "model_dir": args.model_dir,
        "shuffle_embeddings": bool(args.shuffle_embeddings),
        "zero_latents": bool(args.zero_latents),
        "strip_think": bool(args.strip_think), **ctx.strip_think_check,
        "seed": args.seed,
        "limit": args.limit,
        "sampling": sample,
        "max_new_tokens": args.max_new_tokens,
        # what the resampler read; prompt-and-target has no target here, so it read
        # the prompt
        "task_stream": ("prompt" if args.task_stream == "prompt-and-target"
                        else args.task_stream),
        "batch_size": args.batch_size,
        "generations": args.out,
        "n_records": len(all_records),
        "n_impure_conclusion_tokens": n_impure,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": table,
    }
    with open(metrics_out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {metrics_out}")


if __name__ == "__main__":
    main()
