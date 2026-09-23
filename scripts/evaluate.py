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

``--shuffle-embeddings`` is the control that decides whether any of this is about
the waveform. It permutes the cached ECG embeddings across records within the split
(one donor per record, so all five of a record's questions get the same wrong ECG)
and leaves prompts and labels untouched: every question is now asked about someone
else's heart. Metrics that survive that permutation were never coming from the
signal -- they are the label prior plus whatever age/sex leaks. The permutation is a
single cycle over a seeded shuffle, so it has no fixed points, and the seed is
logged and written into every output record.

Every generation, its parsed conclusion, its probability and its label go to jsonl,
so ``--from-jsonl`` recomputes the whole metrics table without touching a GPU.

    # full test split
    python scripts/evaluate.py --model-dir $MODEL_DIR --ckpt $CKPT_DIR/best.pt

    # 500 examples, quick look
    python scripts/evaluate.py --model-dir $MODEL_DIR --ckpt $CKPT_DIR/best.pt --limit 500

    # the control (writes results/eval_test_shuffled.jsonl by default, so it
    # cannot overwrite the run it is meant to be compared against)
    python scripts/evaluate.py --model-dir $MODEL_DIR --ckpt $CKPT_DIR/best.pt \
        --shuffle-embeddings

    # metrics only, no model load
    python scripts/evaluate.py --from-jsonl $EVAL_DIR/test.jsonl

Generation is the cost here: ~200-400 new tokens per example on a 14B sharded over
two A100s. Budget accordingly, use --limit for anything interactive, and --resume to
continue a run that hit its time limit (already-generated examples are skipped).

Compute nodes have no internet: the student and tokenizer load from --model-dir with
local_files_only=True.
"""

import argparse
import json
import os
import sys
import time
from bisect import bisect_right
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for probe_memory

from src.data.dataset import (  # noqa: E402
    DEFAULT_PREFIX_TEXT,
    DatasetConfig,
    _format_prompt,
    build_datasets,
)
from src.model.resampler import PerceiverResamplerConfig, Resampler  # noqa: E402
from src.teacher import parse as teacher_parse  # noqa: E402
from src.teacher.prompt import SUPERCLASSES  # noqa: E402
from probe_memory import load_student  # noqa: E402  -- the same student loader as training

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
                    help="evaluate only the first N examples of the split "
                         "(dataset order: N/5 records, all five superclasses each)")
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
    # the control
    ap.add_argument("--shuffle-embeddings", action="store_true",
                    help="permute ECG embeddings across records within the split "
                         "(prompts and labels untouched): the does-it-use-the-waveform control")
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


def compute_all_metrics(records: Sequence[dict]) -> Dict[str, Dict[str, object]]:
    groups: Dict[str, List[dict]] = {"overall": list(records)}
    for sc in SUPERCLASSES:
        groups[sc] = [r for r in records if r["superclass"] == sc]
    # The two subsets asked for separately: the normality question on its own, and
    # the four abnormal questions pooled.
    groups["NORM (normal subset)"] = [r for r in records if r["superclass"] == "NORM"]
    groups["abnormal subset"] = [r for r in records if r["superclass"] in ABNORMAL]
    return {name: metrics_for(rs) for name, rs in groups.items()}


def _fmt(v, width=6, prec=3):
    if v is None:
        return f"{'n/a':>{width}}"
    return f"{v:>{width}.{prec}f}"


def print_metrics(table: Dict[str, Dict[str, object]]) -> None:
    order = (["overall"] + SUPERCLASSES + ["NORM (normal subset)", "abnormal subset"])
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
    """

    def __init__(self, base, seed: int):
        self.base = base
        ids = sorted({e.ecg_id for e in base.examples})
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


def make_eval_collate(tokenizer, max_prompt_length: int):
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
            prompt_str = _format_prompt(tokenizer, user_content)
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

        return {
            "ecg_embed": ecg_embed,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
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


def load_resampler(ckpt_path: str, dev) -> Resampler:
    """Load the resampler weights from a training checkpoint (or a bare state dict)."""
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "best.pt")
    # our own checkpoint (trusted); it carries numpy/python RNG state that the
    # weights_only=True default (torch>=2.6) refuses to unpickle.
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["resampler"] if isinstance(payload, dict) and "resampler" in payload else payload
    resampler = Resampler(PerceiverResamplerConfig.ecg())
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


@torch.no_grad()
def generate_batch(model, resampler, tokenizer, batch, dev, processors, capture,
                   gen_config):
    """One batch: splice latents in front of the prompt, then greedy-decode to <END>.

    The splice is the training splice (scripts/probe_memory.py splice_forward):
    latents first, then the token embeddings. Left padding means the pads sit between
    the latents and the real prompt; position ids are derived from the attention mask,
    so the real tokens still see the latents at the positions training put them.
    """
    ecg = batch["ecg_embed"].to(dev)
    input_ids = batch["input_ids"].to(dev)
    attn = batch["attention_mask"].to(dev)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tok_embeds = model.get_input_embeddings()(input_ids)
        latents = resampler(ecg, task_embeddings=tok_embeds, task_mask=attn.bool())
    # torch.cat needs one dtype; autocast can hand back fp32 from a LayerNorm-final
    # path, and the student runs in bf16 regardless.
    latents = latents.to(tok_embeds.dtype)

    B, Lq, _ = latents.shape
    full_embeds = torch.cat([latents, tok_embeds], dim=1)
    full_attn = torch.cat(
        [torch.ones(B, Lq, dtype=attn.dtype, device=dev), attn], dim=1
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


def default_out_path(args) -> str:
    name = f"eval_{args.split}{'_shuffled' if args.shuffle_embeddings else ''}.jsonl"
    return os.path.join(REPO_ROOT, "results", name)


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


def main():
    args = parse_args()

    # ---- metrics-only path: no GPU, no model ------------------------------
    if args.from_jsonl:
        records = read_jsonl(args.from_jsonl)
        print(f"loaded {len(records):,} records from {args.from_jsonl}")
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

    cfg_kwargs = {}
    if args.emb_cache:
        cfg_kwargs["emb_cache_dir"] = args.emb_cache
    if args.manifest:
        cfg_kwargs["manifest_path"] = args.manifest
    if args.teacher:
        cfg_kwargs["teacher_path"] = args.teacher
    data_cfg = DatasetConfig(**cfg_kwargs)

    preflight(args, data_cfg)
    done_keys, mode = check_resume(args)
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible; generation needs GPUs")
    print(f"visible GPUs: {torch.cuda.device_count()}")
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    # ---- data --------------------------------------------------------------
    dataset = build_datasets(data_cfg)[args.split]
    print(f"split {args.split}: {len(dataset):,} examples "
          f"({len({e.ecg_id for e in dataset.examples}):,} records)")

    if args.shuffle_embeddings:
        dataset = ShuffledEmbeddings(dataset, args.seed)
        print("*** SHUFFLE-EMBEDDINGS CONTROL ACTIVE ***")
        print(f"    ECG embeddings permuted across {len(dataset.mapping):,} records "
              f"within the split, SEED {args.seed}, "
              f"{dataset.n_fixed_points} record(s) kept their own embedding")
        print("    prompts and labels are untouched: every question is asked about "
              "another record's ECG")

    indices = list(range(len(dataset)))
    if args.limit is not None:
        indices = indices[:args.limit]
        print(f"--limit {args.limit}: evaluating {len(indices):,} examples")

    base_for_meta = dataset.base if isinstance(dataset, ShuffledEmbeddings) else dataset
    if done_keys:
        indices = [
            i for i in indices
            if (base_for_meta.examples[i].ecg_id, base_for_meta.examples[i].superclass)
            not in done_keys
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

    from torch.utils.data import DataLoader, Subset
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_eval_collate(tokenizer, args.max_prompt_length),
    )

    # ---- model -------------------------------------------------------------
    print("loading frozen student (bf16, device_map=auto) ...")
    model = load_student(args.model_dir)
    model.config.use_cache = True   # load_student turns it off for the backward pass
    dev = next(model.get_input_embeddings().parameters()).device
    resampler = load_resampler(args.ckpt, dev)

    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    cand_ids = yes_ids + no_ids
    yes_pos = list(range(len(yes_ids)))
    no_pos = list(range(len(yes_ids), len(cand_ids)))
    print(f"Yes token ids {yes_ids} -> {[tokenizer.decode([i]) for i in yes_ids]}")
    print(f"No  token ids {no_ids} -> {[tokenizer.decode([i]) for i in no_ids]}")
    capture = ConclusionLogitCapture(cand_ids, dev)
    from transformers import LogitsProcessorList
    processors = LogitsProcessorList([capture])

    gen_config = build_generation_config(args.max_new_tokens, eos_id, pad_id)
    print(f"greedy decoding, max_new_tokens={args.max_new_tokens}, stop at "
          f"{END_MARKER}, batch {args.batch_size}; decoding config set explicitly "
          f"(repetition_penalty {gen_config.repetition_penalty}, "
          f"do_sample {gen_config.do_sample})")

    # ---- generate ----------------------------------------------------------
    records: List[dict] = []
    n_impure = 0
    n_printed = 0
    t0 = time.time()
    with open(args.out, mode, encoding="utf-8") as fout:
        for bi, batch in enumerate(loader):
            sequences = generate_batch(model, resampler, tokenizer, batch, dev,
                                       processors, capture, gen_config)
            steps = capture.steps      # steps[t][b] = logits over cand_ids at step t
            for b in range(sequences.shape[0]):
                gen_ids = trim_generated(sequences[b].tolist(), eos_id, pad_id)
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                step, token_text, status = locate_conclusion(tokenizer, gen_ids)

                p_yes = None
                if step is not None and step < len(steps):
                    p_yes = probability_from_logits(steps[step][b], yes_pos, no_pos)
                if status == "impure":
                    n_impure += 1

                rec = {
                    "ecg_id": batch["ecg_id"][b],
                    "superclass": batch["superclass"][b],
                    "strat_fold": batch["strat_fold"][b],
                    "split": args.split,
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
                    "shuffled_embeddings": bool(args.shuffle_embeddings),
                    "donor_ecg_id": batch["donor_ecg_id"][b] if args.shuffle_embeddings else None,
                    "seed": args.seed,
                }
                records.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if n_printed < args.print_first:
                    n_printed += 1
                    print(f"\n--- sample {n_printed}: ecg {rec['ecg_id']} "
                          f"{rec['superclass']} label={rec['label']} "
                          f"parsed={rec['parsed_conclusion']} "
                          f"p_yes={rec['p_yes'] if rec['p_yes'] is None else round(rec['p_yes'], 4)} ---")
                    print(text)
                    print("--- end sample ---", flush=True)

            fout.flush()
            if (bi + 1) % args.log_every == 0:
                done = len(records)
                rate = done / (time.time() - t0)
                left = (len(indices) - done) / max(rate, 1e-9)
                print(f"[{done:>6}/{len(indices)}] {rate:.2f} ex/s  "
                      f"eta {left/60:.1f} min", flush=True)

    elapsed = time.time() - t0
    summarise_run(records, n_impure, elapsed)
    print(f"\nwrote {len(records):,} generations -> {args.out}")

    # ---- metrics -----------------------------------------------------------
    all_records = read_jsonl(args.out) if args.resume else records
    table = compute_all_metrics(all_records)
    print_metrics(table)

    payload = {
        "split": args.split,
        "checkpoint": args.ckpt,
        "model_dir": args.model_dir,
        "shuffle_embeddings": bool(args.shuffle_embeddings),
        "seed": args.seed,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
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
