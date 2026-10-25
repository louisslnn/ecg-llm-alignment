#!/usr/bin/env python
"""The control sweep: three conditions, scored three ways, in one job.

This is the run that says whether a trained resampler is doing anything at all. It
loads the 14B and the resampler ONCE and then makes up to nine passes, so the answer
arrives in a single queue slot instead of nine:

    real                the model as trained
    shuffled            every record gets another record's ECG (--shuffle-embeddings)
    zeroed              the prefix is spliced as zeros (--zero-latents)

each of those three times -- in FAST MODE (below, over the whole split by default),
teacher-forced on the real targets, and generating (over a smaller draw, since a
decode costs far more than a forward pass). They run in that order, cheapest first,
so a job that runs out of time mid-decode has already written the rest. Running
the teacher-forced pass under all three conditions is the part that makes the
losses comparable: a single real-only number says nothing without the shuffled one
beside it, and the zeroed one is the floor.

THE TEACHER-FORCED TABLE reports the mean and the per-position buckets (0-9, 10-24,
25-49, 50-99, 100+) for each condition, the real-vs-shuffled percentage gap, and a
reference BLOCK of run 1's own numbers, row for row -- measured before the
resampler's task stream was narrowed to the prompt, when the target sat inside the
stream the resampler reads. The two blocks print one above the other with the same
row labels, each with its own per-bucket gap, so the comparison is row by row rather
than mean against mean. See --task-stream, and TF_CONTAMINATED below.

All three conditions must cover the same examples in the same order, which is
asserted rather than assumed: the whole table is a difference between columns, and
a differing sample would make it meaningless while looking perfectly normal.

WHICH EXAMPLES. A limited pass is a seeded draw (--sample-seed), never the first N
of the split: the split is ordered by record and fold 10's head is heavily normal,
which once gave a 20-example NORM draw a 0.800 positive rate against the fold's
0.415 and HYP no positives at all. --stratify superclass-label (the default) takes
N/5 per superclass, half Yes and half No; superclass and none are the other two.
Each draw is printed with its positive rate per superclass beside the split's, and
recorded in the summary JSON under "sampling". --seed is only the embedding
permutation's. A balanced draw sits at a ~0.5 positive rate by construction, so its
precision and AUPRC are at that prevalence, not the split's.

FAST MODE is the decision read the way inference should read it: the ECG prefix,
the prompt, then the literal "Conclusion:" -- no Evidence, no Reasoning, no target
text -- and p(Yes) at the next position over the Yes/No token ids generation's
readout uses. One forward per example, so it covers the whole split by default
(--fast-limit to draw). It is evaluate.no_context_records, the same function the
overfit gate's condition 4 and its stop rule call, so the gate and this sweep cannot
measure two different things. Reported per superclass and for the normal and
abnormal subsets in the METRICS table's rows and columns, F1 both at p_yes >= 0.5
and at each row's own best-F1 threshold (in-sample, and labelled so), and written
per example to {tag}_{split}_fast_{condition}.jsonl. On the examples a condition
both generated and scored fast, the two verdicts are compared: a disagreement is
the written reasoning arriving at the opposite of the model's own one-token answer.

The three generation passes write their own jsonl (so they can be re-read, diffed or
re-scored later with `evaluate.py --from-jsonl`), and the run ends with a comparison
table: mean generated length, the fraction of generations that closed with <END>,
the Yes rate, and -- the number to look at first -- how many generations are
BYTE-IDENTICAL across the three conditions.

How to read that last column. If real and shuffled produce the same text for most
examples, the model is not reading the waveform: it is answering from the question,
the age and the sex. If real and zeroed also match, the prefix is not being used at
all and the resampler may as well not be there. Identical text across conditions is
the honest failure signal that per-condition metrics can hide, because three
conditions can each post a respectable F1 while writing the same sentence.

Each pass also prints, for its first batch, the norm of the spliced latents against
the mean norm of the token embeddings they sit next to. Read that before the table:
if the prefix is an order of magnitude shorter than the tokens, it draws almost no
attention weight, and identical generations across conditions are explained by scale
alone -- no amount of training signal in the latents will show up downstream.

LINEAR HEAD. A checkpoint trained with --lambda-linear carries the head, and it is
loaded with the resampler (a payload and meta that disagree on it are fatal). It reads
the same prefix fast mode's student does (evaluate.linear_head_records), so it is
scored on the fast-mode draw and on the generation draw, under each condition, and
the run ends with one table per draw and condition: AUPRC for generation, fast mode
and the head side by side, on the examples every one of them scored, with the mean
|P(Yes) - P_linear(Yes)| of each readout against the head. The linear loss trains
the head and the consistency loss is meant to pull the LLM toward it; this is where
it shows whether it did. Per example to {tag}_{split}_linear_{draw}_{condition}.jsonl.

Generation and teacher forcing default small -- 50 examples per generation
condition at batch 1, 200 per teacher-forced condition -- because they are the
diagnostic; fast mode is the evaluation, and covers the split.
--skip-zeroed-generation drops the zeroed decode, which writes max_new_tokens of
unparseable text for a known 100% failure rate (~43 min per 100 examples).
Batch 1 keeps each condition's decode independent of how the batch happened to be
padded, so a byte-identical comparison means what it says.

    python scripts/eval_controls.py --model-dir $MODEL_DIR --ckpt $CKPT_DIR/best.pt
    python scripts/eval_controls.py ... --limit 100 --tf-limit 500 --out-dir $EVAL_DIR

Compute nodes have no internet: everything loads from --model-dir with
local_files_only=True. See scripts/eval_controls.sh to submit it.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_STRATIFY,
    FAST_THRESHOLD,
    POSITION_BUCKETS,
    STRATIFY_MODES,
    EvalContext,
    apply_shuffle,
    auprc,
    build_data_config,
    checkpoint_objective,
    compute_all_metrics,
    compute_fast_metrics,
    linear_head_records,
    load_linear_head,
    load_split,
    no_context_records,
    print_fast_metrics,
    print_metrics,
    print_sample,
    print_teacher_forced,
    resolve_task_stream,
    run_generation,
    run_teacher_forced,
    sample_indices,
    summarise_teacher_forced,
)
from src.data.dataset import TASK_STREAM_MODES, DatasetConfig  # noqa: E402
from src.teacher.prompt import SUPERCLASSES  # noqa: E402

# The three conditions, in the order they are run and reported. Used for both the
# generation sweep and the teacher-forced sweep, so the two cannot disagree on what
# "shuffled" or "zeroed" means.
CONDITIONS = [
    ("real", dict(shuffled=False, zero_latents=False)),
    ("shuffled", dict(shuffled=True, zero_latents=False)),
    ("zeroed", dict(shuffled=False, zero_latents=True)),
]

# The bucket labels the measured block uses, taken from evaluate.py so the two
# cannot drift; the reference block below is keyed by these same strings.
POSITION_BUCKET_LABELS = tuple(label for _, _, label in POSITION_BUCKETS)

# Run 1's teacher-forced numbers, measured before the resampler's task stream was
# narrowed to the prompt -- i.e. with the target, conclusion included, inside the
# stream the resampler reads. Carried here as a reference BLOCK, row for row against
# the measured one, so the comparison is per bucket rather than mean against mean.
#
# Contaminated is the right word: a resampler that can read "Conclusion: Yes" has no
# need of the waveform, so the real-vs-shuffled gap these rows show is not evidence
# about the ECG. Note where that gap lives -- the 0-9 bucket, real 0.064 against
# shuffled 0.234, is the leak at its most visible: with the answer in the stream the
# opening tokens were nearly free, and they stop being free the moment the ECG is
# the wrong one. A prompt-only run has no such mechanism available, so a large 0-9
# gap there would mean something quite different.
#
# Keyed by bucket label. If POSITION_BUCKETS is ever rebucketed these keys no longer
# describe the same spans, which _reference_rows detects rather than silently
# printing numbers against the wrong rows.
TF_CONTAMINATED = {
    "mean":   {"real": 0.2966, "shuffled": 0.3381, "zeroed": 3.2977},
    "0-9":    {"real": 0.064,  "shuffled": 0.234,  "zeroed": 5.075},
    "10-24":  {"real": 0.391,  "shuffled": 0.466,  "zeroed": 3.835},
    "25-49":  {"real": 0.365,  "shuffled": 0.422,  "zeroed": 3.509},
    "50-99":  {"real": 0.344,  "shuffled": 0.361,  "zeroed": 2.584},
    "100+":   {"real": 0.252,  "shuffled": 0.266,  "zeroed": 3.247},
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="local student checkpoint dir")
    ap.add_argument("--ckpt", required=True,
                    help="resampler checkpoint (.pt file, or a dir holding best.pt)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=50,
                    help="examples per generation condition (default 50), drawn by "
                         "--stratify / --sample-seed")
    ap.add_argument("--tf-limit", type=int, default=200,
                    help="examples for the teacher-forced loss pass (default 200), "
                         "drawn the same way")
    ap.add_argument("--fast-limit", type=int, default=None,
                    help="examples for fast mode, the no-target-context decision "
                         "score (default: the whole split -- one forward per "
                         "example, ~10 min per condition for test)")
    ap.add_argument("--fast-batch-size", type=int, default=16,
                    help="batch size for fast mode (right-padded, one forward, no "
                         "decode)")
    ap.add_argument("--skip-fast", action="store_true", help="leave fast mode out")
    ap.add_argument("--sample-seed", type=int, default=42,
                    help="seed for every limited draw (generation, teacher-forced, "
                         "fast); separate from --seed, the embedding permutation's")
    ap.add_argument("--stratify", choices=STRATIFY_MODES, default=DEFAULT_STRATIFY,
                    help="how a limited pass draws its examples: 'superclass-label' "
                         "(default) is N/5 per superclass, half Yes and half No; "
                         "'superclass' N/5 per superclass, uniform within; 'none' "
                         "uniform over the split. Never the head of the split")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="generation batch size; 1 keeps each condition's decode "
                         "independent of batch padding, which the byte-identical "
                         "comparison depends on")
    ap.add_argument("--tf-batch-size", type=int, default=4,
                    help="batch size for the teacher-forced pass (no generation, so "
                         "it can be larger)")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--max-length", type=int, default=2048,
                    help="prompt+target cap for the teacher-forced pass")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results"),
                    help="where the jsonl files and the summary json go")
    ap.add_argument("--tag", default="controls",
                    help="filename prefix for this sweep's outputs")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the embedding permutation; logged and recorded. "
                         "Which examples are drawn is --sample-seed's")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--print-first", type=int, default=1,
                    help="print the first N generations of each condition in full")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--task-stream", choices=TASK_STREAM_MODES, default=None,
                    help="what the RESAMPLER's task-text stream reads, in both "
                         "sweeps. Default: the mode the checkpoint recorded, or "
                         "'prompt' for one that predates the record. "
                         "'prompt-and-target' reproduces run 1 and is what produced "
                         "the contaminated reference numbers; generation has no "
                         "target, so there it reads the prompt. 'question' is the "
                         "question paraphrase alone, 'none' no text")
    ap.add_argument("--skip-teacher-forced", action="store_true")
    ap.add_argument("--skip-zeroed-generation", action="store_true",
                    help="leave out the zeroed condition's GENERATION pass (its "
                         "teacher-forced and fast passes still run): with nothing in "
                         "the prefix it writes max_new_tokens of unparseable text, "
                         "~43 min per 100 examples for a known 100%% failure rate")
    ap.add_argument("--strip-think", action=argparse.BooleanOptionalAction,
                    default=DatasetConfig.strip_think,
                    help="MUST match what the checkpoint trained with. ON by default; "
                         "pass --no-strip-think for a checkpoint trained with the "
                         "template's open <think> block, as runs 1 and 2 were")
    ap.add_argument("--allow-strip-think-mismatch", action="store_true",
                    help="run even though --strip-think disagrees with the value the "
                         "checkpoint was trained with. Without this the script "
                         "refuses; with it, it warns and marks the output, because "
                         "the student is then scored on a prompt it never trained on")
    # data path overrides (default to DatasetConfig)
    ap.add_argument("--emb-cache", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--teacher", default=None)
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# The comparison                                                              #
# --------------------------------------------------------------------------- #


def condition_summary(records):
    """The per-condition row: length, closure, Yes rate, parse failures."""
    n = len(records)
    parsed = [r for r in records if r["parsed_conclusion"] is not None]
    lens = [r["n_generated_tokens"] for r in records] or [0]
    return {
        "n": n,
        "mean_generated_tokens": float(np.mean(lens)),
        "median_generated_tokens": float(np.median(lens)),
        "frac_closed_with_end": (sum(1 for r in records if r["stopped_at_end"]) / n) if n else None,
        "n_parsed": len(parsed),
        "n_unparsed": n - len(parsed),
        # Yes rate is over generations that produced a readable conclusion; with
        # unparsed ones in the denominator it would move for the wrong reason.
        "yes_rate": (sum(1 for r in parsed if r["parsed_conclusion"]) / len(parsed)
                     if parsed else None),
        "mean_p_yes": (float(np.mean([r["p_yes"] for r in records if r["p_yes"] is not None]))
                       if any(r["p_yes"] is not None for r in records) else None),
        "positive_rate": (float(np.mean([bool(r["label"]) for r in records])) if n else None),
    }


def identical_counts(by_condition):
    """How many examples got byte-identical text across conditions.

    Compared per (ecg_id, superclass) over the examples every condition covered, so a
    condition that generated fewer examples cannot inflate or deflate the count.
    """
    names = list(by_condition)
    keyed = {
        name: {(r["ecg_id"], r["superclass"]): r["generation"] for r in recs}
        for name, recs in by_condition.items()
    }
    common = set.intersection(*(set(k) for k in keyed.values())) if keyed else set()
    # "all three" is every condition that generated; two under --skip-zeroed-generation
    out = {"n_compared": len(common), "identical_all_three": 0, "conditions": names,
           "pairwise": {}}
    for key in common:
        if len({keyed[n][key] for n in names}) == 1:
            out["identical_all_three"] += 1
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out["pairwise"][f"{a} vs {b}"] = sum(
                1 for key in common if keyed[a][key] == keyed[b][key]
            )
    return out


def example_sequence(records):
    """The (ecg_id, superclass) keys in the order the records were produced."""
    return [(r["ecg_id"], r["superclass"]) for r in records]


def assert_same_examples(by_condition, what: str):
    """Fatal unless every condition covered the same examples in the same order.

    The whole sweep is a difference between conditions, so a differing sample makes
    every column incomparable -- and the failure would be invisible in the table,
    which shows only aggregates. Order is checked too, not just membership: the
    per-position buckets and the per-example records are lined up positionally
    downstream, so a reordering would silently pair one record's loss with another's.
    """
    seqs = {name: example_sequence(recs) for name, recs in by_condition.items()}
    names = list(seqs)
    if not names:
        return []
    ref_name, ref = names[0], seqs[names[0]]
    for name in names[1:]:
        other = seqs[name]
        if len(other) != len(ref):
            sys.exit(f"{what}: condition '{name}' covered {len(other)} examples but "
                     f"'{ref_name}' covered {len(ref)}; the conditions are not "
                     "comparable. Check --limit/--tf-limit and the split.")
        for i, (a, b) in enumerate(zip(ref, other)):
            if a != b:
                sys.exit(f"{what}: condition '{name}' diverges from '{ref_name}' at "
                         f"position {i}: {b} vs {a}. Same length, different order -- "
                         "the per-example comparison would pair unrelated records.")
    print(f"  {what}: all {len(names)} conditions cover the same {len(ref)} examples "
          "in the same order")
    return ref


def _pct_gap(real, shuffled):
    """Shuffled relative to real, in percent. Positive = shuffling made it worse."""
    if real in (None, 0) or shuffled is None:
        return None
    return 100.0 * (shuffled - real) / real


def _reference_rows():
    """The contaminated block's rows, in the measured block's row order.

    Returns ``(rows, warning)``. The reference numbers are hardcoded against a
    specific bucketing, so a bucket the reference does not know about comes back as
    None with a warning rather than being quietly omitted or, worse, filled from the
    wrong span.
    """
    rows = [("mean", TF_CONTAMINATED.get("mean", {}))]
    unknown = []
    for label in POSITION_BUCKET_LABELS:
        if label in TF_CONTAMINATED:
            rows.append((f"positions {label}", TF_CONTAMINATED[label]))
        else:
            unknown.append(label)
            rows.append((f"positions {label}", {}))
    warning = None
    if unknown:
        warning = (f"the reference block has no numbers for bucket(s) "
                   f"{', '.join(unknown)}: POSITION_BUCKETS has been changed since "
                   "those were measured, so they cannot be compared row for row")
    return rows, warning


def _tf_block(title, rows, names, label_w, line_w):
    """One block: a titled rule, then a row per label with the real-vs-shuffled gap."""
    # Truncate rather than overflow: a title longer than the table swallows the rule
    # and the block stops looking like a block.
    rule = f"--- {title} "
    print(rule.ljust(line_w, "-") if len(rule) < line_w
          else rule[:line_w - 4] + " ---")
    out = []
    for label, values in rows:
        gap = _pct_gap(values.get("real"), values.get("shuffled"))
        cells = "".join(_fmt(values.get(n), width=11, prec=4) for n in names)
        gap_cell = f"{gap:>+13.1f}%" if gap is not None else f"{'n/a':>14}"
        indent = "" if label == "mean" else "  "
        print(f"{indent + label:<{label_w}}{cells}{gap_cell}")
        out.append({"row": label, "values": {n: values.get(n) for n in names},
                    "real_vs_shuffled_pct": gap})
    return out


def print_teacher_forced_comparison(tf_summaries, task_stream):
    """Two blocks, same row labels, each with its own per-bucket gap.

    Stacked rather than interleaved so a row reads down the page: the measured
    number, then what the same row was when the target sat inside the resampler's
    stream. The gap column is what to compare -- the absolute losses moved for a
    second reason (the resampler is reading a different stream), but the gap is
    within-block and says how much each configuration depended on getting the right
    ECG.
    """
    names = [n for n, _ in CONDITIONS if n in tf_summaries]
    if not names:
        return None
    label_w = 22
    line_w = label_w + 11 * len(names) + 14
    header = (f"{'':<{label_w}}" + "".join(f"{n:>11}" for n in names)
              + f"{'real vs shuf':>14}")

    print("\n" + "=" * line_w)
    print("TEACHER-FORCED LOSS BY CONDITION")
    print("=" * line_w)
    print(header)

    measured_rows = [("mean", {n: tf_summaries[n]["mean_loss"] for n in names})]
    for k, label in enumerate(POSITION_BUCKET_LABELS):
        vals = {}
        for n in names:
            per_pos = tf_summaries[n].get("per_position") or []
            vals[n] = per_pos[k]["mean_loss"] if k < len(per_pos) else None
        measured_rows.append((f"positions {label}", vals))

    measured_title = f"measured now, --task-stream {task_stream}"
    measured = _tf_block(measured_title, measured_rows, names, label_w, line_w)

    reference_rows, warning = _reference_rows()
    reference = _tf_block("run 1 reference, contaminated (target in the stream)",
                          reference_rows, names, label_w, line_w)
    print("-" * line_w)

    if warning:
        print(f"  WARNING: {warning}")
    if task_stream == "prompt-and-target":
        print("  NOTE: this run used the same contaminated stream as the reference, so")
        print("        the two blocks should agree; treat a difference as a change in")
        print("        the data or the checkpoint, not as a finding about the ECG.")

    support = (tf_summaries[names[0]].get("per_position") or [])
    if support:
        print("  bucket support (real): " + "  ".join(
            f"{r['positions']} n={r['n_positions']:,}/{r['n_examples']}ex"
            for r in support))
    print("  examples scored: " + "  ".join(
        f"{n} {tf_summaries[n]['n_scored']:,}" for n in names))

    print("\n  Read the gap column down, row against row. In the reference block it")
    print("  peaks at positions 0-9 (+266%): with the answer inside the stream the")
    print("  opening tokens were nearly free, and only there did the wrong ECG cost")
    print("  anything. That mechanism is gone in a prompt-only run, so a gap in the")
    print("  measured block is about the waveform rather than about the leak.")
    print("  zeroed is the floor: no prefix information at all. real sitting at the")
    print("  zeroed loss means the prefix carries nothing usable; real sitting at the")
    print("  shuffled loss means it carries something that is not this record's ECG.")
    return {"task_stream": task_stream,
            "measured": {"title": measured_title, "rows": measured},
            "reference": {"title": "run 1, contaminated", "rows": reference,
                          "warning": warning}}


def _fmt(v, width=8, prec=3):
    return f"{'n/a':>{width}}" if v is None else f"{v:>{width}.{prec}f}"


def print_comparison(summaries, identical, tf_summary):
    names = list(summaries)
    name_w = max(max(len(n) for n in names), 9)
    header = (f"{'condition':<{name_w}}  {'n':>5} {'mean_len':>8} {'end%':>8} "
              f"{'yes_rate':>8} {'unparsed':>8} {'mean_p':>8} {'pos_rate':>8}")
    print("\n" + "=" * len(header))
    print("CONTROL SWEEP")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name in names:
        s = summaries[name]
        print(f"{name:<{name_w}}  {s['n']:>5} {s['mean_generated_tokens']:>8.1f} "
              f"{_fmt(s['frac_closed_with_end'])} {_fmt(s['yes_rate'])} "
              f"{s['n_unparsed']:>8} {_fmt(s['mean_p_yes'])} {_fmt(s['positive_rate'])}")
    print("-" * len(header))
    print(f"\nbyte-identical generations over {identical['n_compared']} shared examples:")
    print(f"  identical across ALL {len(identical['conditions'])} conditions: "
          f"{identical['identical_all_three']}"
          f"  ({100 * identical['identical_all_three'] / max(identical['n_compared'], 1):.1f}%)")
    for pair, count in identical["pairwise"].items():
        print(f"  {pair:<24} {count:>5}"
              f"  ({100 * count / max(identical['n_compared'], 1):.1f}%)")
    print("\n  real vs shuffled identical => the answer does not depend on WHICH ECG.")
    print("  real vs zeroed   identical => the answer does not depend on the prefix AT ALL.")
    if tf_summary and tf_summary["mean_loss"] is not None:
        print(f"\nteacher-forced loss on {tf_summary['n_scored']} examples: "
              f"mean over examples {tf_summary['mean_loss']:.4f}, "
              f"token-weighted {tf_summary['token_weighted_mean_loss']:.4f}")


def fast_generation_agreement(fast_records, gen_records):
    """Fast mode's Yes/No against the generated Conclusion, per shared example.

    Fast says Yes when p_yes >= FAST_THRESHOLD; generation says what its parsed
    Conclusion says. Compared on the (ecg_id, superclass) pairs both passes covered,
    over the generations that parsed -- an unparsed one has no verdict to agree with,
    and is counted instead. A disagreement is the model's written reasoning arriving
    at the opposite of its own one-token answer. Pure on the records.
    """
    fast = {(r["ecg_id"], r["superclass"]): r for r in fast_records}
    shared = [g for g in gen_records if (g["ecg_id"], g["superclass"]) in fast]
    parsed = [g for g in shared if g["parsed_conclusion"] is not None]

    def tally(gens):
        cells = {"fast Yes, gen Yes": 0, "fast Yes, gen No": 0,
                 "fast No, gen Yes": 0, "fast No, gen No": 0}
        for g in gens:
            f = fast[(g["ecg_id"], g["superclass"])]["p_yes"] >= FAST_THRESHOLD
            cells[f"fast {'Yes' if f else 'No'}, gen "
                  f"{'Yes' if g['parsed_conclusion'] else 'No'}"] += 1
        agree = cells["fast Yes, gen Yes"] + cells["fast No, gen No"]
        return {"n": len(gens), "agree": agree,
                "agreement": agree / len(gens) if gens else None, **cells}

    return {"n_shared": len(shared), "n_unparsed": len(shared) - len(parsed),
            **tally(parsed),
            "per_superclass": {sc: tally([g for g in parsed if g["superclass"] == sc])
                               for sc in SUPERCLASSES}}


def print_agreement(agreement):
    """One row per condition that has both a fast and a generation pass."""
    if not agreement:
        return
    header = (f"{'condition':<10} {'shared':>6} {'unpars':>6} {'agree':>11} "
              f"{'fYes/gNo':>9} {'fNo/gYes':>9}   "
              + " ".join(f"{sc:>5}" for sc in SUPERCLASSES))
    print("\n" + "=" * len(header))
    print(f"FAST MODE vs GENERATED VERDICT  (fast: p_yes >= {FAST_THRESHOLD:g}; "
          "generated: the parsed Conclusion)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, a in agreement.items():
        rate = "n/a" if a["agreement"] is None else f"{a['agreement']:.1%}"
        per = " ".join(f"{_fmt(a['per_superclass'][sc]['agreement'], width=5, prec=2)}"
                       for sc in SUPERCLASSES)
        agree = f"{a['agree']}/{a['n']} {rate}"
        print(f"{name:<10} {a['n_shared']:>6} {a['n_unparsed']:>6} {agree:>11} "
              f"{a['fast Yes, gen No']:>9} {a['fast No, gen Yes']:>9}   {per}")
    print("-" * len(header))
    print("agreement over the generations that parsed; per-superclass columns are the "
          "same rate within each.\nA disagreement is the written reasoning reaching "
          "the opposite of the model's own one-token answer.")


def readout_comparison(readouts, n_draw=None):
    """AUPRC of each readout on the examples EVERY readout scored, by group.

    ``readouts`` is an ordered ``{name: (records, score_key)}``; the last entry is
    the linear head, which every other readout's mean |P - P_linear| is taken
    against. An example counts once every readout has a non-None score for its
    (ecg_id, superclass) -- a generation with no readable conclusion has none -- so
    the columns are on one set of examples and differ only in the readout. Groups
    are "overall" and each superclass. Pure on the records.
    """
    names = list(readouts)
    scores = {name: {(r["ecg_id"], r["superclass"]): r[key] for r in recs}
              for name, (recs, key) in readouts.items()}
    labels = {(r["ecg_id"], r["superclass"]): bool(r["label"])
              for recs, _ in readouts.values() for r in recs}
    first = readouts[names[0]][0]
    shared = [(r["ecg_id"], r["superclass"]) for r in first
              if all(scores[n].get((r["ecg_id"], r["superclass"])) is not None
                     for n in names)]
    linear = names[-1]
    rows = {}
    for group in ["overall", *SUPERCLASSES]:
        keys = [k for k in shared if group == "overall" or k[1] == group]
        y = np.array([labels[k] for k in keys], dtype=bool)
        rows[group] = {"n": len(keys),
                       "positive_rate": float(y.mean()) if keys else None,
                       **{n: (auprc(y, np.array([scores[n][k] for k in keys]))
                              if keys else None) for n in names}}
    gaps = {n: (float(np.mean([abs(scores[n][k] - scores[linear][k]) for k in shared]))
                if shared else None) for n in names[:-1]}
    return {"readouts": names, "n": len(shared),
            "n_draw": len(first) if n_draw is None else n_draw,
            "rows": rows, "mean_abs_gap_to_linear": gaps}


def print_readout_comparison(title, comparison):
    names = comparison["readouts"]
    header = (f"{'group':<9} {'n':>6} {'pos':>6}  "
              + " ".join(f"{n:>11}" for n in names))
    print(f"\n--- {title}: {comparison['n']} of {comparison['n_draw']} examples, "
          "every readout scored ".ljust(len(header), "-"))
    print(header)
    for group, row in comparison["rows"].items():
        print(f"{group:<9} {row['n']:>6} {_fmt(row['positive_rate'], width=6)}  "
              + " ".join(_fmt(row[n], width=11) for n in names))
    print("mean |P(Yes) - P_linear(Yes)|: " + "  ".join(
        f"{n} {_fmt(v, width=0)}" for n, v in comparison["mean_abs_gap_to_linear"].items()))


def run_linear(ctx, head, base, shuffled_view, indices, conditions, draw, args, paths):
    """The linear head's P(Yes) on ``indices`` under each condition, written to
    {tag}_{split}_linear_{draw}_{name}.jsonl; ``{name: records}``. The resampler
    reads the stream fast mode gives it, prompt-and-target read as prompt."""
    stream = "prompt" if args.task_stream == "prompt-and-target" else args.task_stream
    out = {}
    for name, opts in conditions:
        path = os.path.join(args.out_dir, f"{args.tag}_{args.split}_linear_{draw}_{name}.jsonl")
        paths[f"linear_{draw}_{name}"] = path
        records = linear_head_records(
            ctx.model, ctx.resampler, head, ctx.tokenizer, ctx.dev,
            shuffled_view if opts["shuffled"] else base, indices,
            batch_size=args.fast_batch_size, max_length=args.max_length,
            task_stream=stream, seed=args.seed, **opts)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                r.update(split=args.split)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"linear head, {draw} draw, {name}: {len(records):,} examples -> {path}",
              flush=True)
        out[name] = records
    return out


def fast_on(records_by_condition, indices, base, ctx, shuffled_view, conditions, args):
    """Fast-mode records for exactly ``indices``, per condition: taken from the
    fast pass when it covered every one of them (the whole split, by default),
    otherwise scored here with the same function and settings."""
    keys = [(base.examples[i].ecg_id, base.examples[i].superclass) for i in indices]
    stream = "prompt" if args.task_stream == "prompt-and-target" else args.task_stream
    out = {}
    for name, opts in conditions:
        have = {(r["ecg_id"], r["superclass"]): r
                for r in records_by_condition.get(name, [])}
        if all(k in have for k in keys):
            out[name] = [have[k] for k in keys]
            continue
        out[name] = no_context_records(
            ctx.model, ctx.resampler, ctx.tokenizer, ctx.dev,
            shuffled_view if opts["shuffled"] else base, indices,
            batch_size=args.fast_batch_size, max_length=args.max_length,
            task_stream=stream, seed=args.seed, strict=False,
            what=f"fast mode on the generation draw, {name}", **opts)
    return out


# --------------------------------------------------------------------------- #
# Run                                                                         #
# --------------------------------------------------------------------------- #


def preflight(args, data_cfg):
    checks = [
        ("model dir", args.model_dir),
        ("model config", os.path.join(args.model_dir, "config.json")),
        ("embedding index", os.path.join(data_cfg.emb_cache_dir, "index.json")),
        ("manifest", data_cfg.manifest_path),
        ("teacher jsonl", data_cfg.teacher_path),
        ("resampler checkpoint",
         os.path.join(args.ckpt, "best.pt") if os.path.isdir(args.ckpt) else args.ckpt),
    ]
    missing = [f"{name}: {path}" for name, path in checks if not os.path.exists(path)]
    if missing:
        sys.exit("preflight failed, missing input(s):\n  " + "\n  ".join(missing))
    os.makedirs(args.out_dir, exist_ok=True)
    if not os.access(args.out_dir, os.W_OK):
        sys.exit(f"preflight failed: out dir not writable: {args.out_dir}")
    print("preflight OK; all inputs present, out dir writable")


def run_fast(ctx, base, shuffled_view, indices, args, paths):
    """Fast mode under the three conditions: the no-target-context decision score.

    evaluate.no_context_records -- the same function the overfit gate's condition 4
    calls -- non-strict, so a target whose decision token cannot be located keeps
    its p_yes. Written per condition to {tag}_{split}_fast_{name}.jsonl. The
    resampler reads what generation gives it: prompt-and-target has no target at
    inference, so it reads the prompt here as it does there.
    """
    stream = "prompt" if args.task_stream == "prompt-and-target" else args.task_stream
    mismatch = bool(ctx.strip_think_check.get("strip_think_mismatch", False))
    by_condition = {}
    for name, opts in CONDITIONS:
        path = os.path.join(args.out_dir, f"{args.tag}_{args.split}_fast_{name}.jsonl")
        paths[f"fast_{name}"] = path
        print("\n" + "#" * 64)
        print(f"# FAST MODE: {name}  ->  {path}")
        print("#" * 64, flush=True)
        t0 = time.time()
        records = no_context_records(
            ctx.model, ctx.resampler, ctx.tokenizer, ctx.dev,
            shuffled_view if opts["shuffled"] else base, indices,
            batch_size=args.fast_batch_size, max_length=args.max_length,
            task_stream=stream, seed=args.seed, strict=False,
            log_every=args.log_every, what=f"fast mode, {name}", **opts)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                r.update(split=args.split, strip_think_mismatch=mismatch)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        elapsed = time.time() - t0
        print(f"fast mode, {name}: {len(records):,} examples in {elapsed / 60:.1f} min "
              f"({len(records) / max(elapsed, 1e-9):.1f} ex/s) -> {path}")
        by_condition[name] = records
    print("\nsample check:")
    assert_same_examples(by_condition, "fast-mode conditions")
    metrics = {}
    for name, records in by_condition.items():
        metrics[name] = compute_fast_metrics(records)
        print_fast_metrics(metrics[name], title=name)
    return by_condition, metrics, stream


def main():
    args = parse_args()
    data_cfg = build_data_config(args)
    preflight(args, data_cfg)
    args.task_stream = resolve_task_stream(args.ckpt, args.task_stream)
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible; this sweep needs GPUs")
    print(f"visible GPUs: {torch.cuda.device_count()}")
    torch.manual_seed(args.seed)

    t_start = time.time()
    base = load_split(data_cfg, args.split)
    shuffled_view = apply_shuffle(base, args.seed)   # built once, wraps the same split

    # Every pass is a seeded draw (or the whole split), never the head: the split is
    # ordered by record, and its first examples are fold 10's most normal. One draw
    # per pass, shared by its three conditions, so they cover the same examples.
    sampling = {}
    gen_indices, info = sample_indices(base, args.limit, args.sample_seed, args.stratify)
    sampling["generation"] = print_sample(base, gen_indices, info, "generation draw")
    tf_indices, info = sample_indices(base, args.tf_limit, args.sample_seed, args.stratify)
    if not args.skip_teacher_forced:
        sampling["teacher_forced"] = print_sample(base, tf_indices, info,
                                                  "teacher-forced draw")
    fast_indices, info = sample_indices(base, args.fast_limit, args.sample_seed,
                                        args.stratify)
    if not args.skip_fast:
        sampling["fast"] = print_sample(base, fast_indices, info, "fast-mode draw")
    gen_conditions = [(n, o) for n, o in CONDITIONS
                      if not (args.skip_zeroed_generation and n == "zeroed")]
    print(f"\n{len(gen_indices)} examples per generation condition "
          f"({', '.join(n for n, _ in gen_conditions)}; batch {args.batch_size}); "
          f"{len(tf_indices)} per teacher-forced condition "
          f"({len(CONDITIONS)} conditions, batch {args.tf_batch_size}, "
          f"task stream {args.task_stream}); "
          + ("fast mode skipped" if args.skip_fast else
             f"{len(fast_indices):,} per fast-mode condition (batch "
             f"{args.fast_batch_size})"))
    if args.skip_zeroed_generation:
        print("--skip-zeroed-generation: the zeroed condition is not generated; its "
              "teacher-forced and fast passes still run")
    # The same examples in every condition: identical indices over the same split, and
    # the permutation changes only which embedding each record is served.
    keys = [(base.examples[i].ecg_id, base.examples[i].superclass) for i in gen_indices]
    print(f"conditions share {len(set(keys))} distinct (ecg_id, superclass) examples")

    ctx = EvalContext.load(
        args.model_dir, args.ckpt, strip_think=data_cfg.strip_think,
        allow_strip_think_mismatch=args.allow_strip_think_mismatch)
    objective, objective_recorded = checkpoint_objective(args.ckpt)
    head = load_linear_head(args.ckpt, ctx.dev)
    if head is None:
        print("no linear head in this checkpoint: the linear-head readout is skipped")

    paths = {}
    # Fast mode first: it is the cheapest pass and the per-class numbers come from
    # it, so a job that runs out of time in a decode still leaves it written.
    fast_by_condition, fast_metrics, fast_stream = ({}, {}, None)
    if not args.skip_fast:
        fast_by_condition, fast_metrics, fast_stream = run_fast(
            ctx, base, shuffled_view, fast_indices, args, paths)

    # The head is a resampler forward and a dot product per example, no LLM: scored
    # here, on both draws, so a decode that runs out of time leaves it written.
    linear_fast, linear_gen = {}, {}
    if head is not None:
        if not args.skip_fast:
            linear_fast = run_linear(ctx, head, base, shuffled_view, fast_indices,
                                     CONDITIONS, "fast", args, paths)
        linear_gen = run_linear(ctx, head, base, shuffled_view, gen_indices,
                                gen_conditions, "generation", args, paths)

    tf_summaries = {}
    tf_comparison = None
    if not args.skip_teacher_forced:
        tf_by_condition = {}
        for name, opts in CONDITIONS:
            tf_path = os.path.join(
                args.out_dir, f"{args.tag}_{args.split}_teacher_forced_{name}.jsonl")
            paths[f"teacher_forced_{name}"] = tf_path
            print("\n" + "#" * 64)
            print(f"# TEACHER-FORCED LOSS: {name}  ->  {tf_path}")
            print("#" * 64, flush=True)
            # Same indices, same split, same order for all three; only the embedding
            # a record is served and whether the prefix is zeroed change.
            dataset = shuffled_view if opts["shuffled"] else base
            tf_records, _ = run_teacher_forced(
                ctx, dataset, tf_indices, split=args.split, seed=args.seed,
                shuffled=opts["shuffled"], zero_latents=opts["zero_latents"],
                batch_size=args.tf_batch_size, num_workers=args.num_workers,
                max_length=args.max_length, out_path=tf_path,
                log_every=args.log_every, task_stream=args.task_stream,
            )
            tf_by_condition[name] = tf_records
            tf_summaries[name] = summarise_teacher_forced(tf_records)
            print_teacher_forced(tf_summaries[name])

        print("\nsample check:")
        assert_same_examples(tf_by_condition, "teacher-forced conditions")
        # The table itself is printed once, at the end of the run: it is the
        # conclusion, and two blocks of six rows is too much to repeat.

    by_condition = {}
    for name, opts in gen_conditions:
        out_path = os.path.join(args.out_dir, f"{args.tag}_{args.split}_{name}.jsonl")
        paths[name] = out_path
        print("\n" + "#" * 64)
        print(f"# CONDITION {name}  ->  {out_path}")
        print("#" * 64, flush=True)
        dataset = shuffled_view if opts["shuffled"] else base
        records, _, _ = run_generation(
            ctx, dataset, gen_indices, out_path,
            split=args.split, seed=args.seed,
            shuffled=opts["shuffled"], zero_latents=opts["zero_latents"],
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
            num_workers=args.num_workers, print_first=args.print_first,
            log_every=args.log_every, task_stream=args.task_stream,
        )
        by_condition[name] = records

    # ---- per-condition metrics, then the comparison ------------------------
    summaries = {name: condition_summary(recs) for name, recs in by_condition.items()}
    metrics = {}
    for name, recs in by_condition.items():
        print("\n" + "#" * 64)
        print(f"# METRICS: {name}")
        print("#" * 64)
        metrics[name] = compute_all_metrics(recs)
        print_metrics(metrics[name])

    print("\nsample check:")
    assert_same_examples(by_condition, "generation conditions")
    identical = identical_counts(by_condition)
    print_comparison(summaries, identical, tf_summaries.get("real"))
    if tf_summaries:
        tf_comparison = print_teacher_forced_comparison(tf_summaries, args.task_stream)
    agreement = {name: fast_generation_agreement(fast_by_condition[name], recs)
                 for name, recs in by_condition.items() if name in fast_by_condition}
    print_agreement(agreement)

    readouts = {}
    if head is not None:
        bar = "=" * 72
        print(f"\n{bar}\nLINEAR HEAD vs FAST MODE vs GENERATION  (AUPRC, same examples; "
              f"the head's checkpoint: lambda_linear {objective['lambda_linear']:g}, "
              f"lambda_consistency {objective['lambda_consistency']:g})\n{bar}")
        gen_fast = (fast_on(fast_by_condition, gen_indices, base, ctx, shuffled_view,
                            gen_conditions, args) if not args.skip_fast else {})
        for name, recs in by_condition.items():
            sets = {"generation": (recs, "p_yes")}
            if name in gen_fast:
                sets["fast"] = (gen_fast[name], "p_yes")
            sets["linear"] = (linear_gen[name], "p_linear")
            readouts[f"generation draw, {name}"] = readout_comparison(sets)
        for name, recs in fast_by_condition.items():
            readouts[f"fast draw, {name}"] = readout_comparison(
                {"fast": (recs, "p_yes"), "linear": (linear_fast[name], "p_linear")})
        for title, comparison in readouts.items():
            print_readout_comparison(title, comparison)
        print("\nThe linear loss trains the head; the consistency loss pulls the LLM's "
              "Yes/No toward it. Read the\ngap row: a gap that stays where it was "
              "means generation was not dragged toward the head.")

    elapsed = time.time() - t_start
    summary_path = os.path.join(args.out_dir, f"{args.tag}_{args.split}_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "split": args.split, "checkpoint": args.ckpt, "model_dir": args.model_dir,
            "seed": args.seed, "limit": args.limit, "tf_limit": args.tf_limit,
            "fast_limit": args.fast_limit,
            # how every pass chose its examples, and the realised rates
            "sampling": sampling,
            "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
            "task_stream": args.task_stream,
            "skip_zeroed_generation": bool(args.skip_zeroed_generation),
            "strip_think": bool(args.strip_think), **ctx.strip_think_check,
            "files": paths,
            "conditions": summaries,
            "identical": identical,
            "metrics": metrics,
            # one summary per condition now, not a single pass; the old single-dict
            # shape is gone, so anything reading this key needs the condition name
            "teacher_forced": tf_summaries,
            "teacher_forced_comparison": tf_comparison,
            "fast": ({"task_stream": fast_stream, "threshold": FAST_THRESHOLD,
                      "batch_size": args.fast_batch_size, "metrics": fast_metrics}
                     if not args.skip_fast else None),
            "fast_vs_generation": agreement,
            "objective": {**objective, "recorded": objective_recorded},
            "linear_head": readouts if head is not None else None,
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)
    print(f"\ntotal {elapsed/60:.1f} min")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
