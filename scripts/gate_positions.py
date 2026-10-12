#!/usr/bin/env python
"""Teacher-forced loss by target position, from the gate's per-example records.

The gate's condition 1 is one number, a mean over every target token. This reads the
per-example file the gate wrote next to its checkpoints
(gate_teacher_forced_real.jsonl; runs from before the controls landed wrote it as
gate_teacher_forced.jsonl)
and splits that number by where in the target the loss falls: positions 0-9, 10-24,
25-49, 50-99 and 100+, each with its mean loss and the number of examples whose
target reaches it.

The buckets and the pooling are evaluate.py's own (POSITION_BUCKETS,
summarise_positions), imported rather than repeated, so this table and the control
sweep's teacher-forced table cannot drift apart. A bucket mean is pooled over
(example, position) pairs; the "mean" row above the buckets is the mean over
examples, the number condition 1 tests.

Positions are offsets into the TARGET, not into the full sequence: index 0 of
``position_losses`` is the target's first token, whatever the length of the prefix
and the prompt in front of it, and the last index is the EOS the collate appends.
That is how evaluate.teacher_forced_batch writes them, so nothing is shifted here.

Given more than one file, the buckets print side by side. THE FIRST FILE IS TAKEN
AS REAL and every later one as a control, with its gap against real per bucket in
nats and in percent (positive = the control is worse). Each file's condition is
read from its records' own shuffled_embeddings / zero_latents flags and printed
beside its path, so a first file that is itself a control is called out. The files
must cover the same (ecg_id, superclass) examples: a gap between different examples
is not a gap between conditions.

No GPU and no model weights: the jsonl is all it reads, so it runs on a login node.

    python scripts/gate_positions.py $CKPT_DIR/gate_teacher_forced_real.jsonl
    python scripts/gate_positions.py $CKPT_DIR/gate_teacher_forced_real.jsonl \\
                                     $CKPT_DIR/gate_teacher_forced_shuffled.jsonl \\
                                     $CKPT_DIR/gate_teacher_forced_zeroed.jsonl
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_controls import _fmt, _pct_gap  # noqa: E402
from evaluate import POSITION_BUCKETS, read_jsonl, summarise_positions  # noqa: E402

LABEL_W = 18
CELL_W = 17     # one file's "loss  examples" pair
GAP_W = 22      # one control's "nats  percent" pair


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", metavar="JSONL",
                    help="gate_teacher_forced_*.jsonl; with more than one, the first "
                         "is real and the rest are controls")
    return ap.parse_args()


def condition_of(records):
    """real / shuffled / zeroed, from the flags run_teacher_forced wrote on each record."""
    flags = {(bool(r.get("shuffled_embeddings")), bool(r.get("zero_latents")))
             for r in records}
    if len(flags) > 1:
        return "mixed"
    shuffled, zeroed = flags.pop()
    return "+".join(n for n, on in (("shuffled", shuffled), ("zeroed", zeroed)) if on) or "real"


def load(path):
    """One file's rows: the mean over examples, then one row per position bucket."""
    records = read_jsonl(path)
    if not records:
        sys.exit(f"{path}: no records")
    scored = [r for r in records if r.get("position_losses")]
    if not scored:
        sys.exit(f"{path}: none of its {len(records)} records carries position_losses; "
                 "it was written before evaluate.py kept the per-position losses, or "
                 "every target was truncated away")
    # the same length check yn_token_loss.py treats as fatal: a bucket boundary means
    # nothing if the list is not one loss per target token
    bad = [r for r in scored if len(r["position_losses"]) != r["n_target_tokens"]]
    if bad:
        sys.exit(f"{path}: len(position_losses) != n_target_tokens for {len(bad)} of "
                 f"{len(scored)} records, first {bad[0]['ecg_id']} {bad[0]['superclass']}")
    rows = [{"positions": "mean", "n_examples": len(scored),
             "n_positions": sum(r["n_target_tokens"] for r in scored),
             "mean_loss": float(np.mean([r["loss"] for r in scored]))}]
    rows += summarise_positions(scored)
    return {"path": path, "n": len(records), "n_scored": len(scored),
            "condition": condition_of(scored), "rows": rows,
            "examples": Counter((r["ecg_id"], r["superclass"]) for r in scored)}


def name_columns(files):
    """real for the first file; each control by its own condition, numbered if repeated."""
    names = ["real"]
    for f in files[1:]:
        base = f["condition"] if f["condition"] != "real" else "control"
        name, k = base, 1
        while name in names:
            k += 1
            name = f"{base}{k}"
        names.append(name)
    return names


def row_label(row):
    return row["positions"] if row["positions"] == "mean" else f"  positions {row['positions']}"


def print_one(f):
    print(f"\n{'':<{LABEL_W}} {'mean loss':>9} {'tokens':>9} {'examples':>9}")
    for r in f["rows"]:
        print(f"{row_label(r):<{LABEL_W}} {_fmt(r['mean_loss'], width=9, prec=4)} "
              f"{r['n_positions']:>9,} {r['n_examples']:>9,}")


def print_side_by_side(files, names):
    real = files[0]
    for f, name in zip(files[1:], names[1:]):
        if f["examples"] != real["examples"]:
            common = sum((f["examples"] & real["examples"]).values())
            sys.exit(f"\n{name} ({f['path']}) does not cover the examples real does: "
                     f"{f['n_scored']} against {real['n_scored']}, {common} in common. "
                     "A per-bucket gap between different examples is not a gap between "
                     "conditions; run the files one at a time for their own tables.")

    line_w = LABEL_W + CELL_W * len(files) + GAP_W * (len(files) - 1)
    print("\n" + "=" * line_w)
    print(f"TEACHER-FORCED LOSS BY TARGET POSITION, {real['n_scored']} examples")
    print("=" * line_w)
    print(f"{'':<{LABEL_W}}" + "".join(f"{n:>{CELL_W}}" for n in names)
          + "".join(f"{n + ' - real':>{GAP_W}}" for n in names[1:]))
    print(f"{'':<{LABEL_W}}" + f"{'loss':>{CELL_W - 6}}{'ex':>6}" * len(files)
          + f"{'nats':>{GAP_W - 10}}{'%':>10}" * (len(files) - 1))
    for k, r in enumerate(real["rows"]):
        cells = "".join(_fmt(f["rows"][k]["mean_loss"], width=CELL_W - 6, prec=4)
                        + f"{f['rows'][k]['n_examples']:>6,}" for f in files)
        gaps = ""
        for f in files[1:]:
            a, b = r["mean_loss"], f["rows"][k]["mean_loss"]
            pct = _pct_gap(a, b)
            nats = f"{b - a:>+{GAP_W - 10}.4f}" if None not in (a, b) else f"{'n/a':>{GAP_W - 10}}"
            gaps += nats + (f"{pct:>+9.1f}%" if pct is not None else f"{'n/a':>10}")
        print(f"{row_label(r):<{LABEL_W}}{cells}{gaps}")
    print("-" * line_w)
    print("  bucket support (real): " + "  ".join(
        f"{r['positions']} n={r['n_positions']:,}" for r in real["rows"][1:]))


def main():
    args = parse_args()
    files = [load(p) for p in args.files]
    names = name_columns(files)

    name_w = max(len(n) for n in names)
    for f, name in zip(files, names):
        flagged = "" if f["condition"] == name else f"  [records flagged {f['condition']}]"
        print(f"{name:<{name_w}}  {f['n_scored']:>5} of {f['n']} records scored  "
              f"{f['path']}{flagged}")
    if len(files) > 1 and files[0]["condition"] != "real":
        print(f"  NOTE: the first file is taken as real but its records are flagged "
              f"{files[0]['condition']};\n        every gap below is measured against it.")

    if len(files) == 1:
        print_one(files[0])
    else:
        print_side_by_side(files, names)
    print("  buckets are positions within the target (0 = its first token), pooled over "
          "examples:\n  " + ", ".join(label for _, _, label in POSITION_BUCKETS)
          + ". mean is the mean over examples, the gate's condition 1.")


if __name__ == "__main__":
    main()
