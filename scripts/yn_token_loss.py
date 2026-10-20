#!/usr/bin/env python
"""Teacher-forced loss at the Yes/No token, per control condition.

The control sweep's teacher-forced table is a mean over every target token, and
almost all of those tokens are Evidence and Reasoning prose. The one token that
carries the answer is the first one after "Conclusion:", so a model that never
reads the ECG can post a sequence mean barely worse than one that does. This reads
the three per-example files the sweep wrote

    {tag}_{split}_teacher_forced_{real,shuffled,zeroed}.jsonl

and reports the loss at that single position instead.

The jsonl carries ``position_losses`` but not the target text, so the target is
rebuilt here: the same ``parsed_final`` block the dataset serves for that
(ecg_id, superclass), tokenised as the collate does -- add_special_tokens=False,
then EOS appended. ``len(ids) == n_target_tokens`` is asserted for EVERY record.
That assertion is the correctness check on the whole alignment: a position index
into ``position_losses`` means nothing unless the rebuilt token sequence is the one
the loss was computed over. If it fails the script stops and lists the records,
rather than reporting a decision-token loss read off the wrong position.

No GPU and no model weights: the teacher jsonl and the tokenizer are all it loads.

    python scripts/yn_token_loss.py --dir $SCRATCH/ecg-llm-alignment/eval/run2
    python scripts/yn_token_loss.py --dir $EVAL_DIR --model-dir $MODEL_DIR   # offline
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# MARKER, decision_index and tokenise_target are re-exported: older callers import
# them from here.
from decision_token import MARKER, decision_index, tokenise_target  # noqa: E402,F401
from eval_controls import CONDITIONS, _pct_gap, assert_same_examples  # noqa: E402
from evaluate import read_jsonl  # noqa: E402
from src.data.dataset import (  # noqa: E402
    DatasetConfig,
    _extract_teacher_blocks,
    load_student_tokenizer,
)
from src.teacher.prompt import SUPERCLASSES  # noqa: E402

NAMES = [name for name, _ in CONDITIONS]

# A constant-Yes model puts ~all its mass on " Yes" whatever the ECG: loss near zero
# where the target says Yes, several nats where it says No.
NEAR_ZERO = 0.01

PCTLS = (10, 25, 50, 75, 90)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(REPO_ROOT, "results"),
                    help="where eval_controls.py wrote its jsonl (its --out-dir)")
    ap.add_argument("--tag", default="controls")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--model-dir", default=None,
                    help="local student checkpoint dir, for the tokenizer only "
                         "(local_files_only); default loads it by hub name")
    ap.add_argument("--teacher", default=None, help="teacher jsonl (default: the dataset's)")
    ap.add_argument("--known-sequence-mean", type=float, nargs=3, default=None,
                    metavar=("REAL", "SHUFFLED", "ZEROED"),
                    help="the sweep's own whole-target means (its summary JSON's "
                         "teacher_forced.<condition>.mean_loss), printed beside the "
                         "means recomputed from the files; a difference beyond 5e-4 "
                         "says these are not the files those numbers came from. "
                         "Omitted, the row and the check are left out")
    ap.add_argument("--show", type=int, default=10,
                    help="records to print the decoded decision token for")
    return ap.parse_args()


def load_tokenizer(model_dir):
    if model_dir:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    return load_student_tokenizer(DatasetConfig())


def fail_alignment(failures, n_checked):
    records = sorted({(eid, sc) for _, eid, sc, _, _, _ in failures})
    print(f"\nALIGNMENT FAILED: {len(failures)} of {n_checked} file rows, "
          f"{len(records)} distinct (ecg_id, superclass) records.")
    print("Nothing below this line was computed; a decision-token index into "
          "position_losses\nis only valid when the rebuilt target is the one the "
          "loss was computed over.\n")
    print(f"  {'condition':<10} {'ecg_id':>7} {'class':<6} {'n_target_tokens':>15} "
          f"{'rebuilt':>8} {'position_losses':>15}")
    for name, eid, sc, n_file, n_rebuilt, n_pos in failures:
        rebuilt = "no target" if n_rebuilt is None else n_rebuilt
        print(f"  {name:<10} {eid:>7} {sc:<6} {n_file:>15} {rebuilt:>8} {n_pos:>15}")
    sys.exit(1)


def dist_row(label, values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return f"    {label:<10} {0:>5}"
    qs = np.percentile(a, PCTLS)
    return (f"    {label:<10} {a.size:>5} {a.mean():>8.4f} "
            + " ".join(f"{q:>8.4f}" for q in qs) + f" {a.max():>8.4f}")


def main():
    args = parse_args()

    by_condition = {}
    for name in NAMES:
        path = os.path.join(args.dir, f"{args.tag}_{args.split}_teacher_forced_{name}.jsonl")
        by_condition[name] = read_jsonl(path)
        print(f"{name:<9} {len(by_condition[name]):>5} records  {path}")
    assert_same_examples(by_condition, "teacher-forced conditions")
    ref = by_condition[NAMES[0]]

    teacher_path = args.teacher or DatasetConfig().teacher_path
    print(f"\nloading targets (parsed_final) from {teacher_path}")
    usable, _, _ = _extract_teacher_blocks(teacher_path)
    tokenizer = load_tokenizer(args.model_dir)

    # ---- rebuild every target and check it against every file --------------
    ids_list, offsets_list, texts = [], [], []
    failures = []
    for i, r in enumerate(ref):
        text = usable.get(int(r["ecg_id"]), {}).get(r["superclass"])
        ids, offsets = tokenise_target(tokenizer, text) if text is not None else (None, None)
        texts.append(text)
        ids_list.append(ids)
        offsets_list.append(offsets)
        for name in NAMES:
            rec = by_condition[name][i]
            n_file, n_pos = rec["n_target_tokens"], len(rec.get("position_losses") or [])
            if ids is None or len(ids) != n_file or n_pos != n_file:
                failures.append((name, rec["ecg_id"], rec["superclass"], n_file,
                                 None if ids is None else len(ids), n_pos))
    n_checked = len(ref) * len(NAMES)
    if failures:
        fail_alignment(failures, n_checked)
    print(f"alignment: len(ids) == n_target_tokens == len(position_losses) for all "
          f"{n_checked} rows ({len(ref)} records x {len(NAMES)} conditions)")

    # ---- locate the decision token -----------------------------------------
    idx, unlocated = [], []
    for r, text, offsets in zip(ref, texts, offsets_list):
        k, why = decision_index(text, offsets)
        idx.append(k)
        if k is None:
            unlocated.append((r["ecg_id"], r["superclass"], why))
    if unlocated:
        print(f"\nDECISION TOKEN NOT LOCATED in {len(unlocated)} of {len(ref)} records:")
        for eid, sc, why in unlocated:
            print(f"  {eid:>7} {sc:<6} {why}")
        sys.exit(1)

    decoded = [tokenizer.decode([ids[k]]) for ids, k in zip(ids_list, idx)]
    print(f"\ndecision token = first token after {MARKER!r}; {min(args.show, len(ref))} "
          "records, evenly spaced through the file")
    print(f"  {'ecg_id':>7} {'class':<6} {'label':<6} {'index':>5} {'of':>5}  "
          f"{'token':<8} context")
    for i in np.linspace(0, len(ref) - 1, min(args.show, len(ref))).astype(int):
        r, k, ids = ref[i], idx[i], ids_list[i]
        context = tokenizer.decode(ids[max(0, k - 2):k + 4])
        print(f"  {r['ecg_id']:>7} {r['superclass']:<6} {str(r['label']):<6} {k:>5} "
              f"{len(ids):>5}  {decoded[i]!r:<8} {context!r}")

    print("\n  decision token by label, all records:")
    tally = Counter((d, bool(r["label"])) for d, r in zip(decoded, ref))
    for (d, label), n in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {d!r:<8} label {str(label):<6} {n:>5}")

    # ---- the numbers --------------------------------------------------------
    labels = np.array([bool(r["label"]) for r in ref])
    classes = np.array([r["superclass"] for r in ref])
    dec = {name: np.array([rec["position_losses"][k]
                           for rec, k in zip(by_condition[name], idx)], dtype=np.float64)
           for name in NAMES}
    seq = {name: float(np.mean([rec["loss"] for rec in by_condition[name]]))
           for name in NAMES}
    dec_mean = {name: float(dec[name].mean()) for name in NAMES}

    line_w = 26 + 11 * len(NAMES) + 14
    print("\n" + "=" * line_w)
    print(f"MEAN LOSS over {len(ref)} examples")
    print("=" * line_w)
    print(f"{'':<26}" + "".join(f"{n:>11}" for n in NAMES) + f"{'real vs shuf':>14}")
    known = (dict(zip(NAMES, args.known_sequence_mean))
             if args.known_sequence_mean else None)
    rows = ([("sequence mean, known", known)] if known else []) + [
        ("sequence mean, these files", seq), ("DECISION TOKEN", dec_mean)]
    for label, values in rows:
        gap = _pct_gap(values["real"], values["shuffled"])
        print(f"{label:<26}" + "".join(f"{values[n]:>11.4f}" for n in NAMES)
              + f"{gap:>+13.1f}%")
    print(f"  decision token, shuffled - real: {dec_mean['shuffled'] - dec_mean['real']:+.4f} "
          f"nats (sequence mean: {seq['shuffled'] - seq['real']:+.4f})")
    drift = [n for n in NAMES if known and abs(seq[n] - known[n]) > 5e-4]
    if drift:
        print(f"  NOTE: sequence mean differs from --known-sequence-mean for "
              f"{', '.join(drift)}; these are\n        not the files those numbers "
              "came from.")

    groups = [("all", np.ones(len(ref), dtype=bool)),
              ("label True", labels), ("label False", ~labels)]
    groups += [(sc, classes == sc) for sc in SUPERCLASSES]

    print("\nDECISION-TOKEN LOSS, per-example distribution")
    print(f"    {'':<10} {'n':>5} {'mean':>8} "
          + " ".join(f"{'p' + str(p):>8}" for p in PCTLS) + f" {'max':>8}")
    for label, mask in groups:
        print(f"  {label}")
        for name in NAMES:
            print(dist_row(name, dec[name][mask]))

    print(f"\nEXAMPLES WITH DECISION-TOKEN LOSS < {NEAR_ZERO}")
    print(f"    {'':<10}" + "".join(f"{g:>16}" for g, _ in groups[:3]))
    for name in NAMES:
        cells = "".join(
            f"{f'{int((dec[name][mask] < NEAR_ZERO).sum())} / {int(mask.sum())}':>16}"
            for _, mask in groups[:3])
        print(f"    {name:<10}{cells}")


if __name__ == "__main__":
    main()
