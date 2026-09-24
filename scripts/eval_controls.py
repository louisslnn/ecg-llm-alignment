#!/usr/bin/env python
"""The control sweep: three generation conditions and a teacher-forced loss, one job.

This is the run that says whether a trained resampler is doing anything at all. It
loads the 14B and the resampler ONCE and then makes four passes over the same slice
of the split, so the answer arrives in a single queue slot instead of four:

    real                the model as trained
    shuffled            every record gets another record's ECG (--shuffle-embeddings)
    zeroed              the prefix is spliced as zeros (--zero-latents)
    teacher-forced      no generation: loss on the real targets, on a larger slice

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

Defaults are deliberately small -- 50 examples per generation condition at batch 1,
200 for the teacher-forced pass -- because this is a diagnostic, not the evaluation.
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
    EvalContext,
    apply_shuffle,
    build_data_config,
    compute_all_metrics,
    load_split,
    print_metrics,
    print_teacher_forced,
    run_generation,
    run_teacher_forced,
    summarise_teacher_forced,
)

# The three generation conditions, in the order they are run and reported.
CONDITIONS = [
    ("real", dict(shuffled=False, zero_latents=False)),
    ("shuffled", dict(shuffled=True, zero_latents=False)),
    ("zeroed", dict(shuffled=False, zero_latents=True)),
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="local student checkpoint dir")
    ap.add_argument("--ckpt", required=True,
                    help="resampler checkpoint (.pt file, or a dir holding best.pt)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=50,
                    help="examples per generation condition (default 50)")
    ap.add_argument("--tf-limit", type=int, default=200,
                    help="examples for the teacher-forced loss pass (default 200)")
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
                    help="seed for the embedding permutation; logged and recorded")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--print-first", type=int, default=1,
                    help="print the first N generations of each condition in full")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--skip-teacher-forced", action="store_true")
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
    out = {"n_compared": len(common), "identical_all_three": 0, "pairwise": {}}
    for key in common:
        if len({keyed[n][key] for n in names}) == 1:
            out["identical_all_three"] += 1
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out["pairwise"][f"{a} vs {b}"] = sum(
                1 for key in common if keyed[a][key] == keyed[b][key]
            )
    return out


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
    print(f"  identical across ALL THREE conditions: {identical['identical_all_three']}"
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


def main():
    args = parse_args()
    data_cfg = build_data_config(args)
    preflight(args, data_cfg)
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible; this sweep needs GPUs")
    print(f"visible GPUs: {torch.cuda.device_count()}")
    torch.manual_seed(args.seed)

    t_start = time.time()
    base = load_split(data_cfg, args.split)
    shuffled_view = apply_shuffle(base, args.seed)   # built once, wraps the same split

    gen_indices = list(range(min(args.limit, len(base))))
    tf_indices = list(range(min(args.tf_limit, len(base))))
    print(f"\n{len(gen_indices)} examples per generation condition "
          f"({len(CONDITIONS)} conditions, batch {args.batch_size}); "
          f"{len(tf_indices)} for the teacher-forced pass")
    # The same examples in every condition: identical indices over the same split, and
    # the permutation changes only which embedding each record is served.
    keys = [(base.examples[i].ecg_id, base.examples[i].superclass) for i in gen_indices]
    print(f"conditions share {len(set(keys))} distinct (ecg_id, superclass) examples")

    ctx = EvalContext.load(args.model_dir, args.ckpt)

    by_condition = {}
    paths = {}
    for name, opts in CONDITIONS:
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
            log_every=args.log_every,
        )
        by_condition[name] = records

    tf_summary = None
    if not args.skip_teacher_forced:
        tf_path = os.path.join(args.out_dir, f"{args.tag}_{args.split}_teacher_forced.jsonl")
        paths["teacher_forced"] = tf_path
        print("\n" + "#" * 64)
        print(f"# TEACHER-FORCED LOSS  ->  {tf_path}")
        print("#" * 64, flush=True)
        tf_records, _ = run_teacher_forced(
            ctx, base, tf_indices, split=args.split, seed=args.seed,
            shuffled=False, zero_latents=False, batch_size=args.tf_batch_size,
            num_workers=args.num_workers, max_length=args.max_length,
            out_path=tf_path, log_every=args.log_every,
        )
        tf_summary = summarise_teacher_forced(tf_records)
        print_teacher_forced(tf_summary)

    # ---- per-condition metrics, then the comparison ------------------------
    summaries = {name: condition_summary(recs) for name, recs in by_condition.items()}
    metrics = {}
    for name, recs in by_condition.items():
        print("\n" + "#" * 64)
        print(f"# METRICS: {name}")
        print("#" * 64)
        metrics[name] = compute_all_metrics(recs)
        print_metrics(metrics[name])

    identical = identical_counts(by_condition)
    print_comparison(summaries, identical, tf_summary)

    elapsed = time.time() - t_start
    summary_path = os.path.join(args.out_dir, f"{args.tag}_{args.split}_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "split": args.split, "checkpoint": args.ckpt, "model_dir": args.model_dir,
            "seed": args.seed, "limit": args.limit, "tf_limit": args.tf_limit,
            "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
            "files": paths,
            "conditions": summaries,
            "identical": identical,
            "metrics": metrics,
            "teacher_forced": tf_summary,
            "elapsed_seconds": round(elapsed, 1),
        }, f, indent=2)
    print(f"\ntotal {elapsed/60:.1f} min")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
