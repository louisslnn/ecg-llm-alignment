#!/usr/bin/env python
"""Linear probe on the RESAMPLER'S OUTPUT latents -- the prefix the student reads.

scripts/probe_embeddings.py asks what is linearly readable from the cached ECG-FM
embeddings going IN. This asks the same question of what comes OUT: the 64 latents
the resampler produces and splices in front of the student's tokens. Same folds,
same probe, same metrics, so the two numbers sit on one ruler:

    embeddings (N, 312, 768) --[resampler]--> latents (N, 64, 5120)
                                                 |
                              mean-pool over the 64 positions -> (N, 5120)
                              logistic regression, one binary fit per superclass,
                              trained on folds 1-8, scored on fold 10.

WHAT THE COMPARISON MEANS. The probe on the input embeddings measured 0.714 macro
AUPRC (0.882 AUROC). That is what the resampler is handed. If this script comes back
near it, the resampler is preserving the diagnostic content and any failure
downstream is the student not reading the prefix. If it comes back near the positive
rate, the resampler has destroyed the signal before the student ever sees it, and no
amount of work on the prompt or the decoding will recover it. Either way this
separates "the prefix has nothing in it" from "the prefix has something the frozen
student does not use", which the end-to-end metrics cannot do on their own.

THE TASK-TEXT PROBLEM. The resampler is multi-modal: its latents are a function of
the ECG *and* of the question being asked, so there is no single "the latents for
this record". Three ways to reduce that to one vector per record, --question-mode:

    pooled   (default) run all five superclass questions and mean-pool the five
             latent vectors. One vector per record, exactly like the embedding
             probe, and the fairest like-for-like comparison: it asks what the
             resampler carries about the record regardless of what it was asked.
    matched  probe superclass S using the latents produced by S's OWN question.
             Five different feature sets, one per probe. This is the quantity that
             actually matters at inference -- when the student is asked about MI, is
             MI in the prefix it gets? -- but it is not one feature space, so the
             macro number is a summary of five separate experiments.
    fixed    run one question (--fixed-question, default NORM) for every record and
             use it for all five probes. Cheapest (one resampler pass per record
             instead of five) and the closest thing to a text-free control: the
             text stream is held constant, so anything the probe reads is the ECG.

The mode is printed in the header and written into the JSON: a latent probe number
without its question mode is not interpretable.

VARIANTS default to `centered` alone, not the three probe_embeddings.py runs.
Centering is a shift and the probe carries a bias term, so raw and centered share an
optimum -- measured there, they agree to 0.002 AUPRC, with raw needing ~6x the
iterations to get there. At 5120 dims that difference is hours, for a number that is
the same. Pass --variants to run the others anyway.

    # needs a GPU: the 14B's embedding matrix and the 843M resampler
    python scripts/probe_latents.py --model-dir $MODEL_DIR --ckpt $CKPT_DIR/best.pt
    python scripts/probe_latents.py ... --question-mode matched
    python scripts/probe_latents.py ... --limit-records 200   # a quick shape check

COST. Five questions over folds 1-8 and 10 is ~98k resampler forwards; the student
is loaded for its input-embedding matrix only (no transformer forward anywhere in
this script). Budget ~15 min for the model load and ~10-20 min for the pass. The
fits run on the GPU in fp64 (see fit_logreg's `device`): on CPU, a 5120-wide
strong-Wolfe line search takes hours per superclass.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import (  # noqa: E402
    DEFAULT_FOLD_SPLITS,
    DatasetConfig,
    make_collate_fn,
)
from src.teacher.prompt import SUPERCLASSES  # noqa: E402
# The metrics the model itself is scored with, so the probe and the pipeline are
# measured by one implementation rather than two.
from evaluate import EvalContext, auprc, auroc, base_of, load_split  # noqa: E402
# The probe itself: the same LBFGS fit, the same feature transforms and the same
# table the embedding probe prints, so the two reports can be read side by side.
from probe_embeddings import (  # noqa: E402
    GRAD_CONVERGED,
    VARIANTS,
    apply_variant,
    fit_logreg,
    print_variant,
    scores,
)

QUESTION_MODES = ("pooled", "matched", "fixed")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True,
                    help="local student checkpoint dir (its input-embedding matrix "
                         "feeds the resampler's task-text stream)")
    ap.add_argument("--ckpt", required=True, help="resampler checkpoint (best.pt)")
    ap.add_argument("--question-mode", choices=QUESTION_MODES, default="pooled",
                    help="how to reduce the per-question latents to one feature "
                         "vector; see the module docstring (default: pooled)")
    ap.add_argument("--fixed-question", default="NORM", choices=list(SUPERCLASSES),
                    help="which superclass question to use in --question-mode fixed")
    ap.add_argument("--variants", nargs="+", default=["centered"], choices=VARIANTS,
                    help="feature transforms to run (default: centered only)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=2048,
                    help="prompt truncation, as training used it")
    ap.add_argument("--limit-records", type=int, default=None,
                    help="cap the records per split, for a quick shape check")
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fit-device", default="auto",
                    help="where to run the LBFGS fits: auto (the resampler's "
                         "device), cuda, or cpu (hours at this width)")
    ap.add_argument("--compare-to", default=os.path.join(
        REPO_ROOT, "results", "probe_embeddings.json"),
        help="probe_embeddings.py output to print alongside; '' to skip")
    ap.add_argument("--strip-think", action="store_true",
                    help="MUST match the flag the checkpoint trained with")
    ap.add_argument("--emb-cache", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results",
                                                  "probe_latents.json"))
    return ap.parse_args()


def build_data_config(args) -> DatasetConfig:
    kwargs = {"strip_think": bool(args.strip_think)}
    if args.emb_cache:
        kwargs["emb_cache_dir"] = args.emb_cache
    if args.manifest:
        kwargs["manifest_path"] = args.manifest
    if args.teacher:
        kwargs["teacher_path"] = args.teacher
    return DatasetConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Features                                                                    #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def latent_vectors(model, resampler, batch, dev) -> np.ndarray:
    """Mean-pooled resampler output for one batch: (B, embed_dim), float32.

    The forward is the training/eval one (scripts/evaluate.py splice_prefix, minus
    the concatenation): the task-text stream is the student's own input embeddings
    of the prompt, masked exactly as training masked it. Pooling over the 64 latent
    positions mirrors the embedding probe's pooling over its 312 input positions.
    """
    input_ids = batch["input_ids"].to(dev)
    attn = batch["attention_mask"].to(dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tok_embeds = model.get_input_embeddings()(input_ids)
        latents = resampler(batch["ecg_embed"].to(dev),
                            task_embeddings=tok_embeds, task_mask=attn.bool())
    return latents.float().mean(dim=1).cpu().numpy()


def record_features(ctx, dataset, indices: List[int], args, label: str
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Run the resampler over ``indices`` and pool to one vector per record.

    Returns ``(ecg_ids, X)`` with ecg_ids sorted ascending, so features and labels
    can be lined up by record without depending on loader order. When indices cover
    several questions for a record (the pooled mode), their vectors are averaged.
    """
    from torch.utils.data import DataLoader, Subset

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(ctx.tokenizer, max_length=args.max_length,
                                   strip_think=base_of(dataset).config.strip_think),
    )
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    t0 = time.time()
    done = 0
    for batch in loader:
        vecs = latent_vectors(ctx.model, ctx.resampler, batch, ctx.dev)
        for b, eid in enumerate(batch["ecg_id"]):
            eid = int(eid)
            if eid in sums:
                sums[eid] += vecs[b]
                counts[eid] += 1
            else:
                sums[eid] = vecs[b].astype(np.float64)
                counts[eid] = 1
        done += len(vecs)
        if done % (args.batch_size * 100) < args.batch_size:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  [{label}] {done:>7,}/{len(indices):,} forwards "
                  f"({rate:.0f}/s)", flush=True)
    ids = np.array(sorted(sums), dtype=np.int64)
    X = np.stack([sums[i] / counts[i] for i in ids])
    per_record = {counts[i] for i in ids}
    print(f"  [{label}] {len(indices):,} forwards -> {len(ids):,} records "
          f"({sorted(per_record)} question(s) pooled each) in "
          f"{(time.time() - t0) / 60:.1f} min")
    return ids, X


def select_indices(dataset, superclass: Optional[str], limit_records: Optional[int]
                   ) -> List[int]:
    """Example indices for one question (or all of them), capped by record count."""
    examples = base_of(dataset).examples
    keep_ids = None
    if limit_records is not None:
        ordered = sorted({e.ecg_id for e in examples})[:limit_records]
        keep_ids = set(ordered)
    return [i for i, e in enumerate(examples)
            if (superclass is None or e.superclass == superclass)
            and (keep_ids is None or e.ecg_id in keep_ids)]


def label_matrix(dataset, ids: np.ndarray) -> np.ndarray:
    """(n_records, 5) bool labels, read from the examples rather than the batches.

    Taken from the full example list, not from whatever subset was run: in the fixed
    and matched modes the loader only ever sees one question per record, but every
    probe still needs all five labels.
    """
    by_record: Dict[int, Dict[str, bool]] = {}
    for e in base_of(dataset).examples:
        by_record.setdefault(int(e.ecg_id), {})[e.superclass] = bool(e.label)
    missing = [int(i) for i in ids if len(by_record.get(int(i), {})) != len(SUPERCLASSES)]
    if missing:
        raise SystemExit(
            f"{len(missing)} record(s) do not carry all {len(SUPERCLASSES)} "
            f"superclass labels (first: {missing[:5]}); the probe needs the full "
            "label grid per record")
    return np.array([[by_record[int(i)][sc] for sc in SUPERCLASSES] for i in ids],
                    dtype=bool)


def features_for_split(ctx, dataset, args, split: str):
    """Feature matrix (or matrices) plus labels for one split, per question mode.

    Returns ``(features, Y)`` where ``features`` is either a single (n, d) array
    shared by all five probes (pooled, fixed) or a dict superclass -> (n, d) with
    one array per probe (matched). Labels are aligned to the same record order.
    """
    mode = args.question_mode
    if mode == "matched":
        feats, ids_seen = {}, None
        for sc in SUPERCLASSES:
            idx = select_indices(dataset, sc, args.limit_records)
            ids, X = record_features(ctx, dataset, idx, args, f"{split}/{sc}")
            if ids_seen is None:
                ids_seen = ids
            elif not np.array_equal(ids, ids_seen):
                raise SystemExit(
                    f"record sets differ between questions in split {split} "
                    f"({sc} has {len(ids)}, the first question had {len(ids_seen)}); "
                    "the matched mode needs the same records for every probe")
            feats[sc] = X
        return feats, label_matrix(dataset, ids_seen)

    superclass = args.fixed_question if mode == "fixed" else None
    idx = select_indices(dataset, superclass, args.limit_records)
    ids, X = record_features(ctx, dataset, idx, args, split)
    return X, label_matrix(dataset, ids)


# --------------------------------------------------------------------------- #
# Probe                                                                       #
# --------------------------------------------------------------------------- #


def fit_table(train_feats, train_Y, test_feats, test_Y, variant, args, fit_device
              ) -> Dict[str, dict]:
    """One binary probe per superclass, in probe_embeddings.py's table shape.

    ``*_feats`` is one array for every probe, or a dict of one array per probe (the
    matched mode). The variant transform is fitted on train and applied to both, as
    there -- and in the matched mode it is fitted separately per feature set, since
    those are five different spaces.
    """
    per_sc: Dict[str, dict] = {}
    for i, sc in enumerate(SUPERCLASSES):
        tr = train_feats[sc] if isinstance(train_feats, dict) else train_feats
        te = test_feats[sc] if isinstance(test_feats, dict) else test_feats
        tr, te = apply_variant(tr, te, variant)
        y_tr, y_te = train_Y[:, i], test_Y[:, i]
        t0 = time.time()
        w, b, info = fit_logreg(tr, y_tr, args.l2, args.max_iter, args.seed,
                                device=fit_device)
        s = scores(te, w, b)
        ap = auprc(y_te, s)
        pos_rate = float(y_te.mean())
        per_sc[sc] = {
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "train_positive_rate": float(y_tr.mean()),
            "positive_rate": pos_rate,
            "auprc": ap,
            "auroc": auroc(y_te, s),
            "auprc_lift": (ap / pos_rate) if (ap is not None and pos_rate > 0) else None,
            "fit_loss": info["loss"], "fit_initial_loss": info["initial_loss"],
            "fit_grad_norm": info["grad_norm"],
            "fit_seconds": round(time.time() - t0, 2),
        }
        print(f"    {sc:<5} fitted in {per_sc[sc]['fit_seconds']:>7.1f}s  "
              f"AUPRC {ap:.3f}  AUROC {per_sc[sc]['auroc']:.3f}", flush=True)
    stalled = [sc for sc in SUPERCLASSES if per_sc[sc]["fit_grad_norm"] > GRAD_CONVERGED]
    if stalled:
        print(f"  WARNING: {', '.join(stalled)} did not converge (gradient norm > "
              f"{GRAD_CONVERGED:g}). These AUPRCs understate the latents; raise "
              f"--max-iter above {args.max_iter}.")
    aps = [per_sc[sc]["auprc"] for sc in SUPERCLASSES if per_sc[sc]["auprc"] is not None]
    rocs = [per_sc[sc]["auroc"] for sc in SUPERCLASSES if per_sc[sc]["auroc"] is not None]
    per_sc["macro"] = {
        "converged": not stalled,
        "auprc": float(np.mean(aps)) if aps else None,
        "auroc": float(np.mean(rocs)) if rocs else None,
        "positive_rate": float(np.mean([per_sc[sc]["positive_rate"]
                                        for sc in SUPERCLASSES])),
    }
    return per_sc


def print_against_embeddings(latents: Dict[str, dict], embeddings: Dict[str, dict],
                             variant: str) -> None:
    """The line that answers the question: did the resampler keep the signal?"""
    print("\n" + "=" * 69)
    print(f"LATENTS vs INPUT EMBEDDINGS   (AUPRC / AUROC, variant: {variant})")
    print("=" * 69)
    print(f"{'superclass':<12} {'pos_rt':>7} {'embeddings':>18} {'latents':>18} "
          f"{'dAUPRC':>8}")
    print("-" * 69)
    for sc in list(SUPERCLASSES) + ["macro"]:
        lat, emb = latents.get(sc), embeddings.get(sc)
        if not lat or not emb:
            continue
        d = (lat["auprc"] - emb["auprc"]
             if lat["auprc"] is not None and emb["auprc"] is not None else None)
        print(f"{sc:<12} {lat['positive_rate']:>7.3f} "
              f"{emb['auprc']:>8.3f} / {emb['auroc']:>7.3f} "
              f"{lat['auprc']:>8.3f} / {lat['auroc']:>7.3f} "
              f"{(f'{d:+.3f}' if d is not None else 'n/a'):>8}")
    print("-" * 69)
    print("dAUPRC > 0 would mean the resampler made the superclass MORE linearly\n"
          "readable than it was in the cache; ~0 means it preserved it; strongly\n"
          "negative means the signal was lost before the student saw the prefix.")


def load_comparison(path: str, variant: str) -> Optional[Dict[str, dict]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    results = blob.get("results", {})
    table = results.get(variant) or results.get("centered") or next(iter(
        results.values()), None)
    return table


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible; the resampler pass needs a GPU")
    torch.manual_seed(args.seed)

    data_cfg = build_data_config(args)
    train_folds = sorted(DEFAULT_FOLD_SPLITS["train"])
    test_folds = sorted(DEFAULT_FOLD_SPLITS["test"])
    print(f"question mode: {args.question_mode}"
          + (f" (question: {args.fixed_question})" if args.question_mode == "fixed" else "")
          + f"   train folds {train_folds}   test fold {test_folds}")

    ctx = EvalContext.load(args.model_dir, args.ckpt)
    fit_device = (str(ctx.dev) if args.fit_device == "auto" else args.fit_device)
    print(f"fitting on {fit_device} (fp64)")

    train_ds = load_split(data_cfg, "train")
    test_ds = load_split(data_cfg, "test")

    print("\nrunning the resampler over the train split ...")
    train_feats, train_Y = features_for_split(ctx, train_ds, args, "train")
    print("running the resampler over the test split ...")
    test_feats, test_Y = features_for_split(ctx, test_ds, args, "test")

    dim = (next(iter(train_feats.values())) if isinstance(train_feats, dict)
           else train_feats).shape[1]
    print(f"\nfeatures: {dim} dims per record   "
          f"train {len(train_Y):,}   test {len(test_Y):,}")
    print("test positive rates: " + "  ".join(
        f"{sc} {test_Y[:, i].mean():.3f}" for i, sc in enumerate(SUPERCLASSES)))

    results = {}
    for variant in args.variants:
        print(f"\nfitting [{variant}] ...")
        results[variant] = fit_table(train_feats, train_Y, test_feats, test_Y,
                                     variant, args, fit_device)
        print_variant(f"{variant} (latents, {args.question_mode})", results[variant])

    head = args.variants[0]
    embeddings = load_comparison(args.compare_to, head)
    if embeddings:
        print_against_embeddings(results[head], embeddings, head)
    elif args.compare_to:
        print(f"\n(no embedding probe at {args.compare_to} to compare against; "
              "run scripts/probe_embeddings.py first)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "checkpoint": args.ckpt, "model_dir": args.model_dir,
            "question_mode": args.question_mode,
            "fixed_question": (args.fixed_question
                               if args.question_mode == "fixed" else None),
            "train_folds": train_folds, "test_fold": test_folds[0],
            "n_train": int(len(train_Y)), "n_test": int(len(test_Y)),
            "feature_dim": int(dim),
            "pooling": "mean over the 64 output latents",
            "l2": args.l2, "max_iter": args.max_iter, "seed": args.seed,
            "strip_think": bool(args.strip_think),
            "limit_records": args.limit_records,
            "compared_against": args.compare_to if embeddings else None,
            "results": results,
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
