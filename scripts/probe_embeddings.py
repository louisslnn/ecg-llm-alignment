#!/usr/bin/env python
"""Linear probe on the frozen ECG-FM embeddings. CPU only, no LLM anywhere.

How much of each diagnostic superclass is linearly readable from the cached
embeddings before any of this project's machinery touches them? That number is the
ceiling the aligner is working under and the floor it has to beat: if a logistic
regression on mean-pooled embeddings reads MI at 0.45 AUPRC, then a resampler and a
14B that come back at 0.20 are losing information that was already there, and one
that comes back at 0.45 may be doing nothing but relaying the probe.

    embeddings (N, 312, 768)  -> mean-pool over positions -> (N, 768) -> logistic
    regression, one binary fit per superclass, trained on folds 1-8, scored on fold 10.

FIVE BINARY FITS, NOT ONE MULTINOMIAL. The superclasses are not exclusive -- a
record can be MI and STTC at once -- so a multinomial fit over five classes would
model a competition that does not exist in the labels. One-vs-rest per superclass
matches how the task is actually posed to the student: five independent yes/no
questions about one recording.

THREE FEATURE VARIANTS, because the cache has an unusual shape (per-position norms
~350, and any two positions across any two records have cosine similarity ~0.97 --
about 98.6% of each vector's magnitude is one direction shared by everything):

    raw          mean-pooled embeddings as cached
    centered     minus the TRAIN-split mean vector (never the test mean: that would
                 leak the test distribution into the features)
    centered_l2  centered, then scaled to unit L2 norm per record

Read the comparison with this in mind: centering is a pure shift, and a logistic
regression carries a bias term, so raw and centered have the SAME optimum, and any
gap between them is optimiser conditioning rather than information. Measured on
STTC: raw reaches 0.675 AUPRC given 3000 LBFGS iterations, which is what centered
reaches in 500 -- same answer, ~6x the work. That is the finding, not a difference
in what the features carry, and it is the same headwind a resampler trained by SGD
faces on these inputs. The `grad` column reports the final gradient norm of each
fit, and a fit that did not converge is called out, so a slow variant can never be
misread as a weak one. centered_l2 IS a per-sample transform, so a gap there is
real: it discards each record's magnitude, and that magnitude carries signal.

AUPRC is reported against the positive rate, as everywhere else in this project: it
has no fixed floor, and a 0.30 AUPRC means opposite things at a 0.10 and a 0.30
positive rate. `lift` is AUPRC / positive rate.

    python scripts/probe_embeddings.py
    python scripts/probe_embeddings.py --variants raw centered --out results/probe.json

Runs in a few minutes on a laptop (the raw variant needs the most iterations);
no GPU, no model checkpoint, no network.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import DEFAULT_FOLD_SPLITS, DatasetConfig, EmbeddingCache  # noqa: E402
from src.manifest import EXCLUDED_ECG_IDS  # noqa: E402
from src.teacher.prompt import SUPERCLASSES, superclass_answer  # noqa: E402
from src.teacher.view import build_view  # noqa: E402
# The same AUPRC/AUROC the model is scored with (scripts/evaluate.py), so the probe
# and the pipeline are measured on one ruler rather than two implementations.
from evaluate import auprc, auroc  # noqa: E402

VARIANTS = ["raw", "centered", "centered_l2"]

# Above this final gradient norm a fit has not converged and its AUPRC understates
# the features. Calibrated against sklearn: 2e-3 matched it to within 0.002 AUPRC,
# 7e-2 was 0.05 short.
GRAD_CONVERGED = 1e-2


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-cache", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--train-folds", type=int, nargs="+",
                    default=sorted(DEFAULT_FOLD_SPLITS["train"]),
                    help="PTB-XL folds to train on (default 1-8)")
    ap.add_argument("--test-fold", type=int, default=10,
                    help="fold to score on (default 10, the held-out test fold)")
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--l2", type=float, default=1e-4,
                    help="L2 penalty on the weights (not the bias), per example")
    ap.add_argument("--max-iter", type=int, default=3000,
                    help="LBFGS iteration cap. Convergence is governed by the "
                         "gradient tolerance; this is only a ceiling, and the raw "
                         "variant needs most of it (~6x the centered one)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.abspath(
        os.path.join(REPO_ROOT, "results", "probe_embeddings.json")))
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #


def load_manifest(path: str) -> Dict[int, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[int(rec["ecg_id"])] = rec
    return out


def build_matrix(cache: EmbeddingCache, manifest: Dict[int, dict], folds):
    """Mean-pooled embeddings and labels for every record in ``folds``.

    Pooling is a plain mean over the 312 positions. The positions are near-collinear
    (cosine ~0.97), so a mean is a faithful summary of this particular cache rather
    than a lossy one -- and it is what a linear probe on a sequence encoder
    conventionally uses.
    """
    wanted = set(folds)
    X, Y, ids = [], [], []
    for eid, rec in manifest.items():
        if eid in EXCLUDED_ECG_IDS or int(rec["strat_fold"]) not in wanted:
            continue
        if eid not in cache:
            continue
        emb = cache.get(eid)                      # (312, 768) fp32
        X.append(emb.mean(axis=0))                # mean-pool over positions
        view = build_view(rec)
        Y.append([superclass_answer(view, sc) for sc in SUPERCLASSES])
        ids.append(eid)
    return (np.stack(X).astype(np.float64),
            np.array(Y, dtype=bool),
            np.array(ids, dtype=np.int64))


def apply_variant(train_X: np.ndarray, test_X: np.ndarray, variant: str):
    """Feature transform, fitted on TRAIN only and applied to both."""
    if variant == "raw":
        return train_X, test_X
    mean = train_X.mean(axis=0, keepdims=True)     # train mean, never the test mean
    tr, te = train_X - mean, test_X - mean
    if variant == "centered":
        return tr, te
    if variant == "centered_l2":
        def unit(a):
            n = np.linalg.norm(a, axis=1, keepdims=True)
            return a / np.maximum(n, 1e-12)
        return unit(tr), unit(te)
    raise ValueError(f"unknown variant {variant!r}")


# --------------------------------------------------------------------------- #
# Logistic regression (torch LBFGS; sklearn is not in the Alliance wheelhouse)  #
# --------------------------------------------------------------------------- #


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float, max_iter: int, seed: int):
    """One binary logistic regression by full-batch LBFGS.

    Returns ``(w, b, info)``. ``info`` carries the final loss and gradient norm so
    an underfit probe is visible rather than silently reported as a weak embedding:
    this script's whole purpose is measuring how much signal is there, and a probe
    that failed to converge understates exactly that.
    """
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float64)
    yt = torch.as_tensor(y.astype(np.float64), dtype=torch.float64)
    w = torch.zeros(Xt.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        logits = Xt @ w + b
        # mean BCE + L2 on the weights only; penalising the bias would fight the
        # class prior, which is not what regularisation is for here
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
        loss = loss + l2 * w.pow(2).sum()
        loss.backward()
        return loss

    initial = float(opt.step(closure).detach())   # LBFGS returns the loss at the
                                                  # START, not the optimum
    with torch.no_grad():
        grad_norm = float(torch.cat([w.grad.flatten(), b.grad.flatten()]).norm())
        final = float(torch.nn.functional.binary_cross_entropy_with_logits(
            Xt @ w + b, yt) + l2 * w.pow(2).sum())
    return (w.detach().numpy(), float(b.detach()),
            {"loss": final, "initial_loss": initial, "grad_norm": grad_norm})


def scores(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b        # AUPRC/AUROC are rank statistics: no sigmoid needed


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #


def evaluate_variant(train_X, train_Y, test_X, test_Y, args) -> Dict[str, dict]:
    per_sc = {}
    for i, sc in enumerate(SUPERCLASSES):
        y_tr, y_te = train_Y[:, i], test_Y[:, i]
        t0 = time.time()
        w, b, info = fit_logreg(train_X, y_tr, args.l2, args.max_iter, args.seed)
        s = scores(test_X, w, b)
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
    stalled = [sc for sc in SUPERCLASSES
               if per_sc[sc]["fit_grad_norm"] > GRAD_CONVERGED]
    if stalled:
        print(f"  WARNING: {', '.join(stalled)} did not converge (gradient norm > "
              f"{GRAD_CONVERGED:g}). These AUPRCs understate the features; raise "
              f"--max-iter above {args.max_iter}.")
    aps = [m["auprc"] for m in per_sc.values() if m["auprc"] is not None]
    rocs = [m["auroc"] for m in per_sc.values() if m["auroc"] is not None]
    per_sc["macro"] = {
        "converged": not stalled,
        "auprc": float(np.mean(aps)) if aps else None,
        "auroc": float(np.mean(rocs)) if rocs else None,
        "positive_rate": float(np.mean([m["positive_rate"] for sc, m in per_sc.items()
                                        if sc in SUPERCLASSES])),
    }
    return per_sc


def print_variant(name: str, table: Dict[str, dict]) -> None:
    header = (f"{'superclass':<12} {'n_test':>7} {'pos_rt':>7} {'AUPRC':>7} "
              f"{'lift':>6} {'AUROC':>7} {'grad':>9}")
    print(f"\n--- {name} " + "-" * max(0, len(header) - len(name) - 5))
    print(header)
    for sc in SUPERCLASSES:
        m = table[sc]
        print(f"{sc:<12} {m['n_test']:>7} {m['positive_rate']:>7.3f} "
              f"{m['auprc']:>7.3f} {m['auprc_lift']:>6.2f} {m['auroc']:>7.3f} "
              f"{m['fit_grad_norm']:>9.2e}")
    macro = table["macro"]
    print(f"{'macro':<12} {'':>7} {macro['positive_rate']:>7.3f} "
          f"{macro['auprc']:>7.3f} {'':>6} {macro['auroc']:>7.3f}")


def print_comparison(results: Dict[str, Dict[str, dict]]) -> None:
    names = list(results)
    header = f"{'superclass':<12} " + " ".join(f"{n:>18}" for n in names)
    print("\n" + "=" * len(header))
    print("AUPRC / AUROC BY FEATURE VARIANT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for sc in SUPERCLASSES + ["macro"]:
        cells = []
        for n in names:
            m = results[n][sc]
            cells.append(f"{m['auprc']:>8.3f} /{m['auroc']:>7.3f}")
        print(f"{sc:<12} " + " ".join(f"{c:>18}" for c in cells))
    print("-" * len(header))
    print("\nraw and centered share an optimum (centering is a shift, and the model "
          "has a\nbias term), so agreement between those two columns is the expected "
          "result and\nconfirms both fits converged. A gap would mean the raw fit ran "
          "out of\niterations, not that the features differ -- centering buys ~6x "
          "faster\nconvergence, nothing more. centered_l2 IS a per-sample transform and "
          "does change\nthe problem: it discards each record's magnitude, which costs "
          "real AUPRC.")


def main():
    args = parse_args()
    cfg_kwargs = {}
    if args.emb_cache:
        cfg_kwargs["emb_cache_dir"] = args.emb_cache
    if args.manifest:
        cfg_kwargs["manifest_path"] = args.manifest
    cfg = DatasetConfig(**cfg_kwargs)

    for label, path in (("embedding index", os.path.join(cfg.emb_cache_dir, "index.json")),
                        ("manifest", cfg.manifest_path)):
        if not os.path.exists(path):
            sys.exit(f"missing {label}: {path}")

    print(f"cache    {cfg.emb_cache_dir}")
    print(f"manifest {cfg.manifest_path}")
    cache = EmbeddingCache(cfg.emb_cache_dir)
    manifest = load_manifest(cfg.manifest_path)

    t0 = time.time()
    train_X, train_Y, train_ids = build_matrix(cache, manifest, args.train_folds)
    test_X, test_Y, test_ids = build_matrix(cache, manifest, [args.test_fold])
    print(f"\ntrain folds {args.train_folds}: {train_X.shape[0]:,} records  "
          f"test fold {args.test_fold}: {test_X.shape[0]:,} records "
          f"({time.time()-t0:.1f}s to pool)")
    assert not (set(train_ids.tolist()) & set(test_ids.tolist())), "fold overlap"
    print(f"mean-pooled features: {train_X.shape[1]} dims, "
          f"per-record norm {np.linalg.norm(train_X, axis=1).mean():.1f}")
    print("test positive rates: " + "  ".join(
        f"{sc} {test_Y[:, i].mean():.3f}" for i, sc in enumerate(SUPERCLASSES)))

    results = {}
    for variant in args.variants:
        tr, te = apply_variant(train_X, test_X, variant)
        print(f"\n[{variant}] per-record norm: train "
              f"{np.linalg.norm(tr, axis=1).mean():.3f}  test "
              f"{np.linalg.norm(te, axis=1).mean():.3f}", flush=True)
        results[variant] = evaluate_variant(tr, train_Y, te, test_Y, args)
        print_variant(variant, results[variant])

    if len(results) > 1:
        print_comparison(results)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "train_folds": args.train_folds, "test_fold": args.test_fold,
            "n_train": int(train_X.shape[0]), "n_test": int(test_X.shape[0]),
            "l2": args.l2, "max_iter": args.max_iter, "seed": args.seed,
            "pooling": "mean over 312 positions",
            "results": results,
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
