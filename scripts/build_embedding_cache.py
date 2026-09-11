#!/usr/bin/env python
"""Cache ECG-FM encoder outputs for every record in the working set.

Reads data/ptbxl/manifest.jsonl (6 records excluded at load -> 21,793 working
set), runs the batched full-record encoder path, and stores pre-pooling
encoder_out as fp16 in shards of a few thousand records, plus an index mapping
ecg_id -> (shard, row). Records are written in manifest order.

Resumable: a shard whose .npy already exists with the right shape is skipped, so
a restart continues where it left off.

Numerics note: the encoder forward is batch-size-dependent in the last fp32 bits
(each sequence is computed independently, but GEMM kernels round differently for
different batch sizes). The cache is therefore only reproducible with the SAME
batch size; `batch_size` and the per-shard record order are recorded in the index
so scripts/verify_embedding_cache.py can reconstruct each exact mini-batch.

Usage:
    python scripts/build_embedding_cache.py --device cpu --batch-size 16
    python scripts/build_embedding_cache.py --device cuda --limit 200   # smoke test
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)

from src import cache as C
from src import data as D
from src import encoder as E
from src import manifest as M

log = logging.getLogger("build_embedding_cache")


def shard_filename(i: int) -> str:
    return f"shard_{i:03d}.npy"


def shard_is_complete(path: str, n_rows: int) -> bool:
    """A shard is complete iff the file exists with the expected fp16 shape."""
    if not os.path.exists(path):
        return False
    try:
        arr = np.load(path, mmap_mode="r")
    except (ValueError, OSError):
        return False
    return (
        arr.dtype == np.float16
        and arr.shape == (n_rows, E.ENCODER_TOKENS, E.ENCODER_DIM)
    )


def save_atomic(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp"
    np.save(tmp, arr, allow_pickle=False)
    # np.save appends .npy to a path without that suffix; normalise back.
    if not os.path.exists(tmp) and os.path.exists(tmp + ".npy"):
        tmp = tmp + ".npy"
    os.replace(tmp, path)


def encode_shard(model, ecg_ids, ptb_root, device, batch_size, pbar=None):
    """Encode one shard's records into an fp16 (n, 312, 768) array.

    Batches contiguously in the given order, `batch_size` records per forward.
    Advances `pbar` by each mini-batch so the bar moves smoothly within a shard.
    Returns (array, records_done, seconds_spent).
    """
    n = len(ecg_ids)
    out = np.empty((n, E.ENCODER_TOKENS, E.ENCODER_DIM), dtype=np.float16)
    t0 = time.time()
    for start in range(0, n, batch_size):
        chunk = ecg_ids[start : start + batch_size]
        emb = E.encode_full_records(model, chunk, ptb_root, device=device)  # fp32 (b,312,768)
        out[start : start + len(chunk)] = emb.to(torch.float16).numpy()
        if pbar is not None:
            pbar.update(len(chunk))
    return out, n, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/ptbxl/manifest.jsonl")
    ap.add_argument("--ptb", default="data/ptbxl", help="PTB-XL root (waveforms + CSVs)")
    ap.add_argument("--ckpts", default="data/ckpts")
    ap.add_argument("--out", default="data/emb_cache", help="cache output directory")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16, help="records per forward pass")
    ap.add_argument("--shard-size", type=int, default=2000, help="records per shard file")
    ap.add_argument("--limit", type=int, default=None, help="only cache the first N records")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the on-disk .dat/.hea check (when known redundant)")
    ap.add_argument("--force", action="store_true",
                    help="resume even if the cached manifest hash is stale")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    # Working set in manifest order (exclusions applied + logged by load_manifest).
    manifest = M.load_manifest(args.manifest)
    ecg_ids = list(manifest.keys())
    if args.limit is not None:
        ecg_ids = ecg_ids[: args.limit]
    n_total = len(ecg_ids)
    log.info("working set: %d records (limit=%s)", n_total, args.limit)

    # Staleness guard: refuse to resume a cache built from a different manifest.
    try:
        C.check_resume(args.out, args.manifest, force=args.force)
    except C.StaleCacheError as e:
        log.error("%s", e)
        log.error("the cache in %s is stale and must be rebuilt "
                  "(clear the directory) or re-run with --force to override.",
                  args.out)
        sys.exit(1)

    # Preflight: every working-set record must have BOTH .dat and .hea on disk
    # before we pay to load the encoder. Reports all misses, not just the first.
    if args.skip_preflight:
        log.info("preflight skipped (--skip-preflight)")
    else:
        missing = D.find_missing_records(args.ptb, ecg_ids)
        if missing:
            log.error("preflight failed: %d/%d working-set records missing .dat/.hea "
                      "on disk under %s", len(missing), n_total, args.ptb)
            log.error("first %d missing ecg_ids: %s",
                      min(10, len(missing)), missing[:10])
            sys.exit(1)
        log.info("preflight ok: all %d records present (.dat + .hea)", n_total)

    os.makedirs(args.out, exist_ok=True)

    model = E.load_encoder(E.download_checkpoint(args.ckpts), device=args.device)
    log.info("loaded encoder on device=%s", args.device)

    # Deterministic shard layout over the ordered working set.
    shards = [
        ecg_ids[i : i + args.shard_size]
        for i in range(0, n_total, args.shard_size)
    ]

    records = []  # [ecg_id, shard_idx, row] in manifest order
    for si, shard_ids in enumerate(shards):
        for row, eid in enumerate(shard_ids):
            records.append([eid, si, row])

    done_records = 0
    done_seconds = 0.0
    pbar = tqdm(total=n_total, unit="rec", desc="caching", dynamic_ncols=True,
                smoothing=0.1)
    for si, shard_ids in enumerate(shards):
        path = os.path.join(args.out, shard_filename(si))
        if shard_is_complete(path, len(shard_ids)):
            tqdm.write(f"shard {si + 1}/{len(shards)} complete, skipping "
                       f"({len(shard_ids)} records)")
            pbar.update(len(shard_ids))
            continue

        arr, n, secs = encode_shard(
            model, shard_ids, args.ptb, args.device, args.batch_size, pbar=pbar
        )
        save_atomic(path, arr)

        done_records += n
        done_seconds += secs
        rate = n / secs if secs > 0 else float("inf")
        avg_rate = done_records / done_seconds if done_seconds > 0 else float("inf")
        tqdm.write(
            f"shard {si + 1}/{len(shards)}: {n} records in {secs:.1f}s "
            f"({rate:.1f} rec/s | running avg {avg_rate:.1f} rec/s) -> {path} "
            f"| full-pass estimate {n_total / avg_rate / 60:.1f} min"
        )
    pbar.close()

    index = {
        "manifest": args.manifest,
        "manifest_sha256": C.hash_file(args.manifest),
        "git_commit": C.git_commit_if_clean(REPO_ROOT),
        "ptb_root": args.ptb,
        "device_used": args.device,
        "dtype": "float16",
        "token_dim": [E.ENCODER_TOKENS, E.ENCODER_DIM],
        "shard_size": args.shard_size,
        "batch_size": args.batch_size,
        "n_records": n_total,
        "shards": [shard_filename(i) for i in range(len(shards))],
        "records": records,
    }
    C.write_index(args.out, index)
    log.info("wrote index for %d records (manifest %s, git %s) -> %s",
             n_total, index["manifest_sha256"][:12],
             (index["git_commit"] or "dirty/none")[:12], C.index_path(args.out))

    if done_records:
        log.info(
            "encoded %d new records in %.1fs (%.1f rec/s); "
            "full %d-record pass at this rate ~= %.1f min",
            done_records, done_seconds, done_records / done_seconds,
            n_total, n_total / (done_records / done_seconds) / 60,
        )
    else:
        log.info("nothing to encode; all shards were already complete")


if __name__ == "__main__":
    main()
