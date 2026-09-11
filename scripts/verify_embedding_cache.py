#!/usr/bin/env python
"""Verify the ECG-FM embedding cache. Standalone -- not part of the cache build.

Checks:
  1. Coverage: the index maps every working-set ecg_id to a unique (shard, row),
     no gaps, no duplicates (--allow-partial accepts a --limit subset).
  2. Shards: every shard loads with mmap_mode="r", is float16, and has shape
     (n, 312, 768) with n matching the index.
  3. Finiteness: no NaN/Inf across a random sample of rows.
  4. Fidelity: for --n random records, re-encode LIVE in fp32, cast to fp16 and
     assert EXACT equality with the stored bytes. The forward rounds differently
     per batch size, so each record is re-encoded inside its exact original
     mini-batch (reconstructed from the index) -- exact equality is correct here.
  5. Reports total cache size on disk.

Usage:
    python scripts/verify_embedding_cache.py --device cpu --n 5
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import cache as C
from src import encoder as E
from src import manifest as M


def check_coverage(index, working_ids, allow_partial):
    records = index["records"]  # [ecg_id, shard_idx, row]
    index_ids = [r[0] for r in records]

    assert len(index_ids) == len(set(index_ids)), "duplicate ecg_id in index"

    slots = [(r[1], r[2]) for r in records]
    assert len(slots) == len(set(slots)), "duplicate (shard, row) slot in index"

    by_shard = {}
    for eid, si, row in records:
        by_shard.setdefault(si, []).append(row)
    for si, rows in by_shard.items():
        assert sorted(rows) == list(range(len(rows))), f"shard {si} has row gaps/dupes"

    idx_set, work_set = set(index_ids), set(working_ids)
    assert idx_set <= work_set, f"index has {len(idx_set - work_set)} ids outside working set"
    if allow_partial:
        print(f"coverage: {len(idx_set)}/{len(work_set)} working-set records (partial cache allowed)")
    else:
        missing = work_set - idx_set
        assert not missing, f"index missing {len(missing)} working-set ids, e.g. {sorted(missing)[:5]}"
        assert idx_set == work_set
        print(f"coverage: index covers all {len(work_set)} working-set records, no gaps, no duplicates")


def check_shards(cache_dir, index):
    """Every shard mmaps, is float16, and is (n_expected, 312, 768)."""
    expected_rows = {}
    for _eid, si, _row in index["records"]:
        expected_rows[si] = expected_rows.get(si, 0) + 1

    total = 0
    for si, name in enumerate(index["shards"]):
        path = os.path.join(cache_dir, name)
        arr = np.load(path, mmap_mode="r")
        assert arr.dtype == np.float16, f"{name}: dtype {arr.dtype} != float16"
        assert arr.shape == (expected_rows[si], E.ENCODER_TOKENS, E.ENCODER_DIM), (
            f"{name}: shape {arr.shape} != {(expected_rows[si], E.ENCODER_TOKENS, E.ENCODER_DIM)}"
        )
        total += arr.shape[0]
    assert total == len(index["records"]), "shard rows do not sum to index size"
    print(f"shards: {len(index['shards'])} files, all float16 (n, {E.ENCODER_TOKENS}, "
          f"{E.ENCODER_DIM}), {total} rows total")


def check_finite(cache_dir, index, n, seed):
    """No NaN/Inf across a random sample of n rows."""
    rng = random.Random(seed)
    records = index["records"]
    sample = rng.sample(records, min(n, len(records)))
    for eid, si, row in sample:
        path = os.path.join(cache_dir, index["shards"][si])
        arr = np.load(path, mmap_mode="r")
        vec = np.asarray(arr[row], dtype=np.float32)  # widen for a clean isfinite
        assert np.isfinite(vec).all(), f"ecg_id {eid}: non-finite values in cache"
    print(f"finiteness: {len(sample)} random rows checked, no NaN/Inf")


def reconstruct_minibatch(index, target_eid):
    """The exact mini-batch (ordered ecg_ids) the target was encoded in."""
    batch_size = index["batch_size"]
    records = index["records"]

    shard_idx = row = None
    for eid, si, r in records:
        if eid == target_eid:
            shard_idx, row = si, r
    shard_members = {r: eid for eid, si, r in records if si == shard_idx}
    ordered = [shard_members[r] for r in range(len(shard_members))]

    block_start = (row // batch_size) * batch_size
    block = ordered[block_start : block_start + batch_size]
    return block, block.index(target_eid), shard_idx, row


def read_cached(cache_dir, index, shard_idx, row):
    shard_path = os.path.join(cache_dir, index["shards"][shard_idx])
    arr = np.load(shard_path, mmap_mode="r")
    return torch.from_numpy(np.array(arr[row]))  # writable fp16 copy (312, 768)


def check_fidelity(cache_dir, index, ptb_root, ckpts, device, n, seed):
    rng = random.Random(seed)
    all_ids = [r[0] for r in index["records"]]
    sample = rng.sample(all_ids, min(n, len(all_ids)))

    model = E.load_encoder(E.download_checkpoint(ckpts), device=device)

    for eid in sample:
        block, pos, shard_idx, row = reconstruct_minibatch(index, eid)
        cached = read_cached(cache_dir, index, shard_idx, row)

        live = E.encode_full_records(model, block, ptb_root, device=device)[pos].to(torch.float16)

        assert live.shape == cached.shape == (E.ENCODER_TOKENS, E.ENCODER_DIM)
        assert torch.equal(live, cached), (
            f"ecg_id {eid}: live fp16 re-encode != cached "
            f"({int((live != cached).sum())} elements differ)"
        )
        print(f"ecg_id {eid:6d}: shard {shard_idx} row {row} "
              f"(mini-batch of {len(block)}) -> fp16 exact match")

    print(f"fidelity: {len(sample)} random records re-encoded, all exact")


def report_size(cache_dir, index):
    total = 0
    for name in index["shards"] + [C.INDEX_NAME]:
        p = os.path.join(cache_dir, name)
        if os.path.exists(p):
            total += os.path.getsize(p)
    gib = total / (1024 ** 3)
    print(f"cache size on disk: {total:,} bytes ({gib:.2f} GiB) across "
          f"{len(index['shards'])} shards + index")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/emb_cache")
    ap.add_argument("--manifest", default="data/ptbxl/manifest.jsonl")
    ap.add_argument("--ptb", default="data/ptbxl")
    ap.add_argument("--ckpts", default="data/ckpts")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=5, help="records to spot-check (fidelity + finiteness)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-partial", action="store_true",
                    help="accept an index that covers only a subset (e.g. a --limit build)")
    args = ap.parse_args()

    index = C.load_index(args.cache)
    working_ids = list(M.load_manifest(args.manifest).keys())

    check_coverage(index, working_ids, args.allow_partial)
    check_shards(args.cache, index)
    check_finite(args.cache, index, args.n, args.seed)
    check_fidelity(args.cache, index, args.ptb, args.ckpts, args.device, args.n, args.seed)
    report_size(args.cache, index)
    print("OK")


if __name__ == "__main__":
    main()
