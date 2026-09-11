#!/usr/bin/env python
"""Build the PTB-XL label manifest once and write it to disk.

Joins ptbxl_database.csv + scp_statements.csv into a single JSONL file keyed by
ecg_id (see src/manifest.py for the format rationale). Everything downstream
reads that file and never touches the raw CSVs again.

Usage:
    python scripts/build_manifest.py --ptb data/ptbxl
    python scripts/build_manifest.py --ptb data/ptbxl --out data/ptbxl/manifest.jsonl --force
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import manifest as M


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ptb", default="data/ptbxl", help="PTB-XL root (contains the CSVs)")
    ap.add_argument("--out", default=None, help="output path (default: <ptb>/manifest.jsonl)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = ap.parse_args()

    out = args.out or os.path.join(args.ptb, M.DEFAULT_MANIFEST_NAME)

    if os.path.exists(out) and not args.force:
        print(f"{out} already exists; pass --force to overwrite.")
        return

    records = M.build_records(args.ptb)
    M.write_manifest(records, out)
    print(f"wrote {len(records)} records -> {out}\n")

    M.print_summary(records)


if __name__ == "__main__":
    main()
