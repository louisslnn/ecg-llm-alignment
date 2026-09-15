#!/usr/bin/env python
"""Run the v6 fix-#1 strip over an existing teacher output file and report.

Read-only: reads a Mode B responses jsonl, applies
:func:`src.teacher.postprocess.strip_content` to each record's raw content, and
prints every affected block before and after stripping, then the summary counts.
Nothing is written or regenerated.

    python scripts/strip_pilot.py --in data/teacher/pilot50_v5.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.teacher.postprocess import strip_content, strip_reasoning_steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", default="data/teacher/pilot50_v5.jsonl")
    args = ap.parse_args()

    blocks_stripped = 0
    steps_removed = 0
    blocks_emptied = 0
    affected = 0

    with open(args.in_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    for row in rows:
        ecg_id = row["ecg_id"]
        # Content-level strip drives the authoritative counts (it is what would be
        # cached); per-block detail below reuses the parsed reasoning for display.
        _, stats = strip_content(row.get("content"))
        blocks_stripped += stats.blocks_stripped
        steps_removed += stats.steps_removed
        blocks_emptied += stats.blocks_emptied

        for sc, block in row["parsed"]["blocks"].items():
            reasoning = block.get("reasoning")
            res = strip_reasoning_steps(reasoning)
            if not res.changed:
                continue
            affected += 1
            print("=" * 78)
            flag = "  [EMPTIED -> flagged]" if res.emptied else ""
            print(f"ecg_id={ecg_id}  {sc}  removed={res.removed} kept={res.kept}{flag}")
            print("-" * 78)
            print("BEFORE:")
            print(reasoning)
            print("-" * 78)
            print("AFTER:")
            print(res.cleaned.strip() or "(no reasoning steps remain)")

    print("=" * 78)
    print("SUMMARY")
    print(f"  affected blocks (displayed):     {affected}")
    print(f"  blocks stripped (content-level): {blocks_stripped}")
    print(f"  steps removed:                   {steps_removed}")
    print(f"  blocks left with empty Reasoning:{blocks_emptied}")


if __name__ == "__main__":
    main()
