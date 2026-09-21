#!/usr/bin/env python
"""Apply the report-referencing rewrite to a teacher output file and report on it.

For every record: take ``cleaned_content``, rewrite the report-referencing clauses
(:func:`src.teacher.postprocess.rewrite_content`), and re-parse the result into a
fresh ``parsed_clean``. Reports how many blocks each rewrite rule changed, prints a
sample of before/after sentence pairs to eyeball the grammar, and counts any
report-referencing residual left behind (e.g. positive "the report states X"
references, which are deliberately not rewritten).

    python scripts/rewrite_report_refs.py --in data/teacher/full_v6.jsonl --pairs 30
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.teacher import parse as P
from src.teacher import postprocess as PP


def _changed_sentence_pairs(text):
    """Yield (rule_hits, before, after) for each sentence the rewrite changes."""
    for chunk in re.split(r"(?<=[.;\n])", text or ""):
        s = chunk.strip()
        if not s:
            continue
        after, counts = PP.rewrite_report_references(s)
        if counts:
            yield counts, s, after.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="infile", default="data/teacher/full_v6.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="only scan N records")
    ap.add_argument("--pairs", type=int, default=30, help="before/after pairs to print")
    args = ap.parse_args()

    blocks_per_rule = Counter()      # blocks where a rule fired at least once
    subs_per_rule = Counter()        # total substitutions per rule
    blocks_seen = 0
    blocks_rewritten = 0
    residual_before = 0
    residual_after = 0
    blocks_matching_before = 0       # blocks matching any pattern before rewrite
    blocks_matching_after = 0        # blocks STILL matching any pattern afterwards
    residual_blocks_per_pattern = Counter()   # after-rewrite, by residual pattern
    net_new_parse_failures = 0       # parsed clean before, broke after rewrite
    pairs_by_rule = {}               # single-rule name -> list of (before, after)
    residual_samples = []

    with open(args.infile, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cleaned = d.get("cleaned_content") or ""

            # 1. rewrite the whole cleaned_content and re-parse into parsed_clean.
            rewritten, _ = PP.rewrite_content(cleaned)
            before_ok = not P.parse_response(cleaned).errors
            after_ok = not P.parse_response(rewritten).errors
            if before_ok and not after_ok:
                net_new_parse_failures += 1

            # 2. per-block accounting over this record's blocks.
            for sc, block in P.parse_response(cleaned).blocks.items():
                text = block.raw_block or ""
                blocks_seen += 1
                new_text, counts = PP.rewrite_report_references(text)
                if counts:
                    blocks_rewritten += 1
                    for rule, n in counts.items():
                        blocks_per_rule[rule] += 1
                        subs_per_rule[rule] += n
                before_res = PP.count_report_reference_residual(text)
                after_res = PP.count_report_reference_residual(new_text)
                residual_before += before_res
                residual_after += after_res
                blocks_matching_before += 1 if before_res else 0
                blocks_matching_after += 1 if after_res else 0
                for pat in PP.report_reference_pattern_hits(new_text):
                    residual_blocks_per_pattern[pat] += 1
                if after_res and len(residual_samples) < 15:
                    for rx in PP._RESIDUAL_PATTERNS.values():
                        m = rx.search(new_text)
                        if m:
                            residual_samples.append(new_text[max(0, m.start() - 25):m.end() + 25])
                            break

            # 3. collect before/after pairs bucketed by rule for an even spread.
            for counts, before, after in _changed_sentence_pairs(cleaned):
                if len(counts) != 1:
                    continue
                (rule,) = counts
                pairs_by_rule.setdefault(rule, [])
                if len(pairs_by_rule[rule]) < args.pairs:
                    pairs_by_rule[rule].append((before, after))

    # --- report -----------------------------------------------------------
    print(f"scanned blocks: {blocks_seen}   rewritten: {blocks_rewritten} "
          f"({100 * blocks_rewritten / max(blocks_seen, 1):.1f}%)")
    print(f"net-new re-parse failures from the rewrite: {net_new_parse_failures}")
    print(f"report-referencing residual (matches)  before: {residual_before}  after: {residual_after}")
    print(f"blocks still matching any pattern       before: {blocks_matching_before}  "
          f"after: {blocks_matching_after}  "
          f"({100 * blocks_matching_after / max(blocks_seen, 1):.2f}% of blocks)")
    print("\nblocks STILL matching after rewrite, broken down by pattern:")
    for pat in PP._RESIDUAL_PATTERNS:
        print(f"  {pat:18} {residual_blocks_per_pattern[pat]:>6}")

    print("\nblocks rewritten per rule (and total substitutions):")
    order = [name for name, _, _ in PP.REWRITE_RULES]
    print(f"  {'rule':22} {'blocks':>8} {'subs':>8}")
    for name in order:
        print(f"  {name:22} {blocks_per_rule[name]:>8} {subs_per_rule[name]:>8}")

    # Round-robin across rules so the printed pairs span every pattern.
    selected = []
    idx = 0
    while len(selected) < args.pairs and any(
        idx < len(pairs_by_rule.get(r, [])) for r in order
    ):
        for r in order:
            bucket = pairs_by_rule.get(r, [])
            if idx < len(bucket):
                selected.append((r, *bucket[idx]))
                if len(selected) >= args.pairs:
                    break
        idx += 1

    print(f"\n{len(selected)} before/after pairs:")
    for j, (rule, before, after) in enumerate(selected, 1):
        print(f"\n[{j}] {rule}")
        print(f"  BEFORE: {before}")
        print(f"  AFTER : {after}")

    if residual_samples:
        print(f"\nsample of report-referencing residual left un-rewritten "
              f"({len(residual_samples)} shown):")
        for s in residual_samples:
            print(f"  ...{s.strip()}...")


if __name__ == "__main__":
    main()
