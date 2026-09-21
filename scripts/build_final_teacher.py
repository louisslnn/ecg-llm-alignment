#!/usr/bin/env python
"""Produce the final teacher dataset: merge regen, apply the report-reference rewrite.

  1. Merge data/teacher/regen_v6.jsonl into data/teacher/full_v6.jsonl, replacing
     each record whose ecg_id appears in the regen file. full_v6.jsonl is left
     untouched as provenance; the result is written to full_v6_final.jsonl.
  2. For every record, apply the report-reference rewrite to `cleaned_content`,
     store it in a NEW field `final_content`, and re-parse into `parsed_final`.
     `cleaned_content` and `parsed_clean` are kept exactly as they were.
  3. Read the written file back off disk and verify: 21,793 records, no duplicate
     ecg_ids, every regen id present with its new content, the strip invariant, and
     final counts for parse failures, empty Reasoning, residual report-references,
     and label mismatches.

    python scripts/build_final_teacher.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.teacher import parse as P
from src.teacher import postprocess as PP
from src.teacher.prompt import SUPERCLASSES

TEACHER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "teacher")
FULL = os.path.join(TEACHER_DIR, "full_v6.jsonl")
REGEN = os.path.join(TEACHER_DIR, "regen_v6.jsonl")
OUT = os.path.join(TEACHER_DIR, "full_v6_final.jsonl")


def load_regen():
    regen = {}
    with open(REGEN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                regen[int(r["ecg_id"])] = r
    return regen


def build(regen):
    """Stream full_v6, replace regen ids, add final_content + parsed_final; write OUT."""
    applied = set()
    n = 0
    with open(FULL, encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            eid = int(d["ecg_id"])
            if eid in regen:
                d = regen[eid]          # replace the whole record with the regen one
                applied.add(eid)

            cleaned = d.get("cleaned_content") or ""
            final_content, _ = PP.rewrite_content(cleaned)      # report-reference rewrite
            parsed_final = P.parse_response(final_content)      # re-parse
            d["final_content"] = final_content                  # NEW field, kept alongside
            d["parsed_final"] = parsed_final.to_dict()          # NEW field
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return applied, n


def verify(regen):
    """Read OUT back off disk and check everything."""
    ids = []
    parse_fail_blocks = parse_fail_records = 0
    empty_reasoning_blocks = 0
    residual_blocks = residual_records = 0
    label_mismatch_blocks = label_mismatch_records = 0
    strip_leak_blocks = strip_leak_records = 0
    missing_final_field = 0
    regen_verified = 0

    with open(OUT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            eid = int(d["ecg_id"])
            ids.append(eid)

            # the rewrite must be persisted, not just computed
            if "final_content" not in d or "parsed_final" not in d:
                missing_final_field += 1
                continue

            # regen records must carry their regenerated content, plus final_content
            # derived from it (independent recomputation off the written cleaned_content)
            if eid in regen:
                assert d["cleaned_content"] == regen[eid]["cleaned_content"], eid
                expected_final = PP.rewrite_content(d["cleaned_content"])[0]
                assert d["final_content"] == expected_final, eid
                regen_verified += 1

            pf = d["parsed_final"]
            blocks = pf.get("blocks") or {}
            known = d.get("known_answers") or {}

            rec_pf = rec_res = rec_lm = rec_leak = False
            if not pf.get("ok", False):
                parse_fail_records += 1
            parse_fail_blocks += len(pf.get("missing") or []) + len(pf.get("unparseable_conclusion") or [])

            for sc in SUPERCLASSES:
                b = blocks.get(sc)
                if not b:
                    continue
                if not (b.get("reasoning") or "").strip():
                    empty_reasoning_blocks += 1
                # strip invariant: no supervision-pipeline tell survives in parsed_final
                if PP.find_leak_steps(b.get("reasoning")):
                    strip_leak_blocks += 1
                    rec_leak = True
                # residual report-references in the rewritten block
                if PP.count_report_reference_residual(b.get("raw_block")):
                    residual_blocks += 1
                    rec_res = True

            for sc in P.label_mismatches(_as_result(pf), known):
                label_mismatch_blocks += 1
                rec_lm = True

            residual_records += rec_res
            label_mismatch_records += rec_lm
            strip_leak_records += rec_leak

    return {
        "n_records": len(ids),
        "n_unique": len(set(ids)),
        "duplicate_ids": [i for i in set(ids) if ids.count(i) > 1][:10],
        "regen_verified": regen_verified,
        "missing_final_field": missing_final_field,
        "parse_fail_records": parse_fail_records,
        "parse_fail_blocks": parse_fail_blocks,
        "empty_reasoning_blocks": empty_reasoning_blocks,
        "residual_blocks": residual_blocks,
        "residual_records": residual_records,
        "label_mismatch_blocks": label_mismatch_blocks,
        "label_mismatch_records": label_mismatch_records,
        "strip_leak_blocks": strip_leak_blocks,
        "strip_leak_records": strip_leak_records,
    }


class _R:
    """Minimal shim so parse.label_mismatches can read a parsed_final dict."""
    def __init__(self, pf):
        self.blocks = {sc: _B(b) for sc, b in (pf.get("blocks") or {}).items()}


class _B:
    def __init__(self, b):
        self.conclusion = b.get("conclusion")


def _as_result(pf):
    return _R(pf)


def main():
    regen = load_regen()
    print(f"regen records: {len(regen)}")
    applied, n = build(regen)
    print(f"wrote {n} records to {os.path.relpath(OUT)}")
    print(f"regen ids applied: {len(applied)} / {len(regen)}")
    assert applied == set(regen), sorted(set(regen) - applied)

    v = verify(regen)
    print("\n=== VERIFY (read back off disk) ===")
    print(f"records: {v['n_records']}   unique ecg_ids: {v['n_unique']}   "
          f"duplicates: {v['duplicate_ids'] or 'none'}")
    print(f"records missing final_content/parsed_final: {v['missing_final_field']}")
    print(f"regen ids present with new content (verified): {v['regen_verified']} / {len(regen)}")
    print(f"strip invariant violations: {v['strip_leak_blocks']} blocks "
          f"in {v['strip_leak_records']} records  (must be 0)")
    print("\n=== FINAL COUNTS (over parsed_final) ===")
    print(f"parse failures:          {v['parse_fail_blocks']} blocks "
          f"({v['parse_fail_records']} records not ok)")
    print(f"empty Reasoning:         {v['empty_reasoning_blocks']} blocks")
    print(f"residual report-refs:    {v['residual_blocks']} blocks "
          f"({v['residual_records']} records)")
    print(f"label mismatches:        {v['label_mismatch_blocks']} blocks "
          f"({v['label_mismatch_records']} records)")

    ok = (v["n_records"] == 21793 and v["n_unique"] == 21793 and not v["duplicate_ids"]
          and v["missing_final_field"] == 0 and v["regen_verified"] == len(regen)
          and v["strip_leak_blocks"] == 0)
    print("\nRESULT:", "OK" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
