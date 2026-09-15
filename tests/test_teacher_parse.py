"""Parsing of Mode B teacher responses.

Covers the happy path, the <END>-only fallback (no headers), missing blocks,
unparseable conclusions, empty content, and label-mismatch counting.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.teacher.parse import (
    SUPERCLASSES,
    detect_evidence_leakage,
    label_mismatches,
    leak_pattern_counts,
    leaked_superclasses,
    parse_response,
)

WELL_FORMED = """\
### NORM
Evidence: 1) sinus rhythm 2) normal axis
Reasoning:
Step 1: Using Evidence 1 and 2, the trace looks unremarkable.
Conclusion: No, another diagnostic finding is present.
<END>

### MI
Evidence: 1) Q waves anteroseptally
Reasoning:
Step 1: Using Evidence 1, this is consistent with infarction.
Conclusion: Yes, anteroseptal MI.
<END>

### STTC
Evidence: 1) ST/T changes noted
Reasoning:
Step 1: Using Evidence 1.
Conclusion: No, no ST/T diagnostic statement.
<END>

### CD
Evidence: 1) conduction block described
Reasoning:
Step 1: Using Evidence 1.
Conclusion: Yes, conduction disturbance.
<END>

### HYP
Evidence: 1) no hypertrophy criteria
Reasoning:
Step 1: Using Evidence 1.
Conclusion: No, no hypertrophy annotated.
<END>
"""


def test_well_formed_five_blocks():
    r = parse_response(WELL_FORMED)
    assert r.ok
    assert r.split_method == "header"
    assert set(r.blocks) == set(SUPERCLASSES)
    assert r.blocks["MI"].conclusion is True
    assert r.blocks["NORM"].conclusion is False
    assert "Q waves" in r.blocks["MI"].evidence
    assert "consistent with infarction" in r.blocks["MI"].reasoning
    assert "<END>" not in r.blocks["MI"].evidence


def test_label_mismatch_counting():
    r = parse_response(WELL_FORMED)
    known = {"NORM": False, "MI": True, "STTC": False, "CD": True, "HYP": False}
    assert label_mismatches(r, known) == []
    # flip MI's known answer -> one mismatch
    known_bad = {**known, "MI": False}
    assert label_mismatches(r, known_bad) == ["MI"]


def test_fallback_split_on_end_without_headers():
    content = "\n".join(
        f"Evidence: e{i}\nReasoning: r{i}\nConclusion: {'Yes' if i % 2 else 'No'}, x\n<END>"
        for i in range(5)
    )
    r = parse_response(content)
    assert r.split_method == "end"
    assert set(r.blocks) == set(SUPERCLASSES)
    assert r.blocks["NORM"].conclusion is False  # i=0 -> No


def test_missing_block_is_reported():
    content = WELL_FORMED.split("### HYP")[0]  # drop the HYP block
    r = parse_response(content)
    assert "HYP" in r.missing
    assert not r.ok


def test_unparseable_conclusion_is_reported():
    content = """\
### NORM
Evidence: 1) sinus rhythm
Reasoning: Step 1: Using Evidence 1.
Conclusion: probably normal
<END>
### MI
Conclusion: No, x
<END>"""
    r = parse_response(content)
    assert "NORM" in r.unparseable_conclusion
    assert r.blocks["NORM"].conclusion is None
    assert r.blocks["MI"].conclusion is False
    assert not r.ok


def test_empty_content():
    r = parse_response(None)
    assert not r.ok
    assert r.missing == list(SUPERCLASSES)
    assert r.errors
    assert parse_response("   ").errors


def test_prose_mentioning_codes_is_not_a_false_header():
    # A superclass code mid-sentence ("no evidence of CD here") must not be
    # treated as a header; only a code alone on its line is.
    content = """\
### NORM
Evidence: 1) sinus rhythm
Reasoning: Step 1: no evidence of CD here.
Conclusion: Yes, normal ECG.
<END>
### MI
Evidence: 1) none
Reasoning: Step 1: HYP not relevant.
Conclusion: No, no infarction.
<END>"""
    r = parse_response(content)
    assert set(r.blocks) == {"NORM", "MI"}  # CD/HYP mentioned in prose are not headers
    assert r.blocks["NORM"].conclusion is True
    assert "CD" in r.missing and "HYP" in r.missing


def test_to_dict_is_serialisable():
    import json
    d = parse_response(WELL_FORMED).to_dict()
    assert json.loads(json.dumps(d))["ok"] is True
    assert set(d) >= {"blocks", "missing", "unparseable_conclusion", "split_method", "ok"}


# --- annotation-leakage detection --------------------------------------------
# The exact leaked Evidence strings from the v2 pilot outputs
# (data/teacher/pilot_responses.jsonl), reproduced verbatim including their
# unicode (narrow no-break space  , curly quotes, non-breaking hyphen).
LEAK_8505_MI = (
    "1) Clinical report contains no comment on pathologic Q‑waves, "
    "ST‑segment elevation, or T‑wave inversion.  \n2) Recording "
    "properties list “Infarction stadium: not annotated”."
)
LEAK_8505_CD = (
    "1) Annotated finding “incomplete right bundle branch block” "
    "(likelihood 100).  \n2) Clinical report explicitly states "
    "“right bundle branch block”."
)
LEAK_16010_MI = (
    "1) Annotated diagnostic finding “inferior myocardial infarction” "
    "(likelihood 100). 2) Annotated diagnostic finding “ischemic in "
    "anteroseptal leads” (likelihood 100). 3) Annotated diagnostic "
    "finding “ischemic in anterolateral leads” (likelihood 100). "
    "4) Clinical report mentions q waves in II, III, aVF, typical of infarction. "
    "5) Recording property: Infarction stadium III."
)
LEAK_16010_CD = (
    "1) Rhythm annotation “sinus rhythm” (likelihood 0). 2) No "
    "diagnostic statements indicating bundle‑branch block, atrioventricular "
    "block, or other conduction delays. 3) No mention of prolonged QRS duration "
    "or abnormal axis in the report. 4) Signal‑quality annotation is absent, "
    "suggesting the trace is interpretable. 5) Absence of any form or diagnostic "
    "finding labeled as a conduction disturbance."
)
LEAK_1425_NORM = (
    "1) Annotated diagnostic finding “anteroseptal myocardial infarction” "
    "(likelihood 100). 2) Annotated diagnostic finding “inferior "
    "myocardial infarction” (likelihood 50). 3) Clinical report states "
    "“avvikande QRS(T) … inferior infarkt” (abnormal QRS/T course). "
    "4) Signal‑quality note: baseline drift in leads II, III."
)


def _patterns(hits):
    return {h["pattern"] for h in hits}


def test_leak_8505_mi_infarction_stadium_and_absence():
    hits = detect_evidence_leakage(LEAK_8505_MI)
    assert {"infarction stadium", "not annotated", "annotat"} <= _patterns(hits)
    lines = {h["line"] for h in hits}
    # the leak is the second numbered item; the first item is clean
    assert any("Infarction stadium: not annotated" in ln for ln in lines)
    assert all("no comment on pathologic" not in ln for ln in lines)


def test_leak_8505_cd_annotated_finding_and_likelihood():
    assert {"annotat", "likelihood"} <= _patterns(detect_evidence_leakage(LEAK_8505_CD))


def test_leak_16010_mi_diagnostic_finding():
    hits = detect_evidence_leakage(LEAK_16010_MI)
    assert {"annotat", "diagnostic finding", "likelihood", "infarction stadium"} <= _patterns(hits)


def test_leak_16010_cd_diagnostic_statement():
    pats = _patterns(detect_evidence_leakage(LEAK_16010_CD))
    assert "diagnostic statement" in pats   # "No diagnostic statements indicating ..."
    assert "diagnostic finding" in pats     # "... form or diagnostic finding labeled ..."


def test_leak_1425_norm_diagnostic_finding_likelihood():
    hits = detect_evidence_leakage(LEAK_1425_NORM)
    assert {"annotat", "diagnostic finding", "likelihood"} <= _patterns(hits)


def test_clean_evidence_has_no_leak():
    clean = "1) Sinus rhythm. 2) Q waves in II, III, aVF. 3) ST depression in I, aVL, V5, V6."
    assert detect_evidence_leakage(clean) == []
    assert detect_evidence_leakage(None) == []


def test_leak_heart_axis_field_cited_as_evidence():
    # v4: the heart-axis recording-property field is no longer a permitted Evidence
    # source, so citing it is leakage.
    hits = detect_evidence_leakage("1) Heart axis: left axis deviation (LAD).")
    assert "heart axis" in _patterns(hits)


def test_report_axis_observation_is_not_flagged_as_heart_axis_leak():
    # A report-derived axis observation (no "heart axis" field reference) is fine.
    assert detect_evidence_leakage("1) Left axis deviation on the trace.") == []


def test_offending_line_reported_verbatim():
    hits = [h for h in detect_evidence_leakage(LEAK_8505_CD) if h["pattern"] == "likelihood"]
    assert hits and "Annotated finding" in hits[0]["line"]


def test_leakage_surfaced_per_block_and_aggregated():
    content = f"""### NORM
Evidence: {LEAK_1425_NORM}
Reasoning: Step 1: nothing bearing on this question.
Conclusion: No, the recording as described is not a normal ECG.
<END>
### MI
Evidence: 1) q waves in II, III, aVF.
Reasoning: Step 1: consistent with infarction.
Conclusion: Yes, inferior MI.
<END>"""
    r = parse_response(content)
    assert r.blocks["NORM"].leaks          # per-block leak recorded
    assert not r.blocks["MI"].leaks         # clean block has none
    assert leaked_superclasses(r) == ["NORM"]
    counts = leak_pattern_counts(r)
    assert counts.get("annotat", 0) >= 1
    assert counts.get("diagnostic finding", 0) >= 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
