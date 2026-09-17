"""Report-referencing rewrite in src.teacher.postprocess.

The BEFORE strings here are taken from real blocks in data/teacher/full_v6.jsonl
(hyphens are the U+2011 non-breaking form the annotators/model used). The rewrite
must reframe references to the unseen free-text report onto "the listed
observations" while leaving genuine clinical language ("no evidence of X"),
Evidence-referencing language, and characterisations ("described as X") untouched.

    pytest tests/test_teacher_report_rewrite.py
    python  tests/test_teacher_report_rewrite.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.teacher import parse as P
from src.teacher import postprocess as PP

INCLUDE = "the listed observations do not include"
AMONG = "not among the listed observations"


# --- the four canonical shapes the task specifies -------------------------
def test_canonical_the_report_does_not_describe():
    out, counts = PP.rewrite_report_references(
        "the report does not describe ST-segment elevation."
    )
    assert out == "the listed observations do not include ST-segment elevation."
    assert counts["report_active"] == 1


def test_canonical_x_is_not_described():
    out, counts = PP.rewrite_report_references(
        "ST/T abnormalities are not described, so the tracing is normal."
    )
    assert out == "ST/T abnormalities are not among the listed observations, so the tracing is normal."
    assert counts["passive_not_x"] == 1


def test_canonical_no_x_is_reported():
    out, counts = PP.rewrite_report_references("No conduction abnormality is reported.")
    assert out == "The listed observations do not include conduction abnormality."
    assert counts["passive_no_x"] == 1


def test_canonical_the_report_provides_contains_no():
    for verb in ("provides", "contains"):
        out, counts = PP.rewrite_report_references(
            f"the report {verb} no hypertrophy criteria."
        )
        assert out == "the listed observations do not include hypertrophy criteria.", verb
        assert counts["report_verb_no"] == 1


# --- real strings mined from full_v6.jsonl, one per rule ------------------
REAL_CASES = [
    # (before, expected_after, rule)
    ("Step 2: Evidence 2 describes low QRS voltage, but the report does not "
     "mention any accompanying abnormal morphology.",
     "Step 2: Evidence 2 describes low QRS voltage, but the listed observations do not "
     "include any accompanying abnormal morphology.",
     "report_active"),
    ("Step 3: Evidence 3‑4 describe artefacts that may obscure subtle changes, "
     "yet the report contains no description of such changes.",
     "Step 3: Evidence 3‑4 describe artefacts that may obscure subtle changes, "
     "yet the listed observations do not include such changes.",
     "report_verb_no"),
    ("Step 1: Evidence 1 reports only a sinus rhythm; there is no mention of widened "
     "QRS complexes, bundle‑branch block patterns, or other conduction delays.",
     "Step 1: Evidence 1 reports only a sinus rhythm; the listed observations do not "
     "include widened QRS complexes, bundle‑branch block patterns, or other conduction delays.",
     "there_no_mention"),
    ("Step 2: The listed observations contain no mention of pathological Q‑waves, "
     "ST‑segment elevation or depression, or other patterns.",
     "Step 2: The listed observations do not include pathological Q‑waves, "
     "ST‑segment elevation or depression, or other patterns.",
     "aux_no_mention"),
    ("Step 2: Evidence 2 reports the ECG as normal, providing no mention of "
     "Q‑waves or ST‑segment elevation.",
     "Step 2: Evidence 2 reports the ECG as normal, not including "
     "Q‑waves or ST‑segment elevation.",
     "participle_no_mention"),
    ("Step 1: Evidence 2 indicates that the ECG is otherwise normal, with no mention "
     "of increased QRS voltage, tall R‑waves, or other criteria for ventricular hypertrophy.",
     "Step 1: Evidence 2 indicates that the ECG is otherwise normal, with no increased "
     "QRS voltage, tall R‑waves, or other criteria for ventricular hypertrophy among the "
     "listed observations.",
     "with_no_mention"),
    ("Step 3: Since no conduction abnormality is described, the recording is normal.",
     "Step 3: Since the listed observations do not include conduction abnormality, "
     "the recording is normal.",
     "passive_no_x"),
    ("Step 2: Because such infarction‑related abnormalities are not mentioned, "
     "the recording shows no MI.",
     "Step 2: Because such infarction‑related abnormalities are not among the listed "
     "observations, the recording shows no MI.",
     "passive_not_x"),
    ("Step 2: No description of widened QRS, bundle‑branch block patterns, or "
     "intraventricular conduction delays is present.",
     "Step 2: The listed observations do not include widened QRS, bundle‑branch block "
     "patterns, or intraventricular conduction delays.",
     "bare_no_mention"),
    # "in the report" is consumed; the passive rule fires.
    ("Step 2: No ST elevation or depression is mentioned in the report.",
     "Step 2: The listed observations do not include ST elevation or depression.",
     "passive_no_x"),
]


def test_real_strings_per_rule():
    for before, expected, rule in REAL_CASES:
        out, counts = PP.rewrite_report_references(before)
        assert out == expected, f"\nrule {rule}\n got: {out!r}\nwant: {expected!r}"
        assert counts[rule] >= 1, (rule, dict(counts))


# --- what must NOT be rewritten -------------------------------------------
UNCHANGED = [
    # legitimate clinical negative, not a report reference
    "Step 2: There is no evidence of myocardial infarction.",
    # points at the observations / Evidence, not the report
    "Step 2: No ST‑segment or T‑wave abnormalities are reported in the observations.",
    "Step 2: No QRS widening or other conduction delays are described in Evidence 2–3.",
    # Evidence is the subject of an active verb -- correct style, keep it
    "Step 1: Evidence 1 describes a regular sinus rhythm without ectopy.",
    "Step 2: Evidence 1 and 2 describe rhythm and axis only.",
    # a characterisation ("described as"), not an existence claim
    "Step 1: A nonspecific T‑wave change is present; it is not described as ST deviation.",
    # the Conclusion line, which the rewrite must not disturb
    "Conclusion: No, the recording as described shows no findings of hypertrophy.",
]


def test_strings_left_unchanged():
    for s in UNCHANGED:
        out, counts = PP.rewrite_report_references(s)
        assert out == s, f"unexpectedly changed: {s!r} -> {out!r} ({dict(counts)})"
        assert not counts


def test_capitalization_matches_source():
    # sentence-initial "No ..." keeps a capital; mid-sentence "no ..." stays lower
    up, _ = PP.rewrite_report_references("No abnormality is reported.")
    assert up.startswith("The listed observations")
    low, _ = PP.rewrite_report_references("rhythm only; no abnormality is reported.")
    assert "the listed observations do not include" in low
    assert "The listed observations" not in low


def test_idempotent():
    text = ("Step 1: the report does not mention ST changes; no Q-waves are described, "
            "and voltage criteria are not documented.")
    once, _ = PP.rewrite_report_references(text)
    twice, counts2 = PP.rewrite_report_references(once)
    assert once == twice
    assert not counts2  # nothing left to rewrite


def test_rewrite_content_reparses_into_clean_blocks():
    """Apply to a full multi-block cleaned_content and re-parse into parsed_clean."""
    content = (
        "### NORM\n"
        "Evidence: 1) Sinus rhythm.\n"
        "Reasoning:\n"
        "Step 1: Evidence 1 describes a regular sinus rhythm.\n"
        "Step 2: The report does not mention abnormal morphology, and no ST changes "
        "are reported.\n"
        "Conclusion: Yes, the recording as described is a normal ECG.\n"
        "<END>\n"
        "### MI\n"
        "Evidence: 1) Sinus rhythm.\n"
        "Reasoning:\n"
        "Step 1: There is no mention of pathological Q-waves or ST elevation.\n"
        "Conclusion: No, the recording as described shows no findings of myocardial infarction.\n"
        "<END>\n"
    )
    rewritten, stats = PP.rewrite_content(content)
    assert stats.substitutions >= 3
    assert "the report" not in rewritten.lower()
    assert "no mention of" not in rewritten.lower()

    # Re-parse the rewritten content; both blocks stay well-formed.
    result = P.parse_response(rewritten)
    assert set(result.blocks) >= {"NORM", "MI"}
    assert result.blocks["NORM"].conclusion is True
    assert result.blocks["MI"].conclusion is False
    for sc in ("NORM", "MI"):
        reasoning = result.blocks[sc].reasoning
        assert "the report" not in (reasoning or "").lower()
        assert PP.count_report_reference_residual(reasoning) == 0
    # The honest Evidence-referencing step is preserved verbatim.
    assert "Evidence 1 describes a regular sinus rhythm" in result.blocks["NORM"].reasoning


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
