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
     "Step 2: Evidence 2 reports the ECG as normal, and does not include "
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


# --- POSITIVE citations: DROP the attribution, keep the content (real strings) --
# Substituting "the listed observations" for "the report" on a positive citation
# is self-referential ("the listed observations describe the tracing as normal"),
# so these patterns delete the attribution instead.
POSITIVE_CASES = [
    # (before, expected_after, rule)
    # "describes the tracing as Y" -> "the tracing is Y"
    ("2) The report describes the tracing as a normal ECG.",
     "2) The tracing is a normal ECG.",
     "report_positive_as"),
    ("2) The report describes the ECG as “normales ekg” (normal ECG).",
     "2) The ECG is “normales ekg” (normal ECG).",
     "report_positive_as"),
    # general "the report <verb>s REST" -> REST, but ONLY when REST is a clause
    ("the report also states the tracing is a normal ECG.",
     "the tracing is a normal ECG.",
     "report_positive"),
    ("The report states the ECG is otherwise normal.",
     "The ECG is otherwise normal.",
     "report_positive"),
    # possessive with a clean (quote) start -> drop, no article inserted
    ("the report’s “normal ECG” implies absence of such changes.",
     "“normal ECG” implies absence of such changes.",
     "report_possessive"),
    # possessive opening on a bare noun -> insert "the"
    ("but the report’s wording of repolarisation disturbance implies T-wave changes.",
     "but the wording of repolarisation disturbance implies T-wave changes.",
     "report_possessive"),
    ("Step 1: An infarction is identified by the report’s explicit statement of an inferior MI.",
     "Step 1: An infarction is identified by the explicit statement of an inferior MI.",
     "report_possessive"),
    # locative "<finding> [is] <participle> in the report" -> "<finding>"
    ("2) Normal ECG appearance described in the report.",
     "2) Normal ECG appearance.",
     "in_the_report"),
    ("3) T-wave change (t-förändring) noted in the report.",
     "3) T-wave change (t-förändring).",
     "in_the_report"),
    ("2) A bundle-branch block is described in the report.",
     "2) A bundle-branch block.",
     "in_the_report"),
]


def test_positive_citations_dropped_not_substituted():
    for before, expected, rule in POSITIVE_CASES:
        out, counts = PP.rewrite_report_references(before)
        assert out == expected, f"\nrule {rule}\n got: {out!r}\nwant: {expected!r}"
        assert counts[rule] >= 1, (rule, dict(counts))
        # the tell-tale self-referential phrasing must never be produced
        assert "listed observations describe" not in out.lower()
        assert "listed observations'" not in out.lower()
        assert "in the listed observations" not in out.lower()


def test_report_positive_leaves_noun_phrase_remainder():
    """When "the report <verb>s" is followed by a bare noun phrase (no verb of its
    own), deletion would leave a fragment -- so the sentence is left untouched and
    counted in the residual, not mangled."""
    fragments = [
        "Step 1: Using Evidence 1, the report mentions only the rhythm.",
        "the report provides only rhythm information and does not mention QRS voltage.",
        "Evidence: 1) The report notes an “inferiorer infarkt” described as “alter unbest”.",
        "2) The report describes a “normales ekg”.",
    ]
    for s in fragments:
        out, counts = PP.rewrite_report_references(s)
        assert out == s, f"should be untouched: {s!r} -> {out!r}"
        assert "report_positive" not in counts
        assert PP.count_report_reference_residual(out) >= 1  # still flagged as residual


def test_positive_passive_still_reframed():
    # affirmative "X is reported" (not a "the report ..." citation) stays a reframe
    out, counts = PP.rewrite_report_references("Evidence: 1) Atrial fibrillation is reported.")
    assert out == "Evidence: 1) Atrial fibrillation is among the listed observations."
    assert counts["passive_positive"] == 1


def test_fix2_doubled_noun_collapsed():
    # "the report provides no observations of X" must not stack "observations of"
    out, counts = PP.rewrite_report_references(
        "the report provides no observations of QRS-voltage or morphological criteria."
    )
    assert out == "the listed observations do not include QRS-voltage or morphological criteria."
    assert "include observations of" not in out
    assert counts["dedup_double_noun"] >= 1

    out2, _ = PP.rewrite_report_references("No observations of arrhythmia or hypertrophy are reported.")
    assert out2 == "The listed observations do not include arrhythmia or hypertrophy."


def test_fix3_participle_becomes_finite_clause():
    # "providing no mention of X" -> "and does not include X", not "not including X"
    src = "Evidence 2 reports the ECG as normal, providing no mention of Q-waves, ST elevation."
    out, counts = PP.rewrite_report_references(src)
    assert out == "Evidence 2 reports the ECG as normal, and does not include Q-waves, ST elevation."
    assert "not including" not in out
    assert counts["participle_no_mention"] == 1


def test_broken_grammar_characterisation_as():
    """A case where naive 'is reported' -> 'is among the listed observations' breaks.

    "the axis is reported as left-axis deviation" is a characterisation. Naively
    swapping the passive strands "as left-axis deviation" onto a location clause and
    reads as nonsense; the rule must instead drop "reported as" to give "is Y".
    """
    src = "Step 2: The electrical axis is reported as left-axis deviation."

    naive = src.replace("is reported", "is among the listed observations")
    assert naive == (
        "Step 2: The electrical axis is among the listed observations as left-axis deviation."
    )  # grammatically broken -- the "as ..." tail dangles

    out, counts = PP.rewrite_report_references(src)
    assert out == "Step 2: The electrical axis is left-axis deviation."
    assert counts["characterisation_as"] == 1
    assert "among the listed observations" not in out


def test_broken_grammar_no_description_is_present():
    """'No description of X is present' -- naive prefix swap strands 'is present'."""
    src = "Step 2: No description of widened QRS or bundle-branch block is present."

    naive = src.replace("No description of", "The listed observations do not include")
    assert naive.endswith("bundle-branch block is present.")  # dangling tail

    out, counts = PP.rewrite_report_references(src)
    assert out == "Step 2: The listed observations do not include widened QRS or bundle-branch block."
    assert counts["bare_no_mention"] == 1


def test_positive_and_negative_mixed_block_reparses():
    """A block mixing positive and negative references rewrites and re-parses clean."""
    content = (
        "### NORM\n"
        "Evidence: 1) Sinus rhythm.\n"
        "Reasoning:\n"
        "Step 1: Evidence 1 describes a regular sinus rhythm.\n"
        "Conclusion: Yes, the recording as described is a normal ECG.\n"
        "<END>\n"
        "### MI\n"
        "Evidence: 1) Atrial fibrillation is reported. 2) The axis is reported as left-axis deviation.\n"
        "Reasoning:\n"
        "Step 1: The report states the tracing is otherwise normal.\n"
        "Step 2: No pathological Q-waves are reported, and ST elevation is not described.\n"
        "Conclusion: No, the recording as described shows no findings of myocardial infarction.\n"
        "<END>\n"
    )
    rewritten, _ = PP.rewrite_content(content)
    result = P.parse_response(rewritten)
    block = result.blocks["MI"]
    assert block.conclusion is False
    # positive + negative both reframed; no report/passive tell survives
    assert PP.count_report_reference_residual(block.raw_block) == 0
    assert "is among the listed observations" in block.evidence
    assert "is left-axis deviation" in block.evidence
    # positive citation dropped, not substituted (no self-referential phrasing)
    assert "the tracing is otherwise normal" in block.reasoning.lower()
    assert "listed observations state" not in block.reasoning.lower()
    assert "the listed observations do not include pathological q-waves" in block.reasoning.lower()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
