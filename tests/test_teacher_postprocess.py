"""Post-processing of teacher responses (v6 fix #1 strip, plus the fix #2/#3 checks).

The strip cases use the real leaked Reasoning steps observed in the v5 pilot
(``data/teacher/pilot50_v5.jsonl``): the trailing-step removal that leaves the
honest step behind, the renumbering when the tell is not last, and the
empty-Reasoning edge case where every step was a tell.
"""

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.teacher.parse import parse_response
from src.teacher.postprocess import (
    count_diagnostic_evidence_items,
    find_leak_steps,
    has_lead_type,
    is_diagnostic_evidence,
    is_leak_step,
    split_evidence_items,
    strip_content,
    strip_reasoning_steps,
)

PILOT_V5 = os.path.join(REPO_ROOT, "data", "teacher", "pilot50_v5.jsonl")

# --- real leaked steps from the v5 pilot -------------------------------------
# The honest step that precedes the tell in each affected block; must be kept.
HONEST_STEP = (
    "Step 1: The observations describe rhythm and repolarisation changes but do "
    "not mention QRS voltage, R‑wave amplitude, or chamber‑size criteria "
    "required for hypertrophy assessment."
)
# One tell per specified pattern, taken verbatim from the pilot.
TELL_REQUIRED_ANSWER = (
    "Step 2: The listed evidence therefore does not address hypertrophy, yet the "
    "required answer is affirmative."
)  # ecg 2054 HYP
TELL_ANSWER_MUST_BE = (
    "Step 2: The listed observation does not bear on infarction, yet the answer "
    "must be affirmative."
)  # ecg 14708 MI
TELL_KNOWN_ANSWER = (
    "Step 2: Although the listed observations do not address hypertrophy criteria, "
    "the known answer requires a positive designation."
)  # ecg 16357 HYP
TELL_KNOWN_DIAGNOSIS = (
    "Step 2: The listed observations do not bear on myocardial infarction, yet the "
    "answer must reflect the known diagnosis."
)  # ecg 11919 MI (added pattern after pilot review)
# An honest step that mentions "a positive answer" -- must NOT be stripped, since
# the `requires? an?` pattern only fires on "require(s) a(n) positive", not on
# "required for a positive".
HONEST_REQUIRED_FOR = (
    "Step 2: Because no ST/T changes are described, the recording lacks the "
    "findings required for a positive answer."
)  # ecg 5818 STTC


def test_each_pilot_tell_is_detected():
    for tell in (TELL_REQUIRED_ANSWER, TELL_ANSWER_MUST_BE, TELL_KNOWN_ANSWER,
                 TELL_KNOWN_DIAGNOSIS):
        assert is_leak_step(tell), tell


def test_honest_steps_are_not_detected():
    assert not is_leak_step(HONEST_STEP)
    assert not is_leak_step(HONEST_REQUIRED_FOR)


def test_trailing_tell_removed_honest_kept():
    reasoning = HONEST_STEP + "  \n" + TELL_REQUIRED_ANSWER
    res = strip_reasoning_steps(reasoning)
    assert res.removed == 1
    assert res.kept == 1
    assert not res.emptied
    assert "required answer" not in res.cleaned.lower()
    assert HONEST_STEP in res.cleaned


def test_renumbering_when_tell_is_first():
    # Tell as Step 1, honest as Step 2: after stripping, the survivor renumbers to 1.
    tell_as_first = TELL_REQUIRED_ANSWER.replace("Step 2:", "Step 1:")
    honest_as_second = HONEST_STEP.replace("Step 1:", "Step 2:")
    reasoning = tell_as_first + "\n" + honest_as_second
    res = strip_reasoning_steps(reasoning)
    assert res.removed == 1 and res.kept == 1
    assert res.cleaned.strip().startswith("Step 1:")
    assert "Step 2:" not in res.cleaned
    assert "required answer" not in res.cleaned.lower()


def test_renumbering_middle_tell_three_steps():
    # Step1 honest, Step2 tell, Step3 honest -> survivors renumber to Step1, Step2.
    s1 = HONEST_STEP
    s2 = TELL_KNOWN_ANSWER
    s3 = HONEST_REQUIRED_FOR.replace("Step 2:", "Step 3:")
    reasoning = s1 + "\n" + s2 + "\n" + s3
    res = strip_reasoning_steps(reasoning)
    assert res.removed == 1 and res.kept == 2
    assert "Step 1:" in res.cleaned
    assert "Step 2:" in res.cleaned
    assert "Step 3:" not in res.cleaned
    # order preserved: the honest first step stays first
    assert res.cleaned.index("rhythm and repolarisation") < res.cleaned.index(
        "no ST/T changes are described"
    )


def test_empty_reasoning_edge_case_flagged_not_dropped():
    # A block whose only step is a tell: stripping empties Reasoning; flag it.
    res = strip_reasoning_steps(TELL_KNOWN_DIAGNOSIS)
    assert res.removed == 1
    assert res.kept == 0
    assert res.emptied is True
    assert "Step" not in res.cleaned  # no orphan step header left behind


def test_no_steps_returns_unchanged():
    text = "Reasoning without any explicit step markers."
    res = strip_reasoning_steps(text)
    assert res.removed == 0 and res.kept == 0 and not res.emptied
    assert res.cleaned == text


def test_strip_content_touches_only_reasoning():
    block = (
        "### HYP\n"
        "Evidence: 1) Sinus rhythm. 2) Right bundle branch block.\n"
        "Reasoning:\n"
        + HONEST_STEP + "\n" + TELL_REQUIRED_ANSWER + "\n"
        "Conclusion: Yes, the recording shows hypertrophy.\n"
        "<END>"
    )
    cleaned, stats = strip_content(block)
    assert stats.blocks_stripped == 1
    assert stats.steps_removed == 1
    assert stats.blocks_emptied == 0
    # Evidence and Conclusion are untouched
    assert "Evidence: 1) Sinus rhythm. 2) Right bundle branch block." in cleaned
    assert "Conclusion: Yes, the recording shows hypertrophy." in cleaned
    assert "<END>" in cleaned
    # the tell is gone, the honest step remains
    assert "required answer" not in cleaned.lower()
    assert "rhythm and repolarisation" in cleaned


def test_strip_content_empty_block_reported():
    block = (
        "### MI\nEvidence: 1) Sinus rhythm.\nReasoning:\n"
        + TELL_KNOWN_DIAGNOSIS + "\nConclusion: Yes, inferior MI.\n<END>"
    )
    cleaned, stats = strip_content(block)
    assert stats.blocks_stripped == 1
    assert stats.steps_removed == 1
    assert stats.blocks_emptied == 1
    assert "known diagnosis" not in cleaned.lower()


# --- residual supervision-pipeline check (parsed_clean invariant) -------------

def test_find_leak_steps_detects_then_clears():
    reasoning = HONEST_STEP + "\n" + TELL_REQUIRED_ANSWER
    assert find_leak_steps(reasoning) == [TELL_REQUIRED_ANSWER]
    # after stripping, a re-check finds nothing
    cleaned = strip_reasoning_steps(reasoning).cleaned
    assert find_leak_steps(cleaned) == []


def test_find_leak_steps_empty_inputs():
    assert find_leak_steps(None) == []
    assert find_leak_steps("") == []
    assert find_leak_steps(HONEST_STEP) == []


def test_strip_invariant_holds_over_pilot():
    """Strip then re-parse the real pilot: parsed_clean must carry no tell.

    This is the invariant client.py asserts per record and reports in the run
    summary; here it is verified end to end (strip_content -> parse_response ->
    find_leak_steps) on the actual v5 output.
    """
    if not os.path.exists(PILOT_V5):
        pytest.skip(f"pilot output not present at {PILOT_V5}")

    stripped_blocks = 0
    with open(PILOT_V5, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cleaned_content, stats = strip_content(row["content"])
            stripped_blocks += stats.blocks_stripped
            parsed_clean = parse_response(cleaned_content)
            for sc, block in parsed_clean.blocks.items():
                residual = find_leak_steps(block.reasoning)
                assert residual == [], f"{row['ecg_id']} {sc}: {residual}"
    # the pilot is known to contain the 13 tells; make sure we actually stripped.
    assert stripped_blocks == 13


# --- fix #2: lead type check --------------------------------------------------

def test_lead_type_detected():
    assert has_lead_type("Evidence 2 describes a posterior-left lead type.")
    assert has_lead_type("LEAD TYPE normal")


def test_lead_type_not_detected():
    assert not has_lead_type("Evidence 1 describes the electrical axis.")
    assert not has_lead_type(None)


# --- fix #3: diagnosis-in-Evidence check --------------------------------------

def test_diagnostic_evidence_items_counted():
    ev = ("1) Sinus rhythm. 2) Inferior infarct noted. "
          "3) Anterior septal myocardial infarction noted.")
    items = split_evidence_items(ev)
    assert len(items) == 3
    assert count_diagnostic_evidence_items(ev) == 2


def test_observations_not_counted_as_diagnostic():
    # Bundle branch block and ST depression are observations the prompt allows.
    ev = ("1) Sinus rhythm. 2) Right bundle branch block. "
          "3) ST depression in V5, V6.")
    assert count_diagnostic_evidence_items(ev) == 0
    assert not is_diagnostic_evidence("Right bundle branch block")


def test_empty_evidence_is_zero():
    assert count_diagnostic_evidence_items(None) == 0
    assert count_diagnostic_evidence_items("") == 0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
