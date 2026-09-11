"""Teacher view assembly and superclass-answer derivation.

Uses small synthetic manifest entries so the logic is tested in isolation from
the manifest file. The load-bearing rules under test:

* NORM is "Yes" only when the diagnostic classes are exactly {NORM}.
* A diagnostic statement asserts its class even at likelihood 0.
* Non-diagnostic (rhythm/form) statements never change an answer.
* The view drops empty/unknown recording properties and is serialisable.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.teacher.prompt import (
    PROMPT_VERSION,
    SUPERCLASSES,
    render_all_prompts,
    render_prompt,
    superclass_answers,
)
from src.teacher.view import build_view


def _stmt(code, dc, likelihood, diagnostic=True, form=False, rhythm=False, desc=None):
    return {
        "code": code,
        "description": desc or f"{code} description",
        "diagnostic": diagnostic,
        "form": form,
        "rhythm": rhythm,
        "diagnostic_class": dc,
        "diagnostic_subclass": dc,
        "likelihood": likelihood,
    }


def _record(statements, **overrides):
    rec = {
        "ecg_id": 1,
        "statements": statements,
        "age": 62.0,
        "sex": 1,
        "heart_axis": "LAD",
        "infarction_stadium1": "Stadium II-III",
        "infarction_stadium2": "",
        "baseline_drift": "",
        "static_noise": "",
        "burst_noise": "",
        "electrodes_problems": "",
        "report": "sinus rhythm. normal ecg with left axis deviation noted.",
    }
    rec.update(overrides)
    return rec


def test_norm_yes_only_when_diagnostic_classes_exactly_norm():
    rec = _record(
        [_stmt("NORM", "NORM", 100.0), _stmt("SR", None, 0.0, diagnostic=False, rhythm=True)]
    )
    answers = superclass_answers(build_view(rec))
    assert answers == {"NORM": True, "MI": False, "STTC": False, "CD": False, "HYP": False}


def test_norm_no_when_another_diagnostic_class_present():
    rec = _record([_stmt("NORM", "NORM", 100.0), _stmt("IMI", "MI", 15.0)])
    answers = superclass_answers(build_view(rec))
    assert answers["NORM"] is False
    assert answers["MI"] is True


def test_likelihood_zero_diagnostic_still_asserts_class():
    rec = _record([_stmt("ASMI", "MI", 0.0)])
    answers = superclass_answers(build_view(rec))
    assert answers["MI"] is True
    assert answers["NORM"] is False


def test_rhythm_and_form_statements_do_not_set_any_answer():
    rec = _record(
        [
            _stmt("SR", None, 0.0, diagnostic=False, rhythm=True),
            _stmt("ABQRS", None, 0.0, diagnostic=False, form=True),
        ]
    )
    answers = superclass_answers(build_view(rec))
    assert answers == {sc: False for sc in SUPERCLASSES}


def test_view_drops_unknown_and_empty_recording_properties():
    rec = _record(
        [_stmt("NORM", "NORM", 100.0)],
        infarction_stadium1="unknown",
        infarction_stadium2="",
        heart_axis="",
        static_noise=", I-V1,",
    )
    view = build_view(rec)
    assert view.infarction_stadium == []  # "unknown" and "" both dropped
    assert view.heart_axis is None
    assert view.signal_quality == ["static noise in leads I-V1"]
    assert view.sex == "female"


def test_view_is_serialisable_to_plain_dict():
    view = build_view(_record([_stmt("NORM", "NORM", 100.0)]))
    d = view.to_dict()
    assert d["ecg_id"] == 1
    assert d["report_flags"]["floor_hit"] is False
    assert d["statements"][0]["diagnostic_superclass"] == "NORM"
    # round-trips through JSON without custom encoders
    import json

    assert json.loads(json.dumps(d))["ecg_id"] == 1


def test_render_prompt_binds_question_and_known_answer():
    rec = _record([_stmt("IMI", "MI", 100.0)])
    view = build_view(rec)
    prompt = render_prompt(view, "MI")
    assert 'Does this ECG show evidence of myocardial infarction?' in prompt
    assert "The known answer is: Yes" in prompt
    # the NORM question on the same record is a No
    assert "The known answer is: No" in render_prompt(view, "NORM")


def test_norm_question_reads_directly_others_show_evidence():
    view = build_view(_record([_stmt("NORM", "NORM", 100.0)]))
    assert 'Is this a normal ECG?' in render_prompt(view, "NORM")
    assert "show evidence of a normal ECG" not in render_prompt(view, "NORM")
    assert 'Does this ECG show evidence of ST/T changes?' in render_prompt(view, "STTC")
    assert 'Does this ECG show evidence of a conduction disturbance?' in render_prompt(view, "CD")
    assert 'Does this ECG show evidence of hypertrophy?' in render_prompt(view, "HYP")


def test_render_prompt_uses_placeholder_when_report_unusable():
    rec = _record([_stmt("NORM", "NORM", 100.0)], report=" ")
    prompt = render_prompt(build_view(rec), "NORM")
    assert "no usable free-text report" in prompt


def test_render_all_prompts_covers_five_superclasses():
    prompts = render_all_prompts(build_view(_record([_stmt("NORM", "NORM", 100.0)])))
    assert list(prompts) == SUPERCLASSES
    assert all(isinstance(p, str) and p for p in prompts.values())


def test_prompt_version_is_pinned():
    assert PROMPT_VERSION == "teacher-v2"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
