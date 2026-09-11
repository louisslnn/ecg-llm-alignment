"""Signal-quality rendering against the real, messy PTB-XL field values.

The raw strings are pinned here (with their ecg_id) so these stay pure unit tests
of :func:`src.teacher.view.render_quality_field` / :func:`build_view`. The values
were taken from the working set -- PTB-XL stores these fields as comma-separated
lead lists with leading empty entries, trailing padding, ranges, semicolons,
lowercase leads and German descriptors.

Runnable via ``pytest tests/test_teacher_signal_quality.py``.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.teacher.view import build_view, render_quality_field


def _record(**quality):
    rec = {
        "ecg_id": 1,
        "statements": [],
        "age": 60.0,
        "sex": 0,
        "heart_axis": None,
        "infarction_stadium1": None,
        "infarction_stadium2": None,
        "baseline_drift": None,
        "static_noise": None,
        "burst_noise": None,
        "electrodes_problems": None,
        "report": "x" * 40,
    }
    rec.update(quality)
    return rec


# --- the named records -------------------------------------------------------

def test_21282_static_noise_range():
    # ecg_id 21282: static_noise == " , I-AVF,  "
    assert render_quality_field("static_noise", " , I-AVF,  ") == "static noise in leads I-aVF"


def test_1425_baseline_drift_two_leads():
    # ecg_id 1425: baseline_drift == " , II,III"
    assert render_quality_field("baseline_drift", " , II,III") == "baseline drift in leads II, III"


def test_583_electrodes_problems_two_chest_leads():
    # ecg_id 583: electrodes_problems == "V5,V6"
    assert render_quality_field("electrodes_problems", "V5,V6") == "electrode problems in leads V5, V6"


def test_2578_multiple_quality_fields_at_once():
    # ecg_id 2578: baseline_drift " , V1 stark", static_noise " , I-AVL,  ",
    # electrodes_problems "V6". The severity word "stark" is dropped; V1 is kept.
    view = build_view(
        _record(
            baseline_drift=" , V1 stark",
            static_noise=" , I-AVL,  ",
            electrodes_problems="V6",
        )
    )
    assert view.signal_quality == [
        "baseline drift in leads V1",
        "static noise in leads I-aVL",
        "electrode problems in leads V6",
    ]


# --- formatting edge cases actually present in the data ----------------------

def test_leading_empty_and_trailing_padding_are_stripped():
    assert render_quality_field("baseline_drift", "v6,  ") == "baseline drift in leads V6"


def test_lead_names_normalised_to_standard_casing():
    assert render_quality_field("static_noise", " , i-avf,  ") == "static noise in leads I-aVF"
    assert render_quality_field("burst_noise", "avl,avr") == "burst noise in leads aVL, aVR"


def test_spaced_range_and_bare_number_range():
    assert render_quality_field("static_noise", " , I - AVF,  ") == "static noise in leads I-aVF"
    assert render_quality_field("baseline_drift", " , v1-6") == "baseline drift in leads V1-V6"


def test_semicolon_separator():
    assert render_quality_field("electrodes_problems", "V5;V6") == "electrode problems in leads V5, V6"


def test_uncertainty_marks_are_trimmed():
    assert render_quality_field("electrodes_problems", "V1???") == "electrode problems in leads V1"
    assert render_quality_field("electrodes_problems", "Kontaktprobleme aVL ???") == (
        "electrode problems in leads aVL"
    )


def test_all_leads_german_word():
    assert render_quality_field("static_noise", " , alles,  ") == "static noise in all leads"


def test_present_but_no_usable_leads():
    # severity-only annotation names no lead
    assert render_quality_field("static_noise", " , stark,  ") == (
        "static noise annotated (leads not specified)"
    )
    assert render_quality_field("static_noise", " , noisy recording,") == (
        "static noise annotated (leads not specified)"
    )


def test_absent_field_renders_nothing():
    assert render_quality_field("static_noise", None) is None
    assert render_quality_field("static_noise", "") is None
    assert render_quality_field("static_noise", "   ") is None


def test_view_with_no_quality_has_empty_list():
    assert build_view(_record()).signal_quality == []


def test_duplicate_leads_are_deduplicated_in_order():
    assert render_quality_field("baseline_drift", "V1,V1,V2") == "baseline drift in leads V1, V2"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
