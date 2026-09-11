"""Report cleaning on the specific PTB-XL records we found during inspection.

The raw report strings are pinned here (with their ecg_id) so these stay pure
unit tests of :mod:`src.teacher.report`, independent of the manifest file.

Runnable via ``pytest tests/test_teacher_report.py``.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.teacher.report import DEFAULT_MIN_REPORT_CHARS, clean_report

# ecg_id 18822: a curator edit carrying both a bare label and a parenthesised
# label-with-likelihood, plus the autogen boilerplate marker.
RAW_18822 = (
    "sinusrhythmus linkstyp sonst normales ekg 4.46                          "
    "unbestÄtigter bericht Edit: NORM 80, (NORM 100, IVCB)"
)

# ecg_id 4323: validated_by_human=False, ending in the autogen boilerplate.
RAW_4323 = (
    "sinusrhythmus verdacht auf p-sinistrocardiale lagetyp normal mÄssige "
    "amplitudenkriterien fÜr linkshypertrophie 4.46                          "
    "unbestÄtigter bericht"
)

# ecg_id 443 and 11489: the two effectively-empty reports. Each is a single space
# -- report_inspect counts them as both "min non-empty length: 1" (raw length 1)
# and "empty or missing report: 2" (empty after strip). Both must fall through the
# floor to annotations-only evidence.
RAW_443 = " "
RAW_11489 = " "


def test_18822_curator_edit_and_boilerplate_all_stripped():
    result = clean_report(RAW_18822)
    assert result.text == "sinusrhythmus linkstyp sonst normales ekg"
    assert result.has_usable_report is True
    assert result.flags.stripped_boilerplate is True
    assert result.flags.stripped_edit is True
    assert result.flags.stripped_paren_labels is True
    assert result.flags.floor_hit is False
    # nothing leaked from the annotation channel
    assert "Edit" not in result.text
    assert "NORM" not in result.text
    assert "IVCB" not in result.text
    assert "unbest" not in result.text.lower()
    assert "4.46" not in result.text


def test_4323_unvalidated_boilerplate_stripped_report_survives():
    result = clean_report(RAW_4323)
    assert result.text == (
        "sinusrhythmus verdacht auf p-sinistrocardiale lagetyp normal mÄssige "
        "amplitudenkriterien fÜr linkshypertrophie"
    )
    assert result.has_usable_report is True
    assert result.flags.stripped_boilerplate is True
    assert result.flags.stripped_edit is False
    assert result.flags.stripped_paren_labels is False
    assert result.flags.floor_hit is False
    assert "unbestÄtigter bericht" not in result.text
    assert "4.46" not in result.text


def test_443_single_space_report_falls_through_floor():
    result = clean_report(RAW_443)
    assert result.text == ""
    assert result.has_usable_report is False
    assert result.flags.floor_hit is True


def test_11489_second_empty_report_falls_through_floor():
    result = clean_report(RAW_11489)
    assert result.text == ""
    assert result.has_usable_report is False
    assert result.flags.floor_hit is True


def test_alter_unbest_abbreviation_is_not_boilerplate():
    # "alter unbest." means "age indeterminate" -- it must survive; only the full
    # "unbestÄtigter bericht" marker is boilerplate.
    raw = "sinusrhythmus linkstyp qrs(t) abnormal inferiorer infarkt alter unbest."
    result = clean_report(raw)
    assert result.flags.stripped_boilerplate is False
    assert "alter unbest." in result.text
    assert result.has_usable_report is True


def test_legitimate_lowercase_parentheticals_are_kept():
    # "qrs(t)" and similar are genuine report shorthand, not leaked labels.
    raw = "sinusrhythmus lagetyp normal qrs(t) abnorm r-s (er) verschoben nach rechts"
    result = clean_report(raw)
    assert result.flags.stripped_paren_labels is False
    assert "qrs(t)" in result.text
    assert "(er)" in result.text


def test_floor_marks_short_content_report_unusable():
    result = clean_report("trace only.")  # 11 chars < 15
    assert result.has_usable_report is False
    assert result.flags.floor_hit is True


def test_custom_floor_is_respected():
    result = clean_report("trace only.", min_chars=5)
    assert result.has_usable_report is True
    assert result.flags.floor_hit is False
    assert result.text == "trace only."


def test_none_report_is_empty_and_unusable():
    result = clean_report(None)
    assert result.text == ""
    assert result.has_usable_report is False
    assert result.flags.floor_hit is True


def test_to_dict_is_serialisable():
    d = clean_report(RAW_18822).to_dict()
    assert set(d) == {"text", "has_usable_report", "flags"}
    assert set(d["flags"]) == {
        "stripped_boilerplate",
        "stripped_edit",
        "stripped_paren_labels",
        "floor_hit",
    }


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
