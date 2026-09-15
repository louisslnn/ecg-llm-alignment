"""run_teacher.py helpers: id validation and --show-raw record selection.

Loads the script as a module (scripts/ is not a package) and tests the pure
helpers directly, so no API or manifest file is needed.
"""

import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

_spec = importlib.util.spec_from_file_location(
    "run_teacher", os.path.join(REPO_ROOT, "scripts", "run_teacher.py")
)
run_teacher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_teacher)


# --- id validation -----------------------------------------------------------

def test_select_records_returns_in_requested_order():
    manifest = {1: {"ecg_id": 1}, 2: {"ecg_id": 2}, 3: {"ecg_id": 3}}
    recs = run_teacher._select_records(manifest, [3, 1])
    assert [r["ecg_id"] for r in recs] == [3, 1]


def test_select_records_fails_clearly_on_unknown_id():
    manifest = {1: {"ecg_id": 1}, 2: {"ecg_id": 2}}
    with pytest.raises(SystemExit) as exc:
        run_teacher._select_records(manifest, [1, 999])
    assert "999" in str(exc.value)
    assert "not in the working set" in str(exc.value)


def test_dedupe_preserves_order():
    assert run_teacher._dedupe([5, 5, 2, 5, 2, 9]) == [5, 2, 9]


# --- --show-raw selection: only this invocation's requested ids --------------

def _row(ecg_id):
    return {"ecg_id": ecg_id}


def test_show_raw_selects_from_requested_generated_ids_not_stale():
    # 790 is a stale row from a previous run; this invocation requested [38, 473].
    rows = [_row(790), _row(38), _row(473)]
    chosen = run_teacher._select_raw_row(rows, candidate_ids=[38, 473])
    assert chosen["ecg_id"] == 38  # first requested candidate, never the stale 790


def test_show_raw_returns_none_when_nothing_generated_this_invocation():
    # e.g. the only requested id was skipped as already-present, so it is not a
    # candidate -> nothing from this run to show.
    rows = [_row(790)]
    assert run_teacher._select_raw_row(rows, candidate_ids=[38]) is None


def test_show_raw_prefers_most_recent_write_for_an_id():
    old, new = {"ecg_id": 38, "content": "old"}, {"ecg_id": 38, "content": "new"}
    chosen = run_teacher._select_raw_row([old, new], candidate_ids=[38])
    assert chosen["content"] == "new"


def test_show_raw_candidate_order_is_honoured():
    rows = [_row(473), _row(38)]
    assert run_teacher._select_raw_row(rows, [473, 38])["ecg_id"] == 473
    assert run_teacher._select_raw_row(rows, [38, 473])["ecg_id"] == 38


# --- --show-raw restricts to ids generated THIS invocation -------------------

def test_generated_candidates_excludes_unwritten_ids():
    # 790 was requested but not written this run (errored, or stale from before);
    # only 8505 was actually generated now.
    assert run_teacher._generated_candidates([790, 8505], written_ids=[8505]) == [8505]


def test_generated_candidates_preserves_requested_order():
    assert run_teacher._generated_candidates([473, 38], written_ids=[38, 473]) == [473, 38]


def test_generated_candidates_empty_when_nothing_written():
    assert run_teacher._generated_candidates([38, 473], written_ids=[]) == []


def test_show_raw_never_prints_stale_row_from_earlier_run():
    # The reported bug: 790 sits in the output file from a prior run; this run
    # requested [790, 8505] but only 8505 was generated. --show-raw must show 8505.
    rows = [{"ecg_id": 790, "content": "stale"}, {"ecg_id": 8505, "content": "fresh"}]
    candidates = run_teacher._generated_candidates([790, 8505], written_ids=[8505])
    chosen = run_teacher._select_raw_row(rows, candidates)
    assert chosen["ecg_id"] == 8505 and chosen["content"] == "fresh"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
