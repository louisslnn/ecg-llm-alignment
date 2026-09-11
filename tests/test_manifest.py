"""PTB-XL manifest integrity: coverage, uniqueness, and code resolution.

Asserts the manifest has exactly one entry per record in ptbxl_database.csv, no
duplicated ecg_id, and that every code in every record resolves to a non-empty
description. Also checks the JSONL round-trip preserves ecg_ids and likelihood.

Runnable via `pytest tests/test_manifest.py` or `python tests/test_manifest.py`.
"""

import os
import sys

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import manifest as M

PTB_ROOT = os.environ.get("PTB_ROOT", os.path.join(REPO_ROOT, "data/ptbxl"))


@pytest.fixture(scope="module")
def records():
    return M.build_records(PTB_ROOT)


def test_one_row_per_record_no_duplicates(records):
    db = pd.read_csv(os.path.join(PTB_ROOT, "ptbxl_database.csv"))
    ids = [r["ecg_id"] for r in records]

    # exactly one manifest entry per raw record
    assert len(records) == len(db)
    assert set(ids) == set(db["ecg_id"].astype(int))

    # no ecg_id duplicated
    assert len(ids) == len(set(ids))


def test_every_code_resolves_to_a_description(records):
    for r in records:
        for s in r["statements"]:
            desc = s["description"]
            assert desc is not None and desc != "", (
                f"ecg_id {r['ecg_id']}: code {s['code']!r} has no description"
            )


def test_jsonl_round_trip(records, tmp_path):
    out = os.path.join(tmp_path, "manifest.jsonl")
    M.write_manifest(records, out)
    loaded = M.load_manifest(out, apply_exclusions=False)

    assert set(loaded) == {r["ecg_id"] for r in records}

    # nested structure survives the round-trip, including likelihood 0.0
    sample = records[0]
    reloaded = loaded[sample["ecg_id"]]
    assert reloaded["statements"] == sample["statements"]


def test_default_load_excludes_working_set(records, tmp_path):
    out = os.path.join(tmp_path, "manifest.jsonl")
    M.write_manifest(records, out)

    full = M.load_manifest(out, apply_exclusions=False)
    working = M.load_manifest(out)  # exclusions on by default

    assert set(full) - set(working) == set(M.EXCLUDED_ECG_IDS)
    assert len(working) == len(full) - len(M.EXCLUDED_ECG_IDS)
    assert all(eid not in working for eid in M.EXCLUDED_ECG_IDS)


if __name__ == "__main__":
    recs = M.build_records(PTB_ROOT)
    test_one_row_per_record_no_duplicates(recs)
    test_every_code_resolves_to_a_description(recs)
    print(f"ok: {len(recs)} records, all codes resolve, no duplicate ecg_id")
