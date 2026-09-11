"""PTB-XL label manifest.

Builds one self-contained record per ECG by joining ptbxl_database.csv against
scp_statements.csv, so everything downstream (the distillation teacher,
evaluation) reads a single file and never re-parses the raw CSVs.

Format: **JSONL**, one JSON object per line, sorted by ecg_id.

Why JSONL over parquet: each record carries a variable-length list of nested
statement objects (code, description, class flags, likelihood). JSON represents
that natively and round-trips losslessly -- a likelihood of ``0.0`` stays
``0.0``, a missing value becomes ``null`` -- with no schema declaration and no
pyarrow dependency. At ~22k small rows, parquet's columnar/compression wins are
irrelevant, while JSONL stays human-inspectable and diff-friendly.

``report`` (the free-text cardiologist report) is retained for provenance but is
**NOT consumed downstream yet**; it is kept so we never have to touch the CSV to
add it later.
"""

import ast
import json
import logging
import math
import os
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Signal-quality annotation columns (each is a free-text string or NaN).
SIGNAL_QUALITY_FIELDS = [
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
]

DEFAULT_MANIFEST_NAME = "manifest.jsonl"

# Records whose only statement is a single rhythm code at likelihood 0 -- they
# carry no diagnostic label and no confidence, so they are excluded from the
# working set by default (see scripts/manifest_stats.py analysis 3). Dropping
# these 6 leaves 21,793 records.
EXCLUDED_ECG_IDS = frozenset({3796, 3797, 7778, 7783, 9824, 13803})


def _nan_to_none(value: Any) -> Any:
    """pandas NaN -> None (JSON null); everything else passes through."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def load_scp_table(ptb_root: str) -> Dict[str, Dict[str, Any]]:
    """Map each SCP code to its description and class metadata.

    The class flags (diagnostic/form/rhythm) are stored in the CSV as ``1.0`` or
    NaN; we normalise them to plain bools. ``diagnostic_class`` and
    ``diagnostic_subclass`` are strings or NaN -> None.
    """
    scp = pd.read_csv(os.path.join(ptb_root, "scp_statements.csv"), index_col=0)
    table: Dict[str, Dict[str, Any]] = {}
    for code, row in scp.iterrows():
        table[str(code)] = {
            "description": _nan_to_none(row["description"]),
            "diagnostic": bool(row["diagnostic"] == 1.0),
            "form": bool(row["form"] == 1.0),
            "rhythm": bool(row["rhythm"] == 1.0),
            "diagnostic_class": _nan_to_none(row["diagnostic_class"]),
            "diagnostic_subclass": _nan_to_none(row["diagnostic_subclass"]),
        }
    return table


def build_records(ptb_root: str) -> List[Dict[str, Any]]:
    """Build the per-record manifest entries, sorted by ecg_id.

    Each record's ``scp_codes`` string is parsed with ``ast.literal_eval`` into an
    ordered ``{code: likelihood}`` dict; that order is preserved. Every code is
    joined against scp_statements.csv so the teacher sees "atrial fibrillation",
    not "AFIB". Likelihoods are preserved exactly as annotated -- including 0.0,
    which is neither dropped nor binarised.
    """
    scp = load_scp_table(ptb_root)
    db = pd.read_csv(
        os.path.join(ptb_root, "ptbxl_database.csv"), index_col="ecg_id"
    ).sort_index()

    records: List[Dict[str, Any]] = []
    for ecg_id, row in db.iterrows():
        codes = ast.literal_eval(row["scp_codes"])  # ordered {code: likelihood}

        statements = []
        for code, likelihood in codes.items():
            if code not in scp:
                raise KeyError(
                    f"ecg_id {ecg_id}: SCP code {code!r} is absent from "
                    "scp_statements.csv and cannot be resolved to a description"
                )
            info = scp[code]
            statements.append(
                {
                    "code": code,
                    "description": info["description"],
                    "diagnostic": info["diagnostic"],
                    "form": info["form"],
                    "rhythm": info["rhythm"],
                    "diagnostic_class": info["diagnostic_class"],
                    "diagnostic_subclass": info["diagnostic_subclass"],
                    "likelihood": float(likelihood),  # preserved exactly, incl. 0.0
                }
            )

        records.append(
            {
                "ecg_id": int(ecg_id),
                "statements": statements,
                "strat_fold": int(row["strat_fold"]),
                "age": float(row["age"]) if pd.notna(row["age"]) else None,
                "sex": int(row["sex"]),
                **{f: _nan_to_none(row[f]) for f in SIGNAL_QUALITY_FIELDS},
                "validated_by_human": bool(row["validated_by_human"]),
                "initial_autogenerated_report": bool(row["initial_autogenerated_report"]),
                # Retained for provenance; NOT consumed downstream yet.
                "report": _nan_to_none(row["report"]),
            }
        )
    return records


def write_manifest(records: List[Dict[str, Any]], out_path: str) -> None:
    """Write records as JSONL (one object per line), keyed by ecg_id."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def load_manifest(path: str, apply_exclusions: bool = True) -> Dict[int, Dict[str, Any]]:
    """Read the JSONL manifest back into a dict keyed by ecg_id, in file order.

    By default the ``EXCLUDED_ECG_IDS`` are dropped and the exclusion is logged,
    yielding the 21,793-record working set. Pass ``apply_exclusions=False`` to
    load every record faithfully (e.g. for round-trip checks).
    """
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["ecg_id"]] = rec

    if apply_exclusions:
        excluded = sorted(eid for eid in EXCLUDED_ECG_IDS if eid in out)
        for eid in excluded:
            del out[eid]
        logger.info(
            "excluded %d records at manifest load: %s; working set = %d",
            len(excluded), excluded, len(out),
        )

    return out


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the summary statistics reported by the build script."""
    n = len(records)
    fold_counts = dict(sorted(Counter(r["strat_fold"] for r in records).items()))

    stmt_counts = [len(r["statements"]) for r in records]
    median_stmts = statistics.median(stmt_counts)
    max_stmts = max(stmt_counts)

    all_zero = sum(
        1
        for r in records
        if r["statements"] and all(s["likelihood"] == 0.0 for s in r["statements"])
    )
    norm_only = sum(
        1 for r in records if {s["code"] for s in r["statements"]} == {"NORM"}
    )

    code_counts = Counter(s["code"] for r in records for s in r["statements"])
    descriptions = {
        s["code"]: s["description"] for r in records for s in r["statements"]
    }
    top10 = [
        (code, descriptions[code], count)
        for code, count in code_counts.most_common(10)
    ]

    rhythm_and_form = sum(
        1
        for r in records
        if any(s["rhythm"] for s in r["statements"])
        and any(s["form"] for s in r["statements"])
    )

    return {
        "total_records": n,
        "records_per_fold": fold_counts,
        "median_statements": median_stmts,
        "max_statements": max_stmts,
        "all_likelihood_zero": all_zero,
        "norm_only": norm_only,
        "top10_statements": top10,
        "rhythm_and_form": rhythm_and_form,
    }


def print_summary(records: List[Dict[str, Any]]) -> None:
    """Print the manifest summary to stdout."""
    s = summarize(records)
    print(f"total records: {s['total_records']}")

    print("records per fold:")
    for fold, count in s["records_per_fold"].items():
        print(f"  fold {fold:2d}: {count}")

    print(f"statements per record: median {s['median_statements']}, max {s['max_statements']}")
    print(f"records with every likelihood == 0: {s['all_likelihood_zero']}")
    print(f"NORM-only records (statement set == {{NORM}}): {s['norm_only']}")
    print(f"records with >=1 rhythm AND >=1 form statement: {s['rhythm_and_form']}")

    print("ten most frequent statements:")
    for code, desc, count in s["top10_statements"]:
        print(f"  {count:6d}  {code:8s}  {desc}")
