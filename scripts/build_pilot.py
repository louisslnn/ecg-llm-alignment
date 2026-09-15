#!/usr/bin/env python
"""Build the 50-record pilot id set for the teacher run (Phase 1.5).

Stratified per ROADMAP.md's pilot spec: NORM+SR only; multi-label abnormal;
records with a low-likelihood diagnostic finding; records with signal-quality
annotations; German / English / Swedish reports; validated and unvalidated; and
-- enforced last -- at least one Yes and one No for each of the five superclass
questions. Deterministic given --seed.

Writes a JSON list of ecg_ids to --out (default data/pilot_50.json, gitignored).

    python scripts/build_pilot.py --out data/pilot_50.json --seed 0
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import manifest as M
from src.teacher.prompt import SUPERCLASSES, superclass_answers
from src.teacher.view import build_view

TARGET = 50

# Light language heuristic (PTB-XL reports are mostly German). Umlauts imply
# German unless a clear Swedish anchor is present; a clear English anchor with no
# German umlaut flags non-German. Best-effort, only to seed language variety.
_ENGLISH_ANCHORS = {"the", "and", "with", "without", "sinus", "rhythm", "normal",
                    "wave", "waves", "changes", "consistent", "possible"}
_GERMAN_UMLAUTS = set("äöüßÄÖÜ")
_SWEDISH_ANCHORS = {"sinusrytm", "avvikande", "förändringar", "vänster", "höger",
                    "ospecifika", "förlopp"}


def _tokens(rec):
    return (rec.get("report") or "").lower().replace(",", " ").replace(".", " ").split()


def _swedish(rec):
    return bool(set(_tokens(rec)) & _SWEDISH_ANCHORS)


def likely_non_german(rec):
    txt = (rec.get("report") or "")
    if not txt.strip():
        return False
    toks = set(_tokens(rec))
    if toks & _SWEDISH_ANCHORS:
        return True
    if any(c in _GERMAN_UMLAUTS for c in txt):
        return False
    return bool(toks & _ENGLISH_ANCHORS)


def _diag_classes(rec):
    return {s["diagnostic_class"] for s in rec["statements"] if s["diagnostic"]}


def _codes(rec):
    return {s["code"] for s in rec["statements"]}


def _has_signal_quality(rec):
    return any((rec.get(f) or "").strip() for f in M.SIGNAL_QUALITY_FIELDS)


def _low_likelihood_diag(rec):
    return any(s["diagnostic"] and 0 < s["likelihood"] <= 50 for s in rec["statements"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/ptbxl/manifest.jsonl")
    ap.add_argument("--out", default="data/pilot_50.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = M.load_manifest(args.manifest)
    recs = list(manifest.values())
    rng = random.Random(args.seed)

    answers = {r["ecg_id"]: superclass_answers(build_view(r)) for r in recs}

    strata = {
        "NORM+SR only": (lambda r: _codes(r) == {"NORM", "SR"}, 6),
        "multi-label abnormal": (lambda r: "NORM" not in _diag_classes(r)
                                 and sum(s["diagnostic"] for s in r["statements"]) >= 2, 8),
        "low-likelihood diagnostic": (_low_likelihood_diag, 6),
        "signal-quality annotated": (_has_signal_quality, 6),
        "unvalidated": (lambda r: not r["validated_by_human"], 6),
        "swedish report": (_swedish, 3),
        "non-German (English/Swedish)": (likely_non_german, 4),
    }

    selected = []  # ordered, unique
    seen = set()

    def add(rec):
        if rec["ecg_id"] not in seen:
            seen.add(rec["ecg_id"])
            selected.append(rec["ecg_id"])

    for name, (pred, k) in strata.items():
        pool = [r for r in recs if pred(r)]
        for rec in rng.sample(pool, min(k, len(pool))):
            add(rec)

    # Enforce at least one Yes and one No for each superclass.
    for sc in SUPERCLASSES:
        for want in (True, False):
            if any(answers[e][sc] is want for e in selected):
                continue
            pool = [r for r in recs if answers[r["ecg_id"]][sc] is want]
            if pool:
                add(rng.choice(pool))

    # Trim to TARGET without breaking Yes/No coverage; then pad randomly.
    def coverage_ok(ids):
        for sc in SUPERCLASSES:
            vals = {answers[e][sc] for e in ids}
            if True not in vals or False not in vals:
                return False
        return True

    while len(selected) > TARGET:
        for i in range(len(selected) - 1, -1, -1):
            trial = selected[:i] + selected[i + 1:]
            if coverage_ok(trial):
                selected = trial
                break
        else:
            selected = selected[:TARGET]
            break

    if len(selected) < TARGET:
        pool = [r["ecg_id"] for r in recs if r["ecg_id"] not in seen]
        for eid in rng.sample(pool, TARGET - len(selected)):
            seen.add(eid)
            selected.append(eid)

    selected = selected[:TARGET]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sorted(selected), f)
        f.write("\n")

    # Report the composition actually achieved.
    print(f"wrote {len(selected)} ecg_ids to {args.out}")
    for sc in SUPERCLASSES:
        yes = sum(answers[e][sc] for e in selected)
        print(f"  {sc:5s}: Yes={yes:2d}  No={len(selected) - yes:2d}")
    print(f"  signal-quality annotated: {sum(_has_signal_quality(manifest[e]) for e in selected)}")
    print(f"  unvalidated: {sum(not manifest[e]['validated_by_human'] for e in selected)}")
    print(f"  low-likelihood diagnostic: {sum(_low_likelihood_diag(manifest[e]) for e in selected)}")
    print(f"  non-German (heuristic): {sum(likely_non_german(manifest[e]) for e in selected)}")


if __name__ == "__main__":
    main()
