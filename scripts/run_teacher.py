#!/usr/bin/env python
"""Run the Mode B teacher generation over a set of records.

Record ids come from either --ids-file (a JSON list of ints, or an object with an
"ecg_ids" list) or --ecg-ids (one or more ids on the command line); the two are
mutually exclusive. Every requested id must be in the working set or the run
fails clearly before any API call. Each record's Mode B prompt is built, sent to
the OpenAI-compatible endpoint (base URL + key from the environment / .env),
parsed, and written.

    python scripts/run_teacher.py --ids-file data/pilot_50.json --concurrency 8

Smoke test a few specific records and print one full raw response:

    python scripts/run_teacher.py --ecg-ids 38 473 790 --show-raw

Inspect the exact prompt without calling the API (works with no credentials):

    python scripts/run_teacher.py --ecg-ids 38 --dry-run

Outputs (jsonl, gitignored under data/):
    --out  data/teacher/pilot_responses.jsonl   raw content + reasoning + usage
    --err  data/teacher/pilot_errors.jsonl       failures, retried on rerun
"""

import argparse
import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import manifest as M
from src.teacher import client as C
from src.teacher.prompt import render_prompt_mode_b, superclass_answers
from src.teacher.view import build_view


def _load_ids(path):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    ids = obj["ecg_ids"] if isinstance(obj, dict) else obj
    return [int(i) for i in ids]


def _dedupe(ids):
    """Requested ids with duplicates removed, order preserved."""
    seen, out = set(), []
    for eid in ids:
        if eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def _select_records(manifest, ids):
    """Records for ``ids``, in order. Fail clearly if any id is not in the working set."""
    missing = [eid for eid in ids if eid not in manifest]
    if missing:
        raise SystemExit(
            f"error: {len(missing)} requested ecg_id(s) not in the working set: "
            f"{missing}"
        )
    return [manifest[eid] for eid in ids]


def _generated_candidates(requested_ids, written_ids):
    """Requested ids actually written in this invocation, in requested order.

    ``written_ids`` come from run_generation (this run's successful writes), so an
    id that only exists in the file from an earlier run -- or one requested now but
    skipped/errored -- is never a candidate. This is what makes --show-raw show a
    record from the current invocation rather than a stale one.
    """
    written = set(written_ids)
    return [eid for eid in requested_ids if eid in written]


def _select_raw_row(rows, candidate_ids):
    """Pick the row to show: the first ``candidate_ids`` present, most recent write.

    ``rows`` is every line in the output file; ``candidate_ids`` are the ids
    requested and generated in this invocation. Building the lookup last-write-wins
    means a rerun's fresh line is preferred over a stale one for the same id.
    """
    by_id = {}
    for r in rows:
        by_id[r["ecg_id"]] = r
    for eid in candidate_ids:
        if eid in by_id:
            return by_id[eid]
    return None


def _print_raw(out_path, candidate_ids):
    """Print one full stored response drawn only from this invocation's records."""
    if not os.path.exists(out_path):
        print(f"(no output file at {out_path} yet)")
        return
    with open(out_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    row = _select_raw_row(rows, candidate_ids)
    if row is None:
        print("(no record was generated in this invocation to show -- all requested "
              "ids were skipped or errored)")
        return
    print("=" * 78)
    print(f"FULL RAW RESPONSE  ecg_id={row['ecg_id']}  model={row['model']}  "
          f"finish_reason={row.get('finish_reason')}  truncated={row.get('truncated')}")
    print(f"known_answers: {row['known_answers']}")
    print(f"usage: {row.get('usage')}")
    print(f"label_mismatches: {row.get('label_mismatches')}")
    print(f"parse ok: {row['parsed']['ok']}  split={row['parsed']['split_method']}  "
          f"missing={row['parsed']['missing']}  "
          f"unparseable={row['parsed']['unparseable_conclusion']}")
    print("-" * 78 + "\n[reasoning channel]\n")
    print(row.get("reasoning") or "(none returned)")
    print("\n" + "-" * 78 + "\n[content]\n")
    print(row.get("content"))
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--ids-file", default="data/pilot_50.json",
                        help="JSON list of ecg_ids (default: data/pilot_50.json)")
    source.add_argument("--ecg-ids", type=int, nargs="+", default=None,
                        help="one or more ecg_ids on the command line")
    ap.add_argument("--manifest", default="data/ptbxl/manifest.jsonl")
    ap.add_argument("--out", default="data/teacher/pilot_responses.jsonl")
    ap.add_argument("--err", default="data/teacher/pilot_errors.jsonl")
    ap.add_argument("--concurrency", type=int, default=C.DEFAULT_CONCURRENCY)
    ap.add_argument("--max-tokens", type=int, default=C.DEFAULT_MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--reasoning-effort", default=None,
                    help="optional gpt-oss reasoning effort (e.g. low/medium/high)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N ids (use 3 for the smoke test)")
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="regenerate ids already present in --out")
    ap.add_argument("--dry-run", action="store_true",
                    help="render and print the prompt(s); no API call")
    ap.add_argument("--show-raw", action="store_true",
                    help="after the run, print one full raw response")
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    manifest = M.load_manifest(args.manifest)  # working set

    if args.ecg_ids is not None:
        ids, source = _dedupe(args.ecg_ids), "--ecg-ids"
    else:
        ids, source = _dedupe(_load_ids(args.ids_file)), args.ids_file
    if args.limit is not None:
        ids = ids[:args.limit]
    records = _select_records(manifest, ids)  # fails clearly on any unknown id
    logging.info("selected %d records from %s", len(records), source)

    if args.dry_run:
        for rec in records:
            view = build_view(rec)
            print("#" * 78)
            print(f"# ecg_id {rec['ecg_id']}  known_answers={superclass_answers(view)}")
            print("#" * 78)
            print(render_prompt_mode_b(view))
            print()
        return

    config = C.TeacherConfig.from_env(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        concurrency=args.concurrency,
    )
    logging.info("endpoint=%s model=%s max_tokens=%d concurrency=%d",
                 config.endpoint, config.model, config.max_tokens, config.concurrency)

    counters, written_ids = asyncio.run(C.run_generation(
        records, config, args.out, args.err,
        skip_existing=not args.no_skip_existing,
        progress_every=args.progress_every,
    ))
    print("counters:", counters)

    if args.show_raw:
        # Restrict strictly to ids generated in THIS run, regardless of skip mode
        # or stale rows in the output file.
        _print_raw(args.out, _generated_candidates(ids, written_ids))


if __name__ == "__main__":
    main()
