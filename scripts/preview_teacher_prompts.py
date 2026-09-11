#!/usr/bin/env python
"""Render and print the full teacher prompts for N records -- no API calls.

Reads data/ptbxl/manifest.jsonl ONLY. For each selected record it prints all five
superclass prompts exactly as the teacher would receive them, so the input can be
read by hand before any generation. Ends with the working-set coverage figure:
how many records fall back to annotations-only evidence after report cleaning.

Usage:
    python scripts/preview_teacher_prompts.py --n 3
    python scripts/preview_teacher_prompts.py --ecg-ids 18822 4323 443
    python scripts/preview_teacher_prompts.py --n 2 --superclass MI
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import manifest as M
from src.teacher.prompt import (
    PROMPT_VERSION,
    SUPERCLASSES,
    question_text,
    render_prompt,
    superclass_answer,
    superclass_answers,
)
from src.teacher.view import build_view

# A Qwen tokenizer, if cached, is a modern BPE and a reasonable proxy for the
# teacher/student token count (the exact gpt-oss BPE is not available offline).
_TOKENIZER_CANDIDATES = ("Qwen/Qwen2.5-0.5B-Instruct", "gpt2")


def _load_tokenizer():
    """Return (count_fn, label). Prefer a real offline tokenizer, else chars/4.

    ``local_files_only=True`` keeps this strictly offline: an uncached model
    raises and we fall back rather than reaching for the network.
    """
    try:
        from transformers import AutoTokenizer

        for name in _TOKENIZER_CANDIDATES:
            try:
                tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
                return (lambda s: len(tok.encode(s))), f"real tokenizer: {name} (offline proxy)"
            except Exception:
                continue
    except Exception:
        pass
    return (lambda s: max(1, len(s) // 4)), "chars/4 (rough estimate, no tokenizer offline)"


def _print_record(rec, superclasses):
    view = build_view(rec)
    answers = superclass_answers(view)
    flags = view.report_flags

    print("#" * 78)
    print(f"# ecg_id {view.ecg_id}   PROMPT_VERSION={PROMPT_VERSION}")
    print(
        f"# report: {'usable' if view.has_usable_report else 'FALLBACK to annotations-only'}"
        f"  | stripped: boilerplate={flags.stripped_boilerplate} "
        f"edit={flags.stripped_edit} paren_labels={flags.stripped_paren_labels}"
    )
    print(f"# known answers: {answers}")
    print("#" * 78)
    for sc in superclasses:
        print(f"\n----- superclass {sc} -----\n")
        print(render_prompt(view, sc))
        print()


def _fallback_count(records):
    """Working-set records with no usable report after cleaning."""
    return sum(1 for r in records if not build_view(r).has_usable_report)


def _qa_stub(view, sc):
    """The two prompt lines that differ per superclass (question + known answer).

    Used to estimate the marginal cost of an extra superclass in the one-call
    mode, where template and record view are sent only once.
    """
    answer = "Yes" if superclass_answer(view, sc) else "No"
    return (
        f'- The question is: "{question_text(sc)}"\n'
        f"- The label is already known and must be followed. The known answer is: {answer}"
    )


def _token_cost_summary(records, count_tokens, label, sample_size, seed):
    """Estimate input-token cost for the two teacher-call designs.

    Mode A -- five separate calls per record -- resends the full template and
    record view five times. Mode B -- one call per record covering all five
    superclasses -- sends them once plus five short question/answer stubs; we
    estimate its per-record cost as one full prompt plus four marginal stubs.
    """
    records = list(records)
    n_total = len(records)
    if sample_size and sample_size < n_total:
        sample = random.Random(seed).sample(records, sample_size)
    else:
        sample = records
    n_sample = len(sample)

    single_tokens = 0        # sum over sample of one full prompt (all superclasses)
    mode_b_sample = 0        # sum over sample of the one-call design
    for rec in sample:
        view = build_view(rec)
        prompts = {sc: count_tokens(render_prompt(view, sc)) for sc in SUPERCLASSES}
        single_tokens += sum(prompts.values())
        first = SUPERCLASSES[0]
        mode_b_sample += prompts[first] + sum(
            count_tokens(_qa_stub(view, sc)) for sc in SUPERCLASSES[1:]
        )

    mode_a_sample = single_tokens  # five calls == five full prompts per record
    scale = n_total / n_sample
    avg_prompt = single_tokens / (n_sample * len(SUPERCLASSES))
    mode_a_total = mode_a_sample * scale
    mode_b_total = mode_b_sample * scale

    print("=" * 78)
    print("ESTIMATED INPUT-TOKEN COST")
    print(f"  tokenizer: {label}")
    if n_sample < n_total:
        print(f"  estimated from a random sample of {n_sample} of {n_total} records, extrapolated")
    else:
        print(f"  measured over all {n_total} working-set records")
    print(f"  avg input tokens per single-superclass prompt: {avg_prompt:,.0f}")
    print(
        f"  Mode A  five calls/record : {len(SUPERCLASSES) * n_total:,} calls, "
        f"~{mode_a_total:,.0f} input tokens"
    )
    print(
        f"  Mode B  one call/record   : {n_total:,} calls, "
        f"~{mode_b_total:,.0f} input tokens"
    )
    if mode_b_total:
        print(f"  Mode A / Mode B input-token ratio: {mode_a_total / mode_b_total:.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/ptbxl/manifest.jsonl")
    ap.add_argument("--n", type=int, default=3, help="number of random records")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--ecg-ids", type=int, nargs="+", default=None,
        help="specific ecg_ids to render (overrides --n / --seed sampling)",
    )
    ap.add_argument(
        "--superclass", choices=SUPERCLASSES, default=None,
        help="render only this superclass question (default: all five)",
    )
    ap.add_argument(
        "--token-sample", type=int, default=2000,
        help="records sampled for the token-cost estimate, extrapolated to the "
             "full working set (0 = measure every record; slower)",
    )
    args = ap.parse_args()

    records = M.load_manifest(args.manifest)  # working set (exclusions on)
    superclasses = [args.superclass] if args.superclass else SUPERCLASSES

    if args.ecg_ids:
        selected = []
        for eid in args.ecg_ids:
            if eid not in records:
                print(f"warning: ecg_id {eid} not in working set; skipping", file=sys.stderr)
            else:
                selected.append(records[eid])
    else:
        pool = list(records.values())
        rng = random.Random(args.seed)
        selected = rng.sample(pool, min(args.n, len(pool)))

    for rec in selected:
        _print_record(rec, superclasses)

    fallbacks = _fallback_count(records.values())
    total = len(records)
    print("=" * 78)
    print(
        f"working set: {total} records | annotations-only fallback after cleaning: "
        f"{fallbacks} ({100.0 * fallbacks / total:.3f}%)"
    )

    count_tokens, label = _load_tokenizer()
    _token_cost_summary(
        records.values(), count_tokens, label, args.token_sample, args.seed
    )


if __name__ == "__main__":
    main()
