#!/usr/bin/env python
"""The overfit gate's verdict: three conditions on the examples just trained on.

A low training loss is not enough to call the pipeline working. Run 1's gate reached
0.0496 teacher-forced and its free generation was still degenerate, so the gate now
has to show the same thing three ways, all on the examples the overfit trained on:

    1. TEACHER-FORCED LOSS below --gate-threshold. Mean over examples, resampler in
       eval mode, under training's own collate and masking.

    2. FREE GENERATION, greedy, with the decoding config set explicitly
       (evaluate.build_generation_config), at batch 1 so no example is decoded
       behind padding it never trained with. Three checks, all required:
         - at least 90% of generations close with <END>
         - mean generated length within 25% of mean target length
         - the Yes/No parsed from the generated Conclusion matches the label on at
           least 90% (an unparseable generation counts as a mismatch)

    3. DECISION-TOKEN LOSS -- the loss at the single Yes/No token after
       "Conclusion:" -- below 0.1 on the label=True and label=False subsets
       SEPARATELY. Separately is the point: a constant-Yes model sits near zero on
       every positive, so it passes an overall mean whenever positives dominate.
       The token is found with scripts/yn_token_loss.py's locator, and the same
       length check that script treats as fatal is fatal here.

All three are printed with their thresholds whether they pass or not, and the
closing line names every condition that failed. scripts/train.py --overfit calls
:func:`run_gate` once the overfit loop stops and exits 0 only if it returns True.

The overfit loop's STOP RULE is condition 1's own number: :func:`eval_mean_loss`
over :func:`gate_batches`, the eval-mode mean over every overfit example, checked
every --overfit-eval-every steps. A loop that stopped on anything else (it used to
be one training batch) could stop and then fail condition 1 a moment later, which
costs a queue cycle to learn nothing.

Both passes write their per-example records next to the checkpoints
(gate_teacher_forced.jsonl, gate_generations.jsonl), so a failed gate can be read
rather than rerun.
"""

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    END_MARKER,
    EvalContext,
    run_generation,
    run_teacher_forced,
    summarise_teacher_forced,
    teacher_forced_batch,
)
from src.data.dataset import make_collate_fn  # noqa: E402
from yn_token_loss import decision_index, tokenise_target  # noqa: E402

# --gate-threshold is the only one of these exposed as a flag; the rest are the
# gate's definition and are changed here, on purpose, not per submission.
GATE_END_RATE = 0.90          # fraction of generations that must close with <END>
GATE_LENGTH_TOLERANCE = 0.25  # |mean generated - mean target| / mean target
GATE_MATCH_RATE = 0.90        # fraction whose generated Yes/No matches the label
GATE_DECISION_LOSS = 0.1      # per label subset, at the token after "Conclusion:"

CONDITIONS = {1: "teacher-forced loss", 2: "free generation", 3: "decision-token loss"}


def gate_batches(tokenizer, dataset, indices, args):
    """The overfit examples as the gate's teacher-forced pass batches them.

    Same collate, dataset order and batch size as ``run_teacher_forced`` is given in
    :func:`run_gate`, so the stop rule and condition 1 see identical tensors.
    Materialised once: the stop rule walks these every --overfit-eval-every steps,
    and 50 cached embeddings are not worth a worker pool each time.
    """
    from torch.utils.data import DataLoader, Subset

    return list(DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        collate_fn=make_collate_fn(tokenizer, max_length=args.max_length,
                                   strip_think=dataset.config.strip_think)))


@torch.no_grad()
def eval_mean_loss(model, resampler, batches, dev, task_stream: str = "prompt") -> float:
    """Condition 1's quantity: eval-mode teacher-forced loss, mean over examples.

    inf if any example has no target left to score, which condition 1 also fails.
    Leaves the resampler in whichever mode it was found.
    """
    was_training = resampler.training
    resampler.eval()
    losses = []
    for batch in batches:
        losses += [loss for loss, _, _ in teacher_forced_batch(
            model, resampler, batch, dev, task_stream=task_stream)]
    resampler.train(was_training)
    return float("inf") if None in losses else float(np.mean(losses))


def locate_decision_tokens(tokenizer, targets, tf_records):
    """Per example: (decision-token loss, target length in tokens without EOS).

    Fatal if a rebuilt target is not the sequence the loss was computed over, or its
    decision token cannot be located: an index into ``position_losses`` read off a
    misaligned target is a number about some other token.
    """
    out, problems = [], []
    for text, r in zip(targets, tf_records):
        ids, offsets = tokenise_target(tokenizer, text)
        k, why = decision_index(text, offsets)
        n_pos = len(r.get("position_losses") or [])
        if len(ids) != r["n_target_tokens"] or n_pos != len(ids):
            why = (f"rebuilt target has {len(ids)} tokens, n_target_tokens "
                   f"{r['n_target_tokens']}, position_losses {n_pos}")
        if why:
            problems.append(f"  {r['ecg_id']:>7} {r['superclass']:<6} {why}")
            continue
        out.append((r["position_losses"][k], len(offsets)))
    if problems:
        sys.exit(f"GATE FAILED: condition 3 {CONDITIONS[3]} cannot be evaluated; the "
                 f"rebuilt target does not line up with the loss for {len(problems)} "
                 f"of {len(tf_records)} examples:\n" + "\n".join(problems))
    return out


def gate_verdict(tf_records, gen_records, decision_losses, target_lengths, threshold):
    """Print the three conditions with their thresholds; True only if all pass.

    Pure on the records, so the verdict can be tested without a model. The four
    sequences are parallel, one entry per example, in the same order.
    """
    n = len(tf_records)
    if not (len(gen_records) == len(decision_losses) == len(target_lengths) == n):
        raise ValueError("the gate's passes cover different numbers of examples")
    for tf, gen in zip(tf_records, gen_records):
        if (tf["ecg_id"], tf["superclass"]) != (gen["ecg_id"], gen["superclass"]):
            raise ValueError("teacher-forced and generation passes are in a different order")

    rows = []      # (condition, what, measured, need, ok)

    tf = summarise_teacher_forced(tf_records)
    mean_loss = tf["mean_loss"]
    rows.append((1, "mean over examples", _loss(mean_loss), f"< {threshold:g}",
                 mean_loss is not None and tf["n_scored"] == n and mean_loss < threshold))

    closed = sum(1 for r in gen_records if r["stopped_at_end"])
    rows.append((2, f"closed with {END_MARKER}", _rate(closed, n),
                 f">= {GATE_END_RATE:.0%}", closed >= GATE_END_RATE * n))
    gen_len = float(np.mean([r["n_generated_tokens"] for r in gen_records]))
    tgt_len = float(np.mean(target_lengths))
    off = (gen_len - tgt_len) / tgt_len
    rows.append((2, "mean length, tokens",
                 f"{gen_len:.1f} generated vs {tgt_len:.1f} target ({off:+.1%})",
                 f"within {GATE_LENGTH_TOLERANCE:.0%}", abs(off) <= GATE_LENGTH_TOLERANCE))
    matched = sum(1 for r in gen_records
                  if r["parsed_conclusion"] is not None
                  and bool(r["parsed_conclusion"]) == bool(r["label"]))
    rows.append((2, "Conclusion matches label", _rate(matched, n),
                 f">= {GATE_MATCH_RATE:.0%}", matched >= GATE_MATCH_RATE * n))

    dec = np.asarray(decision_losses, dtype=np.float64)
    labels = np.array([bool(r["label"]) for r in tf_records])
    for label in (True, False):
        subset = dec[labels == label]
        # an empty subset cannot show the condition holds, so it does not pass
        mean = float(subset.mean()) if subset.size else None
        measured = (f"{_loss(mean)}  (max {subset.max():.4f})" if subset.size
                    else "no examples")
        rows.append((3, f"label={label} (n {subset.size})", measured,
                     f"< {GATE_DECISION_LOSS:g}",
                     mean is not None and mean < GATE_DECISION_LOSS))

    line_w = 100
    print("\n" + "=" * line_w)
    print(f"OVERFIT GATE on the {n} examples just trained on")
    print("=" * line_w)
    failed = {}
    for cond, name in CONDITIONS.items():
        print(f"{cond}. {name}")
        for c, what, measured, need, ok in rows:
            if c != cond:
                continue
            print(f"     {what:<26} {measured:<44} need {need:<12} "
                  f"{'PASS' if ok else 'FAIL'}")
            if not ok:
                failed.setdefault(cond, []).append(what)
        if cond == 1:
            print(f"     (token-weighted mean {_loss(tf['token_weighted_mean_loss'])}, "
                  "the convention the training loop logs)")
    print("-" * line_w)
    if not failed:
        print("GATE PASSED: all three conditions met")
        return True
    print("GATE FAILED: " + "; ".join(
        f"condition {cond} {CONDITIONS[cond]} ({', '.join(whats)})"
        for cond, whats in failed.items()))
    return False


def _loss(v):
    return "n/a" if v is None else f"{v:.4f}"


def _rate(k, n):
    return f"{k}/{n} ({k / n:.1%})"


def run_gate(model, resampler, tokenizer, dataset, indices, args) -> bool:
    """Run both passes over ``indices`` of ``dataset`` and return the verdict.

    ``args`` is train.py's namespace: the passes use its batch size, max length and
    task stream so the teacher-forced number is measured the way the overfit trained.
    """
    resampler.eval()
    model.config.use_cache = True   # off for the backward pass; generation needs it
    ctx = EvalContext.from_loaded(model, resampler, tokenizer)
    common = dict(split="train", seed=args.seed, shuffled=False, zero_latents=False,
                  num_workers=args.num_workers)

    print("\n" + "#" * 64)
    print("# GATE PASS 1 of 2: teacher-forced loss")
    print("#" * 64, flush=True)
    tf_records, _ = run_teacher_forced(
        ctx, dataset, indices, batch_size=args.batch_size, max_length=args.max_length,
        out_path=os.path.join(args.ckpt_dir, "gate_teacher_forced.jsonl"),
        task_stream=args.task_stream, **common)

    print("\n" + "#" * 64)
    print("# GATE PASS 2 of 2: free generation")
    print("#" * 64, flush=True)
    # Batch 1: training right-pads, so a prompt never sat behind padding. Decoding
    # one at a time keeps every example in the layout it was overfit in.
    gen_records, _, _ = run_generation(
        ctx, dataset, indices, os.path.join(args.ckpt_dir, "gate_generations.jsonl"),
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS, batch_size=1, print_first=3,
        log_every=10, **common)

    targets = [dataset.examples[i].target for i in indices]
    located = locate_decision_tokens(tokenizer, targets, tf_records)
    return gate_verdict(tf_records, gen_records,
                        [loss for loss, _ in located], [n for _, n in located],
                        args.gate_threshold)
