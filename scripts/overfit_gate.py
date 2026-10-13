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
Either way it then writes the resampler the gate judged to overfit_final.pt, with
the verdict in the payload, and says where on one more line after the verdict.

The overfit loop's STOP RULE is condition 1's own number: :func:`eval_mean_loss`
over :func:`gate_batches`, the eval-mode mean over every overfit example, checked
every --overfit-eval-every steps. A loop that stopped on anything else (it used to
be one training batch) could stop and then fail condition 1 a moment later, which
costs a queue cycle to learn nothing.

Both passes write their per-example records next to the checkpoints
(gate_teacher_forced_real.jsonl, gate_generations.jsonl), so a failed gate can be
read rather than rerun.

THE CONTROLS. Once the gate's own two passes are done, the teacher-forced pass is
repeated on the same examples under the two controls of scripts/eval_controls.py --
shuffled embeddings (gate_teacher_forced_shuffled.jsonl) and zeroed latents
(gate_teacher_forced_zeroed.jsonl) -- and a table of the three mean losses is
printed above the verdict. They are DIAGNOSTIC: the verdict is computed from the
real condition alone, with the conditions and thresholds above, so a gate run means
what it meant before the controls existed. What they add is whether a passing loss
depended on the ECG at all.

The shuffle is a derangement over the overfit set's OWN records, not over the train
split: every donor is an ECG the resampler was trained on, so the real-vs-shuffled
gap is about getting the right record and not about seeing an unfamiliar one. The
donor is written into every shuffled record as ``donor_ecg_id``.

UNDER --loss-on decision the training loss lands on one position per example, the
Yes/No token condition 3 reads, and nothing supervises the prose. Conditions 1 and
2 are then not meaningful -- the teacher-forced mean over the whole target will not
fall, and free generation may stop closing with <END> -- so all three are still
measured and printed exactly as above, but the verdict is condition 3 alone, and
the closing line says that rule was the one applied. The stop rule moves with it:
:func:`eval_decision_loss`, the eval-mode decision-token mean per label, read at
the very positions the training loss used. And the controls are reported a second
time ON THE DECISION TOKEN, split by label (:func:`print_decision_controls`): real
against shuffled at that one token is whether knowing which ECG it is changes the
answer. Under --loss-on all, the default, none of this paragraph applies.
"""

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_controls import CONDITIONS as CONTROL_CONDITIONS  # noqa: E402
from eval_controls import assert_same_examples  # noqa: E402
from evaluate import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    END_MARKER,
    EvalContext,
    ShuffledEmbeddings,
    run_generation,
    run_teacher_forced,
    summarise_teacher_forced,
    teacher_forced_batch,
)
from src.data.dataset import make_collate_fn  # noqa: E402
from src.teacher.prompt import superclass_answer  # noqa: E402
from src.teacher.view import build_view  # noqa: E402
from yn_token_loss import decision_index, tokenise_target  # noqa: E402

# --gate-threshold is the only one of these exposed as a flag; the rest are the
# gate's definition and are changed here, on purpose, not per submission.
GATE_END_RATE = 0.90          # fraction of generations that must close with <END>
GATE_LENGTH_TOLERANCE = 0.25  # |mean generated - mean target| / mean target
GATE_MATCH_RATE = 0.90        # fraction whose generated Yes/No matches the label
GATE_DECISION_LOSS = 0.1      # per label subset, at the token after "Conclusion:"

CONDITIONS = {1: "teacher-forced loss", 2: "free generation", 3: "decision-token loss"}


def gate_batches(tokenizer, dataset, indices, args, collate=None):
    """The overfit examples as the gate's teacher-forced pass batches them.

    Same collate, dataset order and batch size as ``run_teacher_forced`` is given in
    :func:`run_gate`, so the stop rule and condition 1 see identical tensors.
    Materialised once: the stop rule walks these every --overfit-eval-every steps,
    and 50 cached embeddings are not worth a worker pool each time.

    ``collate`` replaces the default only to ADD to it: under --loss-on decision
    train.py passes its own, which is this same collate plus the ``decision_mask``
    :func:`eval_decision_loss` reads. Every other tensor is unchanged.
    """
    from torch.utils.data import DataLoader, Subset

    return list(DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        collate_fn=collate or make_collate_fn(
            tokenizer, max_length=args.max_length,
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


@torch.no_grad()
def eval_decision_loss(model, resampler, batches, dev, task_stream: str = "prompt"):
    """Condition 3's quantity: eval-mode loss at the decision token, mean per label.

    The --loss-on decision stop rule. Returns ``{"all": mean, True: mean, False:
    mean}`` over examples, with None for a label that has no examples. The position
    is the one ``decision_mask`` marks -- the mask the training loss was computed
    under -- so the stop rule cannot drift from what is being trained;
    condition 3 then locates the token again from the target text, independently.
    Leaves the resampler in whichever mode it was found.
    """
    was_training = resampler.training
    resampler.eval()
    losses, labels = [], []
    for batch in batches:
        scored = teacher_forced_batch(model, resampler, batch, dev,
                                      task_stream=task_stream)
        for i, (_, _, position_losses) in enumerate(scored):
            supervised = batch["labels"][i] != -100
            # .item() is the check: exactly one marked position inside the target
            k = int(batch["decision_mask"][i][supervised].nonzero().item())
            losses.append(position_losses[k])
            labels.append(bool(batch["head_label"][i]))
    resampler.train(was_training)
    losses, labels = np.asarray(losses, dtype=np.float64), np.asarray(labels)
    out = {"all": float(losses.mean())}
    for label in (True, False):
        subset = losses[labels == label]
        out[label] = float(subset.mean()) if subset.size else None
    return out


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


def gate_verdict(tf_records, gen_records, decision_losses, target_lengths, threshold,
                 loss_on: str = "all"):
    """Print the three conditions with their thresholds; True only if all pass.

    Pure on the records, so the verdict can be tested without a model. The four
    sequences are parallel, one entry per example, in the same order.

    ``loss_on`` is train.py's --loss-on, and picks the rule. Under "all" every
    condition is judged. Under "decision" nothing supervised the prose, so
    conditions 1 and 2 are measured and printed with their thresholds as ever but
    marked not judged, and the verdict is condition 3 alone; the closing line names
    that rule.
    """
    if loss_on not in ("all", "decision"):
        raise ValueError(f"unknown loss_on {loss_on!r}")
    judged = {3} if loss_on == "decision" else set(CONDITIONS)
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
        print(f"{cond}. {name}" + ("" if cond in judged else
                                   "   [printed, NOT judged under --loss-on decision]"))
        for c, what, measured, need, ok in rows:
            if c != cond:
                continue
            print(f"     {what:<26} {measured:<44} need {need:<12} "
                  f"{'PASS' if ok else 'FAIL'}"
                  + ("" if cond in judged else "  (not judged)"))
            if not ok and cond in judged:
                failed.setdefault(cond, []).append(what)
        if cond == 1:
            print(f"     (token-weighted mean {_loss(tf['token_weighted_mean_loss'])}, "
                  "the convention the training loop logs)")
    print("-" * line_w)
    # Which rule produced the verdict, on the verdict's own line. Empty under "all",
    # where the line reads as it always has.
    rule = ("" if loss_on == "all" else
            "  [--loss-on decision rule: judged on condition 3 alone; conditions 1 "
            "and 2 are printed, not judged]")
    if not failed:
        print("GATE PASSED: " + (
            "all three conditions met" if loss_on == "all" else
            f"condition 3 met, {CONDITIONS[3]} < {GATE_DECISION_LOSS:g} for label=True "
            "and label=False separately") + rule)
        return True
    print("GATE FAILED: " + "; ".join(
        f"condition {cond} {CONDITIONS[cond]} ({', '.join(whats)})"
        for cond, whats in failed.items()) + rule)
    return False


def _loss(v):
    return "n/a" if v is None else f"{v:.4f}"


def _rate(k, n):
    return f"{k}/{n} ({k / n:.1%})"


def teacher_forced_pass(ctx, dataset, indices, args, name, *, shuffled, zero_latents):
    """One teacher-forced pass over ``indices``, written to its condition's jsonl.

    Shared by the gate's own pass and the two controls, so the three differ in the
    view they are handed and the two flags and in nothing else.
    """
    records, _ = run_teacher_forced(
        ctx, dataset, indices, split="train", seed=args.seed, shuffled=shuffled,
        zero_latents=zero_latents, batch_size=args.batch_size,
        num_workers=args.num_workers, max_length=args.max_length,
        out_path=os.path.join(args.ckpt_dir, f"gate_teacher_forced_{name}.jsonl"),
        task_stream=args.task_stream)
    return records


def shuffle_within(dataset, indices, seed):
    """The shuffled control's view: a derangement over the gate's OWN records.

    evaluate.apply_shuffle permutes across the whole split, which here would hand
    ten overfit records donors from the other ~17k the resampler never saw. The
    permutation is restricted to the records behind ``indices`` instead, and it is
    over records rather than examples: a record's five questions share one ECG, so
    an example-level permutation at this size hands a question its own embedding
    back through a sibling.

    Fatal unless the mapping is a permutation of exactly those records with no fixed
    point, and unless every donor's cached embedding really differs from the
    record's own -- two records with identical embeddings would be a fixed point the
    ids cannot show. Returns None when the examples span fewer than two records:
    there is then no other ECG to serve, and the control is skipped, not faked.
    """
    records = sorted({int(dataset.examples[i].ecg_id) for i in indices})
    if len(records) < 2:
        print(f"shuffled control skipped: the {len(indices)} examples span "
              f"{len(records)} record(s), so there is no other ECG to serve")
        return None
    view = ShuffledEmbeddings(dataset, seed, ecg_ids=records)
    mapping = view.mapping
    problems = []
    if sorted(mapping) != records or sorted(mapping.values()) != records:
        problems.append("the mapping is not a permutation of the gate's own records")
    for ecg_id, donor in mapping.items():
        if donor == ecg_id:
            problems.append(f"record {ecg_id} keeps its own embedding")
        elif np.array_equal(dataset.cache.get(ecg_id), dataset.cache.get(donor)):
            problems.append(f"record {ecg_id} is served record {donor}, whose cached "
                            "embedding is identical to its own")
    if problems:
        sys.exit("GATE CONTROLS: the embedding shuffle is not a derangement over the "
                 f"{len(records)} records:\n  " + "\n  ".join(problems))
    print(f"shuffled control: derangement over the {len(records)} records behind "
          f"these {len(indices)} examples, SEED {seed}, no record keeps its own "
          "embedding")
    print("    own -> donor: " + "  ".join(f"{k}->{mapping[k]}" for k in records))
    return view


def assert_no_own_embedding(records):
    """Fatal unless every shuffled record names a donor that is not its own ECG.

    Per example, on what the pass wrote: a missing donor means the pass ran on the
    plain dataset under a shuffled label, which is the real condition again.
    """
    kept = [r for r in records
            if r.get("donor_ecg_id") is None or r["donor_ecg_id"] == r["ecg_id"]]
    if kept:
        sys.exit(f"GATE CONTROLS: {len(kept)} of {len(records)} shuffled examples "
                 "kept their own embedding:\n" + "\n".join(
                     f"  {r['ecg_id']:>7} {r['superclass']:<6} donor "
                     f"{r.get('donor_ecg_id')}" for r in kept))
    print(f"  shuffled control: all {len(records)} examples were served another "
          "record's embedding")


def print_controls(by_condition):
    """The three mean losses and each control's gap to real, in nats and percent.

    Pure on the records. The mean is the one condition 1 reports -- over examples --
    so the ``real`` row is the gate's own number. Returns the rows.
    """
    means = {name: summarise_teacher_forced(recs)["mean_loss"]
             for name, recs in by_condition.items()}
    real = means.get("real")
    rows = []
    for name, _ in CONTROL_CONDITIONS:
        mean = means.get(name)
        gap = None if name == "real" or mean is None or real is None else mean - real
        # a real loss of exactly zero leaves the percentage undefined, not infinite
        pct = 100.0 * gap / real if gap is not None and real else None
        rows.append({"condition": name, "ran": name in by_condition,
                     "mean_loss": mean, "gap_nats": gap, "gap_pct": pct})

    line_w = 64
    print("\n" + "=" * line_w)
    print(f"GATE CONTROLS on the same {len(by_condition['real'])} examples "
          "(diagnostic, not in the verdict)")
    print("=" * line_w)
    print(f"{'teacher-forced':<16} {'mean loss':>10} {'gap, nats':>14} {'gap, %':>14}")
    print("-" * line_w)
    for row in rows:
        if not row["ran"]:
            print(f"{row['condition']:<16} {'not run':>10}")
            continue
        gap = "" if row["gap_nats"] is None else f"{row['gap_nats']:+.4f}"
        pct = "" if row["gap_pct"] is None else f"{row['gap_pct']:+.1f}%"
        print(f"{row['condition']:<16} {_loss(row['mean_loss']):>10} {gap:>14} {pct:>14}")
    print("-" * line_w)
    print("gap = control - real, mean over examples; positive means the control "
          "costs loss.")
    print("real vs shuffled near zero => the loss does not depend on WHICH ECG.")
    print("real vs zeroed   near zero => it does not depend on the prefix AT ALL.")
    return rows


def donor_answers(dataset, shuffled_records):
    """Per shuffled example: the DONOR record's known answer to that example's question.

    Computed as the dataset computes an example's own label, from the donor's
    manifest record. It is what a student that reads the prefix should now answer,
    so it splits the shuffled control into the examples where the wrong ECG asks
    for a different answer and those where it asks for the same one.
    """
    return [superclass_answer(build_view(dataset.manifest[int(r["donor_ecg_id"])]),
                              r["superclass"]) for r in shuffled_records]


def print_decision_controls(decision_by_condition, labels, donor_labels=None):
    """The real / shuffled / zeroed triple AT THE DECISION TOKEN, split by label.

    ``decision_by_condition`` maps a condition name to its per-example decision-token
    losses; ``labels`` and ``donor_labels`` (the shuffled donor's answer per example,
    None when the shuffle did not run) are parallel to them. Each label's row is
    followed by the part of it whose donor answers the other way: only there does
    serving the wrong ECG ask for a different token, so a label's whole-row gap is
    diluted by however many of its donors happen to share its answer. Mean over
    examples throughout. Pure on its arguments; returns the rows.
    """
    labels = np.asarray(labels, dtype=bool)
    names = [name for name, _ in CONTROL_CONDITIONS]
    dec = {name: np.asarray(decision_by_condition[name], dtype=np.float64)
           for name in names if name in decision_by_condition}
    subsets = [("all", np.ones(labels.size, dtype=bool))]
    for label in (True, False):
        subsets.append((f"label={label}", labels == label))
        if donor_labels is not None:
            differs = np.asarray(donor_labels, dtype=bool) != labels
            subsets.append((f"  donor answers {'No' if label else 'Yes'}",
                            (labels == label) & differs))

    rows = []
    for what, mask in subsets:
        means = {name: float(v[mask].mean()) if mask.any() else None
                 for name, v in dec.items()}
        real = means.get("real")
        rows.append({"subset": what.strip(), "label": what, "n": int(mask.sum()),
                     "mean_loss": means,
                     "gap_nats": {name: None if None in (mean, real) else mean - real
                                  for name, mean in means.items() if name != "real"}})

    controls = [name for name in names if name != "real"]
    line_w = 22 + 6 + 11 * len(names) + 18 * len(controls)
    print("\n" + "=" * line_w)
    print(f"GATE CONTROLS ON THE DECISION TOKEN, the same {labels.size} examples "
          "(diagnostic, not in the verdict)")
    print("=" * line_w)
    print(f"{'decision-token loss':<22}{'n':>6}" + "".join(f"{n:>11}" for n in names)
          + "".join(f"{n + ' - real':>18}" for n in controls))
    print("-" * line_w)
    for row in rows:
        cells = "".join(f"{_loss(row['mean_loss'][n]) if n in dec else 'not run':>11}"
                        for n in names)
        gaps = "".join(
            f"{'' if row['gap_nats'].get(n) is None else format(row['gap_nats'][n], '+.4f'):>18}"
            for n in controls)
        print(f"{row['label']:<22}{row['n']:>6}{cells}{gaps}")
    print("-" * line_w)
    print("loss at the single Yes/No token after \"Conclusion:\", mean over examples; "
          "gap = control - real.")
    print("real vs shuffled near zero => the answer does not depend on WHICH ECG it "
          "is given.")
    print("real vs zeroed   near zero => it does not depend on the prefix AT ALL.")
    if donor_labels is not None:
        print("'donor answers' rows: the examples of that label whose shuffled-in ECG "
              "has the OTHER known\nanswer to the same question. The rest were served "
              "an ECG with their own answer, where a\nstudent that reads the prefix "
              "loses nothing -- so read the shuffled gap on these rows.")
        if not differs.any():
            print("NOTE: NO example was served an ECG with a different known answer: "
                  "every record in this set\nanswers every question the same way, so "
                  "the answer is a function of the question alone. The\nshuffled "
                  "column cannot show whether the ECG is read here, and a decision-"
                  "token loss near\nzero does not need it to be.")
    return rows


def run_controls(ctx, dataset, indices, args, real_records, shuffled_view):
    """The teacher-forced pass again under each control, and the table.

    DIAGNOSTIC ONLY. Nothing computed here is returned to the verdict's inputs:
    :func:`run_gate` calls this after both of the gate's own passes and hands
    :func:`gate_verdict` the real records it already had.
    """
    by_condition = {"real": real_records}
    for name, opts in CONTROL_CONDITIONS:
        if name == "real" or (opts["shuffled"] and shuffled_view is None):
            continue
        print("\n" + "#" * 64)
        print(f"# GATE CONTROL (diagnostic): teacher-forced loss, {name}")
        print("#" * 64, flush=True)
        # Same indices, same dataset order; only the embedding a record is served
        # and whether the prefix is zeroed change.
        by_condition[name] = teacher_forced_pass(
            ctx, shuffled_view if opts["shuffled"] else dataset, indices, args, name,
            **opts)

    print("\nsample check:")
    assert_same_examples(by_condition, "gate teacher-forced conditions")
    if "shuffled" in by_condition:
        assert_no_own_embedding(by_condition["shuffled"])
    print_controls(by_condition)
    return by_condition


def run_gate(model, resampler, tokenizer, dataset, indices, args) -> bool:
    """Run both passes over ``indices`` of ``dataset`` and return the verdict.

    ``args`` is train.py's namespace: the passes use its batch size, max length and
    task stream so the teacher-forced number is measured the way the overfit trained.

    The controls run after both passes and before the verdict is printed, so
    everything the verdict reads was measured before any control code ran. The
    verdict is the last thing THIS prints; train.py adds one line after it saying
    where the resampler the gate judged was saved.
    """
    resampler.eval()
    model.config.use_cache = True   # off for the backward pass; generation needs it
    ctx = EvalContext.from_loaded(model, resampler, tokenizer)
    real = dict(CONTROL_CONDITIONS)["real"]
    # Built and checked before any pass: it depends only on the indices and the seed.
    shuffled_view = shuffle_within(dataset, indices, args.seed)

    print("\n" + "#" * 64)
    print("# GATE PASS 1 of 2: teacher-forced loss")
    print("#" * 64, flush=True)
    tf_records = teacher_forced_pass(ctx, dataset, indices, args, "real", **real)

    print("\n" + "#" * 64)
    print("# GATE PASS 2 of 2: free generation")
    print("#" * 64, flush=True)
    # Batch 1: training right-pads, so a prompt never sat behind padding. Decoding
    # one at a time keeps every example in the layout it was overfit in.
    gen_records, _, _ = run_generation(
        ctx, dataset, indices, os.path.join(args.ckpt_dir, "gate_generations.jsonl"),
        split="train", seed=args.seed, num_workers=args.num_workers,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS, batch_size=1, print_first=3,
        log_every=10, **real)

    targets = [dataset.examples[i].target for i in indices]
    located = locate_decision_tokens(tokenizer, targets, tf_records)

    by_condition = run_controls(ctx, dataset, indices, args, tf_records, shuffled_view)
    if args.loss_on == "decision":
        # Still diagnostic: the verdict below reads `located`, the real pass alone.
        # Each control goes through the same locator and the same fatal length check.
        decision = {name: [loss for loss, _ in
                           (located if name == "real" else
                            locate_decision_tokens(tokenizer, targets, recs))]
                    for name, recs in by_condition.items()}
        print_decision_controls(
            decision, [r["label"] for r in tf_records],
            donor_answers(dataset, by_condition["shuffled"])
            if "shuffled" in by_condition else None)

    return gate_verdict(tf_records, gen_records,
                        [loss for loss, _ in located], [n for _, n in located],
                        args.gate_threshold, loss_on=args.loss_on)
