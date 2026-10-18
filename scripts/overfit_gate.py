#!/usr/bin/env python
"""The overfit gate's verdict: conditions on the examples just trained on.

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
Yes/No token condition 3 reads, and nothing supervises the prose. The stop rule
moves with the verdict: :func:`eval_no_context_loss`, condition 4 (a)'s own number
-- the eval-mode no-context loss per label, through the same function the gate's
pass calls -- so the loop cannot stop on a number the judged condition then fails.
Condition 3's teacher-forced number (:func:`eval_decision_loss`) is logged beside
it at every check, not stopped on. And the controls are
reported a second time ON THE DECISION TOKEN, split by label
(:func:`print_decision_controls`). Under --loss-on all, the default, none of this
paragraph or the next applies.

The verdict is then CONDITION 4 ALONE (below). Conditions 1-3 are all measured and
printed with their thresholds, marked not judged. 1 and 2 because nothing trained
the prose. 3 because teacher forcing puts the target's last reasoning step right
before "Conclusion:", which states the finding in prose, so the decision token can
be read off the context: a constant prefix reached 0.018 on all fifty without the
ECG. A verdict resting on it can pass a model that never opens the waveform.

UNDER --loss-on decision-no-context everything above holds, and the one thing that
changes is the training sequence: each target is cut to "Conclusion:" and the
decision token (:func:`no_context_target`), so the loss is condition 4's own
quantity. Under --loss-on decision the token is trained with the whole target in
context, which supplies the answer, so condition 4 judges a behaviour that is never
trained: one such run reached 0.0000 on condition 3 and 7.99 on condition 4 (a) for
label=False, answering Yes to everything. Both modes are judged on condition 4, and
both are what DECISION_MODES names.

Two things follow, both enforced by train.py's parser for either mode: it refuses
--overfit-select head (all ten head records give each question the same answer, so
condition 4's shuffled column and donor rows cannot show anything and the verdict
would be meaningless rather than merely weak), and --skip-generation is on by
default there, since the free-generation pass was 21 of the last run's 25 minutes
and measures a condition that is not judged. Skipped, condition 2 is printed as not
run, never as a failure.

4. THE DECISION TOKEN WITH NO TARGET CONTEXT, judged under DECISION_MODES only.
   The student prompt followed by "Conclusion:" and nothing else -- no Evidence, no
   Reasoning -- and the loss and p_yes at the next position (:func:`no_context_pass`).
   With no prose to read the answer off, what predicts the token is the prompt,
   which says nothing about the answer, and the prefix. It runs under real,
   shuffled and zeroed, on the same examples. Two parts, because one number cannot
   say both things, and both must hold:

     (a) the real loss < GATE_DECISION_LOSS (0.1) on label=True and label=False
         separately. The model can answer at all.

     (b) on the donor-answers-opposite rows -- the label=True examples whose
         shuffled-in ECG answers No, and the label=False ones whose donor answers
         Yes -- shuffled minus real >= GATE_SHUFFLE_GAP (0.5 nats), on each row.
         The answer depends on which ECG it was given.

   WHY 0.5. A coin flip is ln 2 = 0.693 nats. If the prefix carries the answer and
   the answer comes from the waveform, handing the model another record's heart --
   one whose answer is the opposite -- must at minimum push it off certainty and
   toward chance. With real near 0.05, a half-nat gap puts shuffled around 0.55,
   just inside chance. Anything smaller is not a model being misled, it is a model
   that was never looking. A model that genuinely flips -- 95% on the donor's
   answer -- sits near -ln(0.05) = 3 nats on those rows and clears this by a wide
   margin: the bar is set at the floor of credible evidence, not at what success
   should look like.

   Each of the two (b) rows needs at least GATE_MIN_DONOR_ROWS (8) examples. With
   fewer, condition 4 is NOT JUDGED: :func:`gate_verdict` returns None rather than
   passing or failing on a number that thin, and the verdict line reads GATE NOT
   JUDGED with the row counts. train.py treats it as not passed (exit 1).

   AUPRC over the examples from p_yes -- the quantity the project is evaluated on --
   is printed against the positive rate, AUPRC's null, and is not judged.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

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
    auprc,
    probability_from_logits,
    run_generation,
    run_teacher_forced,
    splice_prefix,
    summarise_teacher_forced,
    teacher_forced_batch,
    yes_no_token_ids,
)
from src.data.dataset import make_collate_fn, task_stream_ids  # noqa: E402
from src.teacher.prompt import superclass_answer  # noqa: E402
from src.teacher.view import build_view  # noqa: E402
from yn_token_loss import MARKER, decision_index, tokenise_target  # noqa: E402

# --gate-threshold is the only one of these exposed as a flag; the rest are the
# gate's definition and are changed here, on purpose, not per submission.
GATE_END_RATE = 0.90          # fraction of generations that must close with <END>
GATE_LENGTH_TOLERANCE = 0.25  # |mean generated - mean target| / mean target
GATE_MATCH_RATE = 0.90        # fraction whose generated Yes/No matches the label
GATE_DECISION_LOSS = 0.1      # per label subset, at the token after "Conclusion:"
GATE_SHUFFLE_GAP = 0.5        # nats, condition 4 (b); why 0.5: the module docstring
GATE_MIN_DONOR_ROWS = 8       # examples each condition 4 (b) row needs to be judged

# The --loss-on modes whose loss is the Yes/No token alone. Both are judged the same
# way, on condition 4; they differ only in what precedes the token in training.
DECISION_MODES = ("decision", "decision-no-context")

CONDITIONS = {1: "teacher-forced loss", 2: "free generation", 3: "decision-token loss",
              4: "decision token, no target context"}


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

    Logged at every --loss-on decision stop check beside the stop rule
    (:func:`eval_no_context_loss`), not used to stop: condition 3 is not judged.
    Returns :func:`mean_by_label`'s dict. The position is the one ``decision_mask``
    marks -- the mask the training loss was computed under -- and condition 3 then
    locates the token again from the target text, independently. Leaves the
    resampler in whichever mode it was found.
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
    return mean_by_label(losses, labels)


def mean_by_label(losses, labels):
    """``{"all": mean, True: mean, False: mean}`` over examples, with None for a label
    that has no examples. The stop rules' shape, so train.py logs both alike."""
    losses, labels = np.asarray(losses, dtype=np.float64), np.asarray(labels, dtype=bool)
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
                 loss_on: str = "all", no_context=None, donor_labels=None):
    """Print every condition with its threshold; True only if every judged one passes.

    Pure on the records, so the verdict can be tested without a model. The four
    sequences are parallel, one entry per example, in the same order.

    ``loss_on`` is train.py's --loss-on, and picks the rule. Under "all" conditions
    1-3 are judged. Under either of DECISION_MODES condition 4 alone is judged, and 1-3 are
    measured and printed with their thresholds as ever but marked not judged
    (condition 3 with a line saying why); the closing line names that rule.

    ``gen_records`` is None when --skip-generation left the pass out: condition 2 is
    then printed as not run, which is only allowed where it is not judged.

    ``no_context`` maps a condition name to its :func:`no_context_pass` records,
    parallel to the rest; "real" is required under DECISION_MODES, and "shuffled" is
    there unless the shuffle was skipped. ``donor_labels`` is the shuffled donor's
    known answer per example (:func:`donor_answers`). None for both, as under "all",
    leaves condition 4 out entirely.

    Returns None, not False, when condition 4 is not judged because a
    donor-answers-opposite row has fewer than GATE_MIN_DONOR_ROWS examples: that is
    not a pass, and it is not a failure either, and the closing line says which.
    """
    if loss_on not in ("all", *DECISION_MODES):
        raise ValueError(f"unknown loss_on {loss_on!r}")
    decision = loss_on in DECISION_MODES
    judged = {4} if decision else {1, 2, 3}
    skipped = gen_records is None
    if skipped and 2 in judged:
        raise ValueError("free generation was skipped, but condition 2 is judged "
                         f"under --loss-on {loss_on}")
    if decision and (no_context is None or "real" not in no_context):
        raise ValueError(f"condition 4 is judged under --loss-on {loss_on}, but its "
                         "real no-context pass is missing")
    n = len(tf_records)
    if not (len(decision_losses) == len(target_lengths) == n
            and (skipped or len(gen_records) == n)
            and (donor_labels is None or len(donor_labels) == n)):
        raise ValueError("the gate's passes cover different numbers of examples")
    for tf, gen in zip(tf_records, gen_records or []):
        if (tf["ecg_id"], tf["superclass"]) != (gen["ecg_id"], gen["superclass"]):
            raise ValueError("teacher-forced and generation passes are in a different order")

    rows = []      # (condition, what, measured, need, ok); ok None = nothing to judge

    tf = summarise_teacher_forced(tf_records)
    mean_loss = tf["mean_loss"]
    rows.append((1, "mean over examples", _loss(mean_loss), f"< {threshold:g}",
                 mean_loss is not None and tf["n_scored"] == n and mean_loss < threshold))

    if skipped:
        rows.append((2, "free generation", "not run (--skip-generation)", "-", None))
    else:
        closed = sum(1 for r in gen_records if r["stopped_at_end"])
        rows.append((2, f"closed with {END_MARKER}", _rate(closed, n),
                     f">= {GATE_END_RATE:.0%}", closed >= GATE_END_RATE * n))
        gen_len = float(np.mean([r["n_generated_tokens"] for r in gen_records]))
        tgt_len = float(np.mean(target_lengths))
        off = (gen_len - tgt_len) / tgt_len
        rows.append((2, "mean length, tokens",
                     f"{gen_len:.1f} generated vs {tgt_len:.1f} target ({off:+.1%})",
                     f"within {GATE_LENGTH_TOLERANCE:.0%}",
                     abs(off) <= GATE_LENGTH_TOLERANCE))
        matched = sum(1 for r in gen_records
                      if r["parsed_conclusion"] is not None
                      and bool(r["parsed_conclusion"]) == bool(r["label"]))
        rows.append((2, "Conclusion matches label", _rate(matched, n),
                     f">= {GATE_MATCH_RATE:.0%}", matched >= GATE_MATCH_RATE * n))

    labels = np.array([bool(r["label"]) for r in tf_records])

    def by_label(cond, losses, prefix=""):
        for label in (True, False):
            subset = losses[labels == label]
            # an empty subset cannot show the condition holds, so it does not pass
            mean = float(subset.mean()) if subset.size else None
            measured = (f"{_loss(mean)}  (max {subset.max():.4f})" if subset.size
                        else "no examples")
            rows.append((cond, f"{prefix}label={label} (n {subset.size})", measured,
                         f"< {GATE_DECISION_LOSS:g}",
                         mean is not None and mean < GATE_DECISION_LOSS))

    by_label(3, np.asarray(decision_losses, dtype=np.float64))

    thin = []      # condition 4's donor rows with too few examples to judge on
    if no_context is not None:
        keys = [(r["ecg_id"], r["superclass"]) for r in tf_records]
        for recs in no_context.values():
            if [(r["ecg_id"], r["superclass"]) for r in recs] != keys:
                raise ValueError("the no-context probe covers different examples")
        nc = {name: np.asarray([r["loss"] for r in recs], dtype=np.float64)
              for name, recs in no_context.items()}
        by_label(4, nc["real"], prefix="(a) real, ")
        shuffled = "shuffled" in nc and donor_labels is not None
        differs = (np.asarray(donor_labels, dtype=bool) != labels if shuffled
                   else np.zeros(n, dtype=bool))
        for label in (True, False):
            mask = (labels == label) & differs
            k = int(mask.sum())
            gap = float(nc["shuffled"][mask].mean() - nc["real"][mask].mean()) if k else None
            measured = (f"{gap:+.4f}  (shuffled {nc['shuffled'][mask].mean():.4f}, "
                        f"real {nc['real'][mask].mean():.4f})" if k else
                        "no examples" if shuffled else "shuffled control not run")
            what = f"(b) label={label}, donor {'No' if label else 'Yes'} (n {k})"
            rows.append((4, what, measured, f">= {GATE_SHUFFLE_GAP:g} nats",
                         gap is not None and gap >= GATE_SHUFFLE_GAP))
            if k < GATE_MIN_DONOR_ROWS:
                thin.append(f"label={label} donor {'No' if label else 'Yes'} n {k}")
        score = auprc(labels.astype(int), np.array([r["p_yes"] for r in no_context["real"]]))
        rows.append((4, "AUPRC from p_yes, real",
                     f"{_loss(score)}  (positive rate {labels.mean():.4f})", "-", None))
    if thin:
        judged.discard(4)

    line_w = 100
    print("\n" + "=" * line_w)
    print(f"OVERFIT GATE on the {n} examples just trained on")
    print("=" * line_w)
    failed = {}
    for cond, name in CONDITIONS.items():
        if cond == 4 and no_context is None:
            continue
        print(f"{cond}. {name}" + (
            "" if cond in judged else
            f"   [NOT judged: fewer than {GATE_MIN_DONOR_ROWS} examples in a donor row]"
            if cond == 4 and decision else
            f"   [printed, NOT judged under --loss-on {loss_on}]" if cond == 4 else
            f"   [not run, NOT judged under --loss-on {loss_on}]" if cond == 2 and skipped
            else f"   [printed, NOT judged under --loss-on {loss_on}]"))
        for c, what, measured, need, ok in rows:
            if c != cond:
                continue
            status = (("reported" if cond == 4 else "not run") if ok is None else
                      ("PASS" if ok else "FAIL")
                      + ("" if cond in judged else "  (not judged)"))
            print(f"     {what:<34} {measured:<41} need {need:<12} {status}")
            if ok is False and cond in judged:
                failed.setdefault(cond, []).append(what)
        if cond == 1:
            print(f"     (token-weighted mean {_loss(tf['token_weighted_mean_loss'])}, "
                  "the convention the training loop logs)")
        if cond == 3 and decision:
            print("     why not judged: teacher forcing puts the target's last reasoning "
                  "step right before \"Conclusion:\",\n     so the token is readable "
                  "from the context -- a constant prefix reached 0.018 on all fifty "
                  "without\n     the ECG. A verdict resting on it can pass a model that "
                  "never opens the waveform.")
        if cond == 4 and decision:
            print("     (a) real loss: the model can answer at all. (b) shuffled - real, "
                  "on the examples whose\n     donor ECG answers the other way: the answer "
                  "depends on which ECG it was given. Both must hold.")
    print("-" * line_w)
    # Which rule produced the verdict, on the verdict's own line. Empty under "all",
    # where the line reads as it always has.
    rule = ("" if not decision else
            f"  [--loss-on {loss_on} rule: judged on condition 4 alone; conditions 1 and 3 "
            "are printed and condition 2 was not run (--skip-generation), none judged]"
            if skipped else
            f"  [--loss-on {loss_on} rule: judged on condition 4 alone; conditions 1, 2 "
            "and 3 are printed, not judged]")
    if decision and thin:
        print(f"GATE NOT JUDGED: condition 4 needs at least {GATE_MIN_DONOR_ROWS} examples "
              "in each donor-answers-opposite row, and has " + ", ".join(thin)
              + "; too few to pass or fail on" + rule)
        return None
    if not failed:
        print("GATE PASSED: " + (
            "all three conditions met" if loss_on == "all" else
            f"condition 4 met, real loss < {GATE_DECISION_LOSS:g} for label=True and "
            f"label=False separately, and shuffled - real >= {GATE_SHUFFLE_GAP:g} nats on "
            "both donor-answers-opposite rows") + rule)
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


def print_decision_controls(decision_by_condition, labels, donor_labels=None,
                            title="GATE CONTROLS ON THE DECISION TOKEN",
                            scope="diagnostic, not in the verdict"):
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
    print(f"{title}, the same {labels.size} examples ({scope})")
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


def no_context_text(tokenizer, target):
    """``(context, decision_id, why)``: "Conclusion:" as the target spells it, and the
    token condition 3 scores after it.

    ``context`` is the target from the marker up to the decision token's first
    character -- "Conclusion:" itself, whatever the target's spacing. It is checked
    against the full target: tokenised on its own it must be exactly the tokens the
    target has right before the decision token, so the probe asks for the same token
    after the same tokens and differs from condition 3 only in what came before
    them. ``why`` names the problem when it cannot be built, and is None otherwise.
    """
    ids, offsets = tokenise_target(tokenizer, target)
    k, why = decision_index(target, offsets)
    if why:
        return None, None, why
    context = target[target.index(MARKER):offsets[k][0]]
    own = tokenizer(context, add_special_tokens=False).input_ids
    if not own or ids[k - len(own):k] != own:
        return None, None, (f"{context!r} tokenises on its own as {own}, but the target "
                            f"has {ids[max(0, k - len(own)):k]} before its decision token")
    return context, ids[k], None


def no_context_target(tokenizer, target):
    """``(cut, decision_id, why)``: what --loss-on decision-no-context trains on.

    The target cut to :func:`no_context_text`'s context plus the decision token,
    from the target's own text -- "Conclusion: Yes" or "Conclusion: No" as the
    target spells it. Tokenised on its own it must be the context's tokens and then
    ``decision_id``, so the supervised position sees exactly the tokens condition 4
    shows the model and is asked for exactly the token condition 4 scores. The
    logits there depend on nothing after them, so the training loss at that
    position IS condition 4's loss on the same weights. ``why`` names the problem
    when the cut cannot be built, and is None otherwise.
    """
    context, decision_id, why = no_context_text(tokenizer, target)
    if why:
        return None, None, why
    _, offsets = tokenise_target(tokenizer, target)
    k, _ = decision_index(target, offsets)
    start = target.index(MARKER)
    cut = target[start:offsets[k][1]]
    own = tokenizer(cut, add_special_tokens=False).input_ids
    want = tokenizer(context, add_special_tokens=False).input_ids + [decision_id]
    if own != want:
        return None, None, (f"{cut!r} tokenises on its own as {own}, not the context "
                            f"then the decision token {want}")
    return cut, decision_id, None


def no_context_pass(ctx, dataset, indices, args, name, *, shuffled, zero_latents):
    """Condition 4 under one condition, written to gate_no_context_<name>.jsonl.

    :func:`no_context_records` on the gate's context; returns its records.
    """
    records = no_context_records(ctx.model, ctx.resampler, ctx.tokenizer, ctx.dev,
                                 dataset, indices, args, shuffled=shuffled,
                                 zero_latents=zero_latents)
    with open(os.path.join(args.ckpt_dir, f"gate_no_context_{name}.jsonl"), "w",
              encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


@torch.no_grad()
def eval_no_context_loss(model, resampler, tokenizer, dataset, indices, dev, args):
    """Condition 4 (a)'s quantity: eval-mode no-context loss, mean per label.

    The --loss-on decision stop rule. The real condition of :func:`no_context_records`
    -- the very function the gate's condition 4 pass calls -- over the same indices,
    reduced the way :func:`gate_verdict` reduces it, so the loop stops on the number
    the judged condition then re-measures. Returns :func:`mean_by_label`'s dict.
    Fatal, through no_context_records, if an example's context cannot be built.
    Leaves the resampler in whichever mode it was found.
    """
    was_training = resampler.training
    resampler.eval()
    records = no_context_records(model, resampler, tokenizer, dev, dataset, indices,
                                 args, shuffled=False, zero_latents=False)
    resampler.train(was_training)
    return mean_by_label([r["loss"] for r in records], [r["label"] for r in records])


@torch.no_grad()
def no_context_records(model, resampler, tokenizer, dev, dataset, indices, args, *,
                       shuffled, zero_latents):
    """Condition 4: the decision token after the prompt and "Conclusion:" alone.

    Each example goes through training's own collate with its target REPLACED by
    :func:`no_context_text`'s context, so the prompt, the template, strip_think, the
    question stream and the -100 masking are training's, and the LLM reads the prompt
    and then "Conclusion:" (and an EOS after it, which no earlier position can see).
    The task stream is the run's, through task_stream_ids: under ``prompt`` the
    "Conclusion:" tokens are target-side and the resampler does not read them, as in
    training. The position read is the last context token, whose logits predict the
    decision token: its cross-entropy is the loss, and p_yes is the logsumexp-pooled
    Yes against No over the readout's own candidate ids, as evaluation computes it.

    Fatal if any example's context cannot be built or the collate did not lay it out
    as built. Returns the records, in ``indices`` order; writes nothing.
    """
    collate = make_collate_fn(tokenizer, max_length=args.max_length,
                              strip_think=getattr(dataset, "base", dataset).config.strip_think)
    # evaluation's readout: every single-token Yes / No spelling, pooled
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    cand = torch.tensor(yes_ids + no_ids, device=dev)
    yes_pos = list(range(len(yes_ids)))
    no_pos = list(range(len(yes_ids), len(yes_ids) + len(no_ids)))
    donors = dataset.mapping if isinstance(dataset, ShuffledEmbeddings) else None
    records, problems = [], []
    for start in range(0, len(indices), args.batch_size):
        items = [dataset[i] for i in indices[start:start + args.batch_size]]
        built = [no_context_text(tokenizer, it["target"]) for it in items]
        for it, (_, _, why) in zip(items, built):
            if why:
                problems.append(f"  {it['ecg_id']:>7} {it['superclass']:<6} {why}")
        if problems:
            continue
        batch = collate([{**it, "target": context} for it, (context, _, _) in zip(items, built)])
        task_ids, task_mask = task_stream_ids(batch, args.task_stream)
        full_embeds, full_attn, Lq = splice_prefix(
            model, resampler, batch["ecg_embed"].to(dev),
            batch["input_ids"].to(dev), batch["attention_mask"].to(dev), dev,
            zero_latents=zero_latents, task_ids=task_ids.to(dev),
            task_mask=task_mask.to(dev))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inputs_embeds=full_embeds, attention_mask=full_attn,
                           use_cache=False).logits
        for b, (it, (context, decision_id, _)) in enumerate(zip(items, built)):
            span = (batch["labels"][b] != -100).nonzero().flatten()
            n_ctx = len(tokenizer(context, add_special_tokens=False).input_ids)
            if len(span) != n_ctx + 1:            # the context, then the EOS
                problems.append(f"  {it['ecg_id']:>7} {it['superclass']:<6} the collate "
                                f"laid out {len(span)} target tokens, not {n_ctx} + EOS")
                continue
            row = logits[b, Lq + int(span[n_ctx - 1])].float()
            records.append({
                "ecg_id": int(it["ecg_id"]), "superclass": it["superclass"],
                "label": bool(it["label"]),
                "loss": float(F.cross_entropy(row[None], torch.tensor([decision_id],
                                                                      device=row.device))),
                "p_yes": probability_from_logits(row[cand], yes_pos, no_pos),
                "decision_token": tokenizer.decode([decision_id]),
                "context": context,
                "shuffled_embeddings": bool(shuffled), "zero_latents": bool(zero_latents),
                "donor_ecg_id": donors[int(it["ecg_id"])] if donors else None,
                "seed": args.seed,
            })
    if problems:
        sys.exit(f"GATE: condition 4 {CONDITIONS[4]} cannot be evaluated for "
                 f"{len(problems)} of {len(indices)} examples:\n" + "\n".join(problems))
    return records


def print_no_context_auprc(no_context_by_condition, labels):
    """AUPRC from p_yes per condition, against the positive rate (AUPRC's null).

    Over the gate's examples, mixing the five questions, as the project's own metric
    pools them. Pure on its arguments; returns ``{condition: AUPRC or None}``.
    """
    labels = np.asarray(labels, dtype=int)
    out = {name: auprc(labels, np.array([r["p_yes"] for r in recs]))
           for name, recs in no_context_by_condition.items()}
    print(f"AUPRC from p_yes over the {labels.size} examples: " + "   ".join(
        f"{name} {_loss(v)}" for name, v in out.items())
        + f"   (positive rate {labels.mean():.4f}, the null)")
    print("real near the positive rate => p_yes does not rank the Yes examples above "
          "the No ones without\nthe target's prose to read the answer from.")
    return out


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


def run_gate(model, resampler, tokenizer, dataset, indices, args):
    """Run both passes over ``indices`` of ``dataset`` and return the verdict.

    True or False, or None when condition 4 could not be judged (:func:`gate_verdict`).

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
    if args.skip_generation:
        # train.py allows this only where condition 2 is not judged
        print("# GATE PASS 2 of 2: free generation -- SKIPPED (--skip-generation); "
              f"condition 2 is not judged under --loss-on {args.loss_on}")
        print("#" * 64, flush=True)
        gen_records = None
    else:
        print("# GATE PASS 2 of 2: free generation")
        print("#" * 64, flush=True)
        # Batch 1: training right-pads, so a prompt never sat behind padding.
        # Decoding one at a time keeps every example in the layout it was overfit in.
        gen_records, _, _ = run_generation(
            ctx, dataset, indices, os.path.join(args.ckpt_dir, "gate_generations.jsonl"),
            split="train", seed=args.seed, num_workers=args.num_workers,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS, batch_size=1, print_first=3,
            log_every=10, task_stream=args.task_stream, **real)

    targets = [dataset.examples[i].target for i in indices]
    located = locate_decision_tokens(tokenizer, targets, tf_records)

    by_condition = run_controls(ctx, dataset, indices, args, tf_records, shuffled_view)
    if args.loss_on in DECISION_MODES:
        # Diagnostic: condition 3 is printed but not judged under this rule. Each
        # control goes through the same locator and the same fatal length check.
        decision = {name: [loss for loss, _ in
                           (located if name == "real" else
                            locate_decision_tokens(tokenizer, targets, recs))]
                    for name, recs in by_condition.items()}
        donors = (donor_answers(dataset, by_condition["shuffled"])
                  if "shuffled" in by_condition else None)
        labels = [r["label"] for r in tf_records]
        print_decision_controls(decision, labels, donors)

        # Condition 4, under the same three conditions and on the same examples.
        no_context = {}
        for name, opts in CONTROL_CONDITIONS:
            if opts["shuffled"] and shuffled_view is None:
                continue
            print("\n" + "#" * 64)
            print(f"# GATE CONDITION 4: decision token, no target context, {name}")
            print("#" * 64, flush=True)
            no_context[name] = no_context_pass(
                ctx, shuffled_view if opts["shuffled"] else dataset, indices, args,
                name, **opts)
        assert_same_examples(no_context, "gate no-context conditions")
        print_decision_controls(
            {name: [r["loss"] for r in recs] for name, recs in no_context.items()},
            labels, donors,
            title="CONDITION 4: THE DECISION TOKEN AFTER THE PROMPT AND "
                  "\"Conclusion:\" ALONE",
            scope="the verdict reads its real label rows and its donor rows")
        print_no_context_auprc(no_context, labels)

    decision = args.loss_on in DECISION_MODES
    return gate_verdict(tf_records, gen_records,
                        [loss for loss, _ in located], [n for _, n in located],
                        args.gate_threshold, loss_on=args.loss_on,
                        no_context=no_context if decision else None,
                        donor_labels=donors if decision else None)
