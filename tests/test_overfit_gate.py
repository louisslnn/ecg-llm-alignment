"""The overfit gate's verdict (scripts/overfit_gate.py), on synthetic records.

What is pinned here is the reason the gate has three conditions instead of one:

* a teacher-forced mean under the threshold does not pass on its own;
* the decision-token loss is checked per label, so a constant-Yes model -- near zero
  on every positive, high on every negative -- fails even when positives dominate
  and its overall mean is under the threshold;
* a failed gate names the condition that failed, and a passing one needs all three.

``gate_verdict`` is pure on the records, so no model or tokenizer is involved.

    pytest tests/test_overfit_gate.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from overfit_gate import gate_verdict  # noqa: E402

N = 50
TARGET_LEN = 120


def _records(n_pos=10, loss=0.04, closed=N, gen_len=TARGET_LEN, matched=N):
    """A gate run that passes everything unless an argument says otherwise."""
    tf, gen = [], []
    for i in range(N):
        label = i < n_pos
        key = {"ecg_id": i // 5 + 1, "superclass": ["NORM", "MI", "STTC", "CD", "HYP"][i % 5]}
        tf.append({**key, "label": label, "loss": loss, "n_target_tokens": TARGET_LEN + 1})
        gen.append({**key, "label": label, "stopped_at_end": i < closed,
                    "n_generated_tokens": gen_len,
                    "parsed_conclusion": label if i < matched else (not label)})
    return tf, gen


def _verdict(capsys, decision=None, threshold=0.1, loss_on="all", **kw):
    tf, gen = _records(**kw)
    decision = decision or [0.01] * N
    ok = gate_verdict(tf, gen, decision, [TARGET_LEN] * N, threshold, loss_on=loss_on)
    return ok, capsys.readouterr().out.strip().splitlines()[-1]


def test_all_three_conditions_met_passes(capsys):
    ok, last = _verdict(capsys)
    assert ok and last.startswith("GATE PASSED")


def test_teacher_forced_loss_over_threshold_fails_condition_1(capsys):
    ok, last = _verdict(capsys, loss=0.2)
    assert not ok
    assert "condition 1" in last and "condition 2" not in last and "condition 3" not in last


def test_threshold_is_strict(capsys):
    # 0.5 is exact in binary, so the mean of fifty of them is exactly the threshold
    ok, _ = _verdict(capsys, loss=0.5, threshold=0.5)
    assert not ok


def test_low_teacher_forced_loss_with_degenerate_generation_fails_condition_2(capsys):
    # run 1: the loss gate was met and the generations never closed
    ok, last = _verdict(capsys, loss=0.0496, closed=0, gen_len=512, matched=0)
    assert not ok
    assert "condition 2" in last and "condition 1" not in last
    assert "closed with <END>" in last and "mean length" in last and "matches label" in last


def test_generation_rates_pass_at_exactly_90_percent(capsys):
    ok, _ = _verdict(capsys, closed=45, matched=45)
    assert ok
    ok, last = _verdict(capsys, closed=44)
    assert not ok and "closed with <END>" in last
    ok, last = _verdict(capsys, matched=44)
    assert not ok and "matches label" in last


def test_unparsed_generation_counts_as_a_mismatch(capsys):
    tf, gen = _records()
    for r in gen[:6]:
        r["parsed_conclusion"] = None
    assert not gate_verdict(tf, gen, [0.01] * N, [TARGET_LEN] * N, 0.1)
    assert "matches label" in capsys.readouterr().out.strip().splitlines()[-1]


def test_mean_length_tolerance_is_25_percent_either_side(capsys):
    assert _verdict(capsys, gen_len=150)[0] and _verdict(capsys, gen_len=90)[0]
    assert not _verdict(capsys, gen_len=151)[0] and not _verdict(capsys, gen_len=89)[0]


def test_constant_yes_fails_condition_3_even_when_the_overall_mean_passes(capsys):
    # 45 positives near zero, 5 negatives at 0.5: the overall mean is 0.051.
    decision = [0.001] * 45 + [0.5] * 5
    assert sum(decision) / N < 0.1
    ok, last = _verdict(capsys, decision=decision, n_pos=45)
    assert not ok
    assert "condition 3" in last and "label=False" in last and "label=True" not in last


def test_a_label_subset_with_no_examples_does_not_pass(capsys):
    ok, last = _verdict(capsys, n_pos=N)
    assert not ok and "label=False (n 0)" in last


def test_every_condition_is_printed_with_its_threshold_either_way(capsys):
    for kw in ({}, {"loss": 0.2, "closed": 0}):
        tf, gen = _records(**kw)
        gate_verdict(tf, gen, [0.01] * N, [TARGET_LEN] * N, 0.1)
        out = capsys.readouterr().out
        assert out.count("need < 0.1") == 3          # condition 1, and 3 per label
        assert out.count("need >= 90%") == 2
        assert out.count("need within 25%") == 1
        assert sum(line.endswith(("PASS", "FAIL")) for line in out.splitlines()) == 6


# --------------------------------------------------------------------------- #
# The stop rule, and what train.py does around it.                            #
# --------------------------------------------------------------------------- #


def test_the_stop_rule_measures_exactly_what_condition_1_measures():
    """eval_mean_loss over gate_batches IS the gate's teacher-forced mean.

    The loop stops on the first and condition 1 re-measures with the second, so if
    they were different quantities the loop could stop and the gate fail at once.
    Run on a stand-in student and a debug-width resampler deliberately left in TRAIN
    mode: the stop rule has to measure in eval mode and hand the mode back.
    """
    import types

    import torch
    from evaluate import EvalContext, run_teacher_forced, summarise_teacher_forced
    from overfit_gate import eval_mean_loss, gate_batches
    from src.data.dataset import DatasetConfig, build_datasets, load_student_tokenizer
    from src.model.resampler import PerceiverResamplerConfig, Resampler
    from tests.test_task_stream import _StubStudent

    config = DatasetConfig(debug=True)
    dataset = build_datasets(config)["train"]
    tok = load_student_tokenizer(config)
    indices = list(range(10))                      # 3 batches, the last one short
    args = types.SimpleNamespace(batch_size=4, max_length=2048)

    cfg = PerceiverResamplerConfig.debug()
    torch.manual_seed(0)
    model = _StubStudent(len(tok), cfg.embed_dim).eval()
    resampler = Resampler(cfg).train()
    dev = torch.device("cpu")

    batches = gate_batches(tok, dataset, indices, args)
    assert sum(len(b["ecg_id"]) for b in batches) == len(indices)
    stop_rule = eval_mean_loss(model, resampler, batches, dev)
    assert resampler.training, "the stop rule must hand the resampler back in train mode"

    resampler.eval()                               # as run_gate does before its pass
    ctx = EvalContext(model=model, resampler=resampler, tokenizer=tok, dev=dev,
                      eos_id=tok.eos_token_id, pad_id=tok.pad_token_id, yes_pos=[],
                      no_pos=[], capture=None, processors=None)
    records, _ = run_teacher_forced(
        ctx, dataset, indices, split="train", seed=0, shuffled=False,
        zero_latents=False, batch_size=args.batch_size, num_workers=0,
        max_length=args.max_length)
    assert stop_rule == summarise_teacher_forced(records)["mean_loss"]


def _train_args(monkeypatch, *argv):
    import train

    monkeypatch.setattr(sys, "argv", ["train.py", "--model-dir", "x", *argv])
    return train.parse_args()


def test_stop_target_must_sit_below_the_gate_threshold(monkeypatch):
    import pytest

    args = _train_args(monkeypatch, "--overfit", "50")
    assert args.overfit_target_loss < args.gate_threshold      # the defaults agree
    for target in ("0.1", "0.2"):
        with pytest.raises(SystemExit):
            _train_args(monkeypatch, "--overfit", "50", "--overfit-target-loss", target)
    # outside --overfit the two flags are unused and must not block a training run
    _train_args(monkeypatch, "--overfit-target-loss", "0.2")


def test_strip_think_flag_defaults_on_and_can_be_turned_off(monkeypatch):
    assert _train_args(monkeypatch).strip_think is True
    assert _train_args(monkeypatch, "--strip-think").strip_think is True
    assert _train_args(monkeypatch, "--no-strip-think").strip_think is False


def test_git_state_reports_head_and_tracked_changes():
    import subprocess

    import train

    head, dirty, lines = train.git_state()
    expected = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    assert head == expected and len(head) == 40
    assert dirty == any(not line.startswith("??") for line in lines)


# --------------------------------------------------------------------------- #
# The controls: diagnostic passes beside the gate, never inside the verdict.  #
# --------------------------------------------------------------------------- #


def _debug_train():
    from src.data.dataset import DatasetConfig, build_datasets

    return build_datasets(DatasetConfig(debug=True))["train"]


def test_the_gate_shuffle_is_a_derangement_over_its_own_records(capsys):
    """Every donor comes from inside the overfit set and no record keeps its own.

    At ten records a plain permutation has a fixed point more often than not, and a
    permutation over the whole split would serve ECGs the overfit never trained on.
    """
    from evaluate import ShuffledEmbeddings
    from overfit_gate import shuffle_within

    dataset = _debug_train()
    indices = list(range(50))
    own = sorted({dataset.examples[i].ecg_id for i in indices})
    assert len(own) == 10
    for seed in range(25):
        mapping = shuffle_within(dataset, indices, seed).mapping
        assert sorted(mapping) == own and sorted(mapping.values()) == own
        assert all(donor != ecg_id for ecg_id, donor in mapping.items())
    capsys.readouterr()

    # two records is the smallest set with another ECG to serve: a swap
    pair = shuffle_within(dataset, list(range(10)), 0)
    a, b = sorted(pair.mapping)
    assert pair.mapping == {a: b, b: a}
    item = pair[0]
    assert item["donor_ecg_id"] == pair.mapping[item["ecg_id"]]
    assert not (item["ecg_embed"] == dataset[0]["ecg_embed"]).all()

    # one record: nothing to shuffle with, so the control is skipped, not faked
    assert shuffle_within(dataset, list(range(5)), 0) is None
    assert "skipped" in capsys.readouterr().out

    # the unrestricted wrapper is untouched: it still spans the whole split
    assert len(ShuffledEmbeddings(dataset, 0).mapping) == len(
        {e.ecg_id for e in dataset.examples})


def test_a_shuffled_record_that_kept_its_own_embedding_is_fatal():
    import pytest
    from overfit_gate import assert_no_own_embedding

    ok = [{"ecg_id": 1, "superclass": "NORM", "donor_ecg_id": 2},
          {"ecg_id": 2, "superclass": "NORM", "donor_ecg_id": 1}]
    assert_no_own_embedding(ok)
    for bad in (1, None):      # its own ECG, or a pass that never shuffled at all
        with pytest.raises(SystemExit):
            assert_no_own_embedding([{**ok[0], "donor_ecg_id": bad}, ok[1]])


def test_the_controls_table_reports_both_gaps_in_nats_and_percent(capsys):
    from overfit_gate import print_controls

    def recs(loss):
        return [{"ecg_id": 1, "superclass": "NORM", "loss": loss, "n_target_tokens": 9}]

    rows = print_controls({"real": recs(0.05), "shuffled": recs(0.25), "zeroed": recs(2.05)})
    by_name = {r["condition"]: r for r in rows}
    assert by_name["real"]["gap_nats"] is None and by_name["real"]["gap_pct"] is None
    assert abs(by_name["shuffled"]["gap_nats"] - 0.20) < 1e-12
    assert abs(by_name["shuffled"]["gap_pct"] - 400.0) < 1e-9
    assert abs(by_name["zeroed"]["gap_nats"] - 2.00) < 1e-12
    assert abs(by_name["zeroed"]["gap_pct"] - 4000.0) < 1e-9
    out = capsys.readouterr().out
    assert "+0.2000" in out and "+400.0%" in out and "+2.0000" in out and "+4000.0%" in out

    # a control that did not run is named as such, and no gap is invented for it
    rows = print_controls({"real": recs(0.05), "zeroed": recs(2.05)})
    assert [r["ran"] for r in rows] == [True, False, True]
    assert "not run" in capsys.readouterr().out


def test_the_controls_cover_the_same_examples_and_leave_the_real_pass_alone(tmp_path):
    """Three passes, one context: same ordered examples, donors recorded, and the
    real records byte-for-byte what a gate without controls would have measured."""
    import json
    import types

    import torch
    from eval_controls import example_sequence
    from evaluate import EvalContext, run_teacher_forced
    from overfit_gate import run_controls, shuffle_within, teacher_forced_pass
    from src.data.dataset import DatasetConfig, load_student_tokenizer
    from src.model.resampler import PerceiverResamplerConfig, Resampler
    from tests.test_task_stream import _StubStudent

    dataset = _debug_train()
    tok = load_student_tokenizer(DatasetConfig(debug=True))
    indices = list(range(15))                      # 3 records, the last batch short
    args = types.SimpleNamespace(batch_size=4, max_length=2048, seed=0, num_workers=0,
                                 task_stream="prompt", ckpt_dir=str(tmp_path))

    cfg = PerceiverResamplerConfig.debug()
    torch.manual_seed(0)
    model = _StubStudent(len(tok), cfg.embed_dim).eval()
    resampler = Resampler(cfg).eval()
    ctx = EvalContext(model=model, resampler=resampler, tokenizer=tok,
                      dev=torch.device("cpu"), eos_id=tok.eos_token_id,
                      pad_id=tok.pad_token_id, yes_pos=[], no_pos=[], capture=None,
                      processors=None)

    # what the gate measured before the controls existed
    before, _ = run_teacher_forced(
        ctx, dataset, indices, split="train", seed=0, shuffled=False,
        zero_latents=False, batch_size=4, num_workers=0, max_length=2048)

    view = shuffle_within(dataset, indices, args.seed)
    real = teacher_forced_pass(ctx, dataset, indices, args, "real",
                               shuffled=False, zero_latents=False)
    by_condition = run_controls(ctx, dataset, indices, args, real, view)

    assert by_condition["real"] is real and real == before
    assert list(by_condition) == ["real", "shuffled", "zeroed"]
    keys = [(dataset.examples[i].ecg_id, dataset.examples[i].superclass) for i in indices]
    for name, recs in by_condition.items():
        assert example_sequence(recs) == keys
        on_disk = [json.loads(line) for line in
                   open(tmp_path / f"gate_teacher_forced_{name}.jsonl")]
        assert on_disk == recs

    assert all(r["donor_ecg_id"] is None for r in real + by_condition["zeroed"])
    for r in by_condition["shuffled"]:
        assert r["donor_ecg_id"] == view.mapping[r["ecg_id"]] != r["ecg_id"]
        assert r["shuffled_embeddings"] and not r["zero_latents"]
    assert all(r["zero_latents"] for r in by_condition["zeroed"])
    # the controls really are different inputs, not the real pass under another name
    for name in ("shuffled", "zeroed"):
        assert [r["loss"] for r in by_condition[name]] != [r["loss"] for r in real]


# --------------------------------------------------------------------------- #
# --loss-on decision: one supervised position, and a gate judged on it alone. #
# --------------------------------------------------------------------------- #

# what a run with no supervision on the prose looks like to conditions 1 and 2
PROSE_UNTRAINED = {"loss": 2.9, "closed": 0, "gen_len": 512, "matched": 20}


def test_under_loss_on_all_the_closing_line_is_what_it_always_was(capsys):
    assert _verdict(capsys)[1] == "GATE PASSED: all three conditions met"
    ok, last = _verdict(capsys, **PROSE_UNTRAINED)
    assert not ok and "--loss-on" not in last
    assert "condition 1" in last and "condition 2" in last


def test_decision_mode_is_judged_on_condition_3_alone_and_says_so(capsys):
    ok, last = _verdict(capsys, loss_on="decision", **PROSE_UNTRAINED)
    assert ok and last.startswith("GATE PASSED: condition 3 met")
    assert "--loss-on decision rule: judged on condition 3 alone" in last

    # condition 3 still fails per label, and nothing else is named as a failure
    decision = [0.001] * 45 + [0.5] * 5
    ok, last = _verdict(capsys, loss_on="decision", decision=decision, n_pos=45,
                        **PROSE_UNTRAINED)
    assert not ok and last.startswith("GATE FAILED: condition 3")
    assert "label=False" in last and "label=True" not in last
    assert "closed with <END>" not in last and "mean over examples" not in last
    assert "--loss-on decision rule: judged on condition 3 alone" in last

    # passing conditions 1 and 2 does not rescue a failed condition 3
    ok, _ = _verdict(capsys, loss_on="decision", decision=decision, n_pos=45)
    assert not ok


def test_decision_mode_still_prints_all_three_conditions_with_thresholds(capsys):
    tf, gen = _records(**PROSE_UNTRAINED)
    gate_verdict(tf, gen, [0.01] * N, [TARGET_LEN] * N, 0.1, loss_on="decision")
    lines = capsys.readouterr().out.splitlines()
    out = "\n".join(lines)
    assert out.count("need < 0.1") == 3 and out.count("need >= 90%") == 2
    assert out.count("need within 25%") == 1
    # the four rows of conditions 1 and 2 keep their result and are marked; the two
    # of condition 3 are the only ones that read as a verdict
    assert sum(line.endswith("FAIL  (not judged)") for line in lines) == 4
    assert sum(line.endswith("PASS") for line in lines) == 2
    assert sum("NOT judged under --loss-on decision" in line for line in lines) == 2


def test_loss_on_flag_defaults_to_all_and_moves_the_stop_target_check(monkeypatch):
    import pytest

    assert _train_args(monkeypatch).loss_on == "all"
    assert _train_args(monkeypatch, "--loss-on", "decision").loss_on == "decision"
    with pytest.raises(SystemExit):
        _train_args(monkeypatch, "--loss-on", "prose")
    # under decision the loop stops on condition 3's quantity, so the target is
    # held below THAT threshold (0.1) wherever --gate-threshold has been moved to
    for gate_threshold in ("0.1", "5"):
        with pytest.raises(SystemExit):
            _train_args(monkeypatch, "--overfit", "50", "--loss-on", "decision",
                        "--gate-threshold", gate_threshold,
                        "--overfit-target-loss", "0.2")
    _train_args(monkeypatch, "--overfit", "50", "--loss-on", "decision",
                "--gate-threshold", "0.01")
    # and the hyperparameters are the ones --loss-on all runs with
    a = vars(_train_args(monkeypatch, "--overfit", "50"))
    b = vars(_train_args(monkeypatch, "--overfit", "50", "--loss-on", "decision"))
    assert {k for k in a if a[k] != b[k]} == {"loss_on"}


def _decision_setup(n=10):
    import train
    from src.data.dataset import DatasetConfig, load_student_tokenizer, make_collate_fn

    config = DatasetConfig(debug=True)
    dataset = _debug_train()
    tok = load_student_tokenizer(config)
    base = make_collate_fn(tok, max_length=2048, strip_think=config.strip_think)
    return dataset, tok, base, train.make_decision_collate(tok, base), list(range(n))


def test_decision_collate_marks_one_position_and_changes_nothing_else():
    import torch
    from src.data.dataset import prompt_only_task_mask

    dataset, tok, base, collate, indices = _decision_setup(50)
    items = [dataset[i] for i in indices]
    plain, batch = base(items), collate(items)

    assert set(batch) - set(plain) == {"decision_mask"}
    for key, value in plain.items():
        same = (value == batch[key])
        assert bool(same.all()) if torch.is_tensor(value) else same, key
    # so the resampler's task stream is still exactly the prompt
    assert torch.equal(prompt_only_task_mask(batch["attention_mask"], batch["labels"]),
                       prompt_only_task_mask(plain["attention_mask"], plain["labels"]))

    mask = batch["decision_mask"]
    assert mask.dtype == torch.bool and mask.shape == batch["labels"].shape
    assert mask.sum(dim=1).tolist() == [1] * len(items)
    assert bool((batch["labels"][mask] != -100).all())       # inside the target
    seen = set()
    for i, item in enumerate(items):
        pos = int(mask[i].nonzero().item())
        word = tok.decode([int(batch["input_ids"][i, pos])]).strip()
        assert word == ("Yes" if item["label"] else "No")
        before = tok.decode(batch["input_ids"][i, :pos]).rstrip()
        assert before.endswith("Conclusion:")
        seen.add(word)
    assert seen == {"Yes", "No"}


def test_an_example_whose_decision_token_cannot_be_placed_is_fatal_and_named():
    import pytest
    import train
    from src.data.dataset import make_collate_fn

    dataset, tok, _, collate, _ = _decision_setup()
    items = [dataset[i] for i in range(4)]

    # a target cut short by --max-length: --loss-on all trains on what is left
    short = make_collate_fn(tok, max_length=120,
                            strip_think=dataset.config.strip_think)
    assert int((short(items)["labels"] != -100).sum()) > 0
    with pytest.raises(train.DecisionTokenError) as err:
        train.make_decision_collate(tok, short)(items)
    for item in items:
        assert f"{item['ecg_id']:>7} {item['superclass']:<6} rebuilt target has" \
            in str(err.value)

    # one bad example among good ones stops the batch; it is not dropped from it
    for target, why in ((items[1]["target"] + "\nConclusion: Yes", "2 occurrences"),
                        (items[1]["target"].replace("Conclusion:", "Verdict:"),
                         "0 occurrences")):
        with pytest.raises(train.DecisionTokenError) as err:
            collate([items[0], {**items[1], "target": target}, items[2]])
        assert "1 of the 3 examples" in str(err.value) and why in str(err.value)
        assert f"{items[1]['ecg_id']:>7} {items[1]['superclass']:<6}" in str(err.value)


def _loss_student(tok, embed_dim):
    """_StubStudent with the two things splice_forward asks of the real student:
    ``config.use_cache`` and a ``labels`` argument reduced as a causal LM reduces it."""
    import types

    import torch
    from tests.test_task_stream import _StubStudent

    class _LossStudent(_StubStudent):
        config = types.SimpleNamespace(use_cache=False)

        def forward(self, inputs_embeds=None, attention_mask=None, labels=None,
                    use_cache=False):
            out = super().forward(inputs_embeds, attention_mask)
            if labels is not None:
                out.loss = torch.nn.functional.cross_entropy(
                    out.logits[:, :-1].float().flatten(0, 1), labels[:, 1:].flatten(),
                    ignore_index=-100)
            return out

    torch.manual_seed(0)
    return _LossStudent(len(tok), embed_dim).eval()


def test_the_decision_loss_is_the_mean_over_examples_of_that_one_position():
    """Training's number under each --loss-on, against the per-position losses.

    ``decision`` must be the mean over the batch's examples of the loss at the
    marked position and nothing else; ``all`` must be what it was, with or without
    the mask in the batch. The latents are checked through the loss: the task
    stream is read off the full labels in both modes, so the per-position losses
    the two are compared against come from one and the same forward.
    """
    import pytest
    import torch
    from evaluate import teacher_forced_batch
    from probe_memory import splice_forward
    from src.model.resampler import PerceiverResamplerConfig, Resampler

    dataset, tok, base, collate, _ = _decision_setup()
    cfg = PerceiverResamplerConfig.debug()
    model = _loss_student(tok, cfg.embed_dim)
    resampler = Resampler(cfg).eval()
    dev = torch.device("cpu")
    items = [dataset[i] for i in range(4)]
    plain, batch = base(items), collate(items)

    scored = teacher_forced_batch(model, resampler, batch, dev)
    at_decision, everywhere = [], []
    for i, (_, _, position_losses) in enumerate(scored):
        k = int(batch["decision_mask"][i][batch["labels"][i] != -100].nonzero().item())
        at_decision.append(position_losses[k])
        everywhere += position_losses

    with torch.no_grad():
        decision = float(splice_forward(model, resampler, batch, dev, loss_on="decision"))
        whole = float(splice_forward(model, resampler, batch, dev))
        assert whole == float(splice_forward(model, resampler, plain, dev))
    # position_losses are rounded to 4 dp where they are written
    assert decision == pytest.approx(sum(at_decision) / len(at_decision), abs=1e-4)
    assert whole == pytest.approx(sum(everywhere) / len(everywhere), abs=1e-4)
    assert abs(decision - whole) > 1e-3          # and they are different numbers

    # the gradient reaches the resampler through that one position
    resampler.train()
    splice_forward(model, resampler, batch, dev, loss_on="decision").backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in resampler.parameters())

    # a row with no marked position would fall out of the mean without a trace
    with pytest.raises(ValueError):
        holed = {**batch, "decision_mask": batch["decision_mask"].clone()}
        holed["decision_mask"][0] = False
        splice_forward(model, resampler, holed, dev, loss_on="decision")
    with pytest.raises(KeyError):          # the plain collate carries no mask at all
        splice_forward(model, resampler, plain, dev, loss_on="decision")


def test_the_decision_stop_rule_measures_exactly_what_condition_3_measures():
    """eval_decision_loss over the masked gate batches IS condition 3's per-label mean.

    The stop rule reads the position off the collate's mask; condition 3 finds it
    again from the target text over run_teacher_forced's records. If the two were
    different positions the loop could stop and the gate fail at once.
    """
    import types

    import numpy as np
    import torch
    from evaluate import EvalContext, run_teacher_forced
    from overfit_gate import eval_decision_loss, gate_batches, locate_decision_tokens
    from src.model.resampler import PerceiverResamplerConfig, Resampler

    dataset, tok, _, collate, indices = _decision_setup(15)
    args = types.SimpleNamespace(batch_size=4, max_length=2048)
    cfg = PerceiverResamplerConfig.debug()
    model = _loss_student(tok, cfg.embed_dim)
    resampler = Resampler(cfg).train()
    dev = torch.device("cpu")

    batches = gate_batches(tok, dataset, indices, args, collate=collate)
    assert all("decision_mask" in b for b in batches)
    stop_rule = eval_decision_loss(model, resampler, batches, dev)
    assert resampler.training, "the stop rule must hand the resampler back in train mode"

    resampler.eval()
    ctx = EvalContext(model=model, resampler=resampler, tokenizer=tok, dev=dev,
                      eos_id=tok.eos_token_id, pad_id=tok.pad_token_id, yes_pos=[],
                      no_pos=[], capture=None, processors=None)
    records, _ = run_teacher_forced(
        ctx, dataset, indices, split="train", seed=0, shuffled=False,
        zero_latents=False, batch_size=args.batch_size, num_workers=0,
        max_length=args.max_length)
    located = locate_decision_tokens(
        tok, [dataset.examples[i].target for i in indices], records)
    losses = np.array([loss for loss, _ in located])
    labels = np.array([bool(r["label"]) for r in records])
    assert labels.any() and not labels.all()
    assert stop_rule["all"] == float(losses.mean())
    assert stop_rule[True] == float(losses[labels].mean())
    assert stop_rule[False] == float(losses[~labels].mean())


def test_the_decision_controls_table_splits_the_triple_by_label_and_by_donor(capsys):
    from overfit_gate import print_decision_controls

    #           label   donor's answer    real  shuffled  zeroed
    examples = [(True,  False,            0.02, 3.00,     0.70),
                (True,  True,             0.04, 0.04,     0.70),
                (False, True,             0.01, 4.00,     0.60),
                (False, False,            0.03, 0.03,     0.60),
                (False, False,            0.05, 0.05,     0.60)]
    labels = [e[0] for e in examples]
    donors = [e[1] for e in examples]
    dec = {name: [e[2 + j] for e in examples]
           for j, name in enumerate(("real", "shuffled", "zeroed"))}

    rows = {r["subset"]: r for r in print_decision_controls(dec, labels, donors)}
    assert list(rows) == ["all", "label=True", "donor answers No", "label=False",
                          "donor answers Yes"]
    assert [rows[k]["n"] for k in rows] == [5, 2, 1, 3, 1]
    assert abs(rows["label=True"]["mean_loss"]["real"] - 0.03) < 1e-12
    assert abs(rows["label=True"]["gap_nats"]["shuffled"] - 1.49) < 1e-12
    assert abs(rows["label=False"]["gap_nats"]["shuffled"] - (4.08 - 0.09) / 3) < 1e-12
    assert abs(rows["label=False"]["gap_nats"]["zeroed"] - 0.57) < 1e-12
    # the whole of a label's shuffled gap sits in the examples whose donor disagrees
    assert abs(rows["donor answers No"]["gap_nats"]["shuffled"] - 2.98) < 1e-12
    assert abs(rows["donor answers Yes"]["gap_nats"]["shuffled"] - 3.99) < 1e-12
    out = capsys.readouterr().out
    assert "ON THE DECISION TOKEN" in out and "not in the verdict" in out
    assert "+1.4900" in out and "+2.9800" in out

    # no shuffle (one record): the column says so, and no donor rows are invented
    rows = print_decision_controls({"real": dec["real"], "zeroed": dec["zeroed"]}, labels)
    assert [r["subset"] for r in rows] == ["all", "label=True", "label=False"]
    assert all("shuffled" not in r["gap_nats"] for r in rows)
    assert "not run" in capsys.readouterr().out

    # a label with no examples is n/a, not a mean of nothing
    rows = print_decision_controls({k: v[:2] for k, v in dec.items()}, labels[:2],
                                   donors[:2])
    assert rows[3]["n"] == 0 and rows[3]["mean_loss"]["real"] is None
    capsys.readouterr()


def test_donor_answers_are_the_donor_records_own_labels(capsys):
    from overfit_gate import donor_answers, print_decision_controls, shuffle_within

    dataset = _debug_train()
    own = {(e.ecg_id, e.superclass): e.label for e in dataset.examples}

    def served(n):
        indices = list(range(n))
        view = shuffle_within(dataset, indices, 0)
        shuffled = [{"ecg_id": dataset.examples[i].ecg_id,
                     "superclass": dataset.examples[i].superclass,
                     "donor_ecg_id": view.mapping[dataset.examples[i].ecg_id]}
                    for i in indices]
        answers = donor_answers(dataset, shuffled)
        assert answers == [own[(r["donor_ecg_id"], r["superclass"])] for r in shuffled]
        return answers, [dataset.examples[i].label for i in indices]

    answers, labels = served(200)             # 40 records, 6 of them not normal
    assert any(a != b for a, b in zip(answers, labels))      # some donors disagree
    assert any(a == b for a, b in zip(answers, labels))      # and most do not
    capsys.readouterr()

    # The first 50 train examples are ten records that are ALL normal: NORM is Yes
    # and the other four are No for every one of them. No donor can disagree, and
    # the table has to say that its shuffled column tests nothing on such a set.
    answers, labels = served(50)
    assert answers == labels
    capsys.readouterr()
    dec = {name: [0.01] * 50 for name in ("real", "shuffled", "zeroed")}
    rows = {r["subset"]: r for r in print_decision_controls(dec, labels, answers)}
    assert rows["donor answers No"]["n"] == rows["donor answers Yes"]["n"] == 0
    assert "NO example was served an ECG with a different known answer" \
        in capsys.readouterr().out
    print_decision_controls(dec, labels, [not a for a in answers])
    assert "NO example was served" not in capsys.readouterr().out
