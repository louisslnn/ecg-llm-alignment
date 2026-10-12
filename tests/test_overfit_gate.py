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


def _verdict(capsys, decision=None, threshold=0.1, **kw):
    tf, gen = _records(**kw)
    decision = decision or [0.01] * N
    ok = gate_verdict(tf, gen, decision, [TARGET_LEN] * N, threshold)
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
