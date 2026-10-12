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
