"""A checkpoint may only be evaluated on the prompt format it was trained with.

``strip_think`` decides whether the prompt ends in the template's open <think>
block. Training writes its setting into every checkpoint's meta; evaluate.py,
eval_controls.py and probe_latents.py all load checkpoints through
``EvalContext.load``, which refuses a disagreeing setting before the student loads.
Pinned here:

* train.py's checkpoints carry the value, and it reads back;
* a checkpoint with no recorded value (run 1 predates the flag) is treated as False
  and reported as inferred -- never as today's default;
* a mismatch is fatal and names both values and the flag that fixes it;
* --allow-strip-think-mismatch gets through, with a warning, and the mismatch is in
  the returned record that the scripts write into their output;
* the refusal happens before the student is loaded, and no caller can skip it.

No model, no GPU: checkpoints here are a few bytes.

    pytest tests/test_strip_think_check.py
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import evaluate  # noqa: E402
from evaluate import (  # noqa: E402
    EvalContext,
    checkpoint_strip_think,
    require_strip_think_match,
)

# The two loss weights train.py requires on every command line; 0 and 0 is the LM
# loss alone, which every test here is about.
LM_ONLY = ("--lambda-linear", "0", "--lambda-consistency", "0")


def _ckpt(tmp_path, meta="absent", name="ckpt.pt"):
    payload = {"resampler": {"w": torch.zeros(2)}, "global_step": 7}
    if meta != "absent":
        payload["meta"] = meta
    path = str(tmp_path / name)
    torch.save(payload, path)
    return path


def test_training_checkpoints_record_strip_think(tmp_path, monkeypatch):
    """What train.py saves is what the check reads, for both settings."""
    import train

    module = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(module.parameters())
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])   # no GPU here
    for value in (True, False):
        monkeypatch.setattr(sys, "argv", ["train.py", "--model-dir", "x", *LM_ONLY,
                                          "--strip-think" if value else "--no-strip-think"])
        monkeypatch.setitem(train._RUN_META, "strip_think",
                            bool(train.parse_args().strip_think))
        ckpt_dir = tmp_path / str(value)
        ckpt_dir.mkdir()
        path = train.save_checkpoint(str(ckpt_dir), module, optimizer, epoch=0, step=1,
                                     best_val=1.0, is_best=True)
        assert checkpoint_strip_think(path) == (value, True)
        # a directory resolves to its best.pt, as --ckpt accepts
        assert checkpoint_strip_think(str(ckpt_dir)) == (value, True)


def test_overfit_final_reads_back_like_any_training_checkpoint(tmp_path, monkeypatch, capsys):
    """--overfit's final resampler goes through the same two readers, unchanged.

    The gate's resampler is only worth saving if what probes a training checkpoint
    can probe it: same keys bar the optimizer, strip_think in the meta, weights that
    load strictly -- plus the verdict and the arguments, for a failed gate as well.
    """
    import train
    from src.model.resampler import PerceiverResamplerConfig, Resampler

    def tiny(**overrides):          # load_resampler builds .ecg(); 3.4 GB is no unit test
        return PerceiverResamplerConfig(embed_dim=32, ffn_embed_dim=64, attention_heads=4,
                                        num_layers=1, resampled_length=4,
                                        input_embed_dim=16, **overrides)

    monkeypatch.setattr(PerceiverResamplerConfig, "ecg", staticmethod(tiny))
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])   # no GPU here
    monkeypatch.setattr(sys, "argv", ["train.py", "--model-dir", "x", *LM_ONLY, "--overfit", "50",
                                      "--no-strip-think"])
    args = train.parse_args()
    monkeypatch.setitem(train._RUN_META, "strip_think", bool(args.strip_think))
    selection = {"select": "head", "ecg_ids": [1, 2], "label_counts": {}}
    monkeypatch.setitem(train._RUN_META, "overfit_selection", selection)

    module = torch.nn.Linear(2, 2)
    reference = torch.load(
        train.save_checkpoint(str(tmp_path), module, torch.optim.AdamW(module.parameters()),
                              epoch=0, step=1, best_val=1.0, is_best=False),
        weights_only=False)

    ckpt_dir = tmp_path / "gate"
    ckpt_dir.mkdir()
    for passed in (True, False):                   # the second write replaces the first
        resampler = Resampler(tiny())
        path = train.save_overfit_final(str(ckpt_dir), resampler, epoch=3, step=40,
                                        gate_passed=passed, args=args)
        assert os.listdir(ckpt_dir) == ["overfit_final.pt"]
        assert checkpoint_strip_think(path) == (False, True)
        loaded = evaluate.load_resampler(path, torch.device("cpu"))
        assert "epoch 3 step 40" in capsys.readouterr().out
        for key, value in resampler.state_dict().items():
            assert torch.equal(loaded.state_dict()[key], value)

        payload = torch.load(path, weights_only=False)
        assert set(payload) == (set(reference) - {"optimizer"}) | {
            "gate_passed", "args", "overfit_selection"}
        assert payload["gate_passed"] is passed
        assert payload["args"] == vars(args)
        # which examples the verdict is about: beside it, and in the meta
        assert payload["overfit_selection"] == selection == payload["meta"]["overfit_selection"]


@pytest.mark.parametrize("meta", ["absent", {}, {"prefix_scale": 1.0}, None])
def test_missing_value_is_false_and_reported_as_inferred(tmp_path, capsys, meta):
    """Run 1's checkpoint has no entry: that means the think block was in the prompt."""
    path = _ckpt(tmp_path, meta)
    assert checkpoint_strip_think(path) == (False, False)

    result = require_strip_think_match(path, strip_think=False)
    assert result == {"checkpoint_strip_think": False,
                      "checkpoint_strip_think_recorded": False,
                      "strip_think_mismatch": False}
    out = capsys.readouterr().out
    assert "NOT recorded" in out and "predates the flag" in out and "treated as False" in out


def test_bare_state_dict_has_no_record_either(tmp_path):
    path = str(tmp_path / "bare.pt")
    torch.save({"w": torch.zeros(2)}, path)
    assert checkpoint_strip_think(path) == (False, False)


def test_unrecorded_checkpoint_refuses_the_current_default(tmp_path):
    """The case the default flip created: run 1's checkpoint, no flags passed."""
    from src.data.dataset import DatasetConfig

    with pytest.raises(SystemExit) as err:
        require_strip_think_match(_ckpt(tmp_path), DatasetConfig().strip_think)
    message = str(err.value)
    assert "this run has strip_think=True" in message
    assert "trained with strip_think=False" in message and "predates the flag" in message
    assert "Pass --no-strip-think" in message and "--allow-strip-think-mismatch" in message


@pytest.mark.parametrize("trained", [True, False])
def test_mismatch_is_fatal_and_names_both_values(tmp_path, trained):
    path = _ckpt(tmp_path, {"strip_think": trained})
    assert require_strip_think_match(path, trained)["strip_think_mismatch"] is False

    with pytest.raises(SystemExit) as err:
        require_strip_think_match(path, not trained)
    message = str(err.value)
    assert f"this run has strip_think={not trained}" in message
    assert f"trained with strip_think={trained}" in message
    assert "recorded in its meta" in message and path in message
    assert f"Pass {'--strip-think' if trained else '--no-strip-think'} " in message


def test_escape_hatch_warns_and_marks_the_result(tmp_path, capsys):
    path = _ckpt(tmp_path, {"strip_think": False})
    result = require_strip_think_match(path, True, allow_mismatch=True)
    assert result == {"checkpoint_strip_think": False,
                      "checkpoint_strip_think_recorded": True,
                      "strip_think_mismatch": True}
    out = capsys.readouterr().out
    assert "WARNING" in out and "MISMATCH" in out
    assert "strip_think=True" in out and "strip_think=False" in out
    assert "--allow-strip-think-mismatch" in out


def test_no_escape_is_offered_when_the_caller_has_none(tmp_path):
    """train.py's resume check: the message must not point at a flag it lacks."""
    with pytest.raises(SystemExit) as err:
        require_strip_think_match(_ckpt(tmp_path), True, escape_flag=None)
    assert "allow-strip-think-mismatch" not in str(err.value)
    assert "Pass --no-strip-think" in str(err.value)


def test_context_load_refuses_before_the_student_is_loaded(tmp_path, monkeypatch):
    def student_must_not_load(_model_dir):
        raise AssertionError("the student was loaded before the strip_think check")

    monkeypatch.setattr(evaluate, "load_student", student_must_not_load)
    path = _ckpt(tmp_path, {"strip_think": False})
    with pytest.raises(SystemExit) as err:
        EvalContext.load("unused", path, tokenizer=object(), strip_think=True)
    assert "MISMATCH" in str(err.value)


def test_context_load_cannot_be_called_without_stating_the_setting(tmp_path):
    with pytest.raises(TypeError):
        EvalContext.load("unused", _ckpt(tmp_path))


def test_every_script_that_loads_a_checkpoint_goes_through_the_check():
    """The three evaluation scripts pass their setting and expose the escape hatch."""
    scripts = os.path.join(REPO_ROOT, "scripts")
    for name in ("evaluate.py", "eval_controls.py", "probe_latents.py"):
        source = open(os.path.join(scripts, name), encoding="utf-8").read()
        loads = source.count("EvalContext.load(")
        assert loads >= 1, name
        assert source.count("strip_think=data_cfg.strip_think,") == loads, name
        assert source.count(
            "allow_strip_think_mismatch=args.allow_strip_think_mismatch") == loads, name
        assert '"--allow-strip-think-mismatch"' in source, name
        assert "**ctx.strip_think_check" in source, name
    # and nothing else in scripts/ loads a resampler checkpoint around it
    for name in sorted(os.listdir(scripts)):
        if name.endswith(".py") and name != "evaluate.py":
            source = open(os.path.join(scripts, name), encoding="utf-8").read()
            assert "load_resampler(" not in source, f"{name} bypasses EvalContext.load"
    # latent_spread.py reads a checkpoint without the student, so it has no
    # EvalContext to go through. It builds prompts all the same, so it makes the
    # check itself, with the strip_think the checkpoint's run recorded, and offers
    # no way past it.
    source = open(os.path.join(scripts, "latent_spread.py"), encoding="utf-8").read()
    assert ('require_strip_think_match(args.ckpt, run["strip_think"], escape_flag=None)'
            in source)
