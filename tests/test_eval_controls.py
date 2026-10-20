"""The evaluation path: which examples a limited pass draws, and fast mode.

* a --limit is a seeded draw, never the head of the split, and --stratify
  superclass-label balances Yes and No inside each superclass;
* every condition is handed the same draw, so assert_same_examples still holds;
* fast mode's table has the METRICS rows and columns, F1 at 0.5 and at each row's
  own best threshold, and its verdict is compared with the generated one;
* the shell scripts find a run's checkpoint dir under train.sh's own tag, and a
  missing checkpoint fails the preflight before the module load.

    pytest tests/test_eval_controls.py
"""

import os
import subprocess
import sys
import types

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from evaluate import (  # noqa: E402
    METRIC_GROUPS,
    _prf1,
    auprc,
    best_f1_threshold,
    compute_fast_metrics,
    fast_metrics_for,
    print_sample,
    sample_indices,
)
from src.data.dataset import Example  # noqa: E402
from src.teacher.prompt import SUPERCLASSES  # noqa: E402


def _split(pos_rates, n_records=200, seed=0):
    """A synthetic split in the dataset's own order -- record by record, five
    questions each -- with ``pos_rates[sc]`` of the records answering Yes."""
    rng = np.random.default_rng(seed)
    examples = [Example(ecg_id=r, superclass=sc, strat_fold=10, target="",
                        label=bool(rng.random() < pos_rates[sc]))
                for r in range(n_records) for sc in SUPERCLASSES]
    return types.SimpleNamespace(examples=examples)


SKEWED = {"NORM": 0.42, "MI": 0.25, "STTC": 0.24, "CD": 0.22, "HYP": 0.12}


# --------------------------------------------------------------------------- #
# The draw                                                                    #
# --------------------------------------------------------------------------- #

def test_superclass_label_draws_half_yes_half_no_in_every_superclass():
    split = _split(SKEWED)
    idx, info = sample_indices(split, 100, seed=42)
    assert info["stratify"] == "superclass-label" and info["n_drawn"] == 100
    assert idx == sorted(idx) and len(set(idx)) == 100
    for sc in SUPERCLASSES:
        labels = [split.examples[i].label for i in idx
                  if split.examples[i].superclass == sc]
        assert len(labels) == 20 and sum(labels) == 10, sc
    assert not info["shortfalls"]


def test_the_draw_is_not_the_head_and_is_fixed_by_the_sample_seed():
    split = _split(SKEWED)
    idx, _ = sample_indices(split, 50, seed=42)
    assert idx != list(range(50))
    assert sample_indices(split, 50, seed=42)[0] == idx
    assert sample_indices(split, 50, seed=7)[0] != idx
    # a smaller draw with the same seed sits inside a larger one
    assert set(sample_indices(split, 50, seed=42)[0]) <= set(
        sample_indices(split, 200, seed=42)[0])


def test_a_short_cell_gives_all_it_has_and_its_sibling_label_fills_the_rest():
    rates = {**SKEWED, "HYP": 0.0}
    split = _split(rates, n_records=60)
    hyp_yes = [i for i, e in enumerate(split.examples) if e.superclass == "HYP" and e.label]
    assert not hyp_yes
    idx, info = sample_indices(split, 100, seed=42)
    hyp = [split.examples[i] for i in idx if split.examples[i].superclass == "HYP"]
    assert len(hyp) == 20 and not any(e.label for e in hyp)
    assert any(s.startswith("HYP label=True: 0 of 10") for s in info["shortfalls"])


def test_the_other_two_modes_and_the_whole_split():
    split = _split(SKEWED)
    idx, _ = sample_indices(split, 100, seed=1, stratify="superclass")
    for sc in SUPERCLASSES:
        assert sum(split.examples[i].superclass == sc for i in idx) == 20
    idx, _ = sample_indices(split, 100, seed=1, stratify="none")
    assert len(idx) == 100 and idx == sorted(idx) and idx != list(range(100))
    for n in (None, len(split.examples), 10 ** 6):
        idx, info = sample_indices(split, n, seed=1)
        assert idx == list(range(len(split.examples)))
        assert info["stratify"] == "whole split"
    with pytest.raises(ValueError):
        sample_indices(split, 10, seed=1, stratify="label")


def test_the_draw_is_printed_with_the_splits_own_rate_beside_it(capsys):
    split = _split(SKEWED)
    idx, info = sample_indices(split, 100, seed=42)
    out = print_sample(split, idx, info, "generation draw")
    text = capsys.readouterr().out
    assert "--stratify superclass-label, --sample-seed 42" in text
    for sc in SUPERCLASSES:
        rates = out["rates"][sc]
        assert rates["positive_rate"] == 0.5 and rates["n"] == 20
        want = np.mean([e.label for e in split.examples if e.superclass == sc])
        assert rates["split_positive_rate"] == pytest.approx(want)
        assert f"{want:.3f}" in text
    assert out["sample_seed"] == 42 and out["stratify"] == "superclass-label"


def test_every_condition_gets_the_same_draw_and_the_controls_check_passes(capsys):
    """The fast pass under real and shuffled, on one stratified draw of the debug
    split: same examples, same order, so assert_same_examples holds."""
    from eval_controls import assert_same_examples
    from evaluate import ShuffledEmbeddings, no_context_records
    from tests.test_overfit_gate import _debug_train, _stub_ctx

    dataset, ctx = _debug_train(), _stub_ctx()
    idx, _ = sample_indices(dataset, 20, seed=42)
    kw = dict(batch_size=8, max_length=2048, task_stream="prompt", seed=0, strict=False)
    by = {"real": no_context_records(ctx.model, ctx.resampler, ctx.tokenizer, ctx.dev,
                                     dataset, idx, shuffled=False, zero_latents=False,
                                     **kw),
          "shuffled": no_context_records(ctx.model, ctx.resampler, ctx.tokenizer,
                                         ctx.dev, ShuffledEmbeddings(dataset, 0), idx,
                                         shuffled=True, zero_latents=False, **kw)}
    keys = assert_same_examples(by, "fast-mode conditions")
    assert keys == [(dataset.examples[i].ecg_id, dataset.examples[i].superclass)
                    for i in idx]
    assert [r["p_yes"] for r in by["real"]] != [r["p_yes"] for r in by["shuffled"]]


# --------------------------------------------------------------------------- #
# Fast mode                                                                   #
# --------------------------------------------------------------------------- #

def test_the_non_strict_pass_keeps_p_yes_where_the_target_has_no_answer(capsys):
    from evaluate import no_context_records
    from tests.test_overfit_gate import _debug_train, _stub_ctx

    dataset, ctx = _debug_train(), _stub_ctx()

    class Broken:                       # one target with no "Conclusion:" at all
        def __init__(self, base):
            self.base, self.examples = base, base.examples

        def __getitem__(self, i):
            item = self.base[i]
            if i == 1:
                item["target"] = item["target"].replace("Conclusion:", "Verdict:")
            return item

    kw = dict(batch_size=4, max_length=2048, task_stream="prompt", seed=0,
              shuffled=False, zero_latents=False)
    # base_of() reads the config off a ShuffledEmbeddings' base, so the wrapper
    # carries the dataset's own
    view = Broken(dataset)
    view.config = dataset.config
    recs = no_context_records(ctx.model, ctx.resampler, ctx.tokenizer, ctx.dev, view,
                              list(range(4)), strict=False, **kw)
    assert recs[1]["loss"] is None and "0 occurrences" in recs[1]["target_problem"]
    assert 0.0 <= recs[1]["p_yes"] <= 1.0
    assert all(r["loss"] is not None and r["target_problem"] is None
               for i, r in enumerate(recs) if i != 1)
    assert all(r["context"] == "Conclusion:" for r in recs)
    # the input never depended on the target: p_yes is what the intact target gets
    intact = no_context_records(ctx.model, ctx.resampler, ctx.tokenizer, ctx.dev,
                                dataset, list(range(4)), strict=False, **kw)
    assert [r["p_yes"] for r in recs] == [r["p_yes"] for r in intact]
    with pytest.raises(SystemExit, match="0 occurrences"):
        no_context_records(ctx.model, ctx.resampler, ctx.tokenizer, ctx.dev, view,
                           list(range(4)), strict=True, **kw)


def test_best_f1_threshold_is_the_brute_force_best():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.random(60) < 0.3
        p = np.round(rng.random(60), 1)          # ties on purpose
        if not y.any():
            continue
        threshold, best = best_f1_threshold(y, p)
        brute = max(_prf1(y, p >= t)["f1"] for t in np.unique(p))
        assert best["f1"] == pytest.approx(brute)
        assert best == _prf1(y, p >= threshold)
    assert best_f1_threshold(np.zeros(5, bool), np.ones(5)) == (None, None)


def _fast(n=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        label = bool(rng.random() < 0.4)
        out.append({"ecg_id": i // 5, "superclass": SUPERCLASSES[i % 5], "label": label,
                    "p_yes": float(np.clip(rng.normal(0.6 if label else 0.4, 0.2), 0, 1)),
                    "loss": 0.5})
    return out


def test_fast_metrics_are_the_metrics_columns_from_p_yes():
    recs = _fast()
    m = fast_metrics_for(recs)
    y = np.array([r["label"] for r in recs])
    p = np.array([r["p_yes"] for r in recs])
    assert m["n"] == 200 and m["positive_rate"] == pytest.approx(y.mean())
    at = m["at_threshold"]
    assert at["threshold"] == 0.5
    assert {k: at[k] for k in ("precision", "recall", "f1")} == {
        k: _prf1(y, p >= 0.5)[k] for k in ("precision", "recall", "f1")}
    assert m["best_f1"]["f1"] >= at["f1"]
    assert m["auprc"] == auprc(y, p) and m["auprc_lift"] == pytest.approx(
        m["auprc"] / y.mean())
    table = compute_fast_metrics(recs)
    assert list(table) == METRIC_GROUPS
    assert table["NORM (normal subset)"] == table["NORM"]
    assert table["abnormal subset"]["n"] == sum(r["superclass"] != "NORM" for r in recs)


def test_the_fast_table_labels_which_threshold_is_which(capsys):
    from evaluate import print_fast_metrics

    print_fast_metrics(compute_fast_metrics(_fast()), title="real")
    out = capsys.readouterr().out
    assert "F1@.5" in out and "F1*" in out and "thr*" in out
    assert "chosen ON THESE EXAMPLES" in out and "-- real" in out
    for group in METRIC_GROUPS:
        assert group in out


def test_fast_and_generated_verdicts_are_compared_on_shared_parsed_examples():
    from eval_controls import fast_generation_agreement

    fast = [{"ecg_id": i, "superclass": "MI", "p_yes": p}
            for i, p in enumerate((0.9, 0.9, 0.1, 0.1, 0.5))]
    gen = [{"ecg_id": 0, "superclass": "MI", "parsed_conclusion": True},
           {"ecg_id": 1, "superclass": "MI", "parsed_conclusion": False},
           {"ecg_id": 2, "superclass": "MI", "parsed_conclusion": False},
           {"ecg_id": 3, "superclass": "MI", "parsed_conclusion": None},
           {"ecg_id": 4, "superclass": "MI", "parsed_conclusion": True},   # 0.5 is Yes
           {"ecg_id": 9, "superclass": "MI", "parsed_conclusion": True}]   # no fast row
    a = fast_generation_agreement(fast, gen)
    assert a["n_shared"] == 5 and a["n_unparsed"] == 1 and a["n"] == 4
    assert a["agree"] == 3 and a["agreement"] == 0.75
    assert a["fast Yes, gen No"] == 1 and a["fast No, gen Yes"] == 0
    assert a["per_superclass"]["MI"]["agreement"] == 0.75
    assert a["per_superclass"]["NORM"]["agreement"] is None


def test_identical_counts_follow_the_conditions_that_generated():
    from eval_controls import identical_counts

    gen = {name: [{"ecg_id": 1, "superclass": "MI", "generation": "same"}]
           for name in ("real", "shuffled")}
    out = identical_counts(gen)
    assert out["conditions"] == ["real", "shuffled"] and out["identical_all_three"] == 1


# --------------------------------------------------------------------------- #
# Flags                                                                       #
# --------------------------------------------------------------------------- #

def _parse(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    return module.parse_args()


def test_sweep_defaults(monkeypatch):
    import eval_controls

    args = _parse(eval_controls, monkeypatch, "--model-dir", "m", "--ckpt", "c")
    assert (args.stratify, args.sample_seed, args.seed) == ("superclass-label", 42, 42)
    assert args.fast_limit is None and not args.skip_fast
    assert not args.skip_zeroed_generation
    assert _parse(eval_controls, monkeypatch, "--model-dir", "m", "--ckpt", "c",
                  "--skip-zeroed-generation").skip_zeroed_generation


def test_evaluate_and_probe_flags(monkeypatch):
    import evaluate
    import probe_latents
    import yn_token_loss

    args = _parse(evaluate, monkeypatch, "--limit", "500")
    assert (args.stratify, args.sample_seed) == ("superclass-label", 42)
    assert _parse(probe_latents, monkeypatch, "--model-dir", "m", "--ckpt",
                  "c").task_text is None
    assert _parse(yn_token_loss, monkeypatch).known_sequence_mean is None
    assert _parse(yn_token_loss, monkeypatch, "--known-sequence-mean", "1", "2",
                  "3").known_sequence_mean == [1.0, 2.0, 3.0]


def test_probe_latents_resolves_the_stream_as_the_sweep_does(tmp_path, capsys):
    import torch
    from evaluate import resolve_task_stream

    path = str(tmp_path / "best.pt")
    torch.save({"meta": {"task_stream": "question"}}, path)
    assert resolve_task_stream(path, None, flag="--task-text") == "question"
    assert "question (recorded in the checkpoint)" in capsys.readouterr().out
    assert resolve_task_stream(path, "prompt", flag="--task-text") == "prompt"
    out = capsys.readouterr().out
    assert "prompt (--task-text)" in out and "trained with --task-stream question" in out


# --------------------------------------------------------------------------- #
# The job scripts                                                             #
# --------------------------------------------------------------------------- #

def _bash(cmd, env=None):
    return subprocess.run(["bash", "-c", cmd], cwd=REPO_ROOT, capture_output=True,
                          text=True, env={**os.environ, **(env or {})})


def test_one_stream_tag_mapping():
    got = _bash('source scripts/stream_tag.sh; for m in prompt prompt-and-target '
                'question none; do echo "[$(stream_tag $m)]"; done; '
                'stream_tag bogus || echo refused')
    assert got.stdout.split() == ["[]", "[_withtarget]", "[_question]", "[_notext]",
                                  "refused"]
    # and no job script keeps a mapping of its own
    for name in os.listdir(os.path.join(REPO_ROOT, "scripts")):
        if name.endswith(".sh") and name != "stream_tag.sh":
            text = open(os.path.join(REPO_ROOT, "scripts", name)).read()
            assert 'STREAM_TAG="_question"' not in text, name
            assert 'TAG="${TAG}_withtarget"' not in text, name


@pytest.mark.parametrize("script", ["eval_controls.sh", "evaluate.sh",
                                    "probe_latents.sh"])
def test_a_missing_checkpoint_fails_the_preflight_under_trains_tag(tmp_path, script):
    venv, model = tmp_path / "venv", tmp_path / "model"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text("")
    model.mkdir()
    (model / "config.json").write_text("{}")
    env = {"VENV": str(venv), "MODEL_DIR": str(model), "SCRATCH": str(tmp_path),
           "SLURM_SUBMIT_DIR": REPO_ROOT, "RUN": "run2", "TASK_STREAM": "question"}
    got = _bash(f"bash scripts/{script}", env)
    assert got.returncode == 2, got.stderr
    assert (f"resampler checkpoint: {tmp_path}/ecg-llm-alignment/checkpoints/"
            "run2_question/best.pt") in got.stderr
    assert "module" not in got.stdout + got.stderr.split("missing")[0]
    # once it exists, the same preflight passes (and the script goes on to the
    # module load, which this machine does not have)
    ckpt = tmp_path / "ecg-llm-alignment" / "checkpoints" / "run2_question"
    ckpt.mkdir(parents=True)
    (ckpt / "best.pt").write_text("")
    assert "resampler checkpoint" not in _bash(f"bash scripts/{script}", env).stderr
    assert _bash(f"bash scripts/{script}", {**env, "TASK_STREAM": "bogus"}).returncode == 2
