"""The linear and consistency losses (train.py --lambda-linear / --lambda-consistency).

What is pinned here:

* the consistency position is the gold Yes/No token after "Conclusion:", the same
  one --loss-on decision marks, and every one of the reference's guards is fatal;
  a conclusion --max-length cut away sits out the consistency loss and nothing else;
* the consistency loss reaches the resampler and never the head, and the step's
  assertion catches a target that is not detached;
* the LLM's binary logit is the readout's P(Yes), as a log-odds;
* pos_weight is n_neg/n_pos per superclass;
* the command line refuses what the objective cannot mean, and a run states both
  weights;
* checkpoints carry the head and the weights, evaluation refuses a payload that
  disagrees with its meta, and a resume refuses other weights;
* a preempted run requeues itself and never exits 0;
* eval's three-way comparison is on the examples every readout scored.

Uses the debug student's tokenizer and a stub student, as tests/test_overfit_gate.py
does; no GPU.

    pytest tests/test_aux_losses.py
"""

import os
import sys
import types

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import train  # noqa: E402
from src.model.linear_head import AuxiliaryLinearHead  # noqa: E402
from src.teacher.prompt import SUPERCLASSES  # noqa: E402
from tests.test_overfit_gate import _decision_setup, _loss_student  # noqa: E402


def _yes_no(tok):
    from evaluate import yes_no_token_ids
    return yes_no_token_ids(tok)


def _args(**kw):
    base = dict(task_stream="prompt", loss_on="all", lambda_llm=1.0, lambda_linear=1.0,
                lambda_consistency=0.5)
    return types.SimpleNamespace(**{**base, **kw})


def _parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["train.py", "--model-dir", "x", *argv])
    return train.parse_args()


# --------------------------------------------------------------------------- #
# Where the consistency loss reads the LLM                                     #
# --------------------------------------------------------------------------- #

def test_the_consistency_position_is_the_decision_token():
    dataset, tok, base, decision_collate, indices = _decision_setup(20)
    yes, no = _yes_no(tok)
    items = [dataset[i] for i in indices]
    batch = train.make_consistency_collate(tok, base, yes, no)(items)
    marked = decision_collate(items)["decision_mask"]
    assert batch["consistency_pos"].tolist() == \
        [int(marked[i].nonzero().item()) for i in range(len(items))]
    # nothing else in the batch moved
    plain = base(items)
    assert set(batch) - set(plain) == {"consistency_pos"}
    for key, value in plain.items():
        assert bool((value == batch[key]).all()) if torch.is_tensor(value) \
            else value == batch[key], key
    # supervised, and the gold word
    for i, item in enumerate(items):
        p = int(batch["consistency_pos"][i])
        assert int(batch["labels"][i, p]) == int(batch["input_ids"][i, p]) != -100
        assert tok.decode([int(batch["input_ids"][i, p])]).strip() == \
            ("Yes" if item["label"] else "No")


def test_under_decision_no_context_it_reads_the_cut_target():
    dataset, tok, base, _, indices = _decision_setup(10)
    yes, no = _yes_no(tok)
    items = [dataset[i] for i in indices]
    collate = train.make_no_context_collate(
        tok, train.make_consistency_collate(tok, base, yes, no))
    batch = collate(items)
    marked = batch["decision_mask"]
    assert batch["consistency_pos"].tolist() == \
        [int(marked[i].nonzero().item()) for i in range(len(items))]


def test_a_conclusion_cut_by_max_length_sits_out_and_is_not_fatal():
    from src.data.dataset import make_collate_fn

    dataset, tok, _, _, _ = _decision_setup()
    yes, no = _yes_no(tok)
    items = [dataset[i] for i in range(4)]
    short = make_collate_fn(tok, max_length=120, strip_think=dataset.config.strip_think)
    batch = train.make_consistency_collate(tok, short, yes, no)(items)
    assert batch["consistency_pos"].tolist() == [-1] * 4
    assert int((batch["labels"] != -100).sum()) > 0       # the LLM loss still has tokens


@pytest.mark.parametrize("edit, why", [
    (lambda b: {**b, "target": b["target"] + "\nConclusion: Yes"}, "2 occurrences"),
    (lambda b: {**b, "target": b["target"].replace("Conclusion:", "Verdict:")},
     "0 occurrences"),
    (lambda b: {**b, "label": not b["label"]}, "but the gold label is"),
    (lambda b: {**b, "target": b["target"][:b["target"].index("Conclusion:")]
                + "Conclusion: Maybe."}, "not a single Yes/No"),
])
def test_every_guard_is_fatal_and_names_the_example(edit, why):
    dataset, tok, base, _, _ = _decision_setup()
    yes, no = _yes_no(tok)
    collate = train.make_consistency_collate(tok, base, yes, no)
    items = [dataset[i] for i in range(3)]
    bad = edit(items[1])
    with pytest.raises(train.ConsistencyTargetError) as err:
        collate([items[0], bad, items[2]])
    assert "1 of the 3 examples" in str(err.value) and why in str(err.value)
    assert f"{bad['ecg_id']:>7} {bad['superclass']:<6}" in str(err.value)


def test_the_counterfactual_swap_lands_on_the_same_index():
    dataset, tok, _, _, _ = _decision_setup()
    yes, no = _yes_no(tok)
    for i in range(10):
        e = dataset.examples[i]
        k, ids, why = train.consistency_position(tok, e.target, e.label, yes, no)
        assert why is None
        swapped = e.target.replace("Conclusion: Yes" if e.label else "Conclusion: No",
                                   "Conclusion: No" if e.label else "Conclusion: Yes")
        k2, ids2, why2 = train.consistency_position(tok, swapped, not e.label, yes, no)
        assert why2 is None and k2 == k and ids2[:k] == ids[:k] and ids2[k] != ids[k]
    assert train._swapped_word("YES") == "NO" and train._swapped_word("no") == "yes"


def test_the_startup_scan_says_every_bad_target_at_once():
    dataset, tok, _, _, _ = _decision_setup()
    yes, no = _yes_no(tok)
    train.check_consistency_targets(tok, {"train": dataset.examples[:200]}, yes, no)
    broken = [types.SimpleNamespace(ecg_id=e.ecg_id, superclass=e.superclass,
                                    label=e.label, target=e.target + "\nConclusion: Yes")
              for e in dataset.examples[:3]]
    with pytest.raises(train.ConsistencyTargetError, match="3 of 3 targets"):
        train.check_consistency_targets(tok, {"val": broken}, yes, no)


# --------------------------------------------------------------------------- #
# The losses                                                                   #
# --------------------------------------------------------------------------- #

def test_pos_weight_is_n_neg_over_n_pos_per_superclass():
    ex = [types.SimpleNamespace(superclass=sc, label=label)
          for sc, n_pos, n_neg in (("NORM", 4, 6), ("MI", 1, 9), ("STTC", 2, 8),
                                   ("CD", 5, 5), ("HYP", 1, 7))
          for label in [True] * n_pos + [False] * n_neg]
    w = train.superclass_pos_weights(ex)
    assert w == {"NORM": 1.5, "MI": 9.0, "STTC": 4.0, "CD": 1.0, "HYP": 7.0}
    with pytest.raises(ValueError, match="HYP has no positive"):
        train.superclass_pos_weights([e for e in ex if not (e.superclass == "HYP"
                                                            and e.label)])


def test_the_head_pools_the_latents_to_one_logit():
    head = AuxiliaryLinearHead(16)
    x = torch.randn(3, 64, 16)
    out = head(x)
    assert out.shape == (3,)
    assert torch.allclose(out, head.classifier(head.norm(x.mean(dim=1))).squeeze(-1))


def _setup(n=4):
    from src.model.resampler import PerceiverResamplerConfig, Resampler

    dataset, tok, base, _, _ = _decision_setup()
    yes, no = _yes_no(tok)
    cfg = PerceiverResamplerConfig.debug()
    model = _loss_student(tok, cfg.embed_dim)
    torch.manual_seed(0)
    resampler = Resampler(cfg).train()
    head = AuxiliaryLinearHead(cfg.embed_dim).train()
    items = [dataset[i] for i in range(n)]
    batch = train.make_consistency_collate(tok, base, yes, no)(items)
    weights = {sc: 2.0 for sc in SUPERCLASSES}
    return model, resampler, head, batch, yes, no, weights


def test_the_llm_binary_logit_is_the_readouts_p_yes():
    from evaluate import probability_from_logits
    from probe_memory import splice_forward

    model, resampler, _, batch, yes, no, _ = _setup()
    with torch.no_grad():
        _, _, logits, Lq = splice_forward(model, resampler, batch, "cpu", return_aux=True)
    rows = torch.arange(len(batch["ecg_id"]))
    z = train.llm_binary_logits(logits, rows, Lq + batch["consistency_pos"] - 1, yes, no)
    cand = yes + no
    for i in rows.tolist():
        row = logits[i, Lq + int(batch["consistency_pos"][i]) - 1].float()[cand]
        p = probability_from_logits(row, range(len(yes)), range(len(yes), len(cand)))
        assert float(torch.sigmoid(z[i])) == pytest.approx(p, abs=1e-5)


def test_the_total_is_the_weighted_sum_and_each_term_is_what_it_says():
    import torch.nn.functional as F
    from probe_memory import splice_forward

    model, resampler, head, batch, yes, no, weights = _setup()
    args = _args(lambda_llm=0.7, lambda_linear=1.3, lambda_consistency=0.4)
    total, terms = train.make_objective(args, head, yes, no, weights)(
        model, resampler, batch, "cpu")
    with torch.no_grad():
        llm, latents, logits, Lq = splice_forward(model, resampler, batch, "cpu",
                                                  return_aux=True)
        lin = head(latents)
        z = train.llm_binary_logits(logits, torch.arange(4),
                                    Lq + batch["consistency_pos"] - 1, yes, no)
    assert terms["llm"] == pytest.approx(float(llm))
    assert terms["linear"] == pytest.approx(float(F.binary_cross_entropy_with_logits(
        lin, batch["head_label"], pos_weight=torch.full((4,), 2.0))))
    assert terms["consistency"] == pytest.approx(float(
        F.binary_cross_entropy_with_logits(z, torch.sigmoid(lin))), abs=1e-6)
    assert terms["gap"] == pytest.approx(float(
        (torch.sigmoid(z) - torch.sigmoid(lin)).abs().mean()), abs=1e-6)
    assert (terms["n_consistency"], terms["n_skipped"]) == (4, 0)
    assert float(total) == pytest.approx(0.7 * terms["llm"] + 1.3 * terms["linear"]
                                         + 0.4 * terms["consistency"], abs=1e-5)


def test_without_a_head_the_step_is_the_splice_loss_as_before():
    from probe_memory import splice_forward

    model, resampler, _, batch, _, _, _ = _setup()
    with torch.no_grad():
        total, terms = train.make_objective(_args(lambda_linear=0.0,
                                                  lambda_consistency=0.0))(
            model, resampler, batch, "cpu")
        assert float(total) == float(splice_forward(model, resampler, batch, "cpu"))
    assert set(terms) == {"llm"}


def test_consistency_reaches_the_resampler_and_never_the_head():
    from probe_memory import splice_forward

    model, resampler, head, batch, yes, no, _ = _setup()
    _, latents, logits, Lq = splice_forward(model, resampler, batch, "cpu",
                                            return_aux=True)
    z = train.llm_binary_logits(logits, torch.arange(4),
                                Lq + batch["consistency_pos"] - 1, yes, no)
    consistency = train.consistency_loss(z, head(latents))
    train.assert_consistency_spares_head(consistency, head)      # passes
    consistency.backward()
    assert all(p.grad is None for p in head.parameters())
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in resampler.parameters())


def test_a_target_that_is_not_detached_is_caught_in_the_step(monkeypatch):
    import torch.nn.functional as F

    model, resampler, head, batch, yes, no, weights = _setup()
    monkeypatch.setattr(train, "consistency_loss", lambda z, lin: (
        F.binary_cross_entropy_with_logits(z, torch.sigmoid(lin))))
    objective = train.make_objective(_args(), head, yes, no, weights)
    with pytest.raises(train.GradFlowError, match="not detached"):
        objective(model, resampler, batch, "cpu")
    # with a zero weight the term is logged, not added, and not asserted on
    objective = train.make_objective(_args(lambda_consistency=0.0), head, yes, no,
                                     weights)
    objective(model, resampler, batch, "cpu")


def test_rows_past_max_length_drop_out_of_consistency_only():
    model, resampler, head, batch, yes, no, weights = _setup()
    batch = {**batch, "consistency_pos": batch["consistency_pos"].clone()}
    batch["consistency_pos"][1] = -1
    _, terms = train.make_objective(_args(), head, yes, no, weights)(
        model, resampler, batch, "cpu")
    assert (terms["n_consistency"], terms["n_skipped"]) == (3, 1)
    batch["consistency_pos"][:] = -1
    total, terms = train.make_objective(_args(), head, yes, no, weights)(
        model, resampler, batch, "cpu")
    assert terms["consistency"] is None and terms["gap"] is None
    assert float(total) == pytest.approx(terms["llm"] + terms["linear"], abs=1e-5)


def test_the_log_line_gives_each_term_unweighted_and_weighted():
    window = [{"llm": 1.0, "linear": 0.5, "consistency": 0.2, "gap": 0.1,
               "n_consistency": 3, "n_skipped": 1},
              {"llm": 3.0, "linear": 1.5, "consistency": 0.6, "gap": 0.5,
               "n_consistency": 1, "n_skipped": 0}]
    means = train.mean_terms(window)
    assert means["consistency"] == pytest.approx((0.2 * 3 + 0.6) / 4)
    line = train.describe_terms(means, _args(lambda_consistency=0.5), "loss")
    assert "loss 2.0000 (x1 = 2.0000)" in line and "linear 1.0000 (x1 = 1.0000)" in line
    assert "consistency 0.3000 (x0.5 = 0.1500)" in line
    assert line.startswith("total 3.1500") and "|p_llm-p_linear| 0.2000 over 4 rows, " \
        "1 past --max-length" in line
    # a run without a head logs exactly what it always logged
    assert train.describe_terms(train.mean_terms([{"llm": 0.25}]),
                                _args(lambda_linear=0.0, lambda_consistency=0.0),
                                "decision-token loss") == "decision-token loss 0.2500"


# --------------------------------------------------------------------------- #
# The command line                                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("argv, message", [
    ((), "the following arguments are required: --lambda-linear, --lambda-consistency"),
    (("--lambda-linear", "1"), "required: --lambda-consistency"),
    (("--lambda-linear", "0", "--lambda-consistency", "0.5"), "needs --lambda-linear > 0"),
    (("--lambda-linear", "-1", "--lambda-consistency", "0"), "cannot be negative"),
    (("--lambda-llm", "0", "--lambda-linear", "0", "--lambda-consistency", "0"),
     "nothing would train"),
    (("--lambda-linear", "0", "--lambda-consistency", "0", "--linear-head-lr", "1e-3"),
     "silently ignored"),
    (("--lambda-linear", "0", "--lambda-consistency", "0", "--overfit", "50",
      "--save-every-epoch"), "--overfit has no epochs"),
])
def test_the_command_line_refuses_what_the_objective_cannot_mean(monkeypatch, capsys,
                                                                argv, message):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, *argv)
    assert message in capsys.readouterr().err


def test_the_gate_takes_the_new_losses(monkeypatch):
    args = _parse(monkeypatch, "--overfit", "50", "--lambda-linear", "0.25",
                  "--lambda-consistency", "0.1")
    assert (args.lambda_linear, args.lambda_consistency) == (0.25, 0.1)


def test_the_overfit_loop_trains_the_head_and_saves_it_with_the_resampler(
        tmp_path, monkeypatch, capsys):
    """Under --overfit the step is the objective's (the head moves, every term is on
    the step line), the stop rule is still the LM mean, and overfit_final.pt carries
    the head for evaluate.load_linear_head."""
    from evaluate import load_linear_head

    model, resampler, head, batch, yes, no, weights = _setup()
    for p in model.parameters():          # frozen, as load_student leaves the 14B
        p.requires_grad_(False)
    args = types.SimpleNamespace(**vars(_args(lambda_linear=0.25, lambda_consistency=0.1)),
                                 overfit_max_steps=2, overfit_eval_every=2,
                                 overfit_target_loss=1e9, prefix_log_every=100,
                                 grad_clip=1.0)
    groups = [{"params": resampler.parameters(), "lr": 1e-4},
              {"params": head.parameters(), "lr": 1e-4}]
    optimizer = torch.optim.AdamW(groups)
    before = [p.detach().clone() for p in head.parameters()]
    stop_rule = []
    monkeypatch.setattr(train, "eval_mean_loss", lambda *a, **k: stop_rule.append(1) or 0.5)
    steps = train.train_overfit(model, resampler, optimizer, [batch], [batch], "cpu",
                                args, objective=train.make_objective(
                                    args, head, yes, no, weights), head=head)
    out = capsys.readouterr().out
    assert steps == 2 and stop_rule == [1]
    assert "linear" in out and "consistency" in out and "|p_llm-p_linear|" in out
    assert any(not torch.equal(a, b) for a, b in zip(before, head.parameters()))

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(train, "_RUN_META", {**train.objective_lambdas(args)})
    path = train.save_overfit_final(str(tmp_path), resampler, 0, steps, False,
                                    args, head=head)
    loaded = load_linear_head(path, "cpu")
    assert all(torch.equal(a, b) for a, b in zip(head.state_dict().values(),
                                                 loaded.state_dict().values()))


def test_the_head_lr_defaults_to_1e_4(monkeypatch):
    args = _parse(monkeypatch, "--lambda-linear", "1", "--lambda-consistency", "0.5")
    assert args.linear_head_lr == 1e-4 and args.lambda_llm == 1.0
    assert train.objective_lambdas(args) == {
        "lambda_llm": 1.0, "lambda_linear": 1.0, "lambda_consistency": 0.5}


# --------------------------------------------------------------------------- #
# Checkpoints                                                                  #
# --------------------------------------------------------------------------- #

def _save(tmp_path, monkeypatch, lambdas, head):
    module = torch.nn.Linear(2, 2)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])   # no GPU here
    monkeypatch.setattr(train, "_RUN_META", {"strip_think": True, "task_stream": "prompt",
                                             **lambdas})
    opt = torch.optim.AdamW(module.parameters())
    return module, train.save_checkpoint(str(tmp_path), module, opt, epoch=0, step=1,
                                         best_val=1.0, is_best=False, head=head)


def test_the_checkpoint_carries_the_head_and_the_weights_and_eval_loads_it(
        tmp_path, monkeypatch):
    from evaluate import checkpoint_objective, load_linear_head

    head = AuxiliaryLinearHead(8)
    lambdas = {"lambda_llm": 1.0, "lambda_linear": 1.0, "lambda_consistency": 0.5}
    _, path = _save(tmp_path, monkeypatch, lambdas, head)
    assert checkpoint_objective(path) == (lambdas, True)
    loaded = load_linear_head(path, "cpu")
    for (n, a), (_, b) in zip(head.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b), n
    assert not any(p.requires_grad for p in loaded.parameters())


def test_eval_refuses_a_head_its_meta_does_not_account_for(tmp_path, monkeypatch):
    from evaluate import checkpoint_objective, load_linear_head

    _, path = _save(tmp_path, monkeypatch, {"lambda_llm": 1.0, "lambda_linear": 0.0,
                                            "lambda_consistency": 0.0},
                    AuxiliaryLinearHead(8))
    with pytest.raises(SystemExit, match="linear head MISMATCH"):
        load_linear_head(path, "cpu")
    (tmp_path / "b").mkdir()
    _, path = _save(tmp_path / "b", monkeypatch, {"lambda_llm": 1.0, "lambda_linear": 2.0,
                                                  "lambda_consistency": 0.0}, None)
    with pytest.raises(SystemExit, match="has no linear head"):
        load_linear_head(path, "cpu")
    # a checkpoint from before the flags: LM loss alone, no head, said so
    torch.save({"resampler": {}, "meta": {"strip_think": True}}, tmp_path / "old.pt")
    assert checkpoint_objective(str(tmp_path / "old.pt")) == (
        {"lambda_llm": 1.0, "lambda_linear": 0.0, "lambda_consistency": 0.0}, False)
    assert load_linear_head(str(tmp_path / "old.pt"), "cpu") is None


def test_the_epoch_checkpoint_has_weights_and_meta_but_no_optimizer(tmp_path,
                                                                    monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(train, "_RUN_META", {"lambda_linear": 1.0})
    path = train.save_epoch_checkpoint(str(tmp_path), torch.nn.Linear(2, 2), 3, 120,
                                       0.5, head=AuxiliaryLinearHead(8))
    assert os.path.basename(path) == "epoch03.pt"
    payload = torch.load(path, weights_only=False)
    assert "optimizer" not in payload and "linear_head" in payload
    assert payload["meta"]["lambda_linear"] == 1.0
    assert train._rotating(str(tmp_path)) == []        # never resumed from, never pruned


def test_a_resume_under_other_weights_is_refused(tmp_path):
    torch.save({"meta": {"loss_on": "all", "lambda_llm": 1.0, "lambda_linear": 1.0,
                         "lambda_consistency": 0.5}}, tmp_path / "ckpt_step10.pt")
    with pytest.raises(SystemExit) as err:
        train.resume(str(tmp_path), None, None, "cpu", lambdas={
            "lambda_llm": 1.0, "lambda_linear": 1.0, "lambda_consistency": 0.0})
    assert "objective MISMATCH" in str(err.value)
    assert "Pass --lambda-llm 1  --lambda-linear 1  --lambda-consistency 0.5" \
        in str(err.value)
    # a checkpoint from before the flags trained the LM loss alone
    torch.save({"meta": {"loss_on": "all"}}, tmp_path / "ckpt_step10.pt")
    with pytest.raises(SystemExit, match="NOT recorded"):
        train.resume(str(tmp_path), None, None, "cpu", lambdas={
            "lambda_llm": 1.0, "lambda_linear": 1.0, "lambda_consistency": 0.0})


# --------------------------------------------------------------------------- #
# Preemption                                                                   #
# --------------------------------------------------------------------------- #

def test_a_preempted_run_requeues_itself_and_never_exits_0(monkeypatch, capsys):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(train.subprocess, "run", run)
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    with pytest.raises(SystemExit) as err:
        train.requeue_and_exit()
    assert err.value.code == train.PREEMPT_EXIT_CODE != 0
    assert calls == [["scontrol", "requeue", "4242"]]
    assert "requeued" in capsys.readouterr().out

    # a requeue that fails still exits non-zero, so sacct shows FAILED, not COMPLETED
    monkeypatch.setattr(train.subprocess, "run", lambda cmd, **kw: (
        types.SimpleNamespace(returncode=1, stdout="", stderr="Invalid job id")))
    with pytest.raises(SystemExit) as err:
        train.requeue_and_exit()
    assert err.value.code == train.PREEMPT_EXIT_CODE
    assert "FAILED (Invalid job id)" in capsys.readouterr().out

    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(SystemExit) as err:
        train.requeue_and_exit()
    assert err.value.code == train.PREEMPT_EXIT_CODE


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #

def test_the_head_readout_reads_the_prefix_fast_mode_reads():
    from evaluate import linear_head_records, splice_prefix
    from src.data.dataset import make_collate_fn, task_stream_ids
    from decision_token import MARKER

    model, resampler, head, _, _, _, _ = _setup()
    resampler.eval()
    head.eval()
    dataset, tok, _, _, _ = _decision_setup()
    records = linear_head_records(model, resampler, head, tok, "cpu", dataset,
                                  list(range(6)), batch_size=4, max_length=2048,
                                  task_stream="prompt", seed=0, shuffled=False,
                                  zero_latents=False)
    collate = make_collate_fn(tok, max_length=2048, strip_think=dataset.config.strip_think)
    batch = collate([{**dataset[i], "target": MARKER} for i in range(6)])
    ids, mask = task_stream_ids(batch, "prompt")
    with torch.no_grad():
        *_, latents = splice_prefix(model, resampler, batch["ecg_embed"],
                                    batch["input_ids"], batch["attention_mask"], "cpu",
                                    task_ids=ids, task_mask=mask, return_latents=True)
        want = torch.sigmoid(head(latents))
    assert [r["p_linear"] for r in records] == pytest.approx(want.tolist(), abs=1e-6)
    zeroed = linear_head_records(model, resampler, head, tok, "cpu", dataset,
                                 list(range(6)), batch_size=4, max_length=2048,
                                 task_stream="prompt", seed=0, shuffled=False,
                                 zero_latents=True)
    assert len({round(r["p_linear"], 6) for r in zeroed}) == 1      # the floor


def test_the_three_way_comparison_is_on_the_examples_every_readout_scored():
    from eval_controls import readout_comparison

    keys = [(i // 5 + 1, SUPERCLASSES[i % 5]) for i in range(20)]
    labels = [i % 3 == 0 for i in range(20)]

    def recs(key, score):
        return [{"ecg_id": e, "superclass": sc, "label": y, key: score(i, y)}
                for i, ((e, sc), y) in enumerate(zip(keys, labels))]

    gen = recs("p_yes", lambda i, y: None if i == 4 else (0.9 if y else 0.1))
    fast = recs("p_yes", lambda i, y: 0.6 if y else 0.4)
    linear = recs("p_linear", lambda i, y: 0.3)
    out = readout_comparison({"generation": (gen, "p_yes"), "fast": (fast, "p_yes"),
                              "linear": (linear, "p_linear")})
    assert out["readouts"] == ["generation", "fast", "linear"]
    assert (out["n"], out["n_draw"]) == (19, 20)      # the unparsed one is out of all three
    overall = out["rows"]["overall"]
    assert overall["n"] == 19 and overall["generation"] == overall["fast"] == 1.0
    assert overall["linear"] == pytest.approx(overall["positive_rate"])  # a constant
    assert out["rows"][SUPERCLASSES[4]]["n"] == 3
    gaps = out["mean_abs_gap_to_linear"]
    pos = sum(y for i, y in enumerate(labels) if i != 4)
    assert gaps["fast"] == pytest.approx((pos * 0.3 + (19 - pos) * 0.1) / 19)
