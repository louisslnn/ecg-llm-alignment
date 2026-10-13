"""--overfit-select: which examples the gate draws, and that it says which.

``head`` is the first N train examples and must stay exactly that: every gate run
before the flag was one. ``stratified`` draws N/5 whole records so that each
superclass has at least three Yes and three No among them -- on the head set all ten
records are normal, so the answer is a function of the question and the gate can
pass without a waveform being read. Pinned here:

* head is the first N indices, whatever the labels;
* stratified meets the floor with whole records only, in dataset order, and is the
  same draw for the same seed;
* a floor that cannot be met is an error naming what is short, never a relaxed set;
* the flags that make no sense with it are refused at the command line.

No model, no GPU, no data: the examples are built here.

    pytest tests/test_overfit_select.py
"""

import os
import random
import sys
import types

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import train  # noqa: E402
from src.data.dataset import Example  # noqa: E402
from src.teacher.prompt import SUPERCLASSES  # noqa: E402

ABNORMAL = [sc for sc in SUPERCLASSES if sc != "NORM"]


def _split(n_records=400, seed=0, n_leading_normal=10, incomplete=()):
    """A train split's examples: mostly single-label records, a few multi-label.

    The first ``n_leading_normal`` records are normal, as the real split's are.
    Records in ``incomplete`` lose their last question, as a dropped teacher block
    does.
    """
    rng = random.Random(seed)
    examples = []
    for ecg_id in range(1, n_records + 1):
        if ecg_id <= n_leading_normal or rng.random() < 0.4:
            positive = {"NORM"}
        else:
            positive = set(rng.sample(ABNORMAL, rng.choice([1, 1, 1, 2, 3, 4])))
        asked = SUPERCLASSES[:-1] if ecg_id in incomplete else SUPERCLASSES
        examples += [Example(ecg_id=ecg_id, superclass=sc, strat_fold=1, target="",
                             label=sc in positive) for sc in asked]
    return types.SimpleNamespace(examples=examples)


def _args(overfit=50, select="stratified", seed=42):
    return types.SimpleNamespace(overfit=overfit, overfit_select=select, seed=seed)


def test_head_is_the_first_n_examples_and_nothing_else():
    dataset = _split()
    indices, selection = train.select_overfit(dataset, _args(select="head"))
    assert indices == list(range(50))
    assert selection["select"] == "head"
    assert selection["ecg_ids"] == list(range(1, 11))
    # the set the flag exists for: one answer per question, on every record
    assert selection["label_counts"] == {
        sc: {"yes": 10 if sc == "NORM" else 0, "no": 0 if sc == "NORM" else 10}
        for sc in SUPERCLASSES}
    # short of N, head takes what there is, as it always did
    few = types.SimpleNamespace(examples=dataset.examples[:7])
    assert train.select_overfit(few, _args(select="head"))[0] == list(range(7))


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_stratified_meets_the_floor_with_whole_records(seed):
    dataset = _split(incomplete={3, 77, 140})
    indices, selection = train.select_overfit(dataset, _args(seed=seed))

    assert len(indices) == 50 and indices == sorted(indices)
    assert len(selection["ecg_ids"]) == 10
    for counts in selection["label_counts"].values():
        assert counts["yes"] >= 3 and counts["no"] >= 3
        assert counts["yes"] + counts["no"] == 10
    # whole records: each chosen one contributes all five questions, in order
    picked = [dataset.examples[i] for i in indices]
    for k, ecg_id in enumerate(selection["ecg_ids"]):
        unit = picked[5 * k:5 * k + 5]
        assert [e.ecg_id for e in unit] == [ecg_id] * 5
        assert [e.superclass for e in unit] == SUPERCLASSES
    assert not {3, 77, 140} & set(selection["ecg_ids"])


def test_stratified_is_the_same_draw_for_the_same_seed_and_leaves_the_global_rng_alone():
    dataset = _split()
    random.seed(7)
    before = random.getstate()
    first = train.select_overfit(dataset, _args(seed=42))
    assert random.getstate() == before
    assert train.select_overfit(dataset, _args(seed=42)) == first
    assert train.select_overfit(dataset, _args(seed=43))[1]["ecg_ids"] != first[1]["ecg_ids"]


def test_once_nothing_is_short_the_draw_is_the_shuffled_order():
    """The rule's step 3 with every score at zero: no preference left to apply."""
    examples = _split().examples
    ten = train.stratified_records(examples, 10, seed=42)
    forty = train.stratified_records(examples, 40, seed=42)
    assert forty[:10] == ten
    order = sorted({e.ecg_id for e in examples})
    random.Random(42).shuffle(order)
    rest = [r for r in order if r not in ten]
    assert forty[10:] == rest[:30]


def test_a_floor_that_cannot_be_met_is_an_error_not_a_smaller_floor():
    normal_only = types.SimpleNamespace(examples=_split(n_records=10).examples)
    with pytest.raises(ValueError) as err:
        train.select_overfit(normal_only, _args(overfit=30))
    assert "NORM No has 0" in str(err.value) and "MI Yes has 0" in str(err.value)
    assert "NORM Yes" not in str(err.value)

    with pytest.raises(ValueError, match="12 records asked for, 10 have all 5"):
        train.stratified_records(normal_only.examples, 12, seed=0)


@pytest.mark.parametrize("argv, message", [
    (["--overfit-select", "stratified"], "without --overfit N"),
    (["--overfit", "52", "--overfit-select", "stratified"], "whole number of records"),
    (["--overfit", "25", "--overfit-select", "stratified"], "at least 30"),
])
def test_the_command_line_refuses_what_stratified_cannot_mean(monkeypatch, capsys,
                                                            argv, message):
    monkeypatch.setattr(sys, "argv", ["train.py", "--model-dir", "x", *argv])
    with pytest.raises(SystemExit):
        train.parse_args()
    assert message in capsys.readouterr().err


def test_head_is_the_default_and_takes_any_n(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--model-dir", "x", "--overfit", "52"])
    assert train.parse_args().overfit_select == "head"


def test_the_draw_is_printed_ids_first_then_five_by_two(capsys):
    dataset = _split()
    indices, selection = train.select_overfit(dataset, _args())
    train.print_overfit_selection(selection, len(indices), seed=42)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == ("overfit selection: stratified, seed 42 -- 50 examples from "
                        "10 records")
    assert lines[1].split() == ["ecg_ids:"] + [str(i) for i in selection["ecg_ids"]]
    assert lines[2].split() == ["superclass", "Yes", "No"]
    assert [line.split() for line in lines[3:]] == [
        [sc, str(n["yes"]), str(n["no"])] for sc, n in selection["label_counts"].items()]
