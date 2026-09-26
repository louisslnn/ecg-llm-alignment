"""Contract for the (ecg_id, superclass) training dataset.

Runs against the real cache/manifest/teacher artifacts. The three splits are built
once (module-scoped) since that pass streams the 700 MB teacher file. Tokenisation
uses the debug student (Qwen2.5-0.5B), whose tokenizer is cached locally, so the
collate test runs on CPU.

    pytest tests/test_dataset.py
    python  tests/test_dataset.py
"""

import json
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.data.dataset import (
    DEFAULT_PREFIX_TEXT,
    DatasetConfig,
    _format_prompt,
    build_datasets,
    load_student_tokenizer,
    make_collate_fn,
    student_prompt,
)
from src.teacher.postprocess import is_leak_step
from src.teacher.prompt import superclass_answer
from src.teacher.view import build_view

CONFIG = DatasetConfig(debug=True)


def _skip(reason: str) -> None:
    """Skip under pytest, report and continue when run as a script."""
    print(f"SKIP: {reason}")
    if "PYTEST_CURRENT_TEST" in os.environ:
        import pytest

        pytest.skip(reason)

# Expected exclusion accounting over parsed_final (full_v6_final.jsonl, after the
# regen merge + report-reference rewrite). The regen fixed almost all the gaps.
EXPECTED_KEPT = 108_962
EXPECTED_PREEXCLUDED = 6
EXPECTED_BLOCK_MISSING = 2
EXPECTED_PARSE_FAILED = 1
EXPECTED_EMPTY_REASONING = 0

_SPLITS = None


def splits():
    global _SPLITS
    if _SPLITS is None:
        _SPLITS = build_datasets(CONFIG)
    return _SPLITS


def _read_raw_blocks(teacher_path, wanted_ids):
    """Independently read parsed_final.raw_block for a set of ecg_ids."""
    wanted = set(wanted_ids)
    out = {}
    with open(teacher_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            eid = int(d["ecg_id"])
            if eid not in wanted:
                continue
            blocks = (d.get("parsed_final") or {}).get("blocks") or {}
            out[eid] = {
                sc: (b.get("raw_block") or "").strip() for sc, b in blocks.items()
            }
            if len(out) == len(wanted):
                break
    return out


def test_example_count_matches_index_after_exclusions():
    sp = splits()
    stats = sp["train"].stats
    total = sum(len(sp[s]) for s in sp)

    assert total == stats.examples_kept == EXPECTED_KEPT, (total, stats.examples_kept)
    assert stats.preexisting_excluded_records == EXPECTED_PREEXCLUDED
    assert stats.examples_removed_block_missing == EXPECTED_BLOCK_MISSING
    assert stats.examples_removed_parse_failed == EXPECTED_PARSE_FAILED
    assert stats.examples_removed_empty_reasoning == EXPECTED_EMPTY_REASONING

    # Disjoint rules + kept sum to the working-set grid (21,793 records x 5).
    grid = 21_793 * 5
    accounted = (
        stats.examples_removed_record_missing
        + stats.examples_removed_block_missing
        + stats.examples_removed_parse_failed
        + stats.examples_removed_empty_reasoning
        + stats.examples_kept
    )
    assert accounted == grid, (accounted, grid)


def test_no_train_val_test_id_overlap():
    sp = splits()
    ids = {s: {ex.ecg_id for ex in sp[s].examples} for s in sp}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    # And the fold partition is what we asked for.
    assert {ex.strat_fold for ex in sp["train"].examples} == set(range(1, 9))
    assert {ex.strat_fold for ex in sp["val"].examples} == {9}
    assert {ex.strat_fold for ex in sp["test"].examples} == {10}


def test_embeddings_match_direct_cache_read():
    ds = splits()["train"]
    cache_dir = CONFIG.emb_cache_dir
    index = json.load(open(os.path.join(cache_dir, "index.json")))
    locations = {int(e): (int(s), int(o)) for e, s, o in index["records"]}
    shards = index["shards"]

    rng = random.Random(0)
    picks = rng.sample(range(len(ds)), 3)
    for i in picks:
        item = ds[i]
        eid = item["ecg_id"]
        shard_idx, offset = locations[eid]
        direct = np.load(
            os.path.join(cache_dir, shards[shard_idx]), mmap_mode="r"
        )[offset].astype(np.float32)
        assert item["ecg_embed"].shape == (312, 768)
        assert item["ecg_embed"].dtype == torch.float32
        assert np.array_equal(item["ecg_embed"].numpy(), direct), eid


def test_prompt_has_no_report_or_scp_content():
    ds = splits()["train"]
    rng = random.Random(1)
    checked_with_report = 0
    for i in rng.sample(range(len(ds)), 40):
        item = ds[i]
        prompt = item["task_prompt"].lower()
        rec = ds.manifest[item["ecg_id"]]

        report = (rec.get("report") or "").strip().lower()
        if len(report) > 3:
            assert report not in prompt, (item["ecg_id"], report)
            checked_with_report += 1

        # No SCP statement code is dumped into the student prompt.
        for stmt in rec.get("statements", []):
            code = stmt["code"]
            assert f" {code.lower()} " not in f" {prompt} ", (item["ecg_id"], code)

        # No annotation-pipeline vocabulary leaks in.
        for word in ("likelihood", "annotat", "diagnostic_class", "scp"):
            assert word not in prompt, (item["ecg_id"], word)

    assert checked_with_report >= 5  # the check actually exercised real reports


def test_target_from_parsed_final_and_no_supervision_phrases():
    ds = splits()["train"]
    rng = random.Random(2)
    picks = [ds[i] for i in rng.sample(range(len(ds)), 6)]
    raw = _read_raw_blocks(CONFIG.teacher_path, {it["ecg_id"] for it in picks})

    for it in picks:
        # Target is the verbatim parsed_final block, not a reconstruction.
        assert it["target"] == raw[it["ecg_id"]][it["superclass"]], it["ecg_id"]
        # parsed_final is already stripped: no supervision-pipeline tell survives.
        assert not is_leak_step(it["target"]), (it["ecg_id"], it["superclass"])
        assert it["target"].startswith("Evidence")


def test_label_matches_manifest_superclass_answer():
    ds = splits()["train"]
    rng = random.Random(3)
    for i in rng.sample(range(len(ds)), 50):
        item = ds[i]
        rec = ds.manifest[item["ecg_id"]]
        expected = superclass_answer(build_view(rec), item["superclass"])
        assert item["label"] == expected, (item["ecg_id"], item["superclass"])


class _ThinkTemplateTokenizer:
    """Minimal stand-in for a reasoning tokenizer: its generation prompt opens
    a think block, exactly as DeepSeek-R1-Distill's template does."""

    chat_template = "yes"

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        text = f"<|User|>{messages[0]['content']}"
        if add_generation_prompt:
            text += "<|Assistant|><think>\n"
        return text


def test_format_prompt_strips_the_trailing_think_block():
    """The rendered prompt must end at the assistant tag, think block removed."""
    tok = _ThinkTemplateTokenizer()
    out = _format_prompt(tok, "Is this a normal ECG?")
    assert out.endswith("<|Assistant|>"), repr(out)
    assert "<think>" not in out

    # Only a TRAILING open block goes; the same text inside the user's own content
    # is content, not template scaffolding.
    keep = _format_prompt(tok, "the report mentions <think> verbatim")
    assert "<think> verbatim" in keep
    assert keep.endswith("<|Assistant|>")


def test_masked_prompt_decodes_without_a_think_block():
    """Decode the masked (label == -100) span and assert no open think block.

    Run against the REAL student tokenizer, since the think block comes from its
    template and the debug 0.5B has none. This is the check that matters: whatever
    the template renders, what the model is actually conditioned on -- the tokens
    the loss does not supervise -- must contain no unclosed <think>.
    """
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(
            DatasetConfig().student_model, local_files_only=True
        )
    except Exception as err:                     # checkpoint not on this machine
        _skip(f"student tokenizer unavailable ({type(err).__name__}); "
              "run where MODEL_DIR is present")
        return

    ds = splits()["train"]
    batch = make_collate_fn(tok, max_length=2048)([ds[0], ds[1], ds[2]])
    input_ids, labels, attn = batch["input_ids"], batch["labels"], batch["attention_mask"]

    for r in range(input_ids.shape[0]):
        masked = (labels[r] == -100) & (attn[r] == 1)      # the prompt, not padding
        prompt_text = tok.decode(input_ids[r][masked], skip_special_tokens=False)
        assert "<think>" not in prompt_text, prompt_text[-120:]
        assert "</think>" not in prompt_text
        # and it ends where the assistant's turn begins, with nothing after it
        assert prompt_text.rstrip().endswith("<｜Assistant｜>"), prompt_text[-120:]

        # the supervised span is the target, and it carries no think markup either
        target_text = tok.decode(input_ids[r][labels[r] != -100], skip_special_tokens=False)
        assert "think" not in target_text.lower()


def test_collate_masks_prompt_tokens_only():
    ds = splits()["train"]
    tok = load_student_tokenizer(CONFIG)
    collate = make_collate_fn(tok, max_length=1024)

    batch = collate([ds[0], ds[1], ds[2], ds[3]])

    assert batch["ecg_embed"].shape == (4, 312, 768)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape

    input_ids, labels, attn = batch["input_ids"], batch["labels"], batch["attention_mask"]

    supervised = labels != -100
    assert supervised.any()
    # Every supervised position reproduces the input token (it's the target).
    assert torch.equal(input_ids[supervised], labels[supervised])

    for r in range(input_ids.shape[0]):
        row_lbl, row_attn = labels[r], attn[r]
        # The prompt leads every row, so the first real token is masked out.
        assert row_lbl[0].item() == -100
        # Padding (attention 0) is never supervised.
        pad = row_attn == 0
        assert (row_lbl[pad] == -100).all()
        # Some target tokens are supervised in every row.
        assert (row_lbl != -100).sum() > 0

    # head_label is the per-example Yes/No, aligned to the batch.
    assert batch["head_label"].tolist() == [float(ds[i]["label"]) for i in range(4)]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
