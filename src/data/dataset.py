"""Training dataset for the alignment student: one example per (ecg_id, superclass).

The unit of supervision is a single binary question about one ECG: "does this ECG
show <superclass>?". With five superclasses per record and 21,793 records that is
~108,912 candidate examples, filtered at init to the (record, superclass) pairs
that have a usable teacher block.

What one example carries:
  * ``ecg_embed``   the cached ECG-FM embedding (312, 768), fp32, read from a shard
                    by mmap. Shards are opened once at init.
  * ``task_prompt`` the question the STUDENT sees: a deterministic paraphrase for
                    that (ecg_id, superclass) plus patient age/sex. It contains
                    nothing the teacher saw -- no report, no SCP statements, no
                    annotations. This is the whole point: the student must recover
                    the answer from the signal, not read it off the labels.
  * ``target``      the teacher's Evidence/Reasoning/Conclusion text for that
                    superclass, taken verbatim from ``parsed_final`` (the parse of
                    the report-reference-rewritten ``final_content``; never
                    re-parsed here). Loss is computed on this.
  * ``label``       the known Yes/No boolean for (record, superclass), for the
                    linear head that attaches alongside the LM loss.
  * ``ecg_id`` / ``superclass`` / ``strat_fold`` for bookkeeping and splitting.

Splits are by PTB-XL fold (train 1-8, val 9, test 10 by default), exposed as a
parameter rather than hardcoded at the call site.

A note on the paraphrase: the teacher currently renders one fixed phrasing per
superclass (see :mod:`src.teacher.prompt`), so there is no teacher paraphrase to
mirror literally. The deterministic selection specified for the student --
``random.Random(f"{ecg_id}_{superclass}")`` over a small paraphrase bank -- lives
here; using the same seed recipe keeps the student prompt reproducible per example.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..manifest import EXCLUDED_ECG_IDS
from ..teacher.prompt import SUPERCLASS_NAME, SUPERCLASSES, superclass_answer
from ..teacher.view import _SEX_LABELS, build_view

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PTB-XL fold convention: folds 1-8 train, 9 validation, 10 test. Passed in, not
# assumed downstream, so an ablation can repartition without touching call sites.
DEFAULT_FOLD_SPLITS: Dict[str, frozenset] = {
    "train": frozenset(range(1, 9)),
    "val": frozenset({9}),
    "test": frozenset({10}),
}


# --------------------------------------------------------------------------- #
# Student prompt: question paraphrase + patient context. Never label content.  #
# --------------------------------------------------------------------------- #

# NORM asks about normality directly; the four abnormal superclasses ask whether
# the recording shows evidence of the finding. Each bank preserves the question's
# meaning; only the wording varies, selected deterministically per example.
_NORM_PARAPHRASES: List[str] = [
    "Is this a normal ECG?",
    "Would you classify this ECG as normal?",
    "Does this ECG appear normal?",
    "Is this recording within normal limits?",
]
_ABNORMAL_TEMPLATES: List[str] = [
    "Does this ECG show evidence of {name}?",
    "Are there signs of {name} on this ECG?",
    "Is there evidence of {name} in this recording?",
    "Does this tracing indicate {name}?",
]


def question_paraphrases(superclass: str) -> List[str]:
    if superclass not in SUPERCLASS_NAME:
        raise ValueError(f"unknown superclass {superclass!r}")
    if superclass == "NORM":
        return list(_NORM_PARAPHRASES)
    return [t.format(name=SUPERCLASS_NAME[superclass]) for t in _ABNORMAL_TEMPLATES]


def render_question(ecg_id: int, superclass: str) -> str:
    """Deterministic paraphrase for one (ecg_id, superclass), seeded per example."""
    rng = random.Random(f"{ecg_id}_{superclass}")
    return rng.choice(question_paraphrases(superclass))


def _patient_context(age: Optional[float], sex_code: Optional[int]) -> str:
    sex = _SEX_LABELS.get(sex_code, "unknown sex")
    age_str = "unknown" if age is None else str(int(age))
    return f"Patient: age {age_str}, sex {sex}."


def student_prompt(
    ecg_id: int, superclass: str, age: Optional[float], sex_code: Optional[int]
) -> str:
    """The full student-visible prompt for one example (question + patient context)."""
    return f"{render_question(ecg_id, superclass)}\n{_patient_context(age, sex_code)}"


# --------------------------------------------------------------------------- #
# Embedding cache: id -> (shard, offset), shards mmap'd once.                  #
# --------------------------------------------------------------------------- #


class EmbeddingCache:
    """Read-only view over the sharded embedding cache described by index.json."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        with open(os.path.join(cache_dir, "index.json")) as f:
            index = json.load(f)
        self.token_dim = tuple(index["token_dim"])  # (312, 768)
        # id -> (shard_idx, offset), built once.
        self.locations: Dict[int, tuple] = {
            int(eid): (int(shard), int(off)) for eid, shard, off in index["records"]
        }
        # Open every shard once, memory-mapped (no full read).
        self.shards: List[np.ndarray] = [
            np.load(os.path.join(cache_dir, name), mmap_mode="r")
            for name in index["shards"]
        ]

    def __contains__(self, ecg_id: int) -> bool:
        return int(ecg_id) in self.locations

    def get(self, ecg_id: int) -> np.ndarray:
        """(312, 768) fp32 copy of one record's embedding, materialised from mmap."""
        shard_idx, offset = self.locations[int(ecg_id)]
        return np.asarray(self.shards[shard_idx][offset], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Config, exclusion accounting, example records.                              #
# --------------------------------------------------------------------------- #


@dataclass
class DatasetConfig:
    emb_cache_dir: str = os.path.join(REPO_ROOT, "data", "emb_cache")
    manifest_path: str = os.path.join(REPO_ROOT, "data", "ptbxl", "manifest.jsonl")
    teacher_path: str = os.path.join(REPO_ROOT, "data", "teacher", "full_v6_final.jsonl")

    # The student LLM. `debug` swaps in the 0.5B model so tokenisation (and the
    # rest of the training loop) runs on CPU locally.
    debug: bool = False
    student_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    debug_student_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    @property
    def student_model_name(self) -> str:
        return self.debug_student_model if self.debug else self.student_model


@dataclass
class ExclusionStats:
    """How many examples each exclusion rule removed, reported at init.

    Rules are applied in precedence order so the counts are disjoint and sum with
    ``examples_kept`` to the full ``records * len(SUPERCLASSES)`` grid.
    """

    preexisting_excluded_records: int = 0     # the 6 label-less records
    records_missing_from_teacher: int = 0
    examples_removed_record_missing: int = 0  # = records_missing_from_teacher * 5
    examples_removed_block_missing: int = 0
    examples_removed_parse_failed: int = 0
    examples_removed_empty_reasoning: int = 0
    examples_kept: int = 0

    def log(self) -> None:
        logger.info(
            "dataset exclusions | pre-excluded records: %d (%d examples) | "
            "records missing from teacher: %d (%d examples) | block missing: %d | "
            "parse failed: %d | empty reasoning: %d | KEPT: %d",
            self.preexisting_excluded_records,
            self.preexisting_excluded_records * len(SUPERCLASSES),
            self.records_missing_from_teacher,
            self.examples_removed_record_missing,
            self.examples_removed_block_missing,
            self.examples_removed_parse_failed,
            self.examples_removed_empty_reasoning,
            self.examples_kept,
        )


@dataclass
class Example:
    ecg_id: int
    superclass: str
    strat_fold: int
    target: str
    label: bool


@dataclass
class _Build:
    """The shared, split-independent state built once from disk."""

    cache: EmbeddingCache
    manifest: Dict[int, Dict[str, Any]]
    all_examples: List[Example]
    stats: ExclusionStats


def _load_manifest_raw(path: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[int(rec["ecg_id"])] = rec
    return out


# Per-block status when a block is NOT usable. Precedence: missing beats a parse
# failure beats an empty Reasoning section.
_MISSING, _PARSE_FAILED, _EMPTY_REASONING = "missing", "parse_failed", "empty_reasoning"


def _extract_teacher_blocks(path: str):
    """Stream the teacher jsonl once.

    Returns ``(usable, status, teacher_ids)`` where ``usable[ecg_id][sc]`` is the
    verbatim ``raw_block`` for usable blocks, and ``status[ecg_id][sc]`` is the
    exclusion reason for the rest.

    Reads ``parsed_final`` -- the parse of ``final_content``, after the
    report-reference rewrite -- which is the canonical target for training.
    """
    usable: Dict[int, Dict[str, str]] = {}
    status: Dict[int, Dict[str, str]] = {}
    teacher_ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            eid = int(d["ecg_id"])
            teacher_ids.add(eid)
            pc = d.get("parsed_final") or {}
            blocks = pc.get("blocks") or {}
            unparseable = pc.get("unparseable_conclusion") or []
            if isinstance(unparseable, dict):
                unparseable = set(unparseable.keys())
            else:
                unparseable = set(unparseable)
            missing = set(pc.get("missing") or [])

            ok: Dict[str, str] = {}
            bad: Dict[str, str] = {}
            for sc in SUPERCLASSES:
                b = blocks.get(sc)
                if b is None or sc in missing:
                    bad[sc] = _MISSING
                    continue
                conclusion = b.get("conclusion")
                raw_block = (b.get("raw_block") or "").strip()
                if conclusion is None or sc in unparseable or not raw_block:
                    bad[sc] = _PARSE_FAILED
                    continue
                if not (b.get("reasoning") or "").strip():
                    bad[sc] = _EMPTY_REASONING
                    continue
                ok[sc] = raw_block
            usable[eid] = ok
            status[eid] = bad
    return usable, status, teacher_ids


def _build_all(config: DatasetConfig) -> _Build:
    cache = EmbeddingCache(config.emb_cache_dir)

    raw_manifest = _load_manifest_raw(config.manifest_path)
    stats = ExclusionStats()
    stats.preexisting_excluded_records = sum(
        1 for eid in EXCLUDED_ECG_IDS if eid in raw_manifest
    )
    manifest = {eid: rec for eid, rec in raw_manifest.items() if eid not in EXCLUDED_ECG_IDS}

    usable, status, teacher_ids = _extract_teacher_blocks(config.teacher_path)

    all_examples: List[Example] = []
    for eid, rec in manifest.items():
        if eid not in cache:
            # No embedding: cannot form any example for this record. (In the
            # current data the manifest working set and cache coincide exactly.)
            stats.records_missing_from_teacher += 0  # not a teacher issue; skip silently
            continue
        if eid not in teacher_ids:
            stats.records_missing_from_teacher += 1
            stats.examples_removed_record_missing += len(SUPERCLASSES)
            continue

        view = build_view(rec)
        record_status = status.get(eid, {})
        record_usable = usable.get(eid, {})
        strat_fold = int(rec["strat_fold"])
        for sc in SUPERCLASSES:
            raw_block = record_usable.get(sc)
            if raw_block is None:
                reason = record_status.get(sc, _MISSING)
                if reason == _MISSING:
                    stats.examples_removed_block_missing += 1
                elif reason == _PARSE_FAILED:
                    stats.examples_removed_parse_failed += 1
                else:
                    stats.examples_removed_empty_reasoning += 1
                continue
            all_examples.append(
                Example(
                    ecg_id=eid,
                    superclass=sc,
                    strat_fold=strat_fold,
                    target=raw_block,
                    label=superclass_answer(view, sc),
                )
            )

    stats.examples_kept = len(all_examples)
    stats.log()
    return _Build(cache=cache, manifest=manifest, all_examples=all_examples, stats=stats)


# --------------------------------------------------------------------------- #
# The Dataset.                                                                 #
# --------------------------------------------------------------------------- #


class ECGSuperclassDataset(Dataset):
    """One split of the (ecg_id, superclass) examples.

    Construct a single split directly, or all three at once with
    :func:`build_datasets` to parse the teacher file only once.
    """

    def __init__(
        self,
        split: str,
        config: Optional[DatasetConfig] = None,
        fold_splits: Optional[Dict[str, frozenset]] = None,
        _build: Optional[_Build] = None,
    ):
        self.split = split
        self.config = config or DatasetConfig()
        self.fold_splits = fold_splits or DEFAULT_FOLD_SPLITS
        if split not in self.fold_splits:
            raise ValueError(f"unknown split {split!r}; have {list(self.fold_splits)}")

        build = _build or _build_all(self.config)
        self.cache = build.cache
        self.manifest = build.manifest
        self.stats = build.stats

        folds = self.fold_splits[split]
        self.examples: List[Example] = [
            e for e in build.all_examples if e.strat_fold in folds
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ex = self.examples[i]
        rec = self.manifest[ex.ecg_id]
        embed = torch.from_numpy(self.cache.get(ex.ecg_id))  # (312, 768) fp32
        return {
            "ecg_embed": embed,
            "task_prompt": student_prompt(
                ex.ecg_id, ex.superclass, rec.get("age"), rec.get("sex")
            ),
            "target": ex.target,
            "label": ex.label,
            "ecg_id": ex.ecg_id,
            "superclass": ex.superclass,
            "strat_fold": ex.strat_fold,
        }


def build_datasets(
    config: Optional[DatasetConfig] = None,
    fold_splits: Optional[Dict[str, frozenset]] = None,
) -> Dict[str, ECGSuperclassDataset]:
    """Build every split from a single pass over the manifest and teacher file."""
    config = config or DatasetConfig()
    fold_splits = fold_splits or DEFAULT_FOLD_SPLITS
    build = _build_all(config)
    return {
        split: ECGSuperclassDataset(split, config, fold_splits, _build=build)
        for split in fold_splits
    }


# --------------------------------------------------------------------------- #
# Tokenisation / collation.                                                    #
# --------------------------------------------------------------------------- #

DEFAULT_PREFIX_TEXT = "You are given an embedded representation of a 12-lead ECG."


def load_student_tokenizer(config: Optional[DatasetConfig] = None):
    """Load the student tokenizer named by the config (0.5B when ``debug``)."""
    from transformers import AutoTokenizer

    config = config or DatasetConfig()
    return AutoTokenizer.from_pretrained(config.student_model_name)


# A reasoning template's generation prompt opens a think block that our targets
# never close. DeepSeek-R1-Distill renders "<|Assistant|><think>\n" (real tokens:
# <｜Assistant｜>, <think>, \n), so without this the student is asked to start
# inside a think block and then emit Evidence/Reasoning/Conclusion/<END> + EOS with
# no </think> anywhere -- a format it was distilled never to produce, and one the
# loss then teaches it to produce. ChatHealthAI removes the block (confirmed with
# Bo Hong), so the prompt ends at the assistant tag and the target starts there.
# Anchored to the very end of the string, so a "<think>" inside user content is
# left alone; a no-op for templates that add none (e.g. the 0.5B debug student).
_TRAILING_THINK_RE = re.compile(r"<think>\s*\Z")


def _format_prompt(tokenizer, user_content: str) -> str:
    """Wrap the student prompt in the chat template, turn left open for the target.

    The rendered prompt ends at the assistant tag with NO open think block; see
    :data:`_TRAILING_THINK_RE`. Training and evaluation both build prompts here, so
    they cannot drift apart on this.
    """
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return _TRAILING_THINK_RE.sub("", rendered)
    return f"<|user|>\n{user_content}\n<|assistant|>\n"


def make_collate_fn(
    tokenizer,
    max_length: int = 1024,
    prefix_text: str = DEFAULT_PREFIX_TEXT,
) -> Callable[[Sequence[Dict[str, Any]]], Dict[str, Any]]:
    """Build a collate function bound to a tokenizer.

    Produces a batch dict with the embeddings stacked and the prompt+target
    tokenised into ``input_ids`` / ``attention_mask`` / ``labels``. Prompt tokens
    (and padding) are masked to -100 in ``labels`` so the LM loss lands only on the
    target. Right-padded. The resampler latents are NOT prepended here -- the
    training step splices them into the embeddings.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        ecg_embed = torch.stack([b["ecg_embed"] for b in batch], dim=0)

        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        for b in batch:
            user_content = f"{prefix_text}\n\n{b['task_prompt']}"
            prompt_str = _format_prompt(tokenizer, user_content)
            prompt_ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
            target_ids = tokenizer(b["target"], add_special_tokens=False).input_ids
            if eos_id is not None:
                target_ids = target_ids + [eos_id]

            ids = (prompt_ids + target_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
            input_ids_list.append(ids)
            labels_list.append(labels)

        max_len = max(len(x) for x in input_ids_list)
        B = len(batch)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, (ids, lbl) in enumerate(zip(input_ids_list, labels_list)):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            labels[i, :n] = torch.tensor(lbl, dtype=torch.long)
            attention_mask[i, :n] = 1

        return {
            "ecg_embed": ecg_embed,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "head_label": torch.tensor(
                [float(b["label"]) for b in batch], dtype=torch.float32
            ),
            "ecg_id": [b["ecg_id"] for b in batch],
            "superclass": [b["superclass"] for b in batch],
            "strat_fold": [b["strat_fold"] for b in batch],
        }

    return collate
