#!/usr/bin/env python
"""Tokenised length distribution over the training split (DeepSeek-R1-Distill-Qwen-14B).

Mirrors src.data.dataset.make_collate_fn exactly -- same chat-templated prompt,
same target + EOS -- but without truncation, so we see true lengths. Total length
adds the 64 resampler latents that the training step splices in front.

    python scripts/token_length_stats.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data.dataset import (
    DEFAULT_PREFIX_TEXT,
    DatasetConfig,
    _format_prompt,
    build_datasets,
    student_prompt,
)
from src.model.resampler import PerceiverResamplerConfig

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MAX_TEXT_LEN = 2048
N_LATENTS = PerceiverResamplerConfig.ecg().resampled_length  # 64
PCTLS = (50, 90, 95, 99)


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = tok.eos_token_id

    ds = build_datasets(DatasetConfig())["train"]
    print(f"train examples: {len(ds):,}")

    # Reconstruct prompt/target text straight from the example index (no embedding I/O).
    prompt_strs, targets = [], []
    for ex in ds.examples:
        rec = ds.manifest[ex.ecg_id]
        user = f"{DEFAULT_PREFIX_TEXT}\n\n" + student_prompt(
            ex.ecg_id, ex.superclass, rec.get("age"), rec.get("sex")
        )
        prompt_strs.append(_format_prompt(tok, user))
        targets.append(ex.target)

    # Batch-tokenise (fast tokenizer), add_special_tokens=False as the collate does.
    prompt_lens = np.array(
        [len(x) for x in tok(prompt_strs, add_special_tokens=False).input_ids]
    )
    target_lens = np.array(
        [len(x) + (1 if eos is not None else 0)
         for x in tok(targets, add_special_tokens=False).input_ids]
    )
    total_lens = prompt_lens + target_lens + N_LATENTS

    def row(name, a):
        qs = np.percentile(a, PCTLS)
        return (f"{name:16} {int(a.min()):>6} "
                + " ".join(f"{int(round(q)):>6}" for q in qs)
                + f" {int(a.max()):>6}  {float(a.mean()):8.1f}")

    print(f"\nlatents added to every total: {N_LATENTS}\n")
    print(f"{'field':16} {'min':>6} {'p50':>6} {'p90':>6} {'p95':>6} {'p99':>6} {'max':>6}  {'mean':>8}")
    print("-" * 72)
    print(row("prompt", prompt_lens))
    print(row("target", target_lens))
    print(row("total(+64)", total_lens))

    print(f"\nexceeding MAX_TEXT_LEN={MAX_TEXT_LEN}:")
    for name, a in (("prompt", prompt_lens), ("target", target_lens), ("total(+64)", total_lens)):
        n = int((a > MAX_TEXT_LEN).sum())
        print(f"  {name:16} {n:>6} / {len(a):,}  ({100 * n / len(a):.3f}%)")


if __name__ == "__main__":
    main()
