#!/usr/bin/env python
"""Reproduce the pre-training baseline: three generation conditions.

Establishes that an UNTRAINED aligner contributes nothing, which is the
number that training has to move.

  A. text only, no labels          -> generic refusal
  B. soft prompt, no labels        -> generic refusal (identical across ECGs)
  C. soft prompt + labels as text  -> fluent summary, but the LLM is only
                                      paraphrasing the labels; the embedding
                                      is redundant. Kept as an ablation to
                                      show WHY labels must not be serialised.

Usage:
    python scripts/run_baseline.py --ptb /path/to/ptbxl --ckpts /path/to/ckpts \
        --ecgfm-repo /path/to/ECG-FM --n-records 4
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import data as D
from src import encoder as E
from src import generate as G
from src.modules import Aligner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptb", required=True, help="PTB-XL root")
    ap.add_argument("--ckpts", required=True, help="where to store/find ECG-FM ckpt")
    ap.add_argument("--ecgfm-repo", required=True, help="cloned ECG-FM repo root")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n-records", type=int, default=4)
    ap.add_argument("--n-queries", type=int, default=32)
    ap.add_argument("--out", default="baseline_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # --- data
    df = D.load_metadata(args.ptb)
    sub = D.resolve_local(df, args.ptb, rate="hr")
    print(f"{len(sub)} records available locally")
    print(sub.superclass.explode().value_counts().to_dict())

    loader = D.make_loader(sub.head(args.n_records), batch_size=args.n_records)
    source, inps = next(iter(loader))
    print("source:", tuple(source.shape))

    # --- encoder
    ckpt = E.download_checkpoint(args.ckpts)
    model = E.load_encoder(ckpt, device)
    label_names = E.load_label_names(args.ecgfm_repo)

    enc = E.encode(model, source, device)
    h, logits = enc["encoder_out"], enc["logits"]
    print("encoder_out:", tuple(h.shape))
    print("logits     :", tuple(logits.shape))

    # --- untrained aligner
    aligner = Aligner(d_enc=h.shape[-1], d_llm=None, n_queries=args.n_queries)
    # d_llm must match the LLM; resolve after loading it
    tok, llm = G.load_llm(args.llm, device)
    d_llm = llm.config.hidden_size
    aligner = Aligner(d_enc=h.shape[-1], d_llm=d_llm, n_queries=args.n_queries).to(device).eval()
    print(f"aligner trainable params: {aligner.n_trainable():,}")

    with torch.no_grad():
        soft = aligner(h)
    print(f"chain: {tuple(source.shape)} -> {tuple(h.shape)} -> {tuple(soft.shape)}")

    tops = E.top_labels(logits, label_names, k=5)

    results = {
        "shape_chain": {
            "source": list(source.shape),
            "encoder_out": list(h.shape),
            "soft_prompt": list(soft.shape),
        },
        "trainable_params": aligner.n_trainable(),
        "segments": [],
    }

    for i in range(min(4, soft.shape[0])):
        rec = i // 2
        entry = {
            "segment": i,
            "record_superclass": sub.iloc[rec].superclass,
            "ecgfm_top5": [(n, float(p)) for n, p in tops[i]],
        }
        entry["A_text_only"] = G.generate(
            tok, llm, None, "Anything wrong with this ECG?", device=device
        )
        entry["B_soft_prompt"] = G.generate(
            tok, llm, soft[i : i + 1], "Anything wrong with this ECG?", device=device
        )
        q_labels = (
            f"Automated model findings: {G.format_findings(tops[i])}\n\n"
            "Summarise the ECG in two sentences for a clinician."
        )
        entry["C_soft_prompt_plus_labels"] = G.generate(
            tok, llm, soft[i : i + 1], q_labels, device=device
        )
        results["segments"].append(entry)

        print(f"\n=== segment {i} (true: {entry['record_superclass']}) ===")
        print("A text only    :", entry["A_text_only"][:160])
        print("B soft prompt  :", entry["B_soft_prompt"][:160])
        print("C + labels     :", entry["C_soft_prompt_plus_labels"][:160])

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
