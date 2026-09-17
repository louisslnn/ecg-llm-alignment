#!/usr/bin/env python
"""Reproduce the pre-training baseline: three generation conditions.

Establishes that an UNTRAINED aligner conveys nothing from the ECG encoder to
the LLM. That is the number training has to move.

  A. text only, no soft prompt      -> generic refusal
  B. soft prompt, no labels         -> generic refusal, IDENTICAL across ECGs
  C. soft prompt + labels as text   -> fluent summary, but the LLM is only
                                       paraphrasing the labels; the embedding
                                       is redundant (kept as an ablation)

Usage:
    python scripts/run_baseline.py \
        --ptb ~/data/ptbxl \
        --ckpts ~/data/ckpts \
        --ecgfm-repo ~/code/ECG-FM \
        --device cpu \
        --n-records 4
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import encoder as E
from src import generate as G
from src import ptbxl_data as D
from src.modules import Aligner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptb", required=True, help="PTB-XL root")
    ap.add_argument("--ckpts", required=True, help="where to store/find ECG-FM ckpt")
    ap.add_argument("--ecgfm-repo", required=True, help="cloned ECG-FM repo root")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default=None, help="cpu | cuda | mps")
    ap.add_argument("--n-records", type=int, default=4)
    ap.add_argument("--n-queries", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="baseline_results.json")
    ap.add_argument("--stratify", action="store_true",
                    help="one record per diagnostic superclass instead of the first N")
    ap.add_argument("--first-segment-only", action="store_true",
                    help="keep only the first 5 s segment of each record")
    args = ap.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("PRE-TRAINING BASELINE - untrained projection + resampler")
    print("=" * 70)
    print("device:", device)

    # ------------------------------------------------------------------ data
    df = D.load_metadata(args.ptb)
    sub = D.resolve_local(df, args.ptb, rate="hr")
    print(f"\n[data] {len(sub)} records available locally")
    print("[data] superclass spread:",
          sub.superclass.explode().value_counts().to_dict())

    if args.stratify:
        sel = sub.groupby(sub.superclass.str[0], group_keys=False).head(1)
        print(f"[data] stratified: {len(sel)} records")
        print("[data] selected  :", [s[0] for s in sel.superclass])
    else:
        sel = sub.head(args.n_records)
    loader = D.make_loader(sel, batch_size=len(sel))
    source, inps = next(iter(loader))

    seg_per_rec = source.shape[0] // len(inps)
    if args.first_segment_only:
        keep = torch.arange(0, source.shape[0], seg_per_rec)
        source = source[keep]
        seg_to_rec = list(range(len(inps)))
        print(f"[data] keeping 1 of {seg_per_rec} segments per record")
    else:
        seg_to_rec = [i // seg_per_rec for i in range(source.shape[0])]

    print("[data] source tensor:", tuple(source.shape))

    # --------------------------------------------------------------- encoder
    print("\n[encoder] loading ECG-FM ...")
    ckpt = E.download_checkpoint(args.ckpts)
    model = E.load_encoder(ckpt, device)
    label_names = E.load_label_names(args.ecgfm_repo)

    enc = E.encode(model, source, device)
    h, logits = enc["encoder_out"], enc["logits"]
    print("[encoder] encoder_out:", tuple(h.shape), "(pre-pooling)")
    print("[encoder] logits     :", tuple(logits.shape))
    print(f"[encoder] reduction  : {source.shape[-1] / h.shape[1]:.1f}x")

    # ------------------------------------------------------------------- LLM
    print(f"\n[llm] loading {args.llm} ...")
    tok, llm = G.load_llm(args.llm, device)
    d_llm = llm.config.hidden_size
    print("[llm] hidden size:", d_llm)

    # --------------------------------------------------------------- aligner
    aligner = Aligner(
        d_enc=h.shape[-1], d_llm=d_llm, n_queries=args.n_queries
    ).to(device).eval()
    print(f"\n[aligner] trainable parameters: {aligner.n_trainable():,}")
    print("[aligner] RANDOMLY INITIALISED - this is the point of the baseline")

    with torch.no_grad():
        soft = aligner(h)

    chain = f"{tuple(source.shape)} -> {tuple(h.shape)} -> {tuple(soft.shape)}"
    print("[aligner] chain:", chain)

    # ---------------------------------------------------------------- decode
    tops = E.top_labels(logits, label_names, k=5)
    n_show = soft.shape[0]

    results = {
        "device": device,
        "llm": args.llm,
        "shape_chain": chain,
        "trainable_params": aligner.n_trainable(),
        "reduction_factor": source.shape[-1] / h.shape[1],
        "segments": [],
    }

    print("\n" + "=" * 70)
    print("GENERATION")
    print("=" * 70)

    for i in range(n_show):
        rec = seg_to_rec[i]
        entry = {
            "segment": i,
            "record_superclass": sel.iloc[rec].superclass,
            "ecgfm_top5": [(n, round(float(p), 3)) for n, p in tops[i]],
        }

        entry["A_text_only"] = G.generate(
            tok, llm, None, "Anything wrong with this ECG?",
            max_new_tokens=60, device=device,
        )
        entry["B_soft_prompt"] = G.generate(
            tok, llm, soft[i:i + 1], "Anything wrong with this ECG?",
            max_new_tokens=60, device=device,
        )
        q_lab = (f"Automated model findings: {G.format_findings(tops[i])}\n\n"
                 "Summarise the ECG in two sentences for a clinician.")
        entry["C_soft_prompt_plus_labels"] = G.generate(
            tok, llm, soft[i:i + 1], q_lab, max_new_tokens=60, device=device,
        )

        results["segments"].append(entry)

        print(f"\n--- segment {i} | ground truth: {entry['record_superclass']} ---")
        print("  ECG-FM top-3 :",
              ", ".join(f"{n} {p}" for n, p in entry["ecgfm_top5"][:3]))
        print("  A text only  :", entry["A_text_only"][:120].replace("\n", " "))
        print("  B soft prompt:", entry["B_soft_prompt"][:120].replace("\n", " "))
        print("  C + labels   :", entry["C_soft_prompt_plus_labels"][:120].replace("\n", " "))

    # ------------------------------------------------------- the actual result
    b_outputs = [s["B_soft_prompt"] for s in results["segments"]]
    n_unique = len(set(b_outputs))
    results["condition_B_unique_outputs"] = n_unique
    results["condition_B_n_segments"] = len(b_outputs)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Condition B: {n_unique} unique output(s) across {len(b_outputs)} "
          f"different ECG segments.")
    if n_unique == 1:
        print("  -> The soft prompt carries NO information the LLM can read.")
        print("     Different waveforms produce byte-identical text.")
    elif n_unique < len(b_outputs):
        print("  -> Outputs largely collapse; the soft prompt conveys almost nothing.")
    else:
        print("  -> Outputs differ; inspect whether the differences are meaningful.")
    print("\nThis is the pre-training baseline. Training the projection and")
    print(f"resampler ({aligner.n_trainable():,} params) is what has to move it.")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()