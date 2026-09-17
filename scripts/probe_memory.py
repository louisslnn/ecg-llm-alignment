#!/usr/bin/env python
"""Answer one question before any training run: what fits on 2x A100?

Loads the frozen DeepSeek-R1-Distill-Qwen-14B student in bf16, sharded across the
visible GPUs, builds the 843M task-aware resampler, and runs ONE forward + ONE
backward on a real batch from the dataset -- the exact tensor path training takes,
gradients flowing through the frozen backbone into the resampler. Nothing is
optimised; this measures peak memory, not learning.

It sweeps batch sizes {1, 2, 4} x gradient-checkpointing {off, on} x optimizer
{off, on}, catches OOM per configuration and keeps going, then prints a
peak-memory table. The optimizer=on rows allocate AdamW over the resampler and
take one real optimizer.step(), so the fp32 moments are materialised, not merely
declared -- that is the true training peak, shown alongside the fwd/bwd-only one.

After every backward it verifies the gradient actually traversed the whole frozen
14B into the resampler (bio_projection and latent_queries grads present, finite,
non-zero) and that no frozen LLM parameter carries a grad. A truncated backward
would make every memory number meaningless, so this fails the probe loudly.

The composition per step (mixed precision, bf16 autocast over an fp32 resampler):
    tok_embeds = student.embed(input_ids)            # frozen, bf16
    latents    = resampler(ecg_embed, tok_embeds)    # (B, 64, 5120), trainable
    inputs     = cat([latents, tok_embeds])          # latents masked out of loss
    loss       = student(inputs_embeds=inputs, labels=...)   # LM loss
    loss.backward()

The checkpoint must already be on disk (compute nodes have no internet); point
--model-dir at the pre-downloaded cache and this loads with local_files_only=True.

Usage (on a compute node, inside the venv):
    python scripts/probe_memory.py --model-dir $MODEL_DIR
    python scripts/probe_memory.py --model-dir $MODEL_DIR --batch-size 2 --grad-checkpointing on

Caveat: peak here is model weights + activations + resampler weights/grads. It
does NOT include optimizer state; AdamW over the 843M resampler adds ~6.7 GB of
fp32 moments on GPU 0 in a real run. Read the numbers with that headroom in mind.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data.dataset import (  # noqa: E402
    DatasetConfig,
    build_datasets,
    make_collate_fn,
)
from src.model.resampler import PerceiverResamplerConfig, Resampler  # noqa: E402

GiB = 1024 ** 3


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True,
                    help="local path to the pre-downloaded student checkpoint")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="pin a single batch size; default sweeps 1, 2, 4")
    ap.add_argument("--grad-checkpointing", choices=["on", "off"], default=None,
                    help="pin checkpointing; default sweeps both")
    ap.add_argument("--with-optimizer", choices=["on", "off"], default=None,
                    help="pin the optimizer dimension; default sweeps both "
                         "(on = materialise AdamW fp32 moments via one optimizer.step)")
    ap.add_argument("--max-length", type=int, default=1024,
                    help="token cap for the collated prompt+target")
    ap.add_argument("--emb-cache", default=None, help="override embedding cache dir")
    ap.add_argument("--manifest", default=None, help="override manifest.jsonl path")
    ap.add_argument("--teacher", default=None, help="override teacher jsonl path")
    return ap.parse_args()


def gpu_reset():
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def gpu_peaks():
    return [torch.cuda.max_memory_allocated(i) / GiB for i in range(torch.cuda.device_count())]


def gpu_resident():
    return [torch.cuda.memory_allocated(i) / GiB for i in range(torch.cuda.device_count())]


def load_student(model_dir):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",           # shard across every visible GPU
        local_files_only=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False   # required for gradient checkpointing / bwd
    return model


class GradFlowError(RuntimeError):
    """Backward did not reach the resampler as required. Fatal, not an OOM."""


def check_gradients(model, resampler):
    """Prove the backward traversed the frozen 14B into the resampler.

    latent_queries is the first tensor in the whole stack, so a non-zero grad on
    it can only mean the chain loss -> every LLM layer -> spliced latents -> every
    resampler block -> latent_queries stayed connected. bio_projection proves the
    ECG stream path too. Returns the two grad norms; raises on any failure.
    """
    targets = {
        "bio_projection.weight": resampler.bio_projection.weight,
        "latent_queries": resampler.perceiver_resampler.latent_queries,
    }
    norms = {}
    for name, p in targets.items():
        g = p.grad
        if g is None:
            raise GradFlowError(f"backward did not reach {name}: grad is None "
                                "(truncated backward)")
        if not bool(torch.isfinite(g).all()):
            raise GradFlowError(f"{name} grad is non-finite")
        norm = g.detach().float().norm().item()
        if norm == 0.0:
            raise GradFlowError(f"{name} grad norm is zero: backward stopped early")
        norms[name] = norm

    leaked = [n for n, p in model.named_parameters() if p.grad is not None]
    if leaked:
        raise GradFlowError(
            f"{len(leaked)} frozen LLM param(s) carry a grad, e.g. {leaked[:3]}"
        )
    return norms


def one_step(model, resampler, batch, dev):
    """One forward + backward, then verify the backward reached the resampler.

    Raises on OOM (caller handles) or GradFlowError (fatal). Returns loss + norms.
    """
    resampler.zero_grad(set_to_none=True)

    ecg = batch["ecg_embed"].to(dev)
    input_ids = batch["input_ids"].to(dev)
    attn = batch["attention_mask"].to(dev)
    labels = batch["labels"].to(dev)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tok_embeds = model.get_input_embeddings()(input_ids)          # (B, L, 5120)
        latents = resampler(ecg, task_embeddings=tok_embeds, task_mask=attn.bool())
        B, Lq, _ = latents.shape

        # The latents are the prefix; the spliced tensor must carry their grad
        # through to the LLM, or the whole backward path is dead on arrival.
        full_embeds = torch.cat([latents, tok_embeds], dim=1)
        if not full_embeds.requires_grad:
            raise GradFlowError("spliced inputs_embeds.requires_grad is False: "
                                "the resampler prefix is detached from the graph")
        full_attn = torch.cat(
            [torch.ones(B, Lq, dtype=attn.dtype, device=dev), attn], dim=1
        )
        full_labels = torch.cat(
            [torch.full((B, Lq), -100, dtype=labels.dtype, device=dev), labels], dim=1
        )
        # use_cache=False both on config (set at load) and on the call itself.
        assert model.config.use_cache is False, "use_cache must be False for backward"
        out = model(
            inputs_embeds=full_embeds,
            attention_mask=full_attn,
            labels=full_labels,
            use_cache=False,
        )
        loss = out.loss

    loss.backward()
    norms = check_gradients(model, resampler)
    return {"loss": float(loss.detach()), "norms": norms}


def is_oom(err: RuntimeError) -> bool:
    return "out of memory" in str(err).lower()


def run_config(model, resampler, make_batch, dev, bs, ckpt, opt):
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()

    gpu_reset()
    row = {"bs": bs, "ckpt": ckpt, "opt": opt, "status": "OK", "loss": None,
           "peaks": None, "seq_len": None, "norms": None}
    batch = None
    optimizer = None
    try:
        batch = make_batch(bs)
        row["seq_len"] = int(batch["input_ids"].shape[1])
        result = one_step(model, resampler, batch, dev)
        row["loss"] = result["loss"]
        row["norms"] = result["norms"]
        if opt:
            # Allocate AdamW and take one real step so the fp32 moments
            # (exp_avg + exp_avg_sq, ~6.7 GiB over the 843M params) are actually
            # materialised, not just declared. Freed in `finally` so the
            # optimizer=off rows measure clean.
            optimizer = torch.optim.AdamW(resampler.parameters(), lr=1e-5)
            optimizer.step()
        row["peaks"] = gpu_peaks()
    except RuntimeError as err:
        if not is_oom(err):
            raise
        row["status"] = "OOM"
    finally:
        del optimizer
        resampler.zero_grad(set_to_none=True)
        del batch
        torch.cuda.empty_cache()
    return row


def print_table(rows, n_gpus, static):
    print("\n" + "=" * 64)
    print("PEAK MEMORY BY CONFIGURATION  (GiB, includes model + activations)")
    print("=" * 64)
    header = f"{'batch':>5}  {'ckpt':>4}  {'opt':>4}  {'seqlen':>6}  {'status':>6}  " + \
             "  ".join(f"GPU{i:>2}" for i in range(n_gpus))
    print(header)
    print("-" * len(header))
    for r in rows:
        peaks = r["peaks"]
        cells = (
            "  ".join(f"{p:6.1f}" for p in peaks)
            if peaks else "  ".join(f"{'-':>6}" for _ in range(n_gpus))
        )
        print(f"{r['bs']:>5}  {('on' if r['ckpt'] else 'off'):>4}  "
              f"{('on' if r['opt'] else 'off'):>4}  "
              f"{(r['seq_len'] if r['seq_len'] else '-'):>6}  {r['status']:>6}  {cells}")
    print("-" * len(header))
    print("model resident (weights only): " + "  ".join(f"{s:.1f}" for s in static) + " GiB")
    print("opt=on rows include AdamW fp32 moments (true training peak); "
          "opt=off rows are forward/backward only")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible; this probe must run on a GPU node")

    n_gpus = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    print(f"visible GPUs: {n_gpus} -> {names}")

    # --- dataset: one real batch source -----------------------------------
    cfg_kwargs = {}
    if args.emb_cache:
        cfg_kwargs["emb_cache_dir"] = args.emb_cache
    if args.manifest:
        cfg_kwargs["manifest_path"] = args.manifest
    if args.teacher:
        cfg_kwargs["teacher_path"] = args.teacher
    data_cfg = DatasetConfig(**cfg_kwargs)

    print("loading tokenizer + dataset ...")
    tokenizer = load_student_tokenizer_from_dir(args.model_dir)
    train = build_datasets(data_cfg)["train"]
    collate = make_collate_fn(tokenizer, max_length=args.max_length)

    max_bs = args.batch_size or 4
    items = [train[i] for i in range(max_bs)]

    def make_batch(bs):
        return collate(items[:bs])

    # --- model ------------------------------------------------------------
    print(f"loading student from {args.model_dir} (bf16, device_map=auto) ...")
    model = load_student(args.model_dir)
    dev = next(model.get_input_embeddings().parameters()).device
    static = gpu_resident()
    print(f"student embedding device: {dev}; resident per GPU: "
          + "  ".join(f"{s:.1f}" for s in static) + " GiB")

    assert model.config.use_cache is False, "use_cache must be False for backward"
    print("confirmed: model.config.use_cache = False")

    resampler = Resampler(PerceiverResamplerConfig.ecg()).to(dev)
    n_res = sum(p.numel() for p in resampler.parameters())
    print(f"resampler built: {n_res:,} trainable params (fp32) on {dev}")

    # --- sweep: batch x checkpointing x optimizer -------------------------
    batch_sizes = [args.batch_size] if args.batch_size else [1, 2, 4]
    if args.grad_checkpointing == "on":
        ckpt_opts = [True]
    elif args.grad_checkpointing == "off":
        ckpt_opts = [False]
    else:
        ckpt_opts = [False, True]
    if args.with_optimizer == "on":
        opt_opts = [True]
    elif args.with_optimizer == "off":
        opt_opts = [False]
    else:
        opt_opts = [False, True]

    rows = []
    for ckpt in ckpt_opts:
        for bs in batch_sizes:
            for opt in opt_opts:
                print(f"\n>>> batch_size={bs} grad_checkpointing={'on' if ckpt else 'off'} "
                      f"optimizer={'on' if opt else 'off'}")
                row = run_config(model, resampler, make_batch, dev, bs, ckpt, opt)
                status = row["status"]
                if row["peaks"]:
                    peaks = "  ".join(f"{p:.1f}" for p in row["peaks"])
                    n = row["norms"]
                    print(f"    {status}: loss={row['loss']:.3f} seqlen={row['seq_len']} "
                          f"peak GiB = {peaks}")
                    print(f"    grad reached resampler: "
                          f"bio_projection={n['bio_projection.weight']:.3e}  "
                          f"latent_queries={n['latent_queries']:.3e}")
                else:
                    print(f"    {status}")
                rows.append(row)

    print_table(rows, n_gpus, static)


def load_student_tokenizer_from_dir(model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


if __name__ == "__main__":
    main()
