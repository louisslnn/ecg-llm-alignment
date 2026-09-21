#!/usr/bin/env python
"""Train the resampler against the frozen DeepSeek-R1-Distill-Qwen-14B student.

The student is frozen (bf16, sharded across the visible GPUs, use_cache=False); only
the 843M resampler trains (fp32, AdamW). Each step reuses the exact splice validated
in probe_memory.py: resampler latents are prepended to the token embeddings, prompt
and padding are masked to -100, and the loss lands on the teacher target only. There
is no linear head -- the target's Reasoning already carries the Yes/No, so a
conclusion-logit AUPRC would be meaningless; validation reports mean loss only.

Checkpoints (resampler + optimizer + step/epoch/best) go to --ckpt-dir (scratch),
every --ckpt-every steps and at each epoch end. On startup it resumes automatically
from the latest checkpoint if one exists. Only the last 2 rotating checkpoints plus
best.pt (lowest val loss) are kept.

    # the gate: can the resampler overfit 50 examples to ~0 loss?
    python scripts/train.py --model-dir $MODEL_DIR --overfit 50

    # full run (auto-resumes)
    python scripts/train.py --model-dir $MODEL_DIR --ckpt-dir $SCRATCH/ecg/ckpt

Compute nodes have no internet, so the checkpoint + tokenizer load from --model-dir
with local_files_only=True (pre-downloaded by scripts/setup_narval.sh).
"""

import argparse
import glob
import os
import random
import re
import signal
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for probe_memory

from src.data.dataset import DatasetConfig, build_datasets, make_collate_fn  # noqa: E402
from src.model.resampler import PerceiverResamplerConfig, Resampler  # noqa: E402
from probe_memory import (  # noqa: E402  -- reuse the validated splice + checks
    GradFlowError,
    check_gradients,
    gpu_peaks,
    gpu_reset,
    load_student,
    splice_forward,
)

GiB = 1024 ** 3


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="local student checkpoint dir")
    ap.add_argument("--ckpt-dir", default=os.environ.get("CKPT_DIR", "checkpoints"),
                    help="where to write/resume checkpoints (put this on scratch)")
    # optimisation (values from the reference / ROADMAP)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None,
                    help="overrides the default: 1e-5 for training, 1e-4 in --overfit "
                         "(the gate tests that the pipeline learns, not the real LR)")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=2048)  # MAX_TEXT_LEN
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--grad-checkpointing", choices=["on", "off"], default="off",
                    help="LLM activation checkpointing; off by default -- the probe "
                         "showed batch 4 fits (28.5/21.0 GiB) and measured it a no-op")
    # cadence
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    # the gate
    ap.add_argument("--overfit", type=int, default=None,
                    help="overfit the first N train examples (no val); the pipeline gate")
    ap.add_argument("--overfit-target-loss", type=float, default=0.05)
    ap.add_argument("--overfit-max-steps", type=int, default=3000)
    # data path overrides (default to DatasetConfig)
    ap.add_argument("--emb-cache", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--teacher", default=None)
    return ap.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Set by the SIGUSR1/SIGTERM handler (train.sh's `--signal=USR1@120` fires ~120 s
# before the time limit / on preemption). The training loop checkpoints and exits at
# the next step boundary, so --requeue restarts from a fresh checkpoint.
_PREEMPT = {"flag": False}


def _install_signal_handlers():
    def handler(signum, _frame):
        _PREEMPT["flag"] = True
    signal.signal(signal.SIGUSR1, handler)
    signal.signal(signal.SIGTERM, handler)


def preflight(args, data_cfg):
    """Check every input path exists BEFORE loading the 14B model (slow)."""
    checks = [
        ("model dir", args.model_dir),
        ("model config", os.path.join(args.model_dir, "config.json")),
        ("embedding index", os.path.join(data_cfg.emb_cache_dir, "index.json")),
        ("manifest", data_cfg.manifest_path),
        ("teacher jsonl", data_cfg.teacher_path),
    ]
    missing = [f"{name}: {path}" for name, path in checks if not os.path.exists(path)]
    if missing:
        sys.exit("preflight failed, missing input(s):\n  " + "\n  ".join(missing))
    os.makedirs(args.ckpt_dir, exist_ok=True)
    if not os.access(args.ckpt_dir, os.W_OK):
        sys.exit(f"preflight failed: ckpt dir not writable: {args.ckpt_dir}")
    print("preflight OK; all inputs present, ckpt dir writable")


# --------------------------------------------------------------------------- #
# Checkpointing                                                               #
# --------------------------------------------------------------------------- #

_STEP_RE = re.compile(r"ckpt_step(\d+)\.pt$")


def _rotating(ckpt_dir):
    return sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_step*.pt")),
                  key=lambda p: int(_STEP_RE.search(p).group(1)))


def save_checkpoint(ckpt_dir, resampler, optimizer, epoch, step, best_val, is_best):
    payload = {
        "resampler": resampler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "best_val_loss": best_val,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    path = os.path.join(ckpt_dir, f"ckpt_step{step}.pt")
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)          # atomic
    if is_best:
        best = os.path.join(ckpt_dir, "best.pt")
        torch.save(payload, best + ".tmp")
        os.replace(best + ".tmp", best)
    # keep only the last 2 rotating checkpoints (best.pt is separate, never pruned)
    for old in _rotating(ckpt_dir)[:-2]:
        os.remove(old)
    return path


def resume(ckpt_dir, resampler, optimizer, dev):
    ckpts = _rotating(ckpt_dir)
    if not ckpts:
        print("no checkpoint found; starting fresh")
        return 0, 0, float("inf")
    path = ckpts[-1]
    # our own checkpoint (trusted); it carries numpy/python RNG state that the
    # weights_only=True default (torch>=2.6) refuses to unpickle.
    payload = torch.load(path, map_location=dev, weights_only=False)
    resampler.load_state_dict(payload["resampler"])
    optimizer.load_state_dict(payload["optimizer"])
    rng = payload.get("rng")
    if rng:
        torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            pass
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    print(f"resumed from {os.path.basename(path)}: epoch {payload['epoch']} "
          f"step {payload['global_step']} best_val {payload['best_val_loss']:.4f}")
    return payload["epoch"], payload["global_step"], payload["best_val_loss"]


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #

def build_loaders(args, tokenizer, data_cfg):
    from torch.utils.data import DataLoader, Subset

    splits = build_datasets(data_cfg)
    collate = make_collate_fn(tokenizer, max_length=args.max_length)
    g = torch.Generator()
    g.manual_seed(args.seed)

    if args.overfit is not None:
        n = min(args.overfit, len(splits["train"]))
        train = Subset(splits["train"], list(range(n)))
        print(f"OVERFIT mode: {n} train examples, no validation")
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate,
                                  generator=g, drop_last=False)
        return train_loader, None

    train_loader = DataLoader(splits["train"], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              generator=g, drop_last=True)
    val_loader = DataLoader(splits["val"], batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)
    print(f"train {len(splits['train']):,} examples  val {len(splits['val']):,} examples")
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# Train / validate                                                            #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def validate(model, resampler, val_loader, dev):
    resampler.eval()
    losses = []
    for batch in val_loader:
        losses.append(float(splice_forward(model, resampler, batch, dev).detach()))
    resampler.train()
    return sum(losses) / max(len(losses), 1)


def _log_mem():
    return "  ".join(f"gpu{i} {p:.1f}GiB" for i, p in enumerate(gpu_peaks()))


def train_overfit(model, resampler, optimizer, loader, dev, args):
    print(f"gate: overfit to loss <= {args.overfit_target_loss} within "
          f"{args.overfit_max_steps} steps")
    resampler.train()
    step = 0
    checked = False
    while step < args.overfit_max_steps:
        for batch in loader:
            t0 = time.time()
            loss = splice_forward(model, resampler, batch, dev)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not checked:                       # one-time grad-flow gate
                check_gradients(model, resampler)
                checked = True
            torch.nn.utils.clip_grad_norm_(resampler.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            lv = float(loss.detach())
            print(f"step {step:5d}  loss {lv:.4f}  {time.time()-t0:.2f}s  {_log_mem()}",
                  flush=True)
            if lv <= args.overfit_target_loss:
                print(f"GATE PASSED: loss {lv:.4f} <= {args.overfit_target_loss} at step {step}")
                return True
            if step >= args.overfit_max_steps:
                break
    print(f"GATE NOT MET: did not reach {args.overfit_target_loss} in {step} steps")
    return False


def train(model, resampler, optimizer, train_loader, val_loader, dev, args,
          start_epoch, global_step, best_val):
    for epoch in range(start_epoch, args.epochs):
        resampler.train()
        gpu_reset()
        window_t0 = time.time()
        running = []
        for batch in train_loader:
            loss = splice_forward(model, resampler, batch, dev)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if global_step == 0:                  # one-time grad-flow gate
                check_gradients(model, resampler)
            torch.nn.utils.clip_grad_norm_(resampler.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1
            running.append(float(loss.detach()))

            if global_step % args.log_every == 0:
                dt = (time.time() - window_t0) / args.log_every
                print(f"epoch {epoch} step {global_step} loss "
                      f"{sum(running[-args.log_every:]) / args.log_every:.4f} "
                      f"{dt*1000:.0f}ms/step  {_log_mem()}", flush=True)
                gpu_reset()
                window_t0 = time.time()

            if global_step % args.ckpt_every == 0:
                save_checkpoint(args.ckpt_dir, resampler, optimizer, epoch, global_step,
                                best_val, is_best=False)

            if _PREEMPT["flag"]:
                print(f"preemption signal received at step {global_step}; saving "
                      "emergency checkpoint and exiting for requeue", flush=True)
                save_checkpoint(args.ckpt_dir, resampler, optimizer, epoch, global_step,
                                best_val, is_best=False)
                sys.exit(0)

        val_loss = validate(model, resampler, val_loader, dev)
        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)
        print(f"== epoch {epoch} done  train_loss "
              f"{sum(running)/max(len(running),1):.4f}  val_loss {val_loss:.4f}"
              f"{'  (best)' if is_best else ''}", flush=True)
        save_checkpoint(args.ckpt_dir, resampler, optimizer, epoch + 1, global_step,
                        best_val, is_best=is_best)


def main():
    args = parse_args()
    if args.lr is None:                       # 1e-4 in the gate, 1e-5 for real training
        args.lr = 1e-4 if args.overfit is not None else 1e-5
    set_seed(args.seed)
    _install_signal_handlers()

    cfg_kwargs = {}
    if args.emb_cache:
        cfg_kwargs["emb_cache_dir"] = args.emb_cache
    if args.manifest:
        cfg_kwargs["manifest_path"] = args.manifest
    if args.teacher:
        cfg_kwargs["teacher_path"] = args.teacher
    data_cfg = DatasetConfig(**cfg_kwargs)

    preflight(args, data_cfg)

    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible; training needs GPUs")
    print(f"visible GPUs: {torch.cuda.device_count()}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    train_loader, val_loader = build_loaders(args, tokenizer, data_cfg)

    print("loading frozen student (bf16, device_map=auto, use_cache=False) ...")
    model = load_student(args.model_dir)
    if args.grad_checkpointing == "on":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dev = next(model.get_input_embeddings().parameters()).device

    resampler = Resampler(PerceiverResamplerConfig.ecg()).to(dev)
    resampler.train()
    n_train = sum(p.numel() for p in resampler.parameters() if p.requires_grad)
    n_frozen_grad = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"resampler trainable params: {n_train:,} (fp32) on {dev}; "
          f"LLM params with requires_grad: {n_frozen_grad} (must be 0)")

    optimizer = torch.optim.AdamW(resampler.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    print(f"AdamW lr={args.lr} weight_decay={args.weight_decay} "
          f"grad_clip={args.grad_clip} batch={args.batch_size} "
          f"grad_checkpointing={args.grad_checkpointing}")

    if args.overfit is not None:
        ok = train_overfit(model, resampler, optimizer, train_loader, dev, args)
        sys.exit(0 if ok else 1)

    start_epoch, global_step, best_val = resume(args.ckpt_dir, resampler, optimizer, dev)
    train(model, resampler, optimizer, train_loader, val_loader, dev, args,
          start_epoch, global_step, best_val)


if __name__ == "__main__":
    main()
