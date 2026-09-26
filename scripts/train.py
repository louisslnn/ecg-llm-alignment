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
from src.model.resampler import (  # noqa: E402
    PerceiverResamplerConfig,
    Resampler,
    mean_embedding_norm,
)
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
    ap.add_argument("--prefix-scale", type=float, default=1.0,
                    help="multiplier on the output LayerNorm's calibration target: "
                         "the gain becomes (scale * mean_embedding_norm) / "
                         "sqrt(embed_dim). 1.0 puts the latents exactly on the "
                         "student's embedding scale; >1 makes the prefix louder. This "
                         "is the swept variable -- see scripts/gate_scale.sh")
    # Architecture ablations. Both LayerNorms are ON by default and are the two
    # departures from the reference resampler; these turn them off one at a time so a
    # change in the gate can be attributed to one of them. See src/model/resampler.py.
    ap.add_argument("--no-input-norm", action="store_true",
                    help="drop the LayerNorm in front of bio_projection "
                         "(input_layer_norm=False), leaving the cached ECG embeddings "
                         "at their raw ~350 scale, as run 1 trained")
    ap.add_argument("--no-output-norm", action="store_true",
                    help="drop the LayerNorm on the resampler output "
                         "(final_output_norm=False), so the latent scale is whatever "
                         "the block stack produces -- the reference behaviour. There "
                         "is then nothing for --prefix-scale to calibrate, so the two "
                         "flags are mutually exclusive unless the scale is 1.0")
    ap.add_argument("--strip-think", action="store_true",
                    help="strip the template's trailing open <think> block from the "
                         "prompt. OFF by default, matching run 1; evaluation must be "
                         "given the same flag or the student is scored on a prompt it "
                         "never saw")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--grad-checkpointing", choices=["on", "off"], default="off",
                    help="LLM activation checkpointing; off by default -- the probe "
                         "showed batch 4 fits (28.5/21.0 GiB) and measured it a no-op")
    # cadence
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--prefix-log-every", type=int, default=50,
                    help="in --overfit mode, log the latent/token norm ratio and "
                         "||gain|| every N steps, to see the trainable gain drift")
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
    args = ap.parse_args()
    # With no output LayerNorm there is no calibration target and no gain to set, so a
    # non-unit scale would be silently ignored -- and the run would carry a label
    # (log name, checkpoint dir, checkpoint meta) claiming a scale it never applied.
    if args.no_output_norm and args.prefix_scale != 1.0:
        ap.error("--prefix-scale has nothing to calibrate with --no-output-norm; "
                 f"drop one of the two (got --prefix-scale {args.prefix_scale:g})")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Set by the SIGUSR1/SIGTERM handler (train.sh's `--signal=USR1@120` fires ~120 s
# before the time limit / on preemption). The training loop checkpoints and exits at
# the next step boundary, so --requeue restarts from a fresh checkpoint.
_PREEMPT = {"flag": False}

# Settings written into every checkpoint, filled in by main(). A prefix-scale sweep
# yields checkpoints whose weights differ only through one scalar; without this there
# is nothing in the file that says which run produced it.
_RUN_META: dict = {}


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


def save_checkpoint(ckpt_dir, resampler, optimizer, epoch, step, best_val, is_best,
                    meta=None):
    payload = {
        "resampler": resampler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "best_val_loss": best_val,
        # what this run was: a sweep over prefix_scale produces checkpoints that are
        # otherwise indistinguishable, so each one carries its own settings
        "meta": dict(_RUN_META) if meta is None else meta,
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
    collate = make_collate_fn(tokenizer, max_length=args.max_length,
                              strip_think=data_cfg.strip_think)
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


@torch.no_grad()
def prefix_scale_line(model, resampler, batch, dev):
    """Where the prefix actually sits, relative to the tokens it is spliced against.

    The LayerNorm gain is initialised to put the latents on the student's embedding
    scale (times --prefix-scale), but it is a trainable parameter: this measures
    where it has drifted to. Reports the mean per-position latent norm, the mean norm
    of the real prompt/target tokens, their ratio, and ||gain||, which is what the
    latent norm reduces to when the normalised vectors have unit variance -- so a
    gap between ||gain|| and the measured norm says the latents are not unit-variance
    going into the norm, and movement in ||gain|| alone is the optimiser pulling the
    prefix louder or quieter than it was initialised.

    Mirrors the resampler call in probe_memory.splice_forward; one extra resampler
    forward, no LLM forward, so it is cheap next to a training step.
    """
    ecg = batch["ecg_embed"].to(dev)
    input_ids = batch["input_ids"].to(dev)
    attn = batch["attention_mask"].to(dev)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tok_embeds = model.get_input_embeddings()(input_ids)
        latents = resampler(ecg, task_embeddings=tok_embeds, task_mask=attn.bool())
    lat = float(latents.float().norm(dim=-1).mean())
    tok = float(tok_embeds.float().norm(dim=-1)[attn.bool()].mean())
    out_norm = getattr(resampler, "output_norm", None)
    gain = (f"  ||gain|| {float(out_norm.weight.detach().float().norm()):.3f}"
            if out_norm is not None else "")
    return (f"latents {lat:.3f}  tokens {tok:.3f}  ratio {lat / max(tok, 1e-9):.3f}"
            + gain)


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
            if step == 1 or step % args.prefix_log_every == 0:
                print(f"    prefix scale @ step {step}: "
                      f"{prefix_scale_line(model, resampler, batch, dev)}", flush=True)
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
    _RUN_META.update({
        "prefix_scale": args.prefix_scale,
        "input_layer_norm": not args.no_input_norm,
        "final_output_norm": not args.no_output_norm,
        "strip_think": bool(args.strip_think),
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "overfit": args.overfit,
        "seed": args.seed,
    })
    set_seed(args.seed)
    _install_signal_handlers()

    cfg_kwargs = {"strip_think": args.strip_think}
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

    resampler = Resampler(PerceiverResamplerConfig.ecg(
        input_layer_norm=not args.no_input_norm,
        final_output_norm=not args.no_output_norm,
    )).to(dev)
    print(f"resampler architecture: input_layer_norm={not args.no_input_norm} "
          f"final_output_norm={not args.no_output_norm}")
    # Start the latents at the student's own embedding scale. The splice puts them
    # in front of real token embeddings, and attention logits are dot products, so
    # a prefix far shorter than those embeddings draws almost no attention weight
    # and the frozen LLM reads past the ECG. This only sets where training starts:
    # the LayerNorm gain trains like any other parameter, and on resume the
    # checkpoint's learned value replaces it (resume() loads after this).
    embed_norm = mean_embedding_norm(model.get_input_embeddings().weight)
    if args.no_output_norm:
        print(f"student mean embedding norm: {embed_norm:.4f}; output LayerNorm is OFF "
              "(--no-output-norm), so there is no gain to calibrate: the latent scale "
              "is whatever the block stack produces")
    else:
        target = args.prefix_scale * embed_norm
        gain = resampler.calibrate_output_norm(target)
        print(f"student mean embedding norm: {embed_norm:.4f}; prefix_scale "
              f"{args.prefix_scale:g} -> target latent norm {target:.4f}; output "
              f"LayerNorm gain set to {gain:.6f} "
              f"(= {target:.4f} / sqrt({resampler.config.embed_dim}))")
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
