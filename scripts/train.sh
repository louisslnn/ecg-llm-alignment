#!/bin/bash
#SBATCH --account=ctb-liyue
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6-23:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --job-name=ecg-train
#SBATCH --output=logs/train_%j.log
#
# Train the 843M resampler against the frozen 14B student on 2x A100.
#
# 10 epochs: run 1 converged at epoch 8 and turned at 9, so the last five were
# spent overfitting. Override with EPOCHS=... if a run needs longer.
#
# gpubase_bygpu_b5 is Narval's long GPU time-bin (~7 days), which fits the run with
# room to spare; if your allocation uses a shorter bin, checkpointing + auto-resume
# make it safe to just resubmit this script -- it picks up from the latest
# checkpoint on scratch. --requeue lets a preempted job restart and resume automatically;
# --signal=USR1@120 gives train.py ~120 s to write an emergency checkpoint before
# the job is killed (preemption or time limit), so no progress since the last
# periodic checkpoint is lost.
# Override the partition/time with env if the cluster's bins differ:
#   PARTITION=gpubase_bygpu_b3 sbatch --time=23:59:00 scripts/train.sh
#
# Compute nodes have NO internet: run scripts/setup_narval.sh on a login node first
# so the checkpoint is on disk, then submit from the repo root after `mkdir -p logs`:
#   MODEL_DIR=... VENV=... sbatch scripts/train.sh
# Each run gets its own checkpoint directory (RUN=run2 by default):
#   RUN=run3 sbatch scripts/train.sh
set -euo pipefail

module load StdEnv/2023 python/3.11 cuda

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/ctb-liyue/$USER/ecg-llm-alignment}"
VENV="${VENV:-$PROJECT_DIR/venv}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/hf_cache/DeepSeek-R1-Distill-Qwen-14B}"
EPOCHS="${EPOCHS:-10}"
# Checkpoints live on scratch (large, purgeable) so they survive across resubmits.
# One directory per RUN: train.py auto-resumes from whatever is in CKPT_DIR and
# keeps a single best.pt, so sharing a directory with run 1 would overwrite its
# best checkpoint -- and run 1's weights cannot be resumed into run 2 anyway (the
# resampler gained an input and an output LayerNorm, so a strict load fails).
# Run 1 lives in the flat .../checkpoints; point CKPT_DIR there to evaluate it.
RUN="${RUN:-run2}"
CKPT_DIR="${CKPT_DIR:-$SCRATCH/ecg-llm-alignment/checkpoints/$RUN}"

source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # no network on compute nodes
mkdir -p "$CKPT_DIR"

cd "$SLURM_SUBMIT_DIR"
echo "host: $(hostname)   model: $MODEL_DIR   run: $RUN   ckpt: $CKPT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Full run. Auto-resumes from the latest checkpoint in CKPT_DIR if one exists.
srun python scripts/train.py \
    --model-dir "$MODEL_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    --batch-size 4 \
    --epochs "$EPOCHS" \
    --ckpt-every 1000 \
    --log-every 50
