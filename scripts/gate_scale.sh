#!/bin/bash
#SBATCH --account=ctb-liyue
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --job-name=ecg-gate-scale
#SBATCH --output=logs/gate_scale_%j.slurm.log
#
# ONE POINT OF THE PREFIX-SCALE SWEEP: overfit 50 examples with the output
# LayerNorm calibrated to <scale> x the student's embedding norm, and see whether
# the gate still passes. Submit it once per scale:
#
#   mkdir -p logs
#   MODEL_DIR=... VENV=... sbatch scripts/gate_scale.sh 1
#   MODEL_DIR=... VENV=... sbatch scripts/gate_scale.sh 5
#   MODEL_DIR=... VENV=... sbatch scripts/gate_scale.sh 20
#
# Each run gets its own log (logs/gate_scale<N>.log) and its own checkpoint
# directory, so the three can be in the queue at once without touching each other.
#
# WHAT VARIES: --prefix-scale, and nothing else. The prompt keeps the template's
# <think> block (DatasetConfig.strip_think is off by default, as run 1 trained), so
# a difference between these three runs can only come from how loud the prefix is.
#
# WHAT TO READ: the gate verdict (exit 0 = passed), and the "prefix scale @ step N"
# lines, which report the measured latent norm, the token norm, their ratio and
# ||gain|| every 50 steps. The gain is trainable: if it drifts back toward the same
# ratio from all three starting points, the scale is self-correcting and the sweep
# has answered its own question; if each run stays near where it was initialised,
# the initialisation is what matters and the best of the three is the one to keep.
#
# NOTE: --overfit never writes a checkpoint (the gate exits as soon as it passes or
# runs out of steps), so the per-scale directory will be empty. It exists so the
# three jobs cannot share or clobber state, and so a non-gate rerun with the same
# CKPT_DIR lands somewhere of its own.
#
# 400 steps at batch 4 is ~32 passes over the 50 examples; budget ~20 min including
# the model load, well inside the hour.
#
# Compute nodes have NO internet: run scripts/setup_narval.sh on a login node first.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: sbatch scripts/gate_scale.sh <prefix-scale>   e.g. 1, 5, 20" >&2
    exit 2
fi
SCALE="$1"
if ! [[ "$SCALE" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [[ "$SCALE" == "0" ]]; then
    echo "prefix-scale must be a positive number, got '$SCALE'" >&2
    exit 2
fi
# 1 -> "1", 20 -> "20", 0.5 -> "0.5": a stable tag for filenames.
TAG="$(printf '%g' "$SCALE")"

module load StdEnv/2023 python/3.11 cuda

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/ctb-liyue/$USER/ecg-llm-alignment}"
VENV="${VENV:-$PROJECT_DIR/venv}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/hf_cache/DeepSeek-R1-Distill-Qwen-14B}"
# One directory per scale, so three queued jobs cannot collide.
CKPT_DIR="${CKPT_DIR:-$SCRATCH/ecg-llm-alignment/checkpoints/gate_scale$TAG}"
MAX_STEPS="${MAX_STEPS:-400}"
TARGET_LOSS="${TARGET_LOSS:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-4}"

source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # no network on compute nodes
mkdir -p "$CKPT_DIR" logs

cd "$SLURM_SUBMIT_DIR"
LOG="logs/gate_scale${TAG}.log"
echo "host: $(hostname)   prefix_scale: $SCALE   ckpt: $CKPT_DIR   log: $LOG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# tee so the run has a log named by its scale; slurm's own --output file keeps
# whatever happens before this point (module load, venv, preflight failures).
srun python scripts/train.py \
    --model-dir "$MODEL_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    --prefix-scale "$SCALE" \
    --overfit 50 \
    --batch-size "$BATCH_SIZE" \
    --overfit-target-loss "$TARGET_LOSS" \
    --overfit-max-steps "$MAX_STEPS" \
    --prefix-log-every 50 \
    2>&1 | tee "$LOG"
