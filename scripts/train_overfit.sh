#!/bin/bash
#SBATCH --account=ctb-liyue
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --job-name=ecg-overfit
#SBATCH --output=logs/train_overfit_%j.log
#
# THE GATE: can the resampler overfit 50 training examples to ~0 loss? A short
# (30 min, short time-bin) submission that must pass before launching the full run.
# It trains on the first 50 train examples, no validation, logs loss every step,
# and exits non-zero if it doesn't reach the target loss within the step cap.
#
# Compute nodes have NO internet: run scripts/setup_narval.sh on a login node first.
# Submit from the repo root after `mkdir -p logs`:
#   MODEL_DIR=... VENV=... sbatch scripts/train_overfit.sh
set -euo pipefail

module load StdEnv/2023 python/3.11 cuda

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/ctb-liyue/$USER/ecg-llm-alignment}"
VENV="${VENV:-$PROJECT_DIR/venv}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/hf_cache/DeepSeek-R1-Distill-Qwen-14B}"

source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
echo "host: $(hostname)   model: $MODEL_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

srun python scripts/train.py \
    --model-dir "$MODEL_DIR" \
    --overfit 50 \
    --batch-size 4 \
    --overfit-target-loss 0.05 \
    --overfit-max-steps 3000
