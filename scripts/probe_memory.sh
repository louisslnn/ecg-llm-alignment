#!/bin/bash
#SBATCH --account=def-liyue
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --job-name=ecg-mem-probe
#SBATCH --output=logs/probe_memory_%j.log
#
# Peak-memory probe for the 14B student + 843M resampler on 2x A100.
# Compute nodes have NO internet, so the checkpoint must already be on disk
# (run scripts/setup_narval.sh on a login node first) and we load it offline.
#
# Submit from the repo root, after `mkdir -p logs`:
#   MODEL_DIR=... VENV=... sbatch scripts/probe_memory.sh
set -euo pipefail

module load StdEnv/2023 python/3.11 cuda

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/def-liyue/$USER/ecg-llm-alignment}"
VENV="${VENV:-$PROJECT_DIR/venv}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/hf_cache/DeepSeek-R1-Distill-Qwen-14B}"

source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # belt and suspenders: no network

cd "$SLURM_SUBMIT_DIR"
echo "host: $(hostname)   model: $MODEL_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Default behaviour: sweep batch {1,2,4} x checkpointing {off,on}.
python scripts/probe_memory.py --model-dir "$MODEL_DIR"
