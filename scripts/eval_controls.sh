#!/bin/bash
#SBATCH --account=ctb-liyue
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --job-name=ecg-controls
#SBATCH --output=logs/eval_controls_%j.log
#
# The control sweep in ONE job: three generation conditions (real, shuffled ECGs,
# zeroed prefix) plus the teacher-forced loss, on a single load of the 14B.
#
# Same venv, model dir and module stack as scripts/train.sh and scripts/evaluate.sh.
# Pass extra flags straight through:
#   MODEL_DIR=... VENV=... sbatch scripts/eval_controls.sh
#   sbatch scripts/eval_controls.sh --limit 100 --tf-limit 500
#   CKPT=$SCRATCH/ecg-llm-alignment/checkpoints/ckpt_step5000.pt sbatch scripts/eval_controls.sh
#
# The hour: 3 x 50 generations at batch 1 is ~150 decodes of a few hundred greedy
# tokens each, which is the bulk of the time; the 200-example teacher-forced pass is
# a few minutes, and the model load a few more. That fits an hour with room to
# spare, but raising --limit raises it close to linearly -- 3 x 200 will not fit.
# Batch 1 is deliberate: it keeps each condition's decode independent of how a batch
# happened to be padded, which is what makes the byte-identical comparison trustworthy.
#
# Read the output bottom-up: the comparison table is the point, and the first number
# to look at is how many generations are byte-identical across conditions.
#
# Compute nodes have NO internet: run scripts/setup_narval.sh on a login node first.
# Submit from the repo root after `mkdir -p logs`.
set -euo pipefail

module load StdEnv/2023 python/3.11 cuda

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/ctb-liyue/$USER/ecg-llm-alignment}"
VENV="${VENV:-$PROJECT_DIR/venv}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/hf_cache/DeepSeek-R1-Distill-Qwen-14B}"
# Checkpoints are written to scratch by train.sh, one directory per RUN; best.pt
# is the lowest val loss. RUN must match the training run you mean to evaluate --
# the default follows train.sh's default, so `sbatch scripts/train.sh` and this
# script agree without being told. Run 1 predates the convention and sits in the
# flat directory:  CKPT=$SCRATCH/ecg-llm-alignment/checkpoints/best.pt RUN=run1 sbatch ...
RUN="${RUN:-run2}"
CKPT_DIR="${CKPT_DIR:-$SCRATCH/ecg-llm-alignment/checkpoints/$RUN}"
CKPT="${CKPT:-$CKPT_DIR/best.pt}"
# Outputs land on scratch too (project space is at its file quota).
EVAL_DIR="${EVAL_DIR:-$SCRATCH/ecg-llm-alignment/eval/$RUN}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-50}"
TF_LIMIT="${TF_LIMIT:-200}"
BATCH_SIZE="${BATCH_SIZE:-1}"

source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # no network on compute nodes
mkdir -p "$EVAL_DIR"

cd "$SLURM_SUBMIT_DIR"
echo "host: $(hostname)   run: $RUN   ckpt: $CKPT   out: $EVAL_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# "$@" is appended last, so anything passed to sbatch overrides these defaults.
srun python scripts/eval_controls.py \
    --model-dir "$MODEL_DIR" \
    --ckpt "$CKPT" \
    --split "$SPLIT" \
    --limit "$LIMIT" \
    --tf-limit "$TF_LIMIT" \
    --batch-size "$BATCH_SIZE" \
    --out-dir "$EVAL_DIR" \
    "$@"
