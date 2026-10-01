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
# The control sweep in ONE job: three conditions (real, shuffled ECGs, zeroed
# prefix), each both generated and teacher-forced, on a single load of the 14B.
#
# The teacher-forced table carries the mean and the per-position buckets for all
# three conditions with a per-bucket real-vs-shuffled gap, and below it the same
# rows from run 1's contaminated measurement, so the two are compared row by row.
# --task-stream controls what the resampler's task-text stream reads in that pass
# (default prompt; prompt-and-target reproduces run 1):
#   sbatch scripts/eval_controls.sh --task-stream prompt-and-target
#
# Same venv, model dir and module stack as scripts/train.sh and scripts/evaluate.sh.
# Pass extra flags straight through:
#   sbatch scripts/eval_controls.sh
#   sbatch scripts/eval_controls.sh --limit 100 --tf-limit 500
#   CKPT=$SCRATCH/ecg-llm-alignment/checkpoints/ckpt_step5000.pt sbatch scripts/eval_controls.sh
#
# The hour: 3 x 50 generations at batch 1 is ~150 decodes of a few hundred greedy
# tokens each, which is the bulk of the time; the teacher-forced sweep is now 3 x 200
# forward passes rather than one pass of 200, which is still only a few minutes, and
# the model load a few more. That fits an hour with room to spare, but raising
# --limit raises it close to linearly -- 3 x 200 generations will not fit.
# Batch 1 is deliberate: it keeps each condition's decode independent of how a batch
# happened to be padded, which is what makes the byte-identical comparison trustworthy.
#
# Read the output bottom-up: the comparison table is the point, and the first number
# to look at is how many generations are byte-identical across conditions.
#
# Compute nodes have NO internet: run scripts/setup_narval.sh on a login node first.
# Submit from the repo root after `mkdir -p logs`.
set -euo pipefail

# --- paths -------------------------------------------------------------------
# The environment scripts/setup_narval.sh builds, on scratch: /project is at its
# file quota and the student checkpoint is ~28 GB of safetensors. Override either
# with an env var if yours live elsewhere.
VENV="${VENV:-$HOME/scratch/ecg/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/scratch/ecg/hf_cache/DeepSeek-R1-Distill-Qwen-14B}"

# Checked here, before the module load, so a wrong path fails in a second naming
# the path -- not as `activate: No such file` further down, or as a transformers
# traceback minutes into loading the 14B. Both report the file actually needed:
# a directory that exists but was never populated is the likelier failure.
missing=""
[[ -f "$VENV/bin/activate" ]] || missing+="  venv (no bin/activate): $VENV"$'\n'
[[ -f "$MODEL_DIR/config.json" ]] || missing+="  model dir (no config.json): $MODEL_DIR"$'\n'
if [[ -n "$missing" ]]; then
    echo "missing input path(s):" >&2
    printf '%s' "$missing" >&2
    echo "pass VENV=... MODEL_DIR=... or run scripts/setup_narval.sh on a login node" >&2
    exit 2
fi

module load StdEnv/2023 python/3.11 cuda
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
