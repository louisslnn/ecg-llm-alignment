#!/bin/bash
#SBATCH --account=ctb-liyue
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=ecg-controls
#SBATCH --output=logs/eval_controls_%j.log
#
# The control sweep in ONE job: three conditions (real, shuffled ECGs, zeroed
# prefix), each scored in fast mode, teacher-forced and generated, on a single load
# of the 14B. Fast mode is p(Yes) after the prompt and "Conclusion:" alone, over the
# whole split by default, with per-superclass metrics; see eval_controls.py.
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
#   RUN=run2 TASK_STREAM=question sbatch scripts/eval_controls.sh   # checkpoints/run2_question
#   sbatch scripts/eval_controls.sh --limit 100 --tf-limit 500 --skip-zeroed-generation
#   CKPT=$SCRATCH/ecg-llm-alignment/checkpoints/ckpt_step5000.pt sbatch scripts/eval_controls.sh
# Every limited pass is a seeded draw (--sample-seed, --stratify; default N/5 per
# superclass, half Yes and half No), printed with its positive rates.
#
# The two hours: ~15 min for the model load; fast mode over the whole test split,
# 3 x 10,985 forwards at 20+/s, ~30 min; the teacher-forced sweep, 3 x 200 forwards,
# a few minutes; and 3 x 50 generations at batch 1, ~150 decodes of a few hundred
# greedy tokens at ~0.15/s, the rest. Fast mode runs first, so its files are written
# whatever happens after. Raising --limit raises the decode time close to linearly;
# --skip-zeroed-generation takes out a third of it (the zeroed decode is a known
# 100% parse failure), and --skip-fast or --fast-limit takes out fast mode's share.
# Batch 1 is deliberate: it keeps each condition's decode independent of how a batch
# happened to be padded, which is what makes the byte-identical comparison trustworthy.
#
# Read the output bottom-up: the comparison table is the point, and the first number
# to look at is how many generations are byte-identical across conditions.
#
# Compute nodes have NO internet: run scripts/setup_narval.sh on a login node first.
# Submit from the repo root after `mkdir -p logs`.
set -euo pipefail

# The mode -> tag mapping lives in scripts/stream_tag.sh, shared with every job
# script that names a run's directory. sbatch runs a spooled copy of this file, so
# it is found from the submit directory (the repo root), not from $0.
LIB="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/scripts/stream_tag.sh"
[[ -f "$LIB" ]] || { echo "missing $LIB: submit from the repo root" >&2; exit 2; }
source "$LIB"

# --- paths -------------------------------------------------------------------
# The environment scripts/setup_narval.sh builds, on scratch: /project is at its
# file quota and the student checkpoint is ~28 GB of safetensors. Override either
# with an env var if yours live elsewhere.
VENV="${VENV:-$HOME/scratch/ecg/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/scratch/ecg/hf_cache/DeepSeek-R1-Distill-Qwen-14B}"

# Checkpoints are written to scratch by train.sh, one directory per RUN and task
# stream, tagged by scripts/stream_tag.sh: RUN=run2 TASK_STREAM=question trained
# into checkpoints/run2_question. Pass the same two here and this script finds it;
# both default as train.sh's do. TASK_STREAM only locates the directory -- what the
# resampler reads is still the checkpoint's own record (--task-stream overrides).
# best.pt is the lowest val loss. Run 1 predates the convention and sits in the
# flat directory:  CKPT=$SCRATCH/ecg-llm-alignment/checkpoints/best.pt RUN=run1 sbatch ...
TASK_STREAM="${TASK_STREAM:-prompt}"
STREAM_TAG="$(stream_tag "$TASK_STREAM")" || {
    echo "TASK_STREAM must be $STREAM_MODES, got '$TASK_STREAM'" >&2
    exit 2
}
RUN="${RUN:-run2}"
CKPT_DIR="${CKPT_DIR:-$SCRATCH/ecg-llm-alignment/checkpoints/$RUN$STREAM_TAG}"
CKPT="${CKPT:-$CKPT_DIR/best.pt}"
# The Python side accepts a directory holding best.pt as well as a file.
CKPT_FILE="$CKPT"
[[ -d "$CKPT" ]] && CKPT_FILE="$CKPT/best.pt"

# Checked here, before the module load, so a wrong path fails in a second naming
# the path -- not as `activate: No such file` further down, or as a transformers
# traceback minutes into loading the 14B. Both report the file actually needed:
# a directory that exists but was never populated is the likelier failure.
missing=""
[[ -f "$VENV/bin/activate" ]] || missing+="  venv (no bin/activate): $VENV"$'\n'
[[ -f "$MODEL_DIR/config.json" ]] || missing+="  model dir (no config.json): $MODEL_DIR"$'\n'
[[ -f "$CKPT_FILE" ]] || missing+="  resampler checkpoint: $CKPT_FILE"$'\n'
if [[ -n "$missing" ]]; then
    echo "missing input path(s):" >&2
    printf '%s' "$missing" >&2
    echo "pass VENV=... MODEL_DIR=... CKPT=... (or RUN=... TASK_STREAM=... as" \
         "train.sh was given), or run scripts/setup_narval.sh on a login node" >&2
    exit 2
fi

module load StdEnv/2023 python/3.11 cuda
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
