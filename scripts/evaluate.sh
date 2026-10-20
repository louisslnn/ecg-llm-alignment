#!/bin/bash
#SBATCH --account=ctb-liyue
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --job-name=ecg-eval
#SBATCH --output=logs/eval_%j.log
#
# Free-generation evaluation of a trained resampler on 2x A100.
#
# Same venv, model dir and module stack as scripts/train.sh; checkpoints and eval
# outputs live on scratch. Pass extra flags straight through:
#   sbatch scripts/evaluate.sh --limit 500
#   RUN=run2 TASK_STREAM=question sbatch scripts/evaluate.sh   # checkpoints/run2_question
#   CKPT=$SCRATCH/ecg-llm-alignment/checkpoints/best.pt sbatch scripts/evaluate.sh
# --limit N is a seeded draw (--sample-seed, --stratify; default N/5 per superclass,
# half Yes and half No), not the head of the split.
#
# THE CONTROLS -- run these too, and read them next to the main run:
#   sbatch scripts/evaluate.sh --shuffle-embeddings     # wrong ECG per record
#   sbatch scripts/evaluate.sh --zero-latents           # prefix carries nothing
# Each writes its own eval_<split>_<condition>.jsonl, so none can clobber another.
# For all three conditions plus the teacher-forced loss in ONE job (one model load,
# with a comparison table at the end), use scripts/eval_controls.sh instead.
#
# Teacher-forced loss instead of generation, on the real targets:
#   sbatch scripts/evaluate.sh --teacher-forced-loss --limit 2000
#
# ONE HOUR IS NOT THE FULL TEST SPLIT. Test is 10,977 examples, each ~200-400
# greedy tokens from a 14B sharded over two cards; a single hour covers a few
# hundred to a couple of thousand, depending on how fast the student closes its
# block. Two ways to work inside that:
#   * --limit N for a run that is meant to be quick (metrics come out either way,
#     just over fewer examples -- watch the positive rate per superclass at small N);
#   * resubmit the same command with --resume to keep filling the same jsonl,
#     which skips what is already generated. Metrics are then recomputed over the
#     whole file, and `--from-jsonl <file>` recomputes them any time, no GPU.
# Raise --time (up to the partition's bin) for a full uninterrupted pass.
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
# Generations + metrics land on scratch too (project space is at its file quota).
EVAL_DIR="${EVAL_DIR:-$SCRATCH/ecg-llm-alignment/eval/$RUN}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# Every condition writes its own file; overwriting the main run's generations with a
# control's would destroy the only thing worth comparing it against. Mirrors
# evaluate.py's own default naming (condition_suffix).
SUFFIX=""
prev=""
for arg in "$@"; do
    [[ "$arg" == "--shuffle-embeddings" ]] && SUFFIX="${SUFFIX}_shuffled"
    [[ "$arg" == "--zero-latents" ]] && SUFFIX="${SUFFIX}_zeroed"
    [[ "$arg" == "--teacher-forced-loss" ]] && SUFFIX="${SUFFIX}_teacher_forced"
    # An explicit --task-stream is part of the experiment's identity, tagged as
    # train.sh tags checkpoint dirs (scripts/stream_tag.sh); prompt keeps the plain
    # name. Unset, the run reads the checkpoint's own stream, which the directory
    # already names.
    [[ "$prev" == "--task-stream" ]] && SUFFIX="${SUFFIX}$(stream_tag "$arg")"
    [[ "$arg" == --task-stream=* ]] && SUFFIX="${SUFFIX}$(stream_tag "${arg#*=}")"
    prev="$arg"
done
OUT="${OUT:-$EVAL_DIR/eval_${SPLIT}${SUFFIX}.jsonl}"

source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # no network on compute nodes
mkdir -p "$EVAL_DIR"

cd "$SLURM_SUBMIT_DIR"
echo "host: $(hostname)   run: $RUN   ckpt: $CKPT   out: $OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# "$@" is appended last, so anything passed to sbatch overrides these defaults
# (--limit, --shuffle-embeddings, --resume, --out, --split, ...).
srun python scripts/evaluate.py \
    --model-dir "$MODEL_DIR" \
    --ckpt "$CKPT" \
    --split "$SPLIT" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens 512 \
    --out "$OUT" \
    "$@"
