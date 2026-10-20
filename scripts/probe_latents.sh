#!/bin/bash
#SBATCH --account=ctb-liyue
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --job-name=ecg-probe-latents
#SBATCH --output=logs/probe_latents_%j.log
#
# Linear probe on the resampler's OUTPUT latents -- the prefix the student reads --
# against the same probe on the input embeddings (scripts/probe_embeddings.py).
#
# Same venv, model dir, module stack and RUN/CKPT conventions as scripts/train.sh
# and scripts/evaluate.sh. Pass extra flags straight through:
#   sbatch scripts/probe_latents.sh
#   sbatch scripts/probe_latents.sh --question-mode fixed
#   RUN=run2 TASK_STREAM=question sbatch scripts/probe_latents.sh    # checkpoints/run2_question
#   CKPT=$SCRATCH/ecg-llm-alignment/checkpoints/best.pt RUN=run1 sbatch scripts/probe_latents.sh
#
# THE HOUR IS NOT ENOUGH FOR THE DEFAULT MODE. The resampler is multi-modal, so its
# latents depend on which superclass question is being asked, and the question mode
# decides how many forwards that costs over folds 1-8 plus fold 10 (19,611 records):
#
#   pooled   (default)  all five questions per record, mean-pooled   ~98k forwards
#   matched             one question per probe, five passes          ~98k forwards
#   fixed               one question for every record                ~19.6k forwards
#
# Only `fixed` is comfortable in an hour. For pooled or matched, raise the limit and
# the partition bin at submit time -- #SBATCH directives are read before this script
# runs, so they cannot be set from in here:
#
#   sbatch --time=3:00:00 scripts/probe_latents.sh
#   sbatch --partition=gpubase_bygpu_b3 --time=11:59:00 scripts/probe_latents.sh
#
# Budget it as ~15 min for the student load plus the pass itself; there is no
# transformer forward anywhere (the student is loaded for its input-embedding matrix
# alone), so the resampler and the dataloader set the rate. The five fp64 LBFGS fits
# run on the GPU and take minutes, not hours -- on CPU at 5120 dims they would take
# hours, which is why --fit-device defaults to the resampler's device.
#
# Start with `--question-mode fixed --limit-records 500` to confirm the pipeline and
# the checkpoint before spending a long job on the full pooled run.
#
# Each question mode writes its own JSON (and in fixed mode, each question), as do
# --task-text prompt-and-target and the --shuffle-embeddings / --shuffle-labels
# controls, so a null run can never overwrite the number it is the null for:
#   probe_latents_pooled.json                  the run, prompt stream
#   probe_latents_pooled_shuflab.json          its label-shuffle control
#   probe_latents_pooled_question.json         a checkpoint trained on the question
#   probe_latents_pooled_withtarget.json       --task-text prompt-and-target
# The stream is the checkpoint's own unless --task-text says otherwise, and is
# tagged as train.sh tags checkpoint dirs (scripts/stream_tag.sh). Files written
# before that, for a question-stream checkpoint, say _withtarget and are misnamed.
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
# Output lands on scratch too (project space is at its file quota).
EVAL_DIR="${EVAL_DIR:-$SCRATCH/ecg-llm-alignment/eval/$RUN}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# The question mode goes in the filename: pooled, matched and fixed are three
# different experiments, and the second one run must not overwrite the first. In
# fixed mode the question itself goes in too -- fixed/NORM and fixed/MI are as
# different from each other as the modes are. Both defaults track probe_latents.py's
# own (pooled, and NORM for the fixed question).
MODE="pooled"
QUESTION="NORM"
TASK_TEXT=""          # unset: the checkpoint's own, resolved below
CONTROLS=""
prev=""
for arg in "$@"; do
    [[ "$prev" == "--question-mode" ]] && MODE="$arg"
    [[ "$prev" == "--fixed-question" ]] && QUESTION="$arg"
    [[ "$prev" == "--task-text" ]] && TASK_TEXT="$arg"
    case "$arg" in
        --question-mode=*)     MODE="${arg#*=}" ;;
        --fixed-question=*)    QUESTION="${arg#*=}" ;;
        --task-text=*)         TASK_TEXT="${arg#*=}" ;;
        --shuffle-embeddings)  CONTROLS="${CONTROLS}_shufemb" ;;
        --shuffle-labels)      CONTROLS="${CONTROLS}_shuflab" ;;
    esac
    prev="$arg"
done
TAG="$MODE"
SHOWN="$MODE"
if [[ "$MODE" == "fixed" ]]; then
    TAG="fixed_$QUESTION"
    SHOWN="fixed ($QUESTION)"
fi

source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # no network on compute nodes
mkdir -p "$EVAL_DIR"

cd "$SLURM_SUBMIT_DIR"
# Unset, --task-text is what probe_latents.py will resolve it to: the checkpoint's
# recorded --task-stream, or prompt for one with no record. Asked of the same
# function (evaluate.checkpoint_task_stream) so the filename names the stream the
# probe actually feeds.
if [[ -z "$TASK_TEXT" ]]; then
    TASK_TEXT="$(python -c 'import sys; sys.path[:0] = ["scripts", "."]
from evaluate import checkpoint_task_stream
print(checkpoint_task_stream(sys.argv[1]) or "prompt")' "$CKPT")"
fi
# The task-text stream and the controls are part of the experiment's identity too: a
# null run must never land on the file holding the number it is the null for. The
# stream's tag is train.sh's (scripts/stream_tag.sh): _question, _notext,
# _withtarget, nothing for prompt.
TEXT_TAG="$(stream_tag "$TASK_TEXT")" || {
    echo "--task-text must be $STREAM_MODES, got '$TASK_TEXT'" >&2
    exit 2
}
TAG="${TAG}${TEXT_TAG}${CONTROLS}"
OUT="${OUT:-$EVAL_DIR/probe_latents_${TAG}.json}"
echo "host: $(hostname)   run: $RUN   ckpt: $CKPT"
echo "mode: $SHOWN   task-text: $TASK_TEXT   out: $OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# "$@" is appended last, so anything passed to sbatch overrides these defaults
# (--question-mode, --variants, --limit-records, --out, --compare-to, ...).
# --compare-to keeps its default, the repo's results/probe_embeddings.json, which
# resolves because the job runs from the submit directory.
srun python scripts/probe_latents.py \
    --model-dir "$MODEL_DIR" \
    --ckpt "$CKPT" \
    --batch-size "$BATCH_SIZE" \
    --out "$OUT" \
    "$@"
