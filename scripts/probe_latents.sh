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
#   probe_latents_pooled.json                  the run
#   probe_latents_pooled_shuflab.json          its label-shuffle control
#   probe_latents_pooled_withtarget.json       training's own task-text stream
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
# is the lowest val loss. RUN must match the training run you mean to probe -- the
# default follows train.sh's default, so `sbatch scripts/train.sh` and this script
# agree without being told. Run 1 predates the convention and sits in the flat
# directory:  CKPT=$SCRATCH/ecg-llm-alignment/checkpoints/best.pt RUN=run1 sbatch ...
RUN="${RUN:-run2}"
CKPT_DIR="${CKPT_DIR:-$SCRATCH/ecg-llm-alignment/checkpoints/$RUN}"
CKPT="${CKPT:-$CKPT_DIR/best.pt}"
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
TASK_TEXT="prompt"
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
# The task-text stream and the controls are part of the experiment's identity too: a
# null run must never land on the file holding the number it is the null for.
[[ "$TASK_TEXT" == "prompt" ]] || TAG="${TAG}_withtarget"
TAG="${TAG}${CONTROLS}"
OUT="${OUT:-$EVAL_DIR/probe_latents_${TAG}.json}"

source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # no network on compute nodes
mkdir -p "$EVAL_DIR"

cd "$SLURM_SUBMIT_DIR"
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
