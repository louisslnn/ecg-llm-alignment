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
# periodic checkpoint is lost. train.py then runs `scontrol requeue` itself and exits
# 75: an exit 0 would mark the job COMPLETED, and --requeue never fires for that.
# Override the partition/time with env if the cluster's bins differ:
#   PARTITION=gpubase_bygpu_b3 sbatch --time=23:59:00 scripts/train.sh
#
# Compute nodes have NO internet: run scripts/setup_narval.sh on a login node first
# so the checkpoint is on disk, then submit from the repo root after `mkdir -p logs`:
#   sbatch scripts/train.sh
# Each run gets its own checkpoint directory (RUN=run2 by default):
#   RUN=run3 sbatch scripts/train.sh
#
# TASK_STREAM picks what the resampler's second stream reads (--task-stream, default
# prompt): prompt-and-target, question (the question paraphrase alone, no patient
# context) or none (no text). The LLM's input is the same under all four. A
# non-default value is tagged into the default checkpoint dir -- run3_question,
# run3_notext, run3_withtarget -- so a run never auto-resumes another stream's
# checkpoints (train.py refuses that resume anyway, after the module load):
#   RUN=run3 TASK_STREAM=question sbatch scripts/train.sh
#
# LAMBDA_LINEAR and LAMBDA_CONSISTENCY are REQUIRED (train.py's --lambda-linear and
# --lambda-consistency), so every submission states its objective; 0 and 0 is the LM
# loss alone. LAMBDA_LLM (default 1.0) and LINEAR_HEAD_LR (default 1e-4) are
# optional. SAVE_EVERY_EPOCH=1 keeps a ~3.4 GB weights-only epochNN.pt per epoch
# (--save-every-epoch), ~34 GB of scratch per 10-epoch run. train.py refuses to
# resume a CKPT_DIR under different lambdas, so give each setting its own RUN:
#   RUN=run4_lin1_cons05 LAMBDA_LINEAR=1 LAMBDA_CONSISTENCY=0.5 SAVE_EVERY_EPOCH=1 \
#       sbatch scripts/train.sh
set -euo pipefail

# The mode -> tag mapping lives in scripts/stream_tag.sh, shared with every job
# script that names a run's directory. sbatch runs a spooled copy of this file, so
# it is found from the submit directory (the repo root), not from $0.
LIB="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/scripts/stream_tag.sh"
[[ -f "$LIB" ]] || { echo "missing $LIB: submit from the repo root" >&2; exit 2; }
source "$LIB"

# --- which task stream -------------------------------------------------------
# Read and checked here, before the module load: a variable no script reads is
# ignored by bash, and an unknown value should fail in a second, not after the 14B.
TASK_STREAM="${TASK_STREAM:-prompt}"
STREAM_TAG="$(stream_tag "$TASK_STREAM")" || {
    echo "TASK_STREAM must be $STREAM_MODES, got '$TASK_STREAM'" >&2
    exit 2
}

# --- the objective -------------------------------------------------------------
# No defaults for the two new weights: a grid run that forgot one should fail here,
# in a second, not train the wrong objective for a week.
if [[ -z "${LAMBDA_LINEAR:-}" || -z "${LAMBDA_CONSISTENCY:-}" ]]; then
    echo "LAMBDA_LINEAR and LAMBDA_CONSISTENCY must both be set (0 and 0 is the LM loss alone)" >&2
    exit 2
fi
OBJECTIVE=(--lambda-llm "${LAMBDA_LLM:-1.0}" --lambda-linear "$LAMBDA_LINEAR"
           --lambda-consistency "$LAMBDA_CONSISTENCY")
if [[ -n "${LINEAR_HEAD_LR:-}" ]]; then OBJECTIVE+=(--linear-head-lr "$LINEAR_HEAD_LR"); fi
if [[ "${SAVE_EVERY_EPOCH:-0}" == 1 ]]; then OBJECTIVE+=(--save-every-epoch); fi

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
EPOCHS="${EPOCHS:-10}"
# Checkpoints live on scratch (large, purgeable) so they survive across resubmits.
# One directory per RUN: train.py auto-resumes from whatever is in CKPT_DIR and
# keeps a single best.pt, so sharing a directory with run 1 would overwrite its
# best checkpoint -- and run 1's weights cannot be resumed into run 2 anyway (the
# resampler gained an input and an output LayerNorm, so a strict load fails).
# Run 1 lives in the flat .../checkpoints; point CKPT_DIR there to evaluate it.
RUN="${RUN:-run2}"
CKPT_DIR="${CKPT_DIR:-$SCRATCH/ecg-llm-alignment/checkpoints/$RUN$STREAM_TAG}"

source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # no network on compute nodes
mkdir -p "$CKPT_DIR"

cd "$SLURM_SUBMIT_DIR"
echo "host: $(hostname)   model: $MODEL_DIR   run: $RUN   task_stream: $TASK_STREAM   ckpt: $CKPT_DIR"
echo "objective: ${OBJECTIVE[*]}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Full run. Auto-resumes from the latest checkpoint in CKPT_DIR if one exists.
srun python scripts/train.py \
    --model-dir "$MODEL_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    --task-stream "$TASK_STREAM" \
    "${OBJECTIVE[@]}" \
    --batch-size 4 \
    --epochs "$EPOCHS" \
    --ckpt-every 1000 \
    --log-every 50
