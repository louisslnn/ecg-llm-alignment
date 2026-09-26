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
# the gate still passes. Submit it once per point:
#
#   mkdir -p logs
#   MODEL_DIR=... VENV=... sbatch scripts/gate_scale.sh 1
#   MODEL_DIR=... VENV=... sbatch scripts/gate_scale.sh 5
#   MODEL_DIR=... VENV=... sbatch scripts/gate_scale.sh 20
#
# ARCHITECTURE ABLATIONS, one change at a time. Both LayerNorms are on by default
# (they are the two departures from the reference resampler); either can be
# dropped, and the flag goes into the log name and the checkpoint dir so an
# ablation never lands on a baseline's files:
#
#   sbatch scripts/gate_scale.sh 1 --no-input-norm    -> logs/gate_scale1_noinput.log
#   sbatch scripts/gate_scale.sh 1 --no-output-norm   -> logs/gate_scale1_nooutput.log
#   sbatch scripts/gate_scale.sh 1 --no-input-norm --no-output-norm
#                                     -> logs/gate_scale1_noinput_nooutput.log
#
# --no-output-norm removes the thing --prefix-scale calibrates, so it is only
# accepted at scale 1 (train.py refuses the combination too); with both norms off
# the resampler is the reference architecture exactly.
#
# Each run gets its own log (logs/gate_scale<N><flags>.log) and its own checkpoint
# directory, so several can be in the queue at once without touching each other.
#
# WHAT VARIES: --prefix-scale and the two norm flags, nothing else. The prompt
# keeps the template's <think> block (DatasetConfig.strip_think is off by default,
# as run 1 trained), so a difference between runs can only come from how loud the
# prefix is and which norms are present.
#
# WHAT TO READ: the gate verdict (exit 0 = passed), the "resampler architecture"
# line, and the "prefix scale @ step N" lines, which report the measured latent
# norm, the token norm, their ratio and ||gain|| every 50 steps. The gain is
# trainable: if it drifts back toward the same ratio from all three starting
# points, the scale is self-correcting and the sweep has answered its own
# question; if each run stays near where it was initialised, the initialisation
# is what matters and the best of the three is the one to keep. With
# --no-output-norm there is no gain, so that line carries only the norms.
#
# NOTE: --overfit never writes a checkpoint (the gate exits as soon as it passes or
# runs out of steps), so the per-run directory will be empty. It exists so queued
# jobs cannot share or clobber state, and so a non-gate rerun with the same
# CKPT_DIR lands somewhere of its own.
#
# 400 steps at batch 4 is ~32 passes over the 50 examples; budget ~20 min including
# the model load, well inside the hour.
#
# Compute nodes have NO internet: run scripts/setup_narval.sh on a login node first.
set -euo pipefail

usage() {
    echo "usage: sbatch scripts/gate_scale.sh <prefix-scale> [--no-input-norm] [--no-output-norm]"
    echo "   e.g. sbatch scripts/gate_scale.sh 5"
    echo "        sbatch scripts/gate_scale.sh 1 --no-input-norm"
}

SCALE=""
NO_INPUT=0
NO_OUTPUT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-input-norm)  NO_INPUT=1 ;;
        --no-output-norm) NO_OUTPUT=1 ;;
        -h|--help)        usage; exit 0 ;;
        -*)
            echo "unknown flag '$1'" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "$SCALE" ]]; then
                echo "expected one prefix scale, got a second positional '$1'" >&2
                usage >&2
                exit 2
            fi
            SCALE="$1"
            ;;
    esac
    shift
done

if [[ -z "$SCALE" ]]; then
    usage >&2
    exit 2
fi
if ! [[ "$SCALE" =~ ^[0-9]+(\.[0-9]+)?$ ]] || [[ "$SCALE" == "0" ]]; then
    echo "prefix-scale must be a positive number, got '$SCALE'" >&2
    exit 2
fi
# 1 -> "1", 20 -> "20", 0.5 -> "0.5": a stable tag for filenames.
TAG="$(printf '%g' "$SCALE")"

# Build the flag pass-through and the matching filename suffix together, so a run's
# name can never disagree with the architecture it actually ran.
EXTRA=()
if (( NO_INPUT )); then
    EXTRA+=(--no-input-norm)
    TAG+="_noinput"
fi
if (( NO_OUTPUT )); then
    EXTRA+=(--no-output-norm)
    TAG+="_nooutput"
fi
# Caught again in train.py; repeated here so it fails in seconds, before the module
# load and the 14B, instead of after.
if (( NO_OUTPUT )) && [[ "$(printf '%g' "$SCALE")" != "1" ]]; then
    echo "--no-output-norm removes the LayerNorm that --prefix-scale calibrates;" >&2
    echo "it is only meaningful at scale 1, got '$SCALE'" >&2
    exit 2
fi

module load StdEnv/2023 python/3.11 cuda

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/ctb-liyue/$USER/ecg-llm-alignment}"
VENV="${VENV:-$PROJECT_DIR/venv}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/hf_cache/DeepSeek-R1-Distill-Qwen-14B}"
# One directory per (scale, flags) point, so queued jobs cannot collide.
CKPT_DIR="${CKPT_DIR:-$SCRATCH/ecg-llm-alignment/checkpoints/gate_scale$TAG}"
MAX_STEPS="${MAX_STEPS:-400}"
TARGET_LOSS="${TARGET_LOSS:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-4}"

source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # no network on compute nodes
mkdir -p "$CKPT_DIR" logs

cd "$SLURM_SUBMIT_DIR"
LOG="logs/gate_scale${TAG}.log"
echo "host: $(hostname)   prefix_scale: $SCALE   flags: ${EXTRA[*]:-none}"
echo "ckpt: $CKPT_DIR   log: $LOG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# tee so the run has a log named by its scale and flags; slurm's own --output file
# keeps whatever happens before this point (module load, venv, preflight failures).
# ${EXTRA[@]+...} because `set -u` treats an unset empty array as unbound on bash 4.
srun python scripts/train.py \
    --model-dir "$MODEL_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    --prefix-scale "$SCALE" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    --overfit 50 \
    --batch-size "$BATCH_SIZE" \
    --overfit-target-loss "$TARGET_LOSS" \
    --overfit-max-steps "$MAX_STEPS" \
    --prefix-log-every 50 \
    2>&1 | tee "$LOG"
