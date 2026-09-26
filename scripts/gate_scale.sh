#!/bin/bash
#SBATCH --account=ctb-liyue
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --job-name=ecg-gate-scale
#SBATCH --output=logs/gate_scale_%j.slurm.log
#
# ONE POINT OF THE PREFIX-SCALE SWEEP: overfit 50 examples with the output
# LayerNorm calibrated to <scale> x the student's embedding norm, and see whether
# the gate still passes. Submit it once per point:
#
#   mkdir -p logs
#   sbatch scripts/gate_scale.sh 1     -> logs/gate_scale1_400steps.log
#   sbatch scripts/gate_scale.sh 5
#   sbatch scripts/gate_scale.sh 20
#
# --steps N raises the step cap from the 400 a gate check needs to however long it
# takes to actually reach the target loss:
#
#   sbatch scripts/gate_scale.sh 5 --steps 3000   -> logs/gate_scale5_3000steps.log
#
# ARCHITECTURE ABLATIONS, one change at a time. Both LayerNorms are on by default
# (they are the two departures from the reference resampler); either can be
# dropped, and the flag goes into the log name and the checkpoint dir so an
# ablation never lands on a baseline's files:
#
#   sbatch scripts/gate_scale.sh 1 --no-input-norm
#                                     -> logs/gate_scale1_400steps_noinput.log
#   sbatch scripts/gate_scale.sh 1 --no-output-norm
#                                     -> logs/gate_scale1_400steps_nooutput.log
#   sbatch scripts/gate_scale.sh 1 --no-input-norm --no-output-norm
#                                     -> logs/gate_scale1_400steps_noinput_nooutput.log
#
# --no-output-norm removes the thing --prefix-scale calibrates, so it is only
# accepted at scale 1 (train.py refuses the combination too); with both norms off
# the resampler is the reference architecture exactly.
#
# Every run's log and checkpoint directory are named for what produced them --
# logs/gate_scale<scale>_<steps>steps<flags>.log -- so several can be in the queue
# at once without touching each other.
#
# WHAT VARIES: --prefix-scale, the step cap and the two norm flags, nothing else.
# The prompt keeps the template's <think> block (DatasetConfig.strip_think is off by
# default, as run 1 trained), so a difference between runs can only come from how
# loud the prefix is and which norms are present.
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
# THE TIME BUDGET. At ~0.45 s/step and batch 4, plus a ~15 min model load:
#   400 steps  (~32 passes over the 50 examples)  ~3 min   -> ~18 min total
#   3000 steps (~240 passes)                      ~23 min  -> ~38 min total
# The 1:30 bin holds 3000 comfortably. Raise --time before raising --steps much
# past that: the gate is killed mid-run at the limit, and since --overfit writes no
# checkpoint there is nothing to resume from.
#
# Compute nodes have NO internet: run scripts/setup_narval.sh on a login node first.
set -euo pipefail

DEFAULT_STEPS=400

usage() {
    echo "usage: sbatch scripts/gate_scale.sh <prefix-scale> [--steps N]" \
         "[--no-input-norm] [--no-output-norm]"
    echo "   e.g. sbatch scripts/gate_scale.sh 5"
    echo "        sbatch scripts/gate_scale.sh 5 --steps 3000"
    echo "        sbatch scripts/gate_scale.sh 1 --no-input-norm"
    echo "   --steps is the --overfit-max-steps cap (default $DEFAULT_STEPS)"
}

SCALE=""
STEPS="$DEFAULT_STEPS"
NO_INPUT=0
NO_OUTPUT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)
            if [[ $# -lt 2 ]]; then
                echo "--steps needs a value" >&2
                usage >&2
                exit 2
            fi
            STEPS="$2"
            shift
            ;;
        --steps=*)        STEPS="${1#*=}" ;;
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
if ! [[ "$STEPS" =~ ^[0-9]+$ ]] || (( STEPS == 0 )); then
    echo "--steps must be a positive whole number, got '$STEPS'" >&2
    exit 2
fi
# 1 -> "1", 20 -> "20", 0.5 -> "0.5": a stable tag for filenames.
TAG="$(printf '%g' "$SCALE")"
# The step cap is always in the tag, including at the default. Leaving it out when
# it happens to equal DEFAULT_STEPS would make a 400-step log indistinguishable
# from one run after the default changed -- and the whole point of the tag is that
# a filename says what produced it.
TAG+="_${STEPS}steps"

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
# One directory per (scale, steps, flags) point, so queued jobs cannot collide.
CKPT_DIR="${CKPT_DIR:-$SCRATCH/ecg-llm-alignment/checkpoints/gate_scale$TAG}"
TARGET_LOSS="${TARGET_LOSS:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-4}"

source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # no network on compute nodes
mkdir -p "$CKPT_DIR" logs

cd "$SLURM_SUBMIT_DIR"
LOG="logs/gate_scale${TAG}.log"
echo "host: $(hostname)   prefix_scale: $SCALE   steps: $STEPS   flags: ${EXTRA[*]:-none}"
echo "ckpt: $CKPT_DIR   log: $LOG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# tee so the run has a log named by its scale, step cap and flags; slurm's own
# --output file
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
    --overfit-max-steps "$STEPS" \
    --prefix-log-every 50 \
    2>&1 | tee "$LOG"
