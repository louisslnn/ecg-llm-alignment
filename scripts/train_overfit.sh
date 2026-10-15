#!/bin/bash
#SBATCH --account=ctb-liyue
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=ecg-overfit
#SBATCH --output=logs/train_overfit_%j.log
#
# THE GATE: can the resampler overfit 50 training examples, and does that show up
# in what the student then WRITES? A short submission that must pass before
# launching the full run. It trains on the first 50 train examples, no validation,
# logs loss every step, and every 50 steps measures the eval-mode mean loss over all
# 50 -- the loop stops when THAT reaches the target, not when one batch does. When
# the loop stops it checks three conditions on those same 50
# (scripts/overfit_gate.py):
#   1. teacher-forced loss below --gate-threshold (default 0.1)
#   2. free greedy generation: >= 90% close with <END>, mean length within 25% of
#      the targets', and the generated Yes/No matches the label on >= 90%
#   3. loss at the Yes/No token below 0.1 for label=True and label=False separately
# All three print with their thresholds either way; exit 0 only if all pass, exit 1
# with the failed conditions named on the GATE FAILED line. The loss alone is not the
# gate: run 1 reached 0.0496 and its free generation was still degenerate.
#
# The two hours are sized for the WORST case, so that a job killed at the limit is
# not what a failing gate looks like in sacct: ~15 min model load, the overfit run
# to its 3000 step cap (~23 min, plus a few for the stop checks), then 50 greedy
# decodes at batch 1 that all run to max_new_tokens. The gate's own exit is 0 or 1;
# TIMEOUT means the budget was wrong, not that the gate failed. If --overfit-max-steps
# goes up, raise --time with it.
#
# The gate's per-example records (gate_teacher_forced_{real,shuffled,zeroed}.jsonl,
# gate_generations.jsonl) land in CKPT_DIR, on scratch. The shuffled and zeroed
# files are the diagnostic controls; the verdict reads the real one alone.
#
# overfit_final.pt lands there too: the resampler the gate judged, written once the
# gate has run whether it passed or failed (~3.4 GB; no optimizer state, so it is
# for probing with --ckpt, not for resuming). The log's last line repeats the
# verdict and gives its path.
#
# TWO OTHER GATES, each an env override; an unmodified submission is the gate above.
#   LOSS_ON=decision            train on ONE position per example, the Yes/No token
#                               after "Conclusion:" (--loss-on; default all). The
#                               verdict is then condition 3 alone.
#   OVERFIT_SELECT=stratified   draw the 10 records so every superclass has at least
#                               three Yes and three No (--overfit-select; default
#                               head, the first 50 examples, where all ten records
#                               are normal and the answer follows from the question).
#   TASK_STREAM=question        what the resampler's second stream reads
#                               (--task-stream; default prompt): prompt-and-target,
#                               question (the paraphrase alone, no patient
#                               context) or none (no text).
# Each one moves the default CKPT_DIR (overfit_gate_decision, overfit_gate_stratified,
# overfit_gate_question, ...; TASK_STREAM tags as _withtarget, _question, _notext):
# every gate writes the same file names, and a different experiment must not land
# on the baseline's records and overfit_final.pt.
#
# Compute nodes have NO internet: run scripts/setup_narval.sh on a login node first.
# Submit from the repo root after `mkdir -p logs`:
#   sbatch scripts/train_overfit.sh
#   LOSS_ON=decision OVERFIT_SELECT=stratified sbatch scripts/train_overfit.sh
#   TASK_STREAM=question OVERFIT_SELECT=stratified sbatch scripts/train_overfit.sh
set -euo pipefail

# --- which gate --------------------------------------------------------------
LOSS_ON="${LOSS_ON:-all}"
OVERFIT_SELECT="${OVERFIT_SELECT:-head}"
# Caught again in train.py; repeated here so a typo fails in seconds, before the
# module load and the 14B, instead of after.
case "$LOSS_ON" in all|decision) ;; *)
    echo "LOSS_ON must be all or decision, got '$LOSS_ON'" >&2; exit 2 ;;
esac
case "$OVERFIT_SELECT" in head|stratified) ;; *)
    echo "OVERFIT_SELECT must be head or stratified, got '$OVERFIT_SELECT'" >&2; exit 2 ;;
esac
TASK_STREAM="${TASK_STREAM:-prompt}"
case "$TASK_STREAM" in
    prompt)            STREAM_TAG="" ;;
    prompt-and-target) STREAM_TAG="_withtarget" ;;
    question)          STREAM_TAG="_question" ;;
    none)              STREAM_TAG="_notext" ;;
    *)
        echo "TASK_STREAM must be prompt, prompt-and-target, question or none," \
             "got '$TASK_STREAM'" >&2
        exit 2 ;;
esac
# Named into the default checkpoint dir below, and only when not the default, so
# the baseline gate keeps the directory it has always had.
GATE_TAG=""
[[ "$LOSS_ON" == "all" ]] || GATE_TAG+="_$LOSS_ON"
[[ "$OVERFIT_SELECT" == "head" ]] || GATE_TAG+="_$OVERFIT_SELECT"
GATE_TAG+="$STREAM_TAG"

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
CKPT_DIR="${CKPT_DIR:-$SCRATCH/ecg-llm-alignment/checkpoints/overfit_gate$GATE_TAG}"

source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$CKPT_DIR"

cd "$SLURM_SUBMIT_DIR"
echo "host: $(hostname)   model: $MODEL_DIR   loss_on: $LOSS_ON   overfit_select: $OVERFIT_SELECT   task_stream: $TASK_STREAM"
echo "ckpt: $CKPT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

srun python scripts/train.py \
    --model-dir "$MODEL_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    --overfit 50 \
    --overfit-select "$OVERFIT_SELECT" \
    --loss-on "$LOSS_ON" \
    --task-stream "$TASK_STREAM" \
    --batch-size 4 \
    --overfit-target-loss 0.05 \
    --overfit-max-steps 3000
