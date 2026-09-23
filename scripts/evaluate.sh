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
#   MODEL_DIR=... VENV=... sbatch scripts/evaluate.sh --limit 500
#   CKPT=$SCRATCH/ecg-llm-alignment/checkpoints/best.pt sbatch scripts/evaluate.sh
#
# THE CONTROL -- run this second, always, and read it next to the main run:
#   sbatch scripts/evaluate.sh --shuffle-embeddings
# (it writes eval_<split>_shuffled.jsonl, so it cannot clobber the main run).
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

module load StdEnv/2023 python/3.11 cuda

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/ctb-liyue/$USER/ecg-llm-alignment}"
VENV="${VENV:-$PROJECT_DIR/venv}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/hf_cache/DeepSeek-R1-Distill-Qwen-14B}"
# Checkpoints are written to scratch by train.sh; best.pt is the lowest val loss.
CKPT_DIR="${CKPT_DIR:-$SCRATCH/ecg-llm-alignment/checkpoints}"
CKPT="${CKPT:-$CKPT_DIR/best.pt}"
# Generations + metrics land on scratch too (project space is at its file quota).
EVAL_DIR="${EVAL_DIR:-$SCRATCH/ecg-llm-alignment/eval}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# The control writes its own file; overwriting the main run's generations with the
# shuffled ones would destroy the only thing worth comparing it against.
SUFFIX=""
for arg in "$@"; do
    [[ "$arg" == "--shuffle-embeddings" ]] && SUFFIX="_shuffled"
done
OUT="${OUT:-$EVAL_DIR/eval_${SPLIT}${SUFFIX}.jsonl}"

source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # no network on compute nodes
mkdir -p "$EVAL_DIR"

cd "$SLURM_SUBMIT_DIR"
echo "host: $(hostname)   model: $MODEL_DIR   ckpt: $CKPT   out: $OUT"
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
