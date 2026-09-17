#!/bin/bash
# Reproducible Narval environment + checkpoint pre-download.
#
# RUN THIS ON A LOGIN NODE. Login nodes have internet; compute nodes do not, so
# both the wheels and the student checkpoint must be fetched here, up front.
#
# Alliance ships prebuilt wheels; we install with `pip install --no-index` from
# that wheelhouse. No conda, no PyPI. Exact resolved versions are frozen to a
# lockfile so the environment can be rebuilt bit-for-bit.
#
#   bash scripts/setup_narval.sh
#
# Override any path with an env var, e.g.:
#   PROJECT_DIR=/project/def-liyue/$USER/ecg bash scripts/setup_narval.sh
set -euo pipefail

module load StdEnv/2023 python/3.11 cuda

# --- paths (all under project space, which is quota'd and backed up) ----------
PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/def-liyue/$USER/ecg-llm-alignment}"
VENV="${VENV:-$PROJECT_DIR/venv}"
HF_CACHE="${HF_CACHE:-$PROJECT_DIR/hf_cache}"
LOCK="${LOCK:-$PROJECT_DIR/requirements-narval.lock}"

MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MODEL_DIR="$HF_CACHE/DeepSeek-R1-Distill-Qwen-14B"

mkdir -p "$PROJECT_DIR" "$HF_CACHE"

# --- 1. virtualenv from the Alliance wheelhouse ------------------------------
echo "== creating venv at $VENV =="
virtualenv --no-download "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip

echo "== installing wheels (--no-index) =="
pip install --no-index \
    torch transformers accelerate safetensors huggingface_hub \
    numpy pandas scipy

# --- 2. freeze exact versions ------------------------------------------------
pip freeze > "$LOCK"
echo "== recorded exact versions -> $LOCK =="

# --- 3. pre-download the checkpoint into project space (login node only) ------
echo "== pre-downloading $MODEL_ID -> $MODEL_DIR =="
python - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download(
    "$MODEL_ID",
    local_dir="$MODEL_DIR",
    local_dir_use_symlinks=False,          # real files, so compute nodes can read them
    ignore_patterns=["*.pth", "*.gguf"],   # keep only the safetensors weights
)
print("checkpoint materialised at", path)
PY

# --- 4. report on-disk size for quota checking -------------------------------
echo "== checkpoint size on disk (check against your /project quota) =="
du -sh "$MODEL_DIR"

cat <<EOF

done.
  venv:        $VENV
  lockfile:    $LOCK
  checkpoint:  $MODEL_DIR

Next: submit the probe from the repo root (so scripts/ and data/ resolve):
  mkdir -p logs
  MODEL_DIR="$MODEL_DIR" VENV="$VENV" sbatch scripts/probe_memory.sh
EOF
