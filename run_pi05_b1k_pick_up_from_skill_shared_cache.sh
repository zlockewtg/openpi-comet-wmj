#!/usr/bin/env bash
set -euo pipefail

cd /mnt/public/tgy/openpi-comet-wmj

export SHARED_CACHE_ROOT=/mnt/public/tgy/cache
export XDG_CACHE_HOME="${SHARED_CACHE_ROOT}/xdg"
export HF_HOME="${SHARED_CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${SHARED_CACHE_ROOT}/torch"
export TRITON_CACHE_DIR="${SHARED_CACHE_ROOT}/triton"
export PIP_CACHE_DIR="${SHARED_CACHE_ROOT}/pip"
export UV_CACHE_DIR="${SHARED_CACHE_ROOT}/uv"
export WANDB_DIR="${SHARED_CACHE_ROOT}/wandb"
export WANDB_CACHE_DIR="${WANDB_DIR}/cache"
export MPLCONFIGDIR="${SHARED_CACHE_ROOT}/matplotlib"
export NUMBA_CACHE_DIR="${SHARED_CACHE_ROOT}/numba"
export TMPDIR=/mnt/public/tgy/tmp

mkdir -p \
    "$XDG_CACHE_HOME" \
    "$HF_DATASETS_CACHE" \
    "$HF_HUB_CACHE" \
    "$HF_MODULES_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$TORCH_HOME" \
    "$TRITON_CACHE_DIR" \
    "$PIP_CACHE_DIR" \
    "$UV_CACHE_DIR" \
    "$WANDB_CACHE_DIR" \
    "$MPLCONFIGDIR" \
    "$NUMBA_CACHE_DIR" \
    "$TMPDIR"

/mnt/public/tgy/openpi-comet-wmj/.venv/bin/python - <<'PY'
import os
from datasets import config

print("HF_HOME =", os.environ.get("HF_HOME"))
print("HF_DATASETS_CACHE =", config.HF_DATASETS_CACHE)
print("TMPDIR =", os.environ.get("TMPDIR"))
PY

exec /mnt/public/tgy/openpi-comet-wmj/.venv/bin/python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    scripts/train_pytorch.py \
    pi05_b1k_pick_up_from_skill
