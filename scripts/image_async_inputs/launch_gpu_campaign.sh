#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/mnt/LLM
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=''
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export UV_CACHE_DIR=/tmp/doom-uv-cache
control=scripts/image_async_inputs/eval_runs/qwen38_27b_suffix_v2_5gpu_16k
mkdir -p "$control"
exec uv run --no-sync python -u scripts/image_async_inputs/run_gpu_campaign.py >> "$control/campaign.log" 2>&1
