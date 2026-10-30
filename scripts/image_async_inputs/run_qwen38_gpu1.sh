#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/mnt/LLM
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-1}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
args=( \
  --model-name /mnt/LLM/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --output scripts/image_async_inputs/eval_runs/qwen38_27b_gpu1_k64_16k \
  --k-steps 64 --budget 16384 --max-seq-len 65536 --kv-tokens 98304 \
  --distributed-port "$((2370 + CUDA_VISIBLE_DEVICES))" "$@" )
# Existing sweep processes call this wrapper afresh for each condition.
# Lock the output across GPUs; a completed condition is checked and skipped.
output=scripts/image_async_inputs/eval_runs/qwen38_27b_gpu1_k64_16k
for ((i=0;i<${#args[@]};i++)); do
  if [[ "${args[i]}" == --output ]]; then output="${args[i+1]}"; fi
done
mkdir -p "$output"
exec 6>"$output/run.lock"
flock 6
if .venv/bin/python scripts/image_async_inputs/skip_completed_eval.py "${args[@]}"; then
  exit 0
else
  status=$?
  if [[ "$status" != 10 ]]; then exit "$status"; fi
fi
if [[ "$CUDA_VISIBLE_DEVICES" == 6 ]]; then
  occupants=$(nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader)
  if [[ -n "$occupants" ]]; then
    echo 'GPU 6 is occupied; refusing to start an additional model.' >&2
    exit 1
  fi
fi
exec uv run --no-sync python -u scripts/image_async_inputs/image_async_thoughts_eval.py "${args[@]}"
