#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
results=scripts/image_async_inputs/eval_runs/qwen38_27b_gpu1_fixed50_16k
exec 9>"$results/gpu6.lock"
flock -n 9 || { echo 'GPU 6 sweep worker already active.' >&2; exit 1; }
# GPU 1 owns the currently running -1 condition. Work backwards through the rest.
for k in 512 256 128 64 32 16 0; do
  echo "GPU 6 starting fixed50 k=$k at $(date -Is)"
  EVAL_GPU=6 bash scripts/image_async_inputs/run_qwen38_gpu1.sh \
    --sample-manifest scripts/image_async_inputs/fixed_subset_50.json \
    --k-steps "$k" --output "$results/k_$k" > "$results/k_$k.gpu6.log" 2>&1
done
echo 'GPU 6 fixed50 worker complete.'
