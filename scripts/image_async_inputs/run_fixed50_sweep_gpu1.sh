#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
subset=scripts/image_async_inputs/fixed_subset_50.json
results=scripts/image_async_inputs/eval_runs/qwen38_27b_gpu1_fixed50_16k
mkdir -p "$results"
# Single campaign lock prevents accidental concurrent launches on GPU 1.
exec 9>"$results/campaign.lock"
flock -n 9 || { echo 'This sweep is already running.' >&2; exit 1; }
for k in -1 0 16 32 64 128 256 512; do
  echo "Starting fixed50 k=$k at $(date -Is)"
  bash scripts/image_async_inputs/run_qwen38_gpu1.sh \
    --sample-manifest "$subset" --k-steps "$k" \
    --output "$results/k_$k" >> "$results/k_$k.log" 2>&1
  echo "Completed fixed50 k=$k at $(date -Is)"
done
echo 'All eight fixed-50 conditions completed.'
