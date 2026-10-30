#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
results=scripts/image_async_inputs/eval_runs/qwen38_27b_gpu1_full_sweep_16k
mkdir -p "$results"
exec 9>"$results/campaign.lock"
flock -n 9 || { echo 'Full sweep already running.' >&2; exit 1; }
worker() {
  local gpu=$1; shift
  if [[ "$gpu" == 6 ]]; then
    local occupants
    occupants=$(nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader)
    if [[ -n "$occupants" ]]; then
      echo 'GPU 6 occupied: GPU 1 will handle the full sweep alone.'
      return 0
    fi
  fi
  for k in "$@"; do
    echo "Starting full dataset GPU=$gpu k=$k at $(date -Is)"
    EVAL_GPU="$gpu" bash scripts/image_async_inputs/run_qwen38_gpu1.sh \
      --k-steps "$k" --output "$results/k_$k" > "$results/k_$k.gpu$gpu.log" 2>&1
    echo "Completed full dataset GPU=$gpu k=$k at $(date -Is)"
  done
}
worker 1 -1 0 16 32 64 128 256 512 &
gpu1_pid=$!
worker 6 512 256 128 64 32 16 0 -1 &
gpu6_pid=$!
status=0
wait "$gpu1_pid" || status=1
wait "$gpu6_pid" || status=1
if [[ "$status" != 0 ]]; then echo 'A full sweep worker failed.' >&2; exit 1; fi
echo 'All full-dataset conditions completed.'
