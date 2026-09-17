#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
fixed=scripts/image_async_inputs/eval_runs/qwen38_27b_gpu1_fixed50_16k
exec 8>"$fixed/handoff.lock"
flock -n 8 || { echo 'A fixed50 handoff is already queued.' >&2; exit 1; }
echo "Waiting for fixed50 campaign lock at $(date -Is)"
# Wait on the actual sweep lock, not a guessed elapsed time. Retain the lock
# during handoff/full run so the fixed sweep cannot restart on the same GPU.
exec 7>"$fixed/campaign.lock"
flock 7
echo "Fixed50 process released its lock at $(date -Is); validating results"
.venv/bin/python scripts/image_async_inputs/report_fixed50.py \
  --root "$fixed" --manifest scripts/image_async_inputs/fixed_subset_50.json
echo "Validated results saved; launching full sweep at $(date -Is)"
bash scripts/image_async_inputs/run_full_sweep_gpu1.sh
