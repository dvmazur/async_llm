#!/usr/bin/env bash
# Run the AsyncReasoning-style "async thoughts" demo on minisgl.
#
# Mirrors yandex-research/AsyncReasoning's notebooks/demo_async_thoughts.ipynb
# but in-process via minisgl's SharedCacheSession (no notebook, no server).
#
# Usage:
#   ./scripts/run_async_thoughts.sh
#   MODEL=Qwen/Qwen3-8B ./scripts/run_async_thoughts.sh
#   PROBLEM="Why is the sky blue?" ./scripts/run_async_thoughts.sh
#   ./scripts/run_async_thoughts.sh --max-steps 400 --probe-period 20
#
# Env vars (all optional):
#   MODEL                HF model path. Default: Qwen/Qwen3-32B.
#                        Use a smaller model (e.g. Qwen/Qwen3-8B) for quicker
#                        smoke tests; the writer is starved on smaller models
#                        because the probe margin is too small to keep
#                        thinker_and_writer state active for more than one
#                        decode step at a time.
#   PROBLEM              Question for the assistant to solve. Default: the
#                        polynomial-evaluation problem from the original
#                        AsyncReasoning notebook.
#   MAX_STEPS            Total decode steps before forced termination. Default: 800.
#   PROBE_PERIOD         Run the mode-switching probe every N decode steps.
#                        Default: 30.
#   MEMORY_RATIO         Fraction of free GPU memory to reserve for the KV
#                        pool. Default: 0.6 (needed for Qwen3-32B at bf16 on
#                        a single GPU; lower it to ~0.3 for 8B-class models
#                        if you also want to keep an HF model alongside).
#   CUDA_VISIBLE_DEVICES Which GPU to pin to. Default: 0.
#
# Any additional CLI flags after the env-var prefix are forwarded to the
# Python script verbatim.
#
# Prerequisites:
#   * CUDA-capable GPU. The demo prints an error and exits if torch.cuda
#     isn't available.
#   * HF Hub access on first run (model weights download to ~/.cache/huggingface
#     if not already present). 32B at bf16 is ~64 GiB on disk + same on GPU.
#   * `uv` for managing the Python environment (https://docs.astral.sh/uv/).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-32B}"
PROBLEM="${PROBLEM:-Calculate x - x^2 + x^3 for x = 5, 6, 7, 8.  Return all 4 answers in \\boxed{ }.}"
MAX_STEPS="${MAX_STEPS:-800}"
PROBE_PERIOD="${PROBE_PERIOD:-30}"
MEMORY_RATIO="${MEMORY_RATIO:-0.6}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' not found on PATH. Install from https://docs.astral.sh/uv/ ." >&2
    exit 2
fi

echo "Async-thoughts demo (minisgl)"
echo "  model         : $MODEL"
echo "  problem       : $PROBLEM"
echo "  max steps     : $MAX_STEPS"
echo "  probe period  : $PROBE_PERIOD"
echo "  memory ratio  : $MEMORY_RATIO"
echo "  CUDA device   : ${CUDA_VISIBLE_DEVICES}"
echo ""

cd "$REPO_ROOT"
exec uv run python scripts/demo_async_thoughts.py \
    --model "$MODEL" \
    --problem "$PROBLEM" \
    --max-steps "$MAX_STEPS" \
    --probe-period "$PROBE_PERIOD" \
    --memory-ratio "$MEMORY_RATIO" \
    "$@"
