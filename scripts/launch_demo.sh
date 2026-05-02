#!/usr/bin/env bash
# Launch a mini-sglang server with shared-cache enabled for the Hogwild demo.
#
# Usage:
#   ./scripts/launch_demo.sh
#   MODEL=Qwen/Qwen3-9B ./scripts/launch_demo.sh
#   MODEL=/local/path/to/model ./scripts/launch_demo.sh --port 8080

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-9B}"
PORT="${PORT:-1919}"
SC_BUDGET="${SC_BUDGET:-4096}"   # pages reserved for shared-cache blocks
TP="${TP:-1}"                     # tensor-parallel size

echo "Starting mini-sglang server"
echo "  model               : $MODEL"
echo "  port                : $PORT"
echo "  shared-cache budget : $SC_BUDGET pages"
echo "  tensor-parallel     : $TP"
echo ""

cd "$REPO_ROOT"
uv run python -m minisgl \
    --model "$MODEL" \
    --port "$PORT" \
    --shared-cache-page-budget "$SC_BUDGET" \
    --tensor-parallel-size "$TP" \
    "$@"
