#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

for i in 1 2 3; do
    echo "=== EXPERIMENT: starting run $i/3 ==="
    uv run python reset_engine.py
    uv run python run_persistent.py 10 32000
    echo "=== EXPERIMENT: finished run $i/3 (exit $?) ==="
done
echo "=== EXPERIMENT: all 3 runs complete ==="
