#!/usr/bin/env bash
# Default cutoffs are 1,000–8,000 rounds; select physical GPUs explicitly.
set -euo pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec ./run_hogwild.sh --budget-sweep "$@"
