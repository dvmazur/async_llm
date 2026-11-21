#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec python3 robot_sim/scripts/rover_live_launch.py "$@"
