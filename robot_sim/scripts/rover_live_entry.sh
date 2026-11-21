#!/usr/bin/env bash
set -eo pipefail
export PS1=dummy
source /root/.bashrc
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ -f require-imu ]]; then set -- --imu "$@"; fi
if [[ -f front-clearance-10cm ]]; then set -- "$@" --front-clearance 0.10; fi
exec 9>/tmp/rbm-live-qwen.lock
flock -n 9
timeout 95s python3 -m robot_sim.candidate_nav.sensor_startup --lock-fd 9 "$@"
exec python3 -m robot_sim.candidate_nav.live "$@"
