#!/usr/bin/env bash
set -eo pipefail
export PS1=dummy
source /root/.bashrc
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec 8>/tmp/rover-live-view.lock
flock -n 8 || exit 0
printf '%s\n' "$$" > /tmp/rover-live-view.pid
exec > /tmp/rover-live-view.log 2>&1
exec python3 -m robot_sim.candidate_nav.live_view "$@"
