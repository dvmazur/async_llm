#!/usr/bin/env bash
set -euo pipefail
exec 8>/tmp/rbm-live-qwen.lock
flock -n 8 || { echo 'A rover run is active; mapping was not restarted'; exit 1; }
if pgrep -f '^python3 -m robot_sim.candidate_nav.live_waypoint( |$)' >/dev/null; then exit 1; fi
if [[ ${1:-} == --restore ]]; then
    python3 /opt/rover-mapping-quality/mapping_quality.py --restore
fi
if tmux has-session -t rover-mapping-quality 2>/dev/null; then
    tmux send-keys -t rover-mapping-quality C-c
    for i in $(seq 1 50); do
        if ! tmux has-session -t rover-mapping-quality 2>/dev/null; then break; fi
        sleep .2
    done
    if tmux has-session -t rover-mapping-quality 2>/dev/null; then echo 'Previous mapping session did not stop'; exit 1; fi
fi
pids=$(pgrep -f '^/opt/ros/jazzy/lib/(rtabmap_odom/icp_odometry|slam_toolbox/async_slam_toolbox_node)( |$)' || true)
if [[ -n $pids ]]; then kill -TERM $pids; fi
for i in $(seq 1 50); do
    if ! pgrep -f '^/opt/ros/jazzy/lib/(rtabmap_odom/icp_odometry|slam_toolbox/async_slam_toolbox_node)( |$)' >/dev/null; then break; fi
    sleep .2
done
if pgrep -f '^/opt/ros/jazzy/lib/(rtabmap_odom/icp_odometry|slam_toolbox/async_slam_toolbox_node)( |$)' >/dev/null; then exit 1; fi
tmux new-session -d -s rover-mapping-quality 'exec bash /opt/rover-mapping-quality/run_mapping.sh'
echo 'Mapping restarted. No motor commands were sent.'
