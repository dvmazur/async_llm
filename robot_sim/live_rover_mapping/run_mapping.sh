#!/usr/bin/env bash
set -eo pipefail
export PS1=dummy
source /root/.bashrc
cd /opt/rover-mapping-quality
pids=()
cleanup(){ for pid in "${pids[@]}"; do kill -INT "$pid" 2>/dev/null || true; done; wait || true; }
trap cleanup EXIT
trap 'exit 130' INT TERM
ros2 launch main icp.yaml > icp.log 2>&1 &
pids+=("$!")
ros2 launch slam_toolbox online_async_launch.py use_sim_time:=false slam_params_file:=/src/install/main/share/main/config/params.yaml > slam.log 2>&1 &
pids+=("$!")
wait -n "${pids[@]}"
