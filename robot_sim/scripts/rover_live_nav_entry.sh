#!/usr/bin/env bash
set -eo pipefail
export PS1=dummy
source /root/.bashrc
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec 9>/tmp/rbm-live-qwen.lock
flock -n 9
# Check/recover mapping before creating a motion-capable agent.
timeout 95s python3 -m robot_sim.candidate_nav.sensor_startup --lock-fd 9 "$@"
stage=$(mktemp -d /tmp/rover-live-nav.XXXXXX)
pids=()
output=
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
    if [[ ${args[i]} == --output ]]; then output=${args[i+1]}; fi
done
cleanup() {
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
    if [[ -n $output && -d $output ]]; then
        for file in "$stage"/*.log "$stage/nav2.yaml"; do
            if [[ -f $file ]]; then cp "$file" "$output/"; fi
        done
    else
        tail -n 35 "$stage"/*.log >&2 || true
    fi
    rm -rf -- "$stage"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
strategy=1
for ((i=0; i<${#args[@]}; i++)); do
    if [[ ${args[i]} == --strategy ]]; then strategy=${args[i+1]}; fi
done
if [[ $strategy == 3 || $strategy == 4 ]]; then
    python3 -m robot_sim.candidate_nav.live_waypoint "$@"
    exit
fi
python3 -m robot_sim.candidate_nav.live_nav_config "$stage/nav2.yaml"
for component in controller planner; do
    exe=$(ros2 pkg prefix "nav2_$component")/lib/nav2_$component/${component}_server
    "$exe" --ros-args -r __ns:=/rover_live -r cmd_vel:=/rover_live/guarded_cmd_vel \
        --params-file "$stage/nav2.yaml" > "$stage/$component.log" 2>&1 &
    pids+=("$!")
    node=/rover_live/${component}_server
    timeout 15s bash -c 'until ros2 lifecycle get "$1" 2>/dev/null | grep -q unconfigured; do sleep 0.2; done' -- "$node"
    timeout 15s ros2 lifecycle set "$node" configure
    timeout 15s ros2 lifecycle set "$node" activate
done
for plugin in FollowPath Escape; do
    timeout 8s ros2 param get /rover_live/controller_server "$plugin.use_collision_detection" | grep 'Boolean value is: True'
done
private_info=$(timeout 8s ros2 topic info /rover_live/guarded_cmd_vel --verbose)
grep -q 'Type: geometry_msgs/msg/Twist$' <<< "$private_info"
grep -q 'Node namespace: /rover_live$' <<< "$private_info"
motor_info=$(timeout 8s ros2 topic info /cmd_vel --verbose)
if grep -q 'Node namespace: /rover_live$' <<< "$motor_info"; then
    echo 'Nav2 velocity remapping failed' >&2
    exit 1
fi
python3 -m robot_sim.candidate_nav.live_waypoint "$@"
