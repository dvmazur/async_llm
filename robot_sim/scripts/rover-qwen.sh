#!/usr/bin/env bash
set -euo pipefail

server_profile=legacy
gpu_request=auto
model_choice=27b
ssh_opts=(-o IPQoS=none -o ConnectionAttempts=3 -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=2 -o ServerAliveCountMax=2)

# Reuse one owned server transport across recursive prepare/deploy/run calls.
init_server_transport() {
    if [[ -z ${ROVER_QWEN_SERVER_CONTROL:-} ]]; then
        local folder
        folder=$(mktemp -d /tmp/rq-ssh.XXXXXX)
        export ROVER_QWEN_SERVER_CONTROL="$folder/control"
        export ROVER_QWEN_SERVER_OWNER_PID=$$
    fi
    trap cleanup_server_transport EXIT
}

server_ssh() {
    local options=("${ssh_opts[@]}" -o ControlMaster=auto -o ControlPersist=yes
                   -S "$ROVER_QWEN_SERVER_CONTROL")
    if [[ ${1:-} != -O ]]; then
        if ! command ssh "${options[@]}" -O check "$server" >/dev/null 2>&1; then
            local attempt connected=false
            for attempt in 1 2 3; do
                # Retry only opening an empty transport, never a remote command.
                if command ssh "${options[@]}" -fNT "$server"; then connected=true; break; fi
                echo "SSH к $server: попытка $attempt из 3 не удалась." >&2
            done
            if ! $connected; then return 255; fi
        fi
    fi
    command ssh "${options[@]}" "$@"
}

cleanup_server_transport() {
    if [[ ${ROVER_QWEN_SERVER_OWNER_PID:-} == "$$" && $BASH_SUBSHELL == 0 ]]; then
        server_ssh -O exit "$server" >/dev/null 2>&1 || true
        case "$ROVER_QWEN_SERVER_CONTROL" in
            /tmp/rq-ssh.*/control) rm -rf "${ROVER_QWEN_SERVER_CONTROL%/control}" ;;
        esac
    fi
}

usage() {
    cat <<'EOF'
Usage:
  ./rover-qwen.sh start [--server legacy|h100] [--gpu auto|0|1] [--model 27b|35b|122b] [--rover 80|107] --strategy 1|2|3|4 --task '...' [--trajectory-steps 5] [--duration 180] [--dry-run] [--disable-collision-protection] [--no-long-term-memory] [--no-plan-rationale] [--sub-strategy 1|2]
  ./rover-qwen.sh stop
  ./rover-qwen.sh model-stop
  ./rover-qwen.sh deploy
  ./rover-qwen.sh smoke
  ./rover-qwen.sh gate-test
  ./rover-qwen.sh view
  ./rover-qwen.sh view-stop
  ./rover-qwen.sh model
  ./rover-qwen.sh run --instruction '...' [--wall-limit 60] [--enable-motion] [--waypoints] [--validate-only]

Use --rover 80 or --rover 107 before or after the command (default: 80).
--server h100 selects the H100 server (one GPU for 27b/35b, two for 122b); legacy (default) keeps the two-3090 server.
--gpu auto (default), 0 or 1 selects an eligible H100. Sharing requires explicit server authorization; other jobs are never stopped.
--model 27b (default): Qwen3.8-27B-FP8; 35b: Qwen3.6-35B-A3B-FP8; 122b: Qwen3.5-122B-A10B-FP8.
122b requires --server h100 and two idle H100 GPUs; --gpu auto or 0 selects the pair.
Switching unloads our previous model; active runs must finish before switching.
Both profiles use lidar odometry without IMU by default; --imu explicitly requires verified IMU.
Live view: rover 80 at http://127.0.0.1:18770, rover 107 at http://127.0.0.1:18771.
start prepares Qwen, rover and live view automatically. run is the low-level interface.
Strategies: 1 geometric candidates, 2 Qwen-generated candidates, 3 single direction, 4 action sequence.
Strategy 4: persistent CPU memory is enabled by default; --no-long-term-memory disables it.
Strategy 4: --sub-strategy 1 is the default; 2 adds memory and subgoal workers on H100.
Strategy 4 on H100: --no-plan-rationale skips the post-plan explanation and its review.
Strategy 4: --trajectory-steps sets the maximum actions per model request (1..32, default 5).
Strategy 4: --disable-collision-protection disables software obstacle filtering (not link/sensor checks).
Stop ends the sequence early. Sensors and motion checks stay active during execution.
Use --dry-run with start to verify without motion. Run defaults to no motion.
Smoke uses a CPU protocol fixture and never enables motion.
EOF
}

configure_trajectory() {
    trajectory_flags=()
    if [[ -n $sub_strategy ]]; then
        if [[ $strategy != 4 || ! $sub_strategy =~ ^[12]$ ]]; then
            echo '--sub-strategy requires --strategy 4 and a value of 1 or 2' >&2; exit 2
        fi
        if [[ $sub_strategy == 2 && $server_profile != h100 ]]; then
            echo '--sub-strategy 2 requires --server h100' >&2; exit 2
        fi
    fi
    if $memory_disabled && [[ $strategy != 4 ]]; then
        echo '--no-long-term-memory requires --strategy 4' >&2; exit 2
    fi
    if $plan_rationale_disabled && [[ $strategy != 4 || $server_profile != h100 ]]; then
        echo '--no-plan-rationale requires --server h100 and --strategy 4' >&2; exit 2
    fi
    if $collision_disabled && [[ $strategy != 4 ]]; then
        echo "--disable-collision-protection requires --strategy 4" >&2; exit 2
    fi
    if [[ $strategy == 4 ]]; then
        trajectory_steps=${trajectory_steps:-5}
        if [[ ! $trajectory_steps =~ ^([1-9]|[12][0-9]|3[0-2])$ ]]; then
            echo '--trajectory-steps must be an integer from 1 to 32' >&2
            exit 2
        fi
        trajectory_flags=(--trajectory-steps "$trajectory_steps")
        if [[ $sub_strategy == 2 ]]; then trajectory_flags+=(--sub-strategy 2); fi
        if [[ $sub_strategy == 1 && $server_profile == h100 ]] && ! $plan_rationale_disabled; then
            trajectory_flags+=(--no-plan-rationale)
        fi
        if $plan_rationale_disabled; then trajectory_flags+=(--no-plan-rationale); fi
    elif [[ -n $trajectory_steps ]]; then
        echo '--trajectory-steps requires --strategy 4' >&2
        exit 2
    fi
}

collision_disabled=false
memory_disabled=false
plan_rationale_disabled=false
sub_strategy=
rover_id=80
imu=false
remaining=()
while (($#)); do
    case "$1" in
        --no-long-term-memory) memory_disabled=true; shift ;;
        --sub-strategy) sub_strategy=${2:?Specify 1 or 2}; shift 2 ;;
        --no-plan-rationale) plan_rationale_disabled=true; shift ;;
        --disable-collision-protection) collision_disabled=true; shift ;;
        --server) server_profile=${2:?Specify legacy or h100}; shift 2 ;;
        --model) model_choice=${2:?Specify 27b, 35b or 122b}; shift 2 ;;
        --gpu) gpu_request=${2:?Specify auto, 0 or 1}; shift 2 ;;
        --rover) rover_id=${2:?Specify 80 or 107}; shift 2 ;;
        --imu) imu=true; shift ;;
        --no-imu) imu=false; shift ;;
        --task|--instruction|--strategy|--duration|--wall-limit|--trajectory-steps)
            remaining+=("$1" "${2:?Missing option value}"); shift 2 ;;
        *) remaining+=("$1"); shift ;;
    esac
done
set -- "${remaining[@]+"${remaining[@]}"}"
case "$rover_id" in
    80) rover=rover; install_dir=/opt/rover-live; view_port=18770 ;;
    107) rover=rover107; install_dir=/opt/rover-live-107; view_port=18771 ;;
    *) usage >&2; exit 2 ;;
esac
case "$server_profile" in
    legacy|3090) server_profile=legacy; server=5.187.77.37; repo=/home/double/data_scince_expedition/lets/async_llm ;;
    h100) server=h100; repo=/home/ilyamotowilov/data_scince_expedition/lets/async_llm ;;
    *) echo '--server must be legacy or h100' >&2; exit 2 ;;
esac
case "$gpu_request" in auto|0|1) ;; *) echo '--gpu must be auto, 0 or 1' >&2; exit 2 ;; esac
if [[ $server_profile == legacy && $gpu_request != auto ]]; then
    echo '--gpu is only available with --server h100' >&2; exit 2
fi
case "$model_choice" in
    27b|Qwen3.8-27B-FP8|Qwen/Qwen3.8-27B-FP8) model_choice=27b ;;
    35b|Qwen3.6-35B-A3B-FP8|Qwen/Qwen3.6-35B-A3B-FP8) model_choice=35b ;;
    122b|Qwen3.5-122B-A10B-FP8|Qwen/Qwen3.5-122B-A10B-FP8) model_choice=122b ;;
    *) echo '--model must be 27b, 35b or 122b' >&2; exit 2 ;;
esac
if [[ $model_choice == 122b && ( $server_profile != h100 || $gpu_request == 1 ) ]]; then
    echo '122b needs --server h100 and --gpu auto or 0 (uses both GPUs 0,1)' >&2; exit 2
fi
model_service_flags=" --model $model_choice"
if [[ $server_profile == h100 ]]; then model_service_flags+=" --gpu $gpu_request"; fi
view_url="http://127.0.0.1:$view_port"
target="rbm-$rover_id"
check_host="test \"\$(hostname -s)\" = '$target' || exit 1;"
entry="$install_dir/robot_sim/scripts/rover_live_entry.sh"
mode=${1:-help}
if (($#)); then shift; fi
if [[ $mode != run && $mode != start && $# != 0 ]]; then usage >&2; exit 2; fi

case "$mode" in
    help|-h|--help)
        usage
        exit
        ;;
    start)
        strategy=1
        task=
        duration=180
        trajectory_steps=
        dry_run=false
        while (($#)); do
            case "$1" in
                --strategy) strategy=${2:?Missing strategy}; shift 2 ;;
                --task) task=${2:?Missing task}; shift 2 ;;
                --duration) duration=${2:?Missing duration}; shift 2 ;;
                --trajectory-steps) trajectory_steps=${2:?Missing trajectory steps}; shift 2 ;;
                --dry-run) dry_run=true; shift ;;
                -h|--help) usage; exit ;;
                *) usage >&2; exit 2 ;;
            esac
        done
        case "$strategy" in 1|2|3|4) ;; *) echo 'Strategy must be 1, 2, 3 or 4' >&2; exit 2 ;; esac
        configure_trajectory
        if [[ -z $task ]]; then echo '--task is required' >&2; exit 2; fi
        python3 -c 'import math,sys; n=float(sys.argv[1]); raise SystemExit(not (math.isfinite(n) and n>0))' "$duration"
        if ! $dry_run && [[ ! -t 0 || ! -t 1 ]]; then
            echo 'Use an interactive Mac terminal, or add --dry-run' >&2
            exit 2
        fi
        self=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
        self_command=(bash "$self" --server "$server_profile" --gpu "$gpu_request" --model "$model_choice" --rover "$rover_id")
        if $memory_disabled; then self_command+=(--no-long-term-memory); fi
        if $plan_rationale_disabled; then self_command+=(--no-plan-rationale); fi
        if [[ -n $sub_strategy ]]; then self_command+=(--sub-strategy "$sub_strategy"); fi
        if $collision_disabled; then self_command+=(--disable-collision-protection); fi
        if $imu; then self_command+=(--imu); else self_command+=(--no-imu); fi
        init_server_transport
        echo "Preparing rover ${rover_id}…"
        ssh "${ssh_opts[@]}" "$rover" "$check_host if [ \"\$(docker inspect ros --format '{{.State.Running}}')\" != true ]; then docker start ros; fi"
        "${self_command[@]}" deploy
        echo 'Preparing Qwen on the server…'
        server_ssh "$server" "cd '$repo' && python3 robot_sim/scripts/rover_live_model_service.py$model_service_flags"
        "${self_command[@]}" view-stop >/dev/null 2>&1 || true
        "${self_command[@]}" view
        flags=(--enable-motion)
        if $dry_run; then flags=(--validate-only); fi
        echo "Rover $rover_id · strategy $strategy · model $model_choice · task: $task"
        echo "Live map: $view_url · Ctrl+C stops the run"
        exec "${self_command[@]}" run --strategy "$strategy" --instruction "$task" --wall-limit "$duration" \
            ${trajectory_flags[@]+"${trajectory_flags[@]}"} "${flags[@]}"
        ;;
    stop)
        stop_code="export PS1=dummy; source /root/.bashrc; exec python3 $install_dir/robot_sim/scripts/rover_live_stop.py"
        printf -v command '%q ' docker exec ros bash -c "$stop_code"
        if ! ssh "${ssh_opts[@]}" -o ConnectTimeout=3 "$rover" "$check_host $command"; then
            echo 'Remote stop unconfirmed. Switch rover power off physically.' >&2
            exit 1
        fi
        exit
        ;;
    model-stop)
        exec ssh "${ssh_opts[@]}" "$server" "cd '$repo' && python3 robot_sim/scripts/rover_live_model_service.py --stop"
        ;;
    model)
        exec ssh "${ssh_opts[@]}" "$server" "cd '$repo' && python3 robot_sim/scripts/rover_live_model_service.py$model_service_flags"
        ;;
    deploy)
        init_server_transport
        install='set -eu
            dest=$1
            stage=$(mktemp -d /opt/rover-live-stage.XXXXXX)
            trap '\''if [ -d "$stage/previous" ] && [ ! -e "$dest" ]; then mv "$stage/previous" "$dest"; fi; rm -rf "$stage"'\'' EXIT
            mkdir "$stage/bundle"
            tar -xf - -C "$stage/bundle"
            python3 -m compileall -q "$stage/bundle/robot_sim"
            test -x "$stage/bundle/robot_sim/scripts/rover_live_entry.sh"
            printf "%s\n" "$3" > "$stage/bundle/rover-id"
            if [ "$2" = true ]; then
                touch "$stage/bundle/require-imu"
            else
                touch "$stage/bundle/front-clearance-10cm"
            fi
            if [ -e "$dest" ]; then mv "$dest" "$stage/previous"; fi
            mv "$stage/bundle" "$dest"
            echo "Live rover agent installed at $dest"'
        printf -v command '%q ' docker exec -i ros flock -n /tmp/rbm-live-qwen.lock bash -c "$install" -- "$install_dir" "$imu" "$rover_id"
        bundle="${ROVER_QWEN_SERVER_CONTROL%/control}/bundle-$$.tar"
        if ! server_ssh "$server" "cd '$repo' && bash robot_sim/scripts/rover_live_bundle.sh" > "$bundle"; then
            echo "Не удалось получить комплект агента с $server. Установка на ровер не начата." >&2
            exit 1
        fi
        if ! tar -tf "$bundle" >/dev/null 2>&1; then
            echo 'Сервер прислал некорректный архив. Установка на ровер не начата.' >&2
            exit 1
        fi
        ssh "${ssh_opts[@]}" "$rover" "$check_host $command" < "$bundle"
        rm -f "$bundle"
        exit
        ;;
    view|view-stop)
        view_state="${TMPDIR:-/tmp}/rover-live-view-${UID}-${rover_id}"
        mkdir -p "$view_state"
        chmod 700 "$view_state"
        view_control="$view_state/ctl"
        if [[ $mode == view-stop ]]; then
            ssh "${ssh_opts[@]}" -S "$view_control" -O exit "$rover" >/dev/null 2>&1 || true
            stop_view='import os, signal, time
from pathlib import Path
p=Path("/tmp/rover-live-view.pid")
pid=int(p.read_text()) if p.exists() else 0
cmd=Path(f"/proc/{pid}/cmdline")
def ours():
    try:
        return b"robot_sim.candidate_nav.live_view" in cmd.read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
if ours():
    os.kill(pid, signal.SIGTERM)
    deadline=time.monotonic()+4
    while ours() and time.monotonic()<deadline:
        time.sleep(.05)
    if ours():
        raise SystemExit("Live viewer is still shutting down")'
            printf -v command '%q ' docker exec ros python3 -c "$stop_view"
            ssh "${ssh_opts[@]}" "$rover" "$check_host $command"
            echo 'Live view stopped'
            exit
        fi
        printf -v command '%q ' docker exec -d ros "$install_dir/robot_sim/scripts/rover_live_view_entry.sh" --rover-id "$rover_id"
        ssh "${ssh_opts[@]}" "$rover" "$check_host $command"
        container_address=$(ssh "${ssh_opts[@]}" "$rover" "$check_host docker inspect ros --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'")
        if [[ ! $container_address =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo 'Cannot determine the ROS container address' >&2
            exit 1
        fi
        ssh "${ssh_opts[@]}" -S "$view_control" -O exit "$rover" >/dev/null 2>&1 || true
        ssh "${ssh_opts[@]}" -M -S "$view_control" -fNT -o ExitOnForwardFailure=yes \
            -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
            -L "127.0.0.1:$view_port:$container_address:18870" "$rover"
        for ((attempt=0; attempt<30; attempt++)); do
            if curl --silent --fail --max-time 1 "$view_url/api/state" > "$view_state/health.json"; then
                if python3 -c 'import json,sys; raise SystemExit(json.load(open(sys.argv[1])).get("rover_id") != int(sys.argv[2]))' "$view_state/health.json" "$rover_id"; then
                    echo "Live map: $view_url"
                    exit
                fi
            fi
            sleep 0.3
        done
        echo 'Live viewer has not started. Check /tmp/rover-live-view.log in container ros.' >&2
        exit 1
        ;;
    gate-test)
        init_server_transport
        printf -v command '%q ' docker exec -i -e ROS_DOMAIN_ID=219 \
            -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST -w "$install_dir" ros \
            bash -c 'export PS1=dummy; source /root/.bashrc; exec python3 -'
        server_ssh "$server" "cd '$repo' && cat robot_sim/scripts/rover_live_gate_probe.py" |
            ssh "${ssh_opts[@]}" "$rover" "$check_host $command"
        exit
        ;;
    run|smoke)
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

motion=false
waypoints=false
validate=false
strategy=
trajectory_steps=
instruction=
budget=60
if [[ $mode == run ]]; then
    while (($#)); do
        case "$1" in
            --instruction) instruction=${2:?Missing instruction}; shift 2 ;;
            --wall-limit) budget=${2:?Missing wall limit}; shift 2 ;;
            --enable-motion) motion=true; shift ;;
            --waypoints) waypoints=true; strategy=1; shift ;;
            --strategy) strategy=${2:?Missing strategy}; waypoints=true; shift 2 ;;
            --trajectory-steps) trajectory_steps=${2:?Missing trajectory steps}; shift 2 ;;
            --validate-only) validate=true; shift ;;
            *) usage >&2; exit 2 ;;
        esac
    done
    if [[ -z $instruction ]]; then usage >&2; exit 2; fi
    if $waypoints; then
        case "$strategy" in 1|2|3|4) ;; *) echo 'Strategy must be 1, 2, 3 or 4' >&2; exit 2 ;; esac
        entry="$install_dir/robot_sim/scripts/rover_live_nav_entry.sh"
    fi
    configure_trajectory
    if $validate; then
        $waypoints && ! $motion || { echo 'Validation requires --waypoints and disabled motion' >&2; exit 2; }
    fi
    if $motion && [[ ! -t 0 || ! -t 1 ]]; then
        echo "Motion requires an interactive terminal." >&2
        exit 2
    fi
fi

init_server_transport
scratch=$(mktemp -d "${TMPDIR:-/tmp}/rbm-qwen.XXXXXX")
fixture_job=
lease_job=
link_job=
cleanup() {
    if [[ -n $lease_job ]]; then kill "$lease_job" 2>/dev/null || true; fi
    cleanup_server_transport
    if [[ -n $link_job ]]; then kill "$link_job" 2>/dev/null || true; fi
    if [[ -n $fixture_job ]]; then kill "$fixture_job" 2>/dev/null || true; fi
    rm -rf "$scratch"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_id="$(date -u +%Y%m%dT%H%M%S)-$$"
socket_path="/dev/shm/rbm-qwen-$run_id.sock"
output="/tmp/rover-qwen-$mode-$run_id"
port=18766
extra=(--rover-id "$rover_id")
if $memory_disabled; then extra+=(--no-long-term-memory); fi
if $collision_disabled; then extra+=(--disable-collision-protection); fi
if [[ $mode == smoke ]]; then
    port=18866
    fixture_flags=
    if $imu; then port=18867; fixture_flags=--imu; fi
    instruction='Protocol smoke test. Motion is disabled.'
    budget=38
    extra+=(--test-backend)
    server_ssh -n -tt -o ExitOnForwardFailure=yes \
        -L "127.0.0.1:$port:127.0.0.1:$port" "$server" \
        "cd '$repo' && timeout 80s python3 -u robot_sim/scripts/rover_live_smoke_backend.py $fixture_flags" \
        >"$scratch/fixture.log" 2>&1 &
    fixture_job=$!
    for ((attempt=0; attempt<30; attempt++)); do
        if grep -q 'FIXTURE READY' "$scratch/fixture.log"; then break; fi
        if ! kill -0 "$fixture_job" 2>/dev/null; then break; fi
        sleep 0.3
    done
    if ! grep -q 'FIXTURE READY' "$scratch/fixture.log"; then
        cat "$scratch/fixture.log" >&2
        exit 1
    fi
else
    model_target="$scratch/model.sock"
    server_ssh -fNT -o ExitOnForwardFailure=yes \
        -L "$model_target:127.0.0.1:$port" "$server"
fi
if [[ $mode == smoke ]]; then model_target="127.0.0.1:$port"; fi

forward_link=()
if [[ $mode == run ]]; then
    mkfifo "$scratch/lease-input"
    exec 3<>"$scratch/lease-input"
    ssh "${ssh_opts[@]}" -o ControlMaster=auto -o ControlPersist=yes -S "$ROVER_QWEN_SERVER_CONTROL" -T "$server" "cd '$repo' && python3 robot_sim/scripts/rover_model_lease.py --model $model_choice" \
        < "$scratch/lease-input" 3>&- > "$scratch/lease.log" 2>&1 &
    lease_job=$!
    for ((attempt=0; attempt<100; attempt++)); do
        if grep -q '^MODEL LEASE READY' "$scratch/lease.log"; then break; fi
        if ! kill -0 "$lease_job" 2>/dev/null; then cat "$scratch/lease.log" >&2; exit 1; fi
        sleep .1
    done
    if ! grep -q '^MODEL LEASE READY' "$scratch/lease.log"; then cat "$scratch/lease.log" >&2; exit 1; fi
    extra+=(--expected-model "$model_choice")
    link_socket="/dev/shm/rbm-link-$run_id.sock"
    server_ssh "$server" "cd '$repo' && cat robot_sim/candidate_nav/live_link.py" > "$scratch/link.py"
    python3 "$scratch/link.py" --ready-file "$scratch/link-ready" --token "$run_id" > "$scratch/link.log" 2>&1 &
    link_job=$!
    for ((attempt=0; attempt<30; attempt++)); do
        if [[ -s $scratch/link-ready ]]; then break; fi
        if ! kill -0 "$link_job" 2>/dev/null; then cat "$scratch/link.log" >&2; exit 1; fi
        sleep .1
    done
    [[ -s $scratch/link-ready ]] || { echo 'Control link startup failed' >&2; exit 1; }
    link_port=$(cat "$scratch/link-ready")
    forward_link=(-R "$link_socket:127.0.0.1:$link_port")
    extra+=(--link-socket "$link_socket" --link-token "$run_id")
fi
if [[ -n $strategy ]]; then
    extra+=(--strategy "$strategy")
    extra+=(${trajectory_flags[@]+"${trajectory_flags[@]}"})
    if $imu; then extra+=(--imu); fi
fi

tty=()
container=(docker exec -i)
if [[ -t 0 && -t 1 ]]; then tty=(-t); container+=(-t); fi
if $motion; then extra+=(--enable-motion); fi
if $validate; then extra+=(--validate-only); fi
printf -v command '%q ' "${container[@]}" ros "$entry" --model-socket "$socket_path" \
    --output "$output" --instruction "$instruction" --wall-limit "$budget" ${extra[@]+"${extra[@]}"}
echo "Rover logs: $output (inside container ros)"
run_status=0
ssh "${ssh_opts[@]}" ${tty[@]+"${tty[@]}"} -o ExitOnForwardFailure=yes \
    -R "$socket_path:$model_target" ${forward_link[@]+"${forward_link[@]}"} "$rover" \
    "$check_host trap 'rm -f $socket_path ${link_socket:-}' EXIT; $command" || run_status=$?
if ((run_status != 0)); then
    echo "Сеанс завершился с кодом $run_status. Проверь сообщение результата или ошибки выше." >&2
    echo "Журналы на ровере внутри контейнера ros: $output" >&2
    exit "$run_status"
fi

if [[ $mode == smoke ]]; then
    cat "$scratch/fixture.log"
    ssh "${ssh_opts[@]}" "$rover" "$check_host docker exec ros python3 -c \
        'import json; from pathlib import Path; p=Path(\"$output\"); s=json.loads((p / \"agent_summary.json\").read_text()); c=json.loads((p / \"policy_inputs.json\").read_text()); print(\"result:\", s[\"reason\"], \"decisions:\", s[\"decisions\"], \"imu:\", s[\"imu_enabled\"]); assert s[\"imu_enabled\"] == (\"$imu\" == \"true\") and not c[\"motion_enabled\"] and c[\"model_fixture\"]; raise SystemExit(0 if s[\"decisions\"] > 0 and s[\"reason\"] == \"wall_budget\" else 1)'"
    echo 'PASS: CPU fixture, real sensors, no motor publisher'
fi
