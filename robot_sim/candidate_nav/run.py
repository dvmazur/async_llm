"""Isolated online-SLAM + Nav2 + Qwen candidate experiment supervisor."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from robot_sim.rover import ROOT, bridge_configuration, robot_description

from .scenes import SCENARIOS, build_scene


def model_health(port, allow_test_model=False):
    with socket.create_connection(("127.0.0.1", port), timeout=10) as connection:
        connection.settimeout(30)
        connection.sendall(b'{"health":true}\n')
        metadata = json.loads(connection.makefile("rb").readline())
    if not metadata.get("ready") or not metadata.get("shared_cache"):
        raise RuntimeError("A ready shared-cache AsyncLLM backend is required")
    if not allow_test_model and (
        "Qwen3.8-27B" not in metadata.get("model", "")
        or metadata.get("tp_size") != 2
        or metadata.get("quantization") != "fp8"
    ):
        raise RuntimeError(
            "This experiment requires Qwen3.8-27B-FP8 on two GPUs; no model fallback is permitted"
        )
    return metadata


def prepare(output, task, *, imu_enabled=True):
    scenario = build_scene(task["name"], output)
    prefix = Path(os.environ["CONDA_PREFIX"])
    (output / "task.json").write_text(json.dumps(task, indent=2))
    bridges = bridge_configuration()
    rename = {
        "/scan": "/sim/scan",
        "/imu": "/sim/imu",
        "/camera/image_raw": "/sim/camera/image",
        "/camera/camera_info": "/sim/camera/camera_info",
        "/joint_states": "/sim/joints",
        "/ground_truth/pose": "/evaluation/ground_truth",
    }
    excluded_topics = {"/overview/image_raw"}
    if not imu_enabled:
        excluded_topics.add("/imu")
    bridges = [item for item in bridges if item["ros_topic_name"] not in excluded_topics]
    for item in bridges:
        item["ros_topic_name"] = rename.get(item["ros_topic_name"], item["ros_topic_name"])
    (output / "bridge.yaml").write_text(yaml.safe_dump(bridges))
    slam = yaml.safe_load(
        (prefix / "share/slam_toolbox/config/mapper_params_online_async.yaml").read_text()
    )
    slam["slam_toolbox"]["ros__parameters"].update(
        use_sim_time=True,
        scan_topic="/scan/slam",
        resolution=0.04,
        max_laser_range=6.0,
        min_laser_range=0.12,
        minimum_time_interval=0.2,
        minimum_travel_distance=0.04,
        minimum_travel_heading=0.05,
        scan_queue_size=1,
        map_update_interval=0.7,
        transform_publish_period=0.05,
        transform_timeout=0.2,
        enable_interactive_mode=False,
        scan_buffer_size=20,
        loop_match_minimum_chain_size=8,
    )
    (output / "slam.yaml").write_text(yaml.safe_dump(slam))
    (output / "tf.yaml").write_text(
        yaml.safe_dump(
            {
                "robot_state_publisher": {
                    "ros__parameters": {
                        "use_sim_time": True,
                        "robot_description": robot_description(),
                        "publish_frequency": 20.0,
                    }
                }
            }
        )
    )
    nav = yaml.safe_load((ROOT / "robot_sim/config/nav2.yaml").read_text())
    nav.pop("amcl")
    nav.pop("map_server")
    nav["bt_navigator"]["ros__parameters"]["default_nav_to_pose_bt_xml"] = str(
        ROOT / "robot_sim/config/navigate.xml"
    )
    nav["controller_server"]["ros__parameters"]["FollowPath"].update(
        max_vel_x=0.18, max_speed_xy=0.18, acc_lim_x=0.30
    )
    nav["controller_server"]["ros__parameters"]["goal_checker"].update(
        xy_goal_tolerance=0.15, yaw_goal_tolerance=0.30
    )
    for key in ("global_costmap", "local_costmap"):
        cfg = nav[key][key]["ros__parameters"]
        cfg.pop("robot_radius", None)
        cfg.update(
            footprint="[[0.085,0.075],[0.085,-0.075],[-0.095,-0.075],[-0.095,0.075]]",
            footprint_padding=0.005,
            transform_tolerance=1.0,
        )
        cfg["inflation_layer"]["inflation_radius"] = 0.25
        cfg["obstacle_layer"]["scan"].update(
            raytrace_max_range=6.0,
            obstacle_max_range=5.8,
            raytrace_min_range=0.12,
            obstacle_min_range=0.12,
        )
    nav["global_costmap"]["global_costmap"]["ros__parameters"]["static_layer"].update(
        map_topic="/candidate_nav/planning_map",
        subscribe_to_updates=False,
    )
    (output / "nav2.yaml").write_text(yaml.safe_dump(nav))
    ekf_source = ROOT / "robot_sim/config/mapping_ekf.yaml"
    if imu_enabled:
        shutil.copy2(ekf_source, output / "ekf.yaml")
    else:
        ekf = yaml.safe_load(ekf_source.read_text())
        parameters = ekf["ekf_filter_node"]["ros__parameters"]
        for key in tuple(parameters):
            if key == "imu0" or key.startswith("imu0_"):
                del parameters[key]
        (output / "ekf.yaml").write_text(yaml.safe_dump(ekf))
    shutil.copy2(ROOT / "robot_sim/config/mapping_sensors.yaml", output / "sensor_profile.yaml")
    sources = list((ROOT / "robot_sim/candidate_nav").glob("*.py")) + list(
        (ROOT / "robot_sim/mapping").glob("*.py")
    )
    (output / "source_hashes.json").write_text(
        json.dumps(
            {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            indent=2,
        )
    )
    return scenario


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--port", type=int, default=18766)
    p.add_argument("--wall-limit", type=float, default=2400)
    p.add_argument("--sim-limit", type=float)
    p.add_argument("--candidate-mode", choices=["geometric", "qwen"], default="geometric")
    p.add_argument("--control-mode", choices=["waypoint", "direct"], default="waypoint")
    p.add_argument("--no-imu", action="store_true", help="Use wheel odometry without IMU")
    p.add_argument("--seed", type=int, default=29, help="Sensor-noise seed, shared by each pair")
    p.add_argument("--no-video", action="store_true")
    p.add_argument(
        "--allow-test-model",
        action="store_true",
        help="Infrastructure smoke tests only; recorded in metadata",
    )
    args = p.parse_args()
    task = json.loads(args.task.read_text())
    if task["name"] not in SCENARIOS:
        p.error("Unknown evaluation scenario")
    output = (
        args.output
        or ROOT
        / "robot_sim/results/candidate_navigation"
        / f"{task['name']}_{datetime.now():%Y%m%d_%H%M%S}"
    ).resolve()
    if output.exists() and any(output.iterdir()):
        p.error("Output must be new/empty; preserve existing evidence")
    metadata = model_health(args.port, args.allow_test_model)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    scenario = prepare(output, task, imu_enabled=not args.no_imu)
    (output / "run_config.json").write_text(
        json.dumps(
            {
                "candidate_mode": "none" if args.control_mode == "direct" else args.candidate_mode,
                "control_mode": args.control_mode,
                "imu_enabled": not args.no_imu,
                "sensor_seed": args.seed,
                "sim_limit": args.sim_limit or scenario["sim_limit"],
                "wall_limit": args.wall_limit,
                "world_sha256": hashlib.sha256((output / "world.sdf").read_bytes()).hexdigest(),
                "task_sha256": hashlib.sha256((output / "task.json").read_bytes()).hexdigest(),
                "no_automatic_fallback": True,
            },
            indent=2,
        )
    )
    prefix = Path(os.environ["CONDA_PREFIX"])
    processes = []
    logs = []

    def start(name, command):
        log = (output / (name + ".log")).open("w")
        logs.append(log)
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        processes.append((name, process))
        return process

    def interrupt(*_):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    print("OUTPUT", output, flush=True)
    code = 1
    try:
        start(
            "gazebo",
            [
                "xvfb-run",
                "-a",
                "-s",
                "-screen 0 640x480x24 -nolisten tcp",
                "gz",
                "sim",
                "-s",
                "-r",
                str(output / "world.sdf"),
            ],
        )
        start(
            "bridge",
            [
                str(prefix / "lib/ros_gz_bridge/parameter_bridge"),
                "--ros-args",
                "-p",
                "config_file:=" + str(output / "bridge.yaml"),
            ],
        )
        start(
            "drive",
            [
                sys.executable,
                "robot_sim/scripts/rover_drive.py",
                "--output",
                str(output),
                "--odom-topic",
                "/wheel/ideal",
                "--no-odom-tf",
                "--ros-args",
                "-r",
                "/joint_states:=/sim/joints",
            ],
        )
        start(
            "tf",
            [
                str(prefix / "lib/robot_state_publisher/robot_state_publisher"),
                "--ros-args",
                "--params-file",
                str(output / "tf.yaml"),
                "-r",
                "/joint_states:=/sim/joints",
            ],
        )
        start(
            "sensors",
            [
                sys.executable,
                "-m",
                "robot_sim.mapping.sensors",
                "--output",
                str(output),
                "--seed",
                str(args.seed),
            ],
        )
        start(
            "ekf",
            [
                str(prefix / "lib/robot_localization/ekf_node"),
                "--ros-args",
                "--params-file",
                str(output / "ekf.yaml"),
                "-r",
                "odometry/filtered:=/odom",
            ],
        )
        start(
            "slam",
            [
                str(prefix / "lib/slam_toolbox/async_slam_toolbox_node"),
                "--ros-args",
                "--params-file",
                str(output / "slam.yaml"),
            ],
        )
        # The stock launch emits its configure event immediately. On the
        # intentionally constrained single CPU that service response can time
        # out before the node is ready, leaving a live but inactive SLAM node.
        # Drive both lifecycle transitions explicitly and fail early.
        lifecycle_records = []
        for transition in ("configure", "activate"):
            for attempt in range(3):
                time.sleep(2)
                result = subprocess.run(
                    ["ros2", "lifecycle", "set", "/slam_toolbox", transition],
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=60,
                )
                lifecycle_records.append(
                    {
                        "transition": transition,
                        "attempt": attempt + 1,
                        "returncode": result.returncode,
                        "output": result.stdout.strip(),
                    }
                )
                if result.returncode == 0 and "successful" in result.stdout.lower():
                    break
            else:
                raise RuntimeError(f"SLAM lifecycle {transition} failed: {result.stdout.strip()}")
        (output / "slam_lifecycle.json").write_text(json.dumps(lifecycle_records, indent=2))
        start(
            "mapper",
            [
                sys.executable,
                "-m",
                "robot_sim.mapping.mapper",
                "--output",
                str(output),
                "--rolling",
                "--no-preview",
            ],
        )
        start(
            "gateway",
            [sys.executable, "robot_sim/scripts/ros_nodes.py", "gateway", "--output", str(output)],
        )
        if args.control_mode == "waypoint":
            start(
                "nav2",
                [
                    "ros2",
                    "launch",
                    str(ROOT / "robot_sim/launch/candidate_navigation.launch.py"),
                    "params:=" + str(output / "nav2.yaml"),
                ],
            )
        evaluator = start(
            "evaluator",
            [sys.executable, "-m", "robot_sim.candidate_nav.evaluator", "--output", str(output)],
        )
        agent_command = [
            sys.executable,
            "-m",
            "robot_sim.candidate_nav.direct_agent"
            if args.control_mode == "direct"
            else "robot_sim.candidate_nav.agent",
            "--task",
            str(output / "task.json"),
            "--output",
            str(output),
            "--port",
            str(args.port),
            "--wall-limit",
            str(args.wall_limit),
            "--sim-limit",
            str(args.sim_limit or scenario["sim_limit"]),
        ]
        if args.control_mode == "waypoint":
            agent_command += ["--candidate-mode", args.candidate_mode]
        if args.no_imu:
            agent_command.append("--no-imu")
        agent = start(
            "agent",
            agent_command,
        )
        deadline = time.monotonic() + args.wall_limit + 90
        while agent.poll() is None:
            for name, process in processes:
                if process is not agent and process is not evaluator and process.poll() is not None:
                    raise RuntimeError(
                        f"{name} exited with {process.returncode}; see {output / name}.log"
                    )
            if time.monotonic() > deadline:
                raise TimeoutError("Runner wall deadline")
            time.sleep(0.5)
        try:
            evaluator.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        code = agent.returncode
    except (Exception, KeyboardInterrupt) as error:
        (output / "runner_error.txt").write_text(repr(error))
        print("RUNNER_ERROR", repr(error), flush=True)
    finally:
        for _, process in reversed(processes):
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and any(p.poll() is None for _, p in processes):
            time.sleep(0.1)
        for _, process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
        for log in logs:
            log.close()
        (output / "processes.json").write_text(
            json.dumps(
                [{"name": n, "pid": p.pid, "returncode": p.returncode} for n, p in processes],
                indent=2,
            )
        )
    if (output / "summary.json").exists():
        print((output / "summary.json").read_text(), flush=True)
    if (
        not args.no_video
        and (output / "frames.jsonl").exists()
        and (output / "frames.jsonl").stat().st_size
    ):
        from .render import video

        print("VIDEO", video(output), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
