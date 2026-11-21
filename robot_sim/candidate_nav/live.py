"""Run the existing direct Qwen policy on the physical RoboMarvel ROS stack."""

import argparse
import asyncio
import copy
import json
import math
import socket
import threading
import time
from pathlib import Path

import av
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import Twist
from PIL import Image
from rclpy.qos import qos_profile_action_status_default
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time

from .agent import msg_time, yaw
from .direct_agent import DirectAgent
from .live_link import ControlLink
from .direct_control import ACTION_SPEC
from .sensor_grounding import camera_image
from .startup_health import blockers, odom_valid, LABELS


def verify_backend(path, fixture, expected_model=None):
    from .model_catalog import MODELS, model_key
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(5)
        client.connect(str(path))
        client.sendall(b'{"health":true}\n')
        with client.makefile("rb") as reader:
            health = json.loads(reader.readline(65536))
    if fixture:
        compatible = health.get("backend") == "rover_transport_fixture"
    else:
        compatible = (
            health.get("backend") == "native_async_llm"
            and Path(health.get("model", "")).name in MODELS.values()
            and (expected_model is None or Path(health.get("model", "")).name==MODELS[model_key(expected_model)])
            and health.get("tp_size") in (1, 2)
        )
    if health.get("ready") is not True or not compatible:
        raise RuntimeError(f"Incompatible model backend: {health}")
    print(f"Model transport ready ({health['backend']})", flush=True)


class Camera:
    def __init__(self):
        self.latest = None
        self.error = None
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        while not self.closed.is_set():
            try:
                with av.open(
                    "rtsp://127.0.0.1:8554/cam",
                    options={"rtsp_transport": "tcp", "fflags": "nobuffer"},
                    timeout=(5, 2),
                ) as stream:
                    last = -math.inf
                    for frame in stream.decode(video=0):
                        if self.closed.is_set():
                            return
                        received = time.monotonic()
                        if received - last < 0.17:
                            continue
                        self.latest = (camera_image(frame), time.time(), received)
                        self.error = None
                        last = received
            except (av.FFmpegError, OSError) as error:
                self.error = str(error)
            self.closed.wait(0.5)


class MotionGate:
    NAV2_COMMAND_NODES = {("controller_server", "/"), ("behavior_server", "/")}
    NAV2_MOTION_ACTIONS = (
        "navigate_to_pose",
        "navigate_through_poses",
        "follow_path",
        "assisted_teleop",
        "backup",
        "drive_on_heading",
        "spin",
    )

    def __init__(self, agent, enabled, topic="/cmd_vel", *, forward_limit=0.04, front_clearance=None):
        self.agent = agent
        self.topic = topic
        self.forward_limit = forward_limit
        self.front_clearance = front_clearance
        self.output = agent.create_publisher(Twist, topic, 1) if enabled else None
        self.moving = False
        self.last_decision = None
        self.active_actions = set()
        self.timer = None
        self.subscriptions = []
        if self.output is not None:
            for action in self.NAV2_MOTION_ACTIONS:
                subscription = agent.create_subscription(
                    GoalStatusArray,
                    f"/{action}/_action/status",
                    lambda message, action=action: self.observe_action(action, message),
                    qos_profile_action_status_default,
                )
                self.subscriptions.append(subscription)
            self.timer = agent.create_timer(0.05, self.watchdog)

    def close(self):
        self.stop()
        if self.timer is not None:
            self.agent.destroy_timer(self.timer)
            self.timer = None
        for subscription in self.subscriptions:
            self.agent.destroy_subscription(subscription)
        self.subscriptions = []
        if self.output is not None:
            self.agent.destroy_publisher(self.output)
            self.output = None

    def observe_action(self, action, message):
        running = {GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING, GoalStatus.STATUS_CANCELING}
        if any(item.status in running for item in message.status_list):
            self.active_actions.add(action)
        else:
            self.active_actions.discard(action)

    def competing_control(self):
        if self.output is None:
            return False
        allowed = self.NAV2_COMMAND_NODES | {(self.agent.get_name(), self.agent.get_namespace())}
        return bool(self.active_actions) or any(
            (info.node_name, info.node_namespace) not in allowed
            for info in self.agent.get_publishers_info_by_topic(self.topic)
        )

    def stop(self):
        if self.moving:
            self.output.publish(Twist())
            self.moving = False

    def watchdog(self):
        if self.moving and (
            not self.agent.fresh()
            or self.competing_control()
            or self.agent.get_pose() is None
        ):
            self.stop()

    def publish(self, requested):
        v, w = requested.linear.x, requested.angular.z
        if v == w == 0:
            self.stop()
            self.last_decision = None
            return
        link_ok = getattr(self.agent, 'control_link_ok', lambda: True)
        link_healthy = link_ok()
        reason = self.rejection(v, w) if link_healthy else 'control_link_lost'
        if reason is None and self.output is None:
            reason = "dry_run"
        decision = reason or "allowed"
        collision = getattr(self, 'collision_check', None) if link_healthy else None
        collision_reason = collision.reason if collision is not None else None
        if decision != self.last_decision or collision_reason != getattr(self, 'last_collision_reason', None):
            self.last_collision_reason = collision_reason
            self.agent.log(
                "events",
                {"event": "motion_gate", "linear": v, "angular": w, "result": decision,
                 **({"clearance": collision.public()} if collision is not None else {})},
            )
            self.last_decision = decision
        if reason is not None:
            self.stop()
            return
        self.output.publish(requested)
        self.moving = True

    def rejection(self, linear, angular):
        if not (
            (0 < linear <= self.forward_limit and angular == 0)
            or (linear == 0 and 0 < abs(angular) <= 0.08)
        ):
            return "outside_physical_limits"
        if self.competing_control():
            return "other_velocity_publisher"
        if not self.agent.fresh() or self.agent.get_pose() is None:
            return "stale_sensors_or_tf"

        scan = self.agent.scan
        ranges = np.asarray(scan.ranges)
        angles = scan.angle_min + np.arange(ranges.size) * scan.angle_increment
        valid = (ranges >= scan.range_min) & (
            np.isposinf(ranges) | (ranges <= scan.range_max)
        )
        if linear:
            zone = np.abs(np.arctan2(np.sin(angles), np.cos(angles))) <= 0.60
            distance = 0.40
        else:
            zone = np.ones(ranges.size, dtype=bool)
            distance = 0.25
        if not np.any(zone) or np.count_nonzero(valid & zone) < 0.7 * np.count_nonzero(zone):
            return "insufficient_lidar_coverage"
        if linear and self.front_clearance is not None:
            lidar_x, lidar_y = self.agent.scan_origin
            returns = valid & np.isfinite(ranges)
            x = lidar_x + ranges[returns] * np.cos(angles[returns])
            y = lidar_y + ranges[returns] * np.sin(angles[returns])
            measured_speed = self.agent.odom.twist.twist.linear.x
            if not math.isfinite(measured_speed):
                return "invalid_odometry_speed"
            # Hwnode applies 2x gain and has a 0.30 s command watchdog.
            speed = max(2.0 * linear, abs(measured_speed))
            stopping_m = speed * 0.35 + speed * speed / 4.0
            # The configured radius is 0.08 m. Use 0.10 m for the body and lateral margin.
            stop_x = 0.10 + self.front_clearance + stopping_m
            if np.any((x >= 0) & (x <= stop_x) & (np.abs(y) <= 0.10)):
                return "lidar_brake"
            return None
        if np.any(valid & zone & (ranges < distance)):
            return "lidar_brake"
        return None


class LiveAgent(DirectAgent):
    def __init__(self, args):
        super().__init__(args)
        self.destroy_publisher(self.stop_pub)
        self.stop_pub = MotionGate(
            self, args.enable_motion, front_clearance=getattr(args, "front_clearance", None)
        )
        self.link_watch = None
        if getattr(args, 'link_socket', None):
            self.link_watch = ControlLink(args.link_socket, args.link_token)
        self.link_started = time.monotonic()
        self.create_timer(0.05, self.verify_control_link)
        self.stream = Camera()
        self.last_camera_error = None
        self.create_timer(3.0, self.report_sensor_startup)
        self.create_timer(0.10, self.update_camera)
        self.imu_verified = False
        self.imu_samples = 0
        if self.imu_enabled:
            self.create_timer(0.5, self.verify_imu_startup)
            print("Waiting for measured IMU rotation on /hardware/imu (20 s)", flush=True)
        policy_file = self.output / "policy_inputs.json"
        policy = json.loads(policy_file.read_text())
        policy["topics"].remove("/odom")
        policy["topics"].append("/icp/odom")
        if self.imu_enabled:
            policy["topics"].remove("/imu/data")
            policy["topics"].append("/hardware/imu")
        policy.update(
            physical=True,
            camera_source="MediaMTX /cam",
            rgb_lidar_calibrated=False,
            motion_enabled=args.enable_motion,
            model_fixture=args.test_backend,
            front_clearance_m=self.stop_pub.front_clearance,
        )
        policy_file.write_text(json.dumps(policy, indent=2))

    def sensor_readiness(self):
        now = time.monotonic()
        ages = {name: (round(now-self.received[name], 4) if name in self.received else None)
                for name in ('camera', 'scan', 'odom', 'map')}
        try:
            tf = self.buffer.lookup_transform('map', 'base_link', Time())
            ages['pose'] = self.now() - msg_time(tf)
        except Exception:
            ages['pose'] = None
        snapshot = dict(ages=ages, lidar_tf=self.scan is not None,
                        odom_valid=self.odom is not None and odom_valid(self.odom), external_motion=False)
        missing = blockers(snapshot)
        return dict(blocked=missing, blocked_labels=[LABELS[x] for x in missing], sensors=snapshot)

    def report_sensor_startup(self):
        if self.started_sim is not None or self.finished.is_set():
            return
        report = self.sensor_readiness()
        self.log('events', dict(event='waiting_for_sensors', **report))
        if report['blocked']:
            print('Waiting for sensors: '+', '.join(report['blocked_labels']), flush=True)

    def control_link_ok(self):
        return self.link_watch is not None and self.link_watch.healthy

    def verify_control_link(self):
        if (self.args.enable_motion and not self.control_link_ok()
                and time.monotonic() - self.link_started > 2):
            self.finish('control_link_lost')

    def destroy_node(self):
        if self.link_watch is not None:
            self.link_watch.close()
        return super().destroy_node()

    def verify_imu_startup(self):
        if not self.imu_verified and time.monotonic() - self.started_wall > 20:
            print(
                f"IMU verification failed: {self.imu_samples} usable messages, no measured rotation",
                flush=True,
            )
            self.finish("imu_unverified")

    def update_camera(self):
        if self.stream.error and self.stream.error != self.last_camera_error:
            self.last_camera_error = self.stream.error
            self.get_logger().warning(f"Camera stream: {self.stream.error}")
        frame = self.stream.latest
        if frame is None or frame[2] == self.received.get("camera"):
            return
        self.camera = (frame[0], frame[1], "front_rgb")
        self.camera_buffer.append(self.camera)
        self.received["camera"] = frame[2]

    def receive(self, name, message):
        if name == "imu":
            gyro = message.angular_velocity
            rate = math.hypot(gyro.x, gyro.y, gyro.z)
            if (
                not math.isfinite(rate)
                or message.angular_velocity_covariance[0] < 0
                or message.header.frame_id != "base_link"
                or not -0.15 <= self.now() - msg_time(message) < 0.35
            ):
                return
            self.imu_samples += 1
            if not self.imu_verified and rate >= 0.005:
                self.imu_verified = True
                print("Measured IMU rotation verified", flush=True)
        if name == "scan":
            try:
                transform = self.buffer.lookup_transform("base_link", message.header.frame_id, Time())
            except Exception:
                return
            heading = yaw(transform.transform.rotation)
            translation = transform.transform.translation
            self.scan_origin = (translation.x, translation.y)
            message = copy.copy(message)
            message.angle_min += heading
            message.angle_max += heading
        super().receive(name, message)

    def motion_clearance(self, action):
        if action == "forward" and self.stop_pub.front_clearance is not None:
            return self.stop_pub.rejection(ACTION_SPEC[action]["linear_m_s"], 0.0)
        return super().motion_clearance(action)

    def get_pose(self):
        try:
            transform = self.buffer.lookup_transform("map", "base_link", Time())
        except Exception:
            return None
        stamp = Time.from_msg(transform.header.stamp).nanoseconds / 1e9
        if not -0.15 <= self.now() - stamp < 1.0:
            return None
        t = transform.transform
        return t.translation.x, t.translation.y, yaw(t.rotation)

    def fresh(self):
        now = time.monotonic()
        if self.imu_enabled and (
            not self.imu_verified or now - self.received.get("imu", -math.inf) >= 0.30
        ):
            return False
        return super().fresh() and all(
            now - self.received.get(name, -math.inf) < limit
            for name, limit in (("scan", 0.30), ("odom", 0.40), ("map", 12.0))
        )

    def sensor_observation(self, scan_ranges):
        sensors = super().sensor_observation(scan_ranges)
        if self.imu_enabled:
            gyro = self.imu.angular_velocity
            sensors.update(
                imu_available=True,
                yaw_rate_source="imu",
                angular_velocity_rad_s=[gyro.x, gyro.y, gyro.z],
            )
            if self.imu.linear_acceleration_covariance[0] < 0 or not all(
                math.isfinite(value) for value in sensors["acceleration_m_s2"]
            ):
                sensors["acceleration_m_s2"] = None
        return sensors

    def detect_objects(self):
        # RGB is sent to Qwen, but metric object ranging needs measured extrinsics.
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--task", type=Path)
    task.add_argument("--instruction")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-model", default=None)
    parser.add_argument("--model-socket", type=Path, required=True)
    parser.add_argument("--link-socket", type=Path)
    parser.add_argument("--link-token")
    parser.add_argument("--wall-limit", type=float, default=60)
    parser.add_argument("--front-clearance", type=float, help="Forward obstacle margin from the chassis")
    parser.add_argument("--imu", action="store_true", help="Require /hardware/imu and verify rotation")
    parser.add_argument("--rover-id", type=int, choices=[80, 107], default=80)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--test-backend", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.enable_motion and not (args.link_socket and args.link_token):
        parser.error('Motion requires a supervised control link')
    if args.test_backend and args.enable_motion:
        parser.error("A test backend cannot enable motion")
    if not math.isfinite(args.wall_limit) or args.wall_limit <= 0:
        parser.error("--wall-limit must be finite and positive")
    if args.front_clearance is not None and not (
        math.isfinite(args.front_clearance) and args.front_clearance >= 0.10
    ):
        parser.error("--front-clearance must be at least 0.10 m")

    verify_backend(args.model_socket, args.test_backend, args.expected_model)
    if args.instruction is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        args.task = args.output / "task.json"
        args.task.write_text(json.dumps({"instruction": args.instruction}, ensure_ascii=False))
    args.physical, args.no_imu, args.wall_clock = True, not args.imu, True
    args.candidate_mode, args.control_mode = "none", "direct"
    args.port, args.sim_limit = 18766, args.wall_limit
    args.max_decision_age, args.model_timeout = 20.0, 22.0
    ACTION_SPEC["forward"].update(linear_m_s=0.04, max_sim_s=3.0, max_distance_m=0.20)
    ACTION_SPEC["left"].update(angular_rad_s=0.08, max_sim_s=0.8, angle_rad=0.35)
    ACTION_SPEC["right"].update(angular_rad_s=-0.08, max_sim_s=0.8, angle_rad=0.35)
    ros_args = ["--ros-args", "-r", "__node:=rover_live_qwen", "-r", "/odom:=/icp/odom"]
    if args.imu:
        ros_args += ["-r", "/imu/data:=/hardware/imu"]
    rclpy.init(
        args=ros_args,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = None
    try:
        node = LiveAgent(args)
        asyncio.run(node.run())
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop_pub.stop()
            node.stream.closed.set()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
