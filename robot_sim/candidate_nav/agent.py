"""Concurrent ROS observation, candidate planning, LLM choice and Nav2 execution.

This process never opens world geometry or evaluation target coordinates.
Blocking model I/O and proposal generation do not block ROS callbacks. A slow
LLM cannot prevent the local controller/watchdog from responding to sensors.
"""
from . import artifact_writer as archive

import argparse
import asyncio
import base64
import io
import json
import math
import time
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from PIL import Image as PILImage
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from .candidates import CandidateGenerator, planning_window_open
from .geometry import (
    Grid,
    colored_returns,
    fuse_observations,
    new_nearby_evidence,
    reachable,
)
from .task_policy import is_task_mode, MODE
from .io import write_json_atomic
from .proposals import proposal_map, validate_proposals
from .render import render_map


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def msg_time(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9


def image64(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def grid_message(msg, version):
    return Grid(
        np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width).copy(),
        msg.info.resolution,
        (msg.info.origin.position.x, msg.info.origin.position.y),
        msg.header.frame_id,
        version,
        msg_time(msg),
        yaw(msg.info.origin.orientation),
    )


from .model_transport import request_model


class Agent(Node):
    def __init__(self, args):
        super().__init__("candidate_llm_navigator")
        self.set_parameters(
            [
                rclpy.parameter.Parameter(
                    "use_sim_time", value=not getattr(args, "wall_clock", False)
                )
            ]
        )
        self.args = args
        self.imu_enabled = not getattr(args, "no_imu", False)
        self.required_sensors = ("odom", "scan", "camera")
        if self.imu_enabled:
            self.required_sensors += ("imu",)
        self.task = json.loads(args.task.read_text())
        self.output = args.output
        forbidden = {"target_xy", "world", "objects", "spawn", "evaluation", "target_pose"}
        if forbidden.intersection(self.task):
            raise ValueError("Privileged metadata in policy task")
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.nav = (
            None
            if getattr(args, "control_mode", "waypoint") == "direct"
            else ActionClient(self, NavigateToPose, "navigate_to_pose")
        )
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.status_pub = self.create_publisher(String, "/candidate_nav/status", 10)
        self.planning_map_pub = self.create_publisher(
            OccupancyGrid, "/candidate_nav/planning_map", latched
        )
        self.stop_pub = self.create_publisher(Twist, "/cmd_vel_requested", 1)
        self.map = self.dynamic = self.camera = self.odom = self.imu = self.scan = self.info = None
        self.version = 0
        self.camera_buffer = deque(maxlen=20)
        self.scan_buffer = deque(maxlen=40)
        self.policy_camera = None
        self.received = {}
        self.pose = None
        self.path = []
        self.path_m = 0.0
        self.objects = []
        self.visible = []
        self.candidates = []
        self.reach = None
        self.active = None
        self.generator = CandidateGenerator()
        self.pending = asyncio.Queue(maxsize=1)
        self.need_plan = asyncio.Event()
        self.need_plan.set()
        self.urgent_plan = False
        self.finished = asyncio.Event()
        self.state = "waiting_for_sensors"
        self.reason = None
        self.contacts = 0
        self.decisions = 0
        self.snapshot_serial = 0
        self.started_wall = time.monotonic()
        self.started_sim = None
        self.last_frame = -math.inf
        self.last_plan = -math.inf
        self.last_object_plan = -math.inf
        self.goal_handle = None
        self.preempt_requested = False
        self.executed = 0
        self.completed = 0
        self.rejected = 0
        self.failed = 0
        self.latencies = []
        self.generation_latencies = []
        self.raw_proposals = self.accepted_proposals = 0
        self.last_generation_id = None
        self.last_progress = time.monotonic()
        self._last_object_stamp = -1
        self._next_pose_update = 0.0
        self._last_pair_error = 0.0
        self.pair_stats = {"success": 0, "reason": "startup"}
        for folder in ("camera_frames", "map_frames", "observations", "generations"):
            (self.output / folder).mkdir(parents=True, exist_ok=True)
        self.logs = {
            name: (self.output / (name + ".jsonl")).open("w")
            for name in ("frames", "decisions", "events", "observations", "generations")
        }
        subscriptions = [
            (OccupancyGrid, "/map", "map", latched),
            (OccupancyGrid, "/mapping/dynamic_grid", "dynamic", latched),
        ]
        if not getattr(args, "physical", False):
            subscriptions.extend(
                [
                    (Image, "/camera/image_raw", "camera", qos_profile_sensor_data),
                    (CameraInfo, "/camera/camera_info", "info", 10),
                ]
            )
        subscriptions.extend(
            [
                (Odometry, "/odom", "odom", 10),
                (LaserScan, "/scan", "scan", qos_profile_sensor_data),
            ]
        )
        if self.imu_enabled:
            subscriptions.append((Imu, "/imu/data", "imu", qos_profile_sensor_data))
        for typ, topic, name, qos in subscriptions:
            self.create_subscription(
                typ, topic, lambda msg, name=name: self.receive(name, msg), qos
            )
        extra_topics = ["/tf", "/tf_static"]
        if not getattr(args, "physical", False):
            from ros_gz_interfaces.msg import Contacts

            self.create_subscription(Contacts, "/contacts", self.contact, qos_profile_sensor_data)
            extra_topics.append("/contacts")
        (self.output / "policy_inputs.json").write_text(
            json.dumps(
                {
                    "topics": [topic for _, topic, _, _ in subscriptions]
                    + extra_topics,
                    "imu_enabled": self.imu_enabled,
                    "task": self.task,
                    "ground_truth_subscription": False,
                    "preloaded_map": False,
                    "candidate_mode": args.candidate_mode,
                    "automatic_geometric_fallback": False,
                    "model_images": ["front_rgb", "online_slam_with_candidates"],
                    "coroutines": [
                        "ros_observer",
                        "proposal_builder",
                        "model_planner",
                        "nav2_executor",
                        "recorder",
                    ],
                    "model_coroutines": ["camera_observer", "spatial_reasoner", "goal_selector"],
                },
                indent=2,
            )
        )

    def log(self, name, record):
        self.logs[name].write(json.dumps(record, allow_nan=False) + "\n")
        self.logs[name].flush()

    def receive(self, name, msg):
        self.received[name] = time.monotonic()
        if name == "map":
            self.version += 1
            self.map = grid_message(msg, self.version)
        elif name == "dynamic":
            self.dynamic = grid_message(msg, self.version)
        elif name == "camera":
            channels = 3 if msg.encoding in ("rgb8", "bgr8") else 4
            array = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
            rgb = array[:, : msg.width * channels].reshape(msg.height, msg.width, channels)[
                :, :, :3
            ]
            if msg.encoding in ("bgr8", "bgra8"):
                rgb = rgb[:, :, ::-1]
            self.camera = (PILImage.fromarray(rgb.copy()), msg_time(msg), msg.header.frame_id)
            self.camera_buffer.append(self.camera)
        else:
            setattr(self, name, msg)
            if name == "scan":
                self.scan_buffer.append(msg)

    def contact(self, msg):
        if msg.contacts:
            self.contacts += 1
            if getattr(self.args,'strategy',None) == 4 and getattr(self.args,'disable_collision_protection',False):
                self.log('events',dict(event='body_contact_observed',collision_protection='disabled'))
            else:
                self.finish("body_contact")

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def get_pose(self):
        try:
            tf = self.buffer.lookup_transform("map", "base_footprint", Time())
            p = tf.transform.translation
            return (p.x, p.y, yaw(tf.transform.rotation))
        except Exception:
            return None

    def fresh(self):
        return all(
            time.monotonic() - self.received.get(name, -math.inf) < 2.5
            for name in self.required_sensors
        )

    def sensor_observation(self, scan_ranges):
        twist = self.odom.twist.twist
        covariance = self.odom.pose.covariance
        now = time.monotonic()
        imu = self.imu if self.imu_enabled else None
        sensors = {
            "lidar_min_m": scan_ranges,
            "linear_velocity_m_s": twist.linear.x,
            "yaw_rate_rad_s": imu.angular_velocity.z if imu is not None else twist.angular.z,
            "acceleration_m_s2": (
                [getattr(imu.linear_acceleration, axis) for axis in ("x", "y", "z")]
                if imu is not None
                else None
            ),
            "pose_covariance_xy_yaw": [covariance[i] for i in (0, 7, 35)],
            "sensor_age_wall_s": {
                name: round(now - self.received[name], 3) for name in self.required_sensors
            },
        }
        if not self.imu_enabled:
            sensors.update(imu_available=False, yaw_rate_source="odometry")
        return sensors

    def finish(self, reason):
        if self.finished.is_set():
            return
        self.reason = reason
        self.state = reason
        self.stop_pub.publish(Twist())
        self.finished.set()
        self.status_pub.publish(
            String(
                data=json.dumps(
                    {
                        "finished": True,
                        "reason": reason,
                        "sim_time": self.now(),
                        "model_claimed_found": reason == "model_found",
                        "decisions": self.decisions,
                    }
                )
            )
        )
        self.log("events", {"event": "finished", "reason": reason, "sim_time": self.now()})

    def detect_objects(self):
        """Generic color-region proposals grounded by timestamped lidar points.

        No task-color filtering: distractors and target receive identical handling.
        Classification/choice/completion remain model decisions.
        """
        if self.camera is None or self.scan is None or self.info is None:
            return []
        # Latest messages can be newer than the available TF while moving.
        # Pair buffered measurements after a small simulation-time delay,
        # as the existing calibrated mapper does; never use latest TF for old RGB.
        eligible = [entry for entry in self.camera_buffer if 0.18 <= self.now() - entry[1] < 1.0]
        if not eligible or not self.scan_buffer:
            self.pair_stats["reason"] = "buffer_wait"
            return []
        image, stamp, frame = eligible[-1]
        scan = min(self.scan_buffer, key=lambda item: abs(msg_time(item) - stamp))
        if abs(stamp - msg_time(scan)) > 0.15:
            self.pair_stats["reason"] = "timestamp_gap"
            return []
        try:
            scan_time = Time.from_msg(scan.header.stamp)
            camera_time = Time(seconds=int(stamp), nanoseconds=int((stamp % 1) * 1e9))
            tf = self.buffer.lookup_transform_full(
                frame, camera_time, scan.header.frame_id, scan_time, "odom"
            )
            tf_map = self.buffer.lookup_transform("map", scan.header.frame_id, scan_time)
        except Exception as error:
            self.pair_stats.update(reason="tf_unavailable", error=str(error))
            if time.monotonic() - self._last_pair_error > 5:
                self._last_pair_error = time.monotonic()
                self.log(
                    "events",
                    {
                        "event": "rgb_lidar_pair_wait",
                        "error": str(error),
                        "sim_time": self.now(),
                        "camera_stamp": stamp,
                        "scan_stamp": msg_time(scan),
                    },
                )
            return []
        self.policy_camera = (image, stamp, frame)
        self.pair_stats.update(reason="paired", success=self.pair_stats["success"] + 1)
        self.pair_stats.pop("error", None)
        from robot_sim.mapping.core import transform_points

        r = np.asarray(scan.ranges)
        a = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        valid = np.isfinite(r) & (r >= scan.range_min) & (r < scan.range_max)
        points = np.column_stack(
            [r[valid] * np.cos(a[valid]), r[valid] * np.sin(a[valid]), np.zeros(valid.sum())]
        )

        def apply(t, p):
            v, q = t.transform.translation, t.transform.rotation
            return transform_points(p, [v.x, v.y, v.z], [q.x, q.y, q.z, q.w])

        optical = apply(tf, points)
        world = apply(tf_map, points)
        k = self.info.k
        good = optical[:, 2] > 0.05
        u = np.full(len(points), np.nan)
        v = u.copy()
        u[good] = k[0] * optical[good, 0] / optical[good, 2] + k[2]
        v[good] = k[4] * optical[good, 1] / optical[good, 2] + k[5]
        hsv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2HSV)
        found = []
        for color, low, high in (
            ("red", (0, 110, 70), (12, 255, 255)),
            ("blue", (95, 100, 60), (135, 255, 255)),
            ("green", (35, 100, 60), (85, 255, 255)),
            ("yellow", (18, 110, 70), (35, 255, 255)),
        ):
            mask = cv2.inRange(hsv, np.array(low), np.array(high))
            if color == "red":
                mask |= cv2.inRange(hsv, np.array((170, 110, 70)), np.array((180, 255, 255)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
            for index in range(1, count):
                x, y, w, h, area = map(int, stats[index])
                if area < 25 or w < 4 or h < 4:
                    continue
                associated = colored_returns(u, v, good, labels == index)
                if not associated.any():
                    continue
                point = np.median(world[associated, :2], axis=0)
                distance = float(np.linalg.norm(point - np.asarray(self.pose[:2])))
                found.append(
                    {
                        "description": f"{color} RGB region; object class unconfirmed",
                        "color": color,
                        "xy": point.tolist(),
                        "range_m": distance,
                        "bbox": [x, y, w, h],
                        "visible": True,
                        "stamp": stamp,
                    }
                )
        return found

    def object_memory(self):
        visible = self.detect_objects()
        for old in self.objects:
            old["visible"] = False
        for new in visible:
            match = next(
                (
                    old
                    for old in self.objects
                    if old["color"] == new["color"] and math.dist(old["xy"], new["xy"]) < 0.6
                ),
                None,
            )
            if match is None:
                self.objects.append(new)
            else:
                match.update(new)
        self.objects = self.objects[-30:]
        for item in self.objects:
            item["range_m"] = math.dist(item["xy"], self.pose[:2])
            item["within_completion_range"] = item["range_m"] <= 0.8
        self.visible = visible

    def build_reach(self, grid, pose, dynamic, tf):
        planning = (
            fuse_observations(grid, dynamic, tf) if dynamic is not None and tf is not None else grid
        )
        return reachable(planning, pose, radius=0.13, max_distance=10)

    def publish_planning_map(self, grid):
        msg = OccupancyGrid()
        msg.header.frame_id = grid.frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = grid.resolution
        msg.info.width = grid.values.shape[1]
        msg.info.height = grid.values.shape[0]
        msg.info.origin.position.x, msg.info.origin.position.y = grid.origin
        msg.info.origin.orientation.z = math.sin(grid.yaw / 2)
        msg.info.origin.orientation.w = math.cos(grid.yaw / 2)
        msg.data = grid.values.astype(np.int8).ravel().tolist()
        self.planning_map_pub.publish(msg)

    async def ros_observer(self):
        while not self.finished.is_set():
            for _ in range(12):
                rclpy.spin_once(self, timeout_sec=0)
            if time.monotonic() >= self._next_pose_update:
                self._next_pose_update = time.monotonic() + 0.1
                pose = self.get_pose()
                if pose is not None:
                    self.pose = pose
                    if not self.path or math.dist(self.path[-1], pose[:2]) > 0.03:
                        if self.path:
                            self.path_m += math.dist(self.path[-1], pose[:2])
                        self.path.append(pose[:2])
                        self.last_progress = time.monotonic()
                    self.object_memory()
            if time.monotonic() - self.started_wall > self.args.wall_limit:
                self.finish("wall_budget")
            if self.started_sim is None and time.monotonic() - self.started_wall > 120:
                self.finish("sensor_startup_timeout")
            if self.started_sim is not None and self.now() - self.started_sim > self.args.sim_limit:
                self.finish("simulation_budget")
            await asyncio.sleep(0.01)

    async def proposal_builder(self):
        while not self.finished.is_set():
            pose = self.get_pose()
            if self.map is None or self.version < 3 or pose is None or not self.fresh():
                await asyncio.sleep(0.2)
                continue
            if self.started_sim is None:
                self.started_sim = self.now()
                self.state = "exploring"
            grid = self.map
            dynamic = self.dynamic
            transform = None
            if dynamic is not None and self.now() - dynamic.stamp < 2.0:
                try:
                    tf = self.buffer.lookup_transform(grid.frame, dynamic.frame, Time())
                    transform = (
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        yaw(tf.transform.rotation),
                    )
                except Exception:
                    dynamic = None
            else:
                dynamic = None
            proposal_started = time.monotonic()
            reach = await asyncio.to_thread(self.build_reach, grid, pose, dynamic, transform)
            candidates = (
                await asyncio.to_thread(
                    self.generator.generate, reach, pose, [dict(o) for o in self.objects]
                )
                if self.args.candidate_mode == "geometric"
                else self.candidates
            )
            self.reach = reach
            self.publish_planning_map(reach.grid)
            self.candidates = candidates
            self.log(
                "events",
                {
                    "event": "proposals_updated",
                    "map_version": grid.version,
                    "seconds": round(time.monotonic() - proposal_started, 3),
                    "count": len(candidates),
                    "sim_time": self.now(),
                },
            )
            if self.active and planning_window_open(self.active, pose):
                self.need_plan.set()
            if (
                any(o["range_m"] < 2.0 for o in self.visible)
                and time.monotonic() - self.last_object_plan > 20
            ):
                self.last_object_plan = time.monotonic()
                self.urgent_plan = True
                self.need_plan.set()
            await asyncio.sleep(1.0)

    def snapshot(self, generation=False):
        self.snapshot_serial += 1
        pose = tuple(self.pose)
        path = list(self.path)
        objects = [dict(item) for item in self.objects]
        reach = self.reach
        self._snapshot_grid = reach.grid
        candidates = [
            c
            for c in self.candidates
            if self.active is None
            or math.dist((c.x, c.y), (self.active.x, self.active.y))
            > (0.2 if c.kind == "inspect" else 0.55)
        ]
        grid = reach.grid
        map_image, p = render_map(grid, pose, candidates, path, self.active, objects)
        generation_facts = None
        if generation:
            candidates = []
            map_image, generation_facts = proposal_map(
                reach, pose, path, self.active, objects
            )
        scan_ranges = {}
        r = np.asarray(self.scan.ranges)
        angles = self.scan.angle_min + np.arange(len(r)) * self.scan.angle_increment
        for name, center in (
            ("front", 0),
            ("left", math.pi / 2),
            ("right", -math.pi / 2),
            ("rear", math.pi),
        ):
            delta = np.arctan2(np.sin(angles - center), np.cos(angles - center))
            values = r[(abs(delta) < 0.35) & np.isfinite(r) & (r >= self.scan.range_min)]
            scan_ranges[name] = round(
                float(values.min()) if len(values) else float(self.scan.range_max), 3
            )
        camera = (
            self.policy_camera
            if self.policy_camera is not None and 0 <= self.now() - self.policy_camera[1] < 1.0
            else self.camera
        )
        if hasattr(self, 'select_snapshot_camera'):
            camera = self.select_snapshot_camera(camera, pose)
        transform = self.buffer.lookup_transform("map", "odom", Time())
        map_from_odom = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            yaw(transform.transform.rotation),
        ]
        public_candidates = []
        for candidate in candidates:
            item = candidate.public()
            angle = (
                math.atan2(candidate.y - pose[1], candidate.x - pose[0]) - pose[2]
            )
            item["relative_bearing_rad"] = round(math.atan2(math.sin(angle), math.cos(angle)), 3)
            public_candidates.append(item)
        observation = {
            "protocol": "candidate_generation_v1" if generation else "candidate_navigation_v1",
            "snapshot_id": self.snapshot_serial,
            "episode_id": self.output.name,
            "candidate_generation_snapshot": self.last_generation_id if not generation else None,
            "map_version": grid.version,
            "frame_id": grid.frame,
            "map_from_odom": map_from_odom,
            "sim_time": self.now(),
            "camera_stamp": camera[1],
            "camera_pairing": dict(self.pair_stats),
            "instruction": self.task["instruction"],
            "pose": list(pose),
            "candidates": public_candidates,
            "objects": [
                {
                    **{k: v for k, v in obj.items() if k != "stamp"},
                    "last_seen_age_sim_s": round(self.now() - obj["stamp"], 3),
                }
                for obj in objects
            ],
            "sensors": self.sensor_observation(scan_ranges),
            "recent_outcomes": [list(h) for h in self.generator.history[-12:]],
            "camera": image64(camera[0]),
            "map_image": image64(map_image),
        }
        if generation:
            observation["generation_map"] = generation_facts
        if getattr(self.args, "control_mode", "waypoint") == "direct":
            from .direct_control import ACTION_SPEC

            observation.update(
                protocol="direct_navigation_v1",
                action_spec=ACTION_SPEC,
                recent_outcomes=list(self.action_history),
                odom_pose=self.odom_pose(),
            )
            observation["sensors"]["lidar_polar_min_m"] = [
                {
                    "bearing_rad": round(float(center), 3),
                    "min_m": round(float(values.min()), 3) if len(values) else None,
                }
                for center in np.linspace(-math.pi, math.pi, 16, endpoint=False)
                for values in [
                    np.minimum(r, self.scan.range_max)[
                        (
                            abs(np.arctan2(np.sin(angles - center), np.cos(angles - center)))
                            < math.pi / 16
                        )
                        & ~np.isnan(r)
                        & (r >= self.scan.range_min)
                    ]
                ]
            ]
        section = "generations" if generation else "observations"
        folder = self.output / section / f"{self.snapshot_serial:04d}"
        folder.mkdir()
        archive.image(self,camera[0],folder / "camera.png")
        archive.image(self,map_image,folder / "map.png")
        archive.grid(self,
            folder / "grid.npz",
            occupancy=grid.values,
            resolution=grid.resolution,
            origin=grid.origin,
            yaw=grid.yaw,
            stamp=grid.stamp,
            version=grid.version,
        )
        public = {k: v for k, v in observation.items() if k not in {"camera", "map_image"}}
        archive.text(self,folder / "observation.json",json.dumps(public, indent=2, allow_nan=False))
        self.log(section, public)
        return observation, candidates, time.monotonic()

    async def model_planner(self):
        errors = 0
        while not self.finished.is_set():
            await self.need_plan.wait()
            if self.pose is None or self.reach is None or not self.fresh():
                await asyncio.sleep(0.2)
                continue
            if not self.pending.empty():
                await asyncio.sleep(0.2)
                continue
            if not planning_window_open(self.active, self.pose, self.urgent_plan):
                self.need_plan.clear()
                continue
            self.need_plan.clear()
            self.urgent_plan = False
            try:
                if self.args.candidate_mode == "qwen":
                    self.state = (
                        "generating_candidates" if self.active is None else "driving_and_generating"
                    )
                    source, _, _ = self.snapshot(generation=True)
                    generated = await request_model(source, self.args.port)
                    self.generation_latencies.append(generated["latency_s"])
                    # Recheck SLAM frame correction before using old image coordinates.
                    tf = self.buffer.lookup_transform("map", "odom", Time())
                    current_tf = [
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        yaw(tf.transform.rotation),
                    ]
                    previous_tf = source["map_from_odom"]
                    correction = abs(
                        math.atan2(
                            math.sin(current_tf[2] - previous_tf[2]),
                            math.cos(current_tf[2] - previous_tf[2]),
                        )
                    )
                    frame_valid = (
                        math.dist(current_tf[:2], previous_tf[:2]) <= 0.25 and correction <= 0.15
                    )
                    candidates, checks = (
                        validate_proposals(
                            generated["proposals"],
                            self.reach,
                            source["generation_map"]["projection"],
                            self.pose,
                            self.generator,
                        )
                        if frame_valid
                        else (
                            [],
                            [
                                {"accepted": False, "reason": "map_frame_changed", "raw": p}
                                for p in generated["proposals"]
                            ],
                        )
                    )
                    self.raw_proposals += len(generated["proposals"])
                    self.accepted_proposals += len(candidates)
                    folder = self.output / "generations" / f"{source['snapshot_id']:04d}"
                    write_json_atomic(folder / "response.json", generated)
                    write_json_atomic(folder / "validation.json", checks)
                    self.log(
                        "events",
                        {
                            "event": "llm_candidates_generated",
                            "sim_time": self.now(),
                            "snapshot_id": source["snapshot_id"],
                            "raw": len(generated["proposals"]),
                            "accepted": len(candidates),
                            "latency_s": generated["latency_s"],
                        },
                    )
                    self.candidates = candidates
                    self.last_generation_id = source["snapshot_id"]
                observation, candidates, created = self.snapshot()
                self.state = "thinking" if self.active is None else "driving_and_thinking"
                response = await request_model(observation, self.args.port)
                errors = 0
                self.decisions += 1
                self.latencies.append(response["latency_s"])
                response.update(sim_time=self.now(), snapshot_wall_age=time.monotonic() - created)
                self.log("decisions", response)
                if response["goal_id"] == "Z":
                    if (
                        not self.fresh()
                        or math.dist(self.pose[:2], observation["pose"][:2]) > 0.35
                        or not any(o["visible"] and o["range_m"] < 0.95 for o in self.objects)
                    ):
                        self.rejected += 1
                        self.need_plan.set()
                        self.log(
                            "events",
                            {
                                "event": "stale_completion_rejected",
                                "sim_time": self.now(),
                                "movement_m": math.dist(self.pose[:2], observation["pose"][:2]),
                                "fresh": self.fresh(),
                                "visible_nearby": sum(
                                    o["visible"] and o["range_m"] < 0.95 for o in self.objects
                                ),
                            },
                        )
                        continue
                    self.finish("model_found")
                    break
                if response["goal_id"] == "W":
                    self.state = "waiting_for_new_evidence"
                    await asyncio.sleep(2)
                    self.need_plan.set()
                    continue
                candidate = next(c for c in candidates if c.id == response["goal_id"])
                await self.pending.put((candidate, observation, created))
                if (
                    candidate.kind == "inspect"
                    and self.active is not None
                    and self.active.kind == "frontier"
                    and self.goal_handle is not None
                    and self.goal_handle.accepted
                ):
                    self.preempt_requested = True
                    self.goal_handle.cancel_goal_async()
                    self.log(
                        "events", {"event": "model_preempted_frontier", "sim_time": self.now()}
                    )
                self.last_plan = time.monotonic()
            except Exception as error:
                errors += 1
                self.log(
                    "events", {"event": "model_error", "error": str(error), "sim_time": self.now()}
                )
                if errors >= 3:
                    self.finish("model_errors")
                    break
                await asyncio.sleep(2)
                self.need_plan.set()

    async def ros_future(self, future, timeout):
        deadline = time.monotonic() + timeout
        while not future.done():
            if self.finished.is_set():
                raise asyncio.CancelledError()
            if time.monotonic() > deadline:
                raise TimeoutError("ROS action deadline")
            await asyncio.sleep(0.03)
        return future.result()

    async def nav2_executor(self):
        while not self.finished.is_set():
            candidate, observation, created = await self.pending.get()
            if not self.fresh() or self.reach is None or time.monotonic() - created > 180:
                self.rejected += 1
                self.need_plan.set()
                continue
            if new_nearby_evidence(self.objects, observation.get("objects", [])):
                self.rejected += 1
                self.stop_pub.publish(Twist())
                self.need_plan.set()
                self.log(
                    "events",
                    {
                        "event": "new_nearby_evidence_replan",
                        "sim_time": self.now(),
                        "source_snapshot": observation["snapshot_id"],
                    },
                )
                continue
            if self.reach.grid.frame != observation["frame_id"] or not self.reach.path(
                (candidate.x, candidate.y)
            ):
                self.rejected += 1
                self.generator.record(candidate, "invalidated")
                self.need_plan.set()
                continue
            try:
                transform = self.buffer.lookup_transform("map", "odom", Time())
                current = [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    yaw(transform.transform.rotation),
                ]
                previous = observation["map_from_odom"]
                correction = abs(
                    math.atan2(
                        math.sin(current[2] - previous[2]), math.cos(current[2] - previous[2])
                    )
                )
                if math.dist(current[:2], previous[:2]) > 0.25 or correction > 0.15:
                    self.rejected += 1
                    self.need_plan.set()
                    self.log(
                        "events", {"event": "map_correction_rejected_goal", "sim_time": self.now()}
                    )
                    continue
            except Exception:
                self.rejected += 1
                self.need_plan.set()
                continue
            deadline = time.monotonic() + 90
            while not self.nav.server_is_ready():
                if time.monotonic() > deadline:
                    self.finish("nav2_unavailable")
                    return
                if self.finished.is_set():
                    return
                await asyncio.sleep(0.2)
            goal = NavigateToPose.Goal()
            goal.pose.header.frame_id = "map"
            goal.pose.header.stamp = self.get_clock().now().to_msg()
            goal.pose.pose.position.x = candidate.x
            goal.pose.pose.position.y = candidate.y
            goal.pose.pose.orientation.z = math.sin(candidate.yaw / 2)
            goal.pose.pose.orientation.w = math.cos(candidate.yaw / 2)
            self.active = candidate
            self.state = "driving"
            self.executed += 1
            self.log(
                "events",
                {
                    "event": "goal_sent",
                    "candidate": candidate.public(),
                    "sim_time": self.now(),
                    "source_snapshot": observation["snapshot_id"],
                    "validated_map_version": self.reach.grid.version,
                },
            )
            try:
                self.goal_handle = await self.ros_future(self.nav.send_goal_async(goal), 15)
                if not self.goal_handle.accepted:
                    raise RuntimeError("Nav2 rejected goal")
                result = await self.ros_future(self.goal_handle.get_result_async(), 180)
                outcome = (
                    "superseded"
                    if self.preempt_requested
                    else "reached"
                    if result.status == 4
                    else "failed"
                )
                self.completed += int(outcome == "reached")
                self.failed += int(outcome == "failed")
                self.generator.record(candidate, outcome)
                self.log(
                    "events",
                    {
                        "event": "goal_result",
                        "status": result.status,
                        "outcome": outcome,
                        "sim_time": self.now(),
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.failed += 1
                self.generator.record(candidate, "failed")
                self.log(
                    "events",
                    {"event": "navigation_error", "error": str(error), "sim_time": self.now()},
                )
                if self.goal_handle is not None and self.goal_handle.accepted:
                    self.goal_handle.cancel_goal_async()
            finally:
                self.active = None
                self.goal_handle = None
                self.preempt_requested = False
                if self.pending.empty():
                    self.need_plan.set()

    async def recorder(self):
        index = 0
        while not self.finished.is_set():
            if self.pose is not None and self.map is not None and self.camera is not None:
                stamp = self.now()
                if stamp - self.last_frame >= 0.45:
                    self.last_frame = stamp
                    camera = f"camera_frames/{index:06d}.jpg"
                    map_file = f"map_frames/policy_{index:06d}.png"
                    self.camera[0].save(self.output / camera, quality=85)
                    display_grid = self.reach.grid if self.reach is not None else self.map
                    map_image, p = render_map(
                        display_grid,
                        self.pose,
                        self.candidates,
                        self.path,
                        self.active,
                        self.objects,
                    )
                    map_image.save(self.output / map_file)
                    self.log(
                        "frames",
                        {
                            "sim_time": stamp,
                            "camera": camera,
                            "map": map_file,
                            "projection": p,
                            "pose": list(self.pose),
                            "state": self.state,
                            "path_m": self.path_m,
                            "decisions": self.decisions,
                            "action": getattr(self, "action_label", None),
                        },
                    )
                    index += 1
            await asyncio.sleep(0.1)

    async def run(self):
        tasks = [
            asyncio.create_task(coro(), name=name)
            for name, coro in (
                ("ros_observer", self.ros_observer),
                (
                    "map_fusion"
                    if getattr(self.args, "control_mode", "waypoint") == "direct"
                    else "proposal_builder",
                    self.proposal_builder,
                ),
                ("model_planner", self.model_planner),
                (
                    "command_executor"
                    if getattr(self.args, "control_mode", "waypoint") == "direct"
                    else "nav2_executor",
                    self.nav2_executor,
                ),
                ("recorder", self.recorder),
            )
        ]
        waiter = asyncio.create_task(self.finished.wait())
        try:
            done, _ = await asyncio.wait(tasks + [waiter], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task is not waiter and not task.cancelled() and task.exception():
                    self.log(
                        "events",
                        {
                            "event": "coroutine_error",
                            "task": task.get_name(),
                            "error": repr(task.exception()),
                            "traceback": "".join(traceback.format_exception(
                                type(task.exception()), task.exception(), task.exception().__traceback__)),
                        },
                    )
                    self.finish("coroutine_error")
        finally:
            if self.goal_handle is not None and self.goal_handle.accepted:
                self.goal_handle.cancel_goal_async()
            self.stop_pub.publish(Twist())
            if not self.finished.is_set():
                self.finish("interrupted")
            for task in tasks + [waiter]:
                task.cancel()
            await asyncio.gather(*tasks, waiter, return_exceptions=True)
            report = {
                "reason": self.reason,
                "model_claimed_found": self.reason == "model_found",
                "decisions": self.decisions,
                "executed_goals": self.executed,
                "reached_goals": self.completed,
                "failed_goals": self.failed,
                "rejected_goals": self.rejected,
                "odom_path_m": self.path_m,
                "contacts": self.contacts,
                "candidate_mode": self.args.candidate_mode,
                "control_mode": getattr(self.args, "control_mode", "waypoint"),
                "imu_enabled": self.imu_enabled,
                "generation_requests": len(self.generation_latencies),
                "raw_proposals": self.raw_proposals,
                "accepted_proposals": self.accepted_proposals,
                "generation_seconds": sum(self.generation_latencies),
                "generation_latency_median_s": float(np.median(self.generation_latencies))
                if self.generation_latencies
                else None,
                "selection_seconds": sum(self.latencies),
                "sim_seconds": self.now() - (self.started_sim or self.now()),
                "latency_median_s": float(np.median(self.latencies)) if self.latencies else None,
                "latency_p95_s": float(np.percentile(self.latencies, 95))
                if self.latencies
                else None,
            }
            if getattr(self.args, "control_mode", "waypoint") == "direct":
                report.update(
                    executed_commands=self.executed,
                    completed_commands=self.completed,
                    rejected_commands=self.rejected,
                    executed_goals=0,
                    reached_goals=0,
                    failed_goals=0,
                    rejected_goals=0,
                )
            if getattr(self.args, 'physical', False):
                report['rover_id'] = getattr(self.args, 'rover_id', 80)
                report['model_claimed_task_complete'] = self.reason == 'task_complete'
                report['completion_mode'] = getattr(self.args, 'completion_mode', None)
            write_json_atomic(self.output / "agent_summary.json", report)
            for log in self.logs.values():
                log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=18766)
    p.add_argument("--sim-limit", type=float, default=400)
    p.add_argument("--wall-limit", type=float, default=1800)
    p.add_argument("--candidate-mode", choices=["geometric", "qwen"], default="geometric")
    p.add_argument("--no-imu", action="store_true", help="Use odometry yaw rate without IMU")
    p.add_argument(
        "--wall-clock", action="store_true", help="Use system time on a physical ROS stack"
    )
    args = p.parse_args()
    rclpy.init()
    agent = Agent(args)
    try:
        asyncio.run(agent.run())
    finally:
        agent.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
