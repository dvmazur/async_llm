"""LLM direction control on the same sensor-built maps, with no Nav2 or candidates."""

import argparse
import asyncio
import json
import math
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import rclpy
from geometry_msgs.msg import Twist
from rclpy.time import Time

from .agent import Agent, request_model, yaw
from .direct_control import ACTION_SPEC, camera_clock_ready, scan_clearance, signed_turn
from .geometry import fuse_observations


class DirectAgent(Agent):
    def motion_clearance(self, action):
        return scan_clearance(self.scan, action) if action != "stop" else None

    def fresh(self):
        return (
            super().fresh()
            and self.camera is not None
            and camera_clock_ready(self.now(), self.camera[1])
        )

    def __init__(self, args):
        super().__init__(args)
        self.action_history = deque(maxlen=8)
        self.action_label = "stop"
        policy_file = self.output / "policy_inputs.json"
        policy = json.loads(policy_file.read_text())
        policy.update(
            control_mode="direct",
            candidate_mode="none",
            nav2_execution=False,
            model_images=["front_rgb", "online_slam_without_candidates"],
            coroutines=[
                "ros_observer",
                "map_fusion",
                "model_planner",
                "command_executor",
                "recorder",
            ],
            model_coroutines=["camera_observer", "spatial_reasoner", "action_selector"],
            action_spec=ACTION_SPEC,
            inference_motion="stopped; each bounded primitive completes before the next snapshot",
        )
        policy_file.write_text(json.dumps(policy, indent=2))

    async def proposal_builder(self):
        """Only fuse measured maps; no reachability, frontiers or candidate generation."""
        while not self.finished.is_set():
            if self.map is None or self.version < 3 or self.pose is None or not self.fresh():
                await asyncio.sleep(0.1)
                continue
            if self.started_sim is None:
                self.started_sim = self.now()
                self.state = "ready"
            grid, dynamic = self.map, self.dynamic
            transform = None
            if dynamic is not None and self.now() - dynamic.stamp < 2:
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
            if dynamic is not None:
                grid = await asyncio.to_thread(fuse_observations, grid, dynamic, transform)
            self.reach = SimpleNamespace(grid=grid)
            await asyncio.sleep(0.7)

    async def model_planner(self):
        errors = 0
        while not self.finished.is_set():
            await self.need_plan.wait()
            if self.reach is None or self.pose is None or not self.fresh():
                await asyncio.sleep(0.1)
                continue
            self.need_plan.clear()
            try:
                observation, _, created = self.snapshot()
                self.state = "thinking_stopped"
                response = await request_model(
                    observation,
                    self.args.port,
                    timeout=getattr(self.args, "model_timeout", 300),
                    socket_path=getattr(self.args, "model_socket", None),
                )
                errors = 0
                self.decisions += 1
                self.latencies.append(response["latency_s"])
                response.update(sim_time=self.now(), snapshot_wall_age=time.monotonic() - created)
                self.log("decisions", response)
                await self.pending.put((response, observation, created))
            except Exception as error:
                errors += 1
                self.stop_pub.publish(Twist())
                self.log(
                    "events", {"event": "model_error", "error": str(error), "sim_time": self.now()}
                )
                if errors >= 3:
                    self.finish("model_errors")
                else:
                    await asyncio.sleep(2)
                    self.need_plan.set()

    async def nav2_executor(self):
        """Compatibility hook for base coroutine supervisor; never opens a ROS action."""
        while not self.finished.is_set():
            response, observation, created = await self.pending.get()
            start_pose = self.odom_pose()
            source = observation["odom_pose"]
            drift = math.dist(start_pose[:2], source[:2])
            turn = abs(
                math.atan2(math.sin(start_pose[2] - source[2]), math.cos(start_pose[2] - source[2]))
            )
            max_age = getattr(self.args, "max_decision_age", 180)
            if (
                not self.fresh()
                or time.monotonic() - created > max_age
                or drift > 0.12
                or turn > 0.12
            ):
                self.rejected += 1
                self.log(
                    "events",
                    {
                        "event": "direct_stale_rejected",
                        "source_snapshot": observation["snapshot_id"],
                        "sim_time": self.now(),
                    },
                )
                self.need_plan.set()
                continue
            if response["task_complete"]:
                if any(o["visible"] and o["range_m"] < 0.95 for o in self.objects):
                    self.finish("model_found")
                    break
                self.rejected += 1
                self.need_plan.set()
                continue
            action = response["action"]
            spec = ACTION_SPEC[action]
            self.action_label = action
            self.state = "action_" + action
            self.executed += 1
            started, wall_started = self.now(), time.monotonic()
            self.log(
                "events",
                {
                    "event": "command_started",
                    "source_snapshot": observation["snapshot_id"],
                    "action": action,
                    "sim_time": started,
                    "odom_pose": start_pose,
                },
            )
            outcome = "duration_complete"
            while not self.finished.is_set() and self.now() - started < spec["max_sim_s"]:
                pose = self.odom_pose()
                if not self.fresh() or time.monotonic() - wall_started > 30:
                    outcome = "sensor_or_wall_watchdog"
                    break
                if (
                    action in {"left", "right"}
                    and signed_turn(start_pose[2], pose[2], action) >= spec["angle_rad"] - 0.025
                ):
                    outcome = "angle_complete"
                    break
                if (
                    action == "forward"
                    and math.dist(start_pose[:2], pose[:2]) >= spec["max_distance_m"]
                ):
                    outcome = "distance_complete"
                    break
                brake = self.motion_clearance(action)
                if brake:
                    outcome = brake
                    break
                msg = Twist()
                msg.linear.x, msg.angular.z = spec["linear_m_s"], spec["angular_rad_s"]
                self.stop_pub.publish(msg)
                await asyncio.sleep(0.04)
            self.stop_pub.publish(Twist())
            self.action_label = "stop"
            # Wait for physical braking before taking the next model snapshot.
            await asyncio.sleep(0.4)
            end_pose = self.odom_pose()
            record = {
                "action": action,
                "outcome": outcome,
                "source_snapshot": observation["snapshot_id"],
                "sim_time": self.now(),
                "duration_sim_s": self.now() - started,
                "distance_m": math.dist(start_pose[:2], end_pose[:2]),
                "start_odom": start_pose,
                "end_odom": end_pose,
            }
            self.action_history.append(record)
            self.log("events", {"event": "command_finished", **record})
            self.completed += 1
            self.state = "ready"
            self.need_plan.set()

    def odom_pose(self):
        p = self.odom.pose.pose
        return [round(p.position.x, 4), round(p.position.y, 4), round(yaw(p.orientation), 4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=18766)
    p.add_argument("--sim-limit", type=float, required=True)
    p.add_argument("--wall-limit", type=float, required=True)
    p.add_argument("--no-imu", action="store_true", help="Use odometry yaw rate without IMU")
    p.add_argument(
        "--wall-clock", action="store_true", help="Use system time on a physical ROS stack"
    )
    args = p.parse_args()
    args.candidate_mode, args.control_mode = "none", "direct"
    rclpy.init()
    node = DirectAgent(args)
    try:
        asyncio.run(node.run())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
