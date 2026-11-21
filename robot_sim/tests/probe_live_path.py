"""Run only in ROS domain 219 with a private Nav2 controller and synthetic sensors."""

import asyncio
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from geometry_msgs.msg import TransformStamped, Twist
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Odometry
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from robot_sim.candidate_nav.agent import Agent
from robot_sim.candidate_nav.live_nav_config import NS, PRIVATE_VELOCITY
from robot_sim.candidate_nav.live_waypoint import WaypointAgent


class NoMotorRelay:
    def __init__(self):
        self.armed = False
        self.blocked = None
        self.stops = 0

    def stop(self):
        self.stops += 1


class Probe(Node):
    make_path = WaypointAgent.make_path
    ros_future = Agent.ros_future

    def __init__(self):
        super().__init__('live_path_probe')
        self.finished = asyncio.Event()
        self.stop_pub = NoMotorRelay()
        self.goal_handle = None
        self.active_route = []
        self.replan_count = 0
        self.action_label = 'stop'
        self.samples = []
        self.events = []
        self.position = 0.0
        self.last_tick = time.monotonic()
        self.velocity = 0.0
        self.reject_plan = False
        self.controller = ActionClient(self, FollowPath, NS + '/follow_path')
        self.create_subscription(Twist, PRIVATE_VELOCITY, self.command, qos_profile_sensor_data)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, '/icp/odom', 10)
        self.tf = TransformBroadcaster(self)
        self.static = StaticTransformBroadcaster(self)
        identity = TransformStamped()
        identity.header.frame_id = 'map'
        identity.child_frame_id = 'odom'
        identity.transform.rotation.w = 1.0
        self.static.sendTransform(identity)
        self.create_timer(0.02, self.tick)

    def command(self, message):
        if self.stop_pub.armed:
            self.samples.append((time.monotonic(), message.linear.x))
        self.velocity = message.linear.x

    def tick(self):
        now = time.monotonic()
        if self.stop_pub.armed:
            self.position += self.velocity * (now - self.last_tick)
        self.last_tick = now
        stamp = self.get_clock().now().to_msg()
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x = self.position
        transform.transform.rotation.w = 1.0
        self.tf.sendTransform(transform)
        odom = Odometry()
        odom.header = transform.header
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.position
        odom.pose.pose.orientation.w = 1.0
        odom.twist.twist.linear.x = self.velocity
        self.odom_pub.publish(odom)
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = 'base_link'
        scan.angle_min = -math.pi
        scan.angle_max = math.pi
        scan.angle_increment = 2 * math.pi / 360
        scan.range_min, scan.range_max = 0.01, 5.0
        scan.ranges = [3.0] * 360
        self.scan_pub.publish(scan)

    def get_pose(self):
        return (self.position, 0.0, 0.0)

    def fresh(self):
        return True

    def now(self):
        return time.time()

    def log(self, _name, event):
        self.events.append(event)

    async def plan(self, _candidate):
        await asyncio.sleep(0.12)
        return [] if self.reject_plan else [(self.position, 0.0), (1.2, 0.0)]

    async def observe(self):
        while not self.finished.is_set():
            for _ in range(12):
                rclpy.spin_once(self, timeout_sec=0)
            await asyncio.sleep(0.002)

    async def run_probe(self):
        observer = asyncio.create_task(self.observe())
        try:
            deadline = time.monotonic() + 35
            active_file = Path(__file__).with_name('controller-active')
            while not self.controller.server_is_ready() or not active_file.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('Private controller did not become ready')
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.7)
            assert not self.get_publishers_info_by_topic('/cmd_vel'), 'Motor publisher in isolated domain'
            candidate = SimpleNamespace(public=lambda: {'id': 'A'})
            outcome = await WaypointAgent.drive_path(
                self, [(0.0, 0.0), (1.2, 0.0)], 0.0, seconds=4.6, candidate=candidate)
            updates = sum(event['event'] == 'path_updated' for event in self.events)
            assert outcome == 'goal_budget', (outcome, self.events)
            assert updates >= 2, updates
            assert len(self.samples) > 50, len(self.samples)
            assert all(speed > 0 for _, speed in self.samples), self.samples
            gaps = [b[0] - a[0] for a, b in zip(self.samples, self.samples[1:])]
            assert max(gaps) < 0.20, max(gaps)
            assert self.stop_pub.stops == 1, self.stop_pub.stops
            self.reject_plan = True
            blocked = await WaypointAgent.drive_path(
                self, [(self.position, 0.0), (1.2, 0.0)], 0.0, seconds=3, candidate=candidate)
            assert blocked == 'blocked', blocked
            assert not self.stop_pub.armed and self.goal_handle is None
            assert self.stop_pub.stops == 2, self.stop_pub.stops
            print(json.dumps({'result': 'PASS', 'path_updates': updates,
                              'velocity_samples': len(self.samples),
                              'invalid_replan_stops_controller': True,
                              'zero_commands_before_final_cancel': 0,
                              'max_command_gap_s': round(max(gaps), 3),
                              'simulated_distance_m': round(self.position, 3),
                              'motor_publishers': 0}), flush=True)
        finally:
            self.finished.set()
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)


if __name__ == '__main__':
    assert os.environ.get('ROS_DOMAIN_ID') == '219'
    rclpy.init()
    probe = Probe()
    try:
        asyncio.run(probe.run_probe())
    finally:
        probe.destroy_node()
        rclpy.shutdown()
