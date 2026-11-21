"""Exercise real ROS publishing in a localhost-only domain with no hardware."""

import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_action_status_default
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from robot_sim.candidate_nav.live import LiveAgent, MotionGate


def test_imu_policy():
    # Exercise the real adapter with a synthetic ROS message in memory only.
    agent = object.__new__(LiveAgent)
    agent.now = lambda: 100.0
    agent.imu_enabled = True
    agent.imu_verified = False
    agent.imu_samples = 0
    agent.required_sensors = ("odom", "scan", "camera", "imu")
    agent.odom = Odometry()
    agent.odom.twist.twist.angular.z = 0.75
    agent.camera = (None, 100.0, "front_rgb")
    agent.received = dict.fromkeys(("odom", "scan", "camera", "map"), time.monotonic())

    msg = Imu()
    msg.header.frame_id = "base_link"
    msg.header.stamp.sec = 100
    msg.linear_acceleration_covariance[0] = -1.0
    agent.receive("imu", msg)
    assert not agent.imu_verified and not agent.fresh()

    msg.angular_velocity.z = -0.20
    agent.receive("imu", msg)
    assert agent.imu_verified and agent.fresh()
    result = agent.sensor_observation({})
    assert result["yaw_rate_rad_s"] == -0.20
    assert result["angular_velocity_rad_s"] == [0.0, 0.0, -0.20]
    assert result["yaw_rate_source"] == "imu" and result["imu_available"] is True
    assert result["acceleration_m_s2"] is None

    msg.linear_acceleration_covariance[0] = 0.01
    msg.linear_acceleration.z = 9.8
    assert agent.sensor_observation({})["acceleration_m_s2"] == [0.0, 0.0, 9.8]
    agent.received["imu"] -= 1.0
    assert not agent.fresh()
    print("PASS: IMU startup check, gyro precedence, unavailable acceleration, stale IMU")


def twist(v=0.0, w=0.0):
    cmd = Twist()
    cmd.linear.x = v
    cmd.angular.z = w
    return cmd


class TestAgent(Node):
    def __init__(self):
        super().__init__("live_gate_probe")
        self.ready = True
        self.events = []
        self.scan = SimpleNamespace(
            ranges=[2.0] * 360,
            angle_min=-math.pi,
            angle_increment=math.pi / 180,
            range_min=0.05,
            range_max=10.0,
        )

    def fresh(self):
        return self.ready

    def get_pose(self):
        return (0, 0, 0)

    def log(self, _file, event):
        self.events.append(event["result"])


assert os.environ["ROS_DOMAIN_ID"] == "219"
assert os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] == "LOCALHOST"
rclpy.init()
agent = TestAgent()
observer = Node("live_gate_listener")
topic = "/rbm_live_gate_probe/cmd_vel"
received = []
observer.create_subscription(Twist, topic, lambda msg: received.append(msg.linear.x), 5)


def spin(seconds):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        rclpy.spin_once(agent, timeout_sec=0.01)
        rclpy.spin_once(observer, timeout_sec=0.01)


try:
    if Path("require-imu").is_file():
        test_imu_policy()
    spin(0.35)
    assert {name for name, _ in agent.get_node_names_and_namespaces()} <= {
        "live_gate_probe", "live_gate_listener"
    }, "Unexpected node in isolated test domain"
    dry = MotionGate(agent, False, topic)
    dry.publish(twist(0.04))
    assert dry.output is None and agent.events[-1] == "dry_run"

    if Path("front-clearance-10cm").is_file():
        agent.scan_origin = (0.06, 0.0)
        agent.odom = Odometry()
        close = MotionGate(agent, False, topic, front_clearance=0.10)
        # At 0.04 m/s requested the envelope ends about 0.23 m from base_link,
        # or 0.17 m directly ahead of the lidar.
        agent.scan.ranges[180] = 0.18
        assert close.rejection(0.04, 0.0) is None
        agent.odom.twist.twist.linear.x = 0.15
        assert close.rejection(0.04, 0.0) == "lidar_brake"
        agent.odom.twist.twist.linear.x = 0.0
        agent.scan.ranges[180] = 0.16
        assert close.rejection(0.04, 0.0) == "lidar_brake"
        agent.scan.ranges[180] = 2.0
        agent.scan.ranges[225] = 0.20
        assert close.rejection(0.04, 0.0) is None
        agent.scan.ranges[225] = 2.0
        print("PASS: 10 cm chassis clearance, lidar offset, stopping allowance, side return")

    live = MotionGate(agent, True, topic)
    spin(0.35)
    live.publish(twist(0.04))
    spin(0.25)
    assert received[-1] == 0.04
    assert live.moving and not live.competing_control()

    agent.scan.ranges[180] = 0.20
    live.publish(twist(0.04))
    spin(0.25)
    assert received[-1] == 0.0 and agent.events[-1] == "lidar_brake"
    assert live.rejection(0.0, 0.08) == "lidar_brake"

    agent.scan.ranges[:] = [float("nan")] * 360
    assert live.rejection(0.04, 0.0) == "insufficient_lidar_coverage"
    assert live.rejection(0.1, 0.0) == "outside_physical_limits"
    agent.scan.ranges[:] = [2.0] * 360
    live.publish(twist(0.04))
    spin(0.10)
    agent.ready = False
    spin(0.20)
    assert received[-1] == 0.0 and not live.moving
    agent.ready = True

    status = observer.create_publisher(
        GoalStatusArray, "/follow_path/_action/status", qos_profile_action_status_default
    )
    spin(0.35)
    status.publish(GoalStatusArray(status_list=[GoalStatus(status=GoalStatus.STATUS_EXECUTING)]))
    spin(0.20)
    assert live.rejection(0.04, 0.0) == "other_velocity_publisher"
    status.publish(GoalStatusArray())
    spin(0.20)
    assert live.rejection(0.04, 0.0) is None

    external = observer.create_publisher(Twist, topic, 1)
    spin(0.20)
    assert live.rejection(0.04, 0.0) == "other_velocity_publisher"
    print("PASS: dry-run, obstacle, blind lidar, speed limits, stale sensors, Nav2 activity, external publisher")
finally:
    observer.destroy_node()
    agent.destroy_node()
    rclpy.shutdown()
