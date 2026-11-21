"""Stop only our live control agents and send repeated zero velocities."""

import os
from pathlib import Path
import signal
import time

import rclpy
from geometry_msgs.msg import Twist


rclpy.init()
node = rclpy.create_node('rover_live_emergency_stop')
output = node.create_publisher(Twist, '/cmd_vel', 10)
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        args = (entry / 'cmdline').read_bytes().split(b'\0')
        if any(module in args for module in
               (b'robot_sim.candidate_nav.live_waypoint', b'robot_sim.candidate_nav.live')):
            os.kill(int(entry.name), signal.SIGTERM)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
end = time.monotonic() + 1.0
while time.monotonic() < end:
    output.publish(Twist())
    rclpy.spin_once(node, timeout_sec=.02)
node.destroy_node()
rclpy.shutdown()
print('STOP SENT', flush=True)
