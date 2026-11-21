"""Read-only privileged evaluator; never publishes navigation commands or goals."""

import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import String

from .io import write_json_atomic


class Evaluator(Node):
    def __init__(self, output):
        super().__init__("candidate_navigation_evaluator")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.output = output
        self.scenario = json.loads((output / "evaluation_scenario.json").read_text())
        self.path = 0.0
        self.last = None
        self.distance = None
        self.contacts = 0
        self.finished = False
        self.status = {}
        self.minimum = math.inf
        self.started = time.monotonic()
        self.log = (output / "evaluation_trajectory.jsonl").open("w")
        self.create_subscription(
            PoseStamped, "/evaluation/ground_truth", self.truth, qos_profile_sensor_data
        )
        self.create_subscription(Contacts, "/contacts", self.contact, qos_profile_sensor_data)
        self.create_subscription(String, "/candidate_nav/status", self.result, 10)

    def contact(self, msg):
        self.contacts += int(bool(msg.contacts))

    def result(self, msg):
        status = json.loads(msg.data)
        if status.get("finished"):
            self.status = status
            self.finished = True

    def truth(self, msg):
        xy = (msg.pose.position.x, msg.pose.position.y)
        if self.last is not None:
            self.path += math.dist(self.last, xy)
        self.last = xy
        self.distance = math.dist(xy, self.scenario["target_xy"])
        self.minimum = min(self.minimum, self.distance)
        self.log.write(
            json.dumps(
                {
                    "sim_time": self.get_clock().now().nanoseconds / 1e9,
                    "truth": xy,
                    "distance_to_target_m": self.distance,
                    "truth_path_m": self.path,
                }
            )
            + "\n"
        )
        self.log.flush()

    def save(self):
        absent = self.scenario.get("absent", False)
        claimed = self.status.get("model_claimed_found", False)
        passed = (
            (
                not claimed
                and self.contacts == 0
                and self.path >= 2.0
                and self.status.get("decisions", 0) >= 3
                and self.status.get("reason") in {"simulation_budget", "search_exhausted"}
            )
            if absent
            else (
                claimed
                and self.distance is not None
                and self.distance < 1.15
                and self.contacts == 0
                and self.path >= self.scenario["minimum_path_m"]
            )
        )
        result = {
            "passed": passed,
            "reason": self.status.get("reason", "interrupted"),
            "model_claimed_found": claimed,
            "truth_path_m": self.path,
            "final_distance_to_target_m": self.distance,
            "minimum_distance_to_target_m": self.minimum if math.isfinite(self.minimum) else None,
            "contacts": self.contacts,
            "target_absent": absent,
            "minimum_required_path_m": self.scenario["minimum_path_m"],
            "ground_truth_for_evaluation_only": True,
            "wall_seconds": time.monotonic() - self.started,
        }
        if (self.output / "agent_summary.json").exists():
            result["agent"] = json.loads((self.output / "agent_summary.json").read_text())
        write_json_atomic(self.output / "summary.json", result)
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rclpy.init()
    node = Evaluator(args.output)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        # Let the agent save its own final report before assembling metrics.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not (args.output / "agent_summary.json").exists():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        print(json.dumps(node.save()), flush=True)
        node.log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
