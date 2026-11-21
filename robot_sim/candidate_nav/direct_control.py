"""Bounded direction commands. Safety may stop, never select a different direction."""

import math

from .task_policy import is_task_mode, MODE

ACTION_SPEC = {
    "forward": {"linear_m_s": 0.18, "angular_rad_s": 0.0, "max_sim_s": 6.0, "max_distance_m": 1.08},
    "left": {"linear_m_s": 0.0, "angular_rad_s": 0.45, "max_sim_s": 2.4, "angle_rad": math.pi / 4},
    "right": {
        "linear_m_s": 0.0,
        "angular_rad_s": -0.45,
        "max_sim_s": 2.4,
        "angle_rad": math.pi / 4,
    },
    "stop": {"linear_m_s": 0.0, "angular_rad_s": 0.0, "max_sim_s": 1.0},
}

# Actual physical speeds after hardware gain compensation; shared by strategies 3 and 4.
LIVE_ACTION_SPEC = {
    'forward': {'linear_m_s': 0.10, 'angular_rad_s': 0.0, 'max_sim_s': 3.5, 'max_distance_m': 0.25},
    'left': {'linear_m_s': 0.0, 'angular_rad_s': 0.8, 'max_sim_s': 1.4, 'angle_rad': math.pi / 4},
    'right': {'linear_m_s': 0.0, 'angular_rad_s': -0.8, 'max_sim_s': 1.4, 'angle_rad': math.pi / 4},
    'stop': {'linear_m_s': 0.0, 'angular_rad_s': 0.0, 'max_sim_s': 0.3},
}


def camera_clock_ready(sim_time, camera_stamp):
    """ROS can deliver stamped images before the first /clock callback."""
    return (
        math.isfinite(sim_time)
        and math.isfinite(camera_stamp)
        and sim_time > 0
        and -0.01 <= sim_time - camera_stamp < 1.0
    )


def validate_response(response, observation):
    action = response.get("action")
    if action not in ACTION_SPEC or type(response.get("task_complete")) is not bool:
        raise ValueError("Invalid direct action/completion schema")
    labels = {"F": "forward", "L": "left", "R": "right", "S": "stop", "Z": "stop"}
    if (
        response.get("protocol") != "direct_navigation_v1"
        or labels.get(response.get("goal_id")) != action
        or response["task_complete"] != (response.get("goal_id") == "Z")
    ):
        raise ValueError("Inconsistent direct action label")
    if response["task_complete"]:
        if is_task_mode(observation):
            confirmed = response.get('completion_mode') == MODE
        else:
            confirmed = any(o.get("visible") and o["range_m"] < 0.95 for o in observation["objects"])
        if action != "stop" or not confirmed:
            raise ValueError("Completion requires stop and a consistent completion response")


def scan_clearance(scan, action):
    """Conservative live scan brake; invalid coverage stops the primitive."""
    values = []
    for index, distance in enumerate(scan.ranges):
        angle = scan.angle_min + index * scan.angle_increment
        if math.isnan(distance) or distance < scan.range_min:
            continue
        if action == "forward" and math.isfinite(distance):
            x, y = distance * math.cos(angle), distance * math.sin(angle)
            if -0.10 < x < 0.32 and abs(y) < 0.12:
                return "lidar_brake"
        if action == "forward" and abs(math.atan2(math.sin(angle), math.cos(angle))) > 0.45:
            continue
        values.append(scan.range_max if math.isinf(distance) else distance)
    if not values:
        return "missing_scan_coverage"
    limit = 0.32 if action == "forward" else 0.16
    return "lidar_brake" if min(values) < limit else None


def signed_turn(start, current, action):
    delta = math.atan2(math.sin(current - start), math.cos(current - start))
    return delta if action == "left" else -delta
