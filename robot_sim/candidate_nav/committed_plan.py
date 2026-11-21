"""Single authoritative plan, body directions and admissible primitive prefixes."""

import math
from .action_plan import ACTION_PLAN_SPEC, LABELS, ACTION_LABELS, rollout, turn_delta

TRANSLATIONS = {"F", "G", "B", "C"}


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def orientation(pose, maps):
    yaw = pose[2]
    c, s = math.cos(yaw), math.sin(yaw)
    result = dict(
        frame="map",
        yaw_degrees=round(math.degrees(yaw), 2),
        forward_xy=[round(c, 3), round(s, 3)],
        backward_xy=[round(-c, 3), round(-s, 3)],
    )
    for name in ("local", "accumulated"):
        projection = (maps.get(name) or {}).get("projection") or {}
        if projection:
            angle = yaw - projection.get("yaw", 0)
            result[name + "_forward_image_xy"] = [
                round(math.cos(angle), 3),
                round(-math.sin(angle), 3),
            ]
    return result


def admissible(prefix, options, spec, turn_labels):
    """Current geometry admits a turn prefix then one straight direction.

    Turns remain available as in the previous physical policy. After translation
    followed by a turn, another translation needs new geometry. Forward and reverse
    are both supported, including short steps and equivalent turn decompositions.
    """
    angle = 0.0
    distance = 0.0
    direction = None
    can_translate = True
    for code in prefix:
        item = spec[LABELS[code]]
        if code in TRANSLATIONS:
            sign = 1 if item["linear_m_s"] > 0 else -1
            if direction is not None and direction != sign:
                can_translate = False
            direction = sign
            distance += item["max_distance_m"]
        elif "angle_rad" in item:
            if direction is not None:
                can_translate = False
            angle += turn_delta(LABELS[code], spec)
        elif code == "S":
            return []
    allowed = []
    if can_translate:
        for code in ("F", "G", "B", "C"):
            name = LABELS[code]
            if name not in spec:
                continue
            item = spec[name]
            sign = 1 if item["linear_m_s"] > 0 else -1
            if direction is not None and direction != sign:
                continue
            for row in options:
                base_code = row.get("translation_label", "F")
                base = spec.get(LABELS[base_code], {})
                if "max_distance_m" not in base or (base["linear_m_s"] > 0) != (
                    sign > 0
                ):
                    continue
                row_angle = sum(turn_delta(a, spec) for a in row["turn_actions"])
                budget = row["then_forward_steps"] * base["max_distance_m"]
                if (
                    abs(wrap(angle - row_angle)) < 1e-6
                    and distance + item["max_distance_m"] <= budget + 1e-8
                ):
                    allowed.append(code)
                    break
    return allowed + list(turn_labels) + ["S"]


def make_plan(actions, pose, spec, unprotected, complete=False):
    """Compile selected primitives into display and execution data, without replanning."""
    poses = rollout(pose, actions, spec)
    steps = []
    descriptions = []
    for index, name in enumerate(actions):
        item = spec[name]
        step = dict(index=index + 1, action=name)
        if "max_distance_m" in item:
            direction = "forward" if item["linear_m_s"] > 0 else "backward"
            angle = poses[index][2] + (0 if direction == "forward" else math.pi)
            step.update(
                direction=direction,
                nominal_distance_m=item["max_distance_m"],
                travel_map_bearing_degrees=round(math.degrees(wrap(angle)), 2),
            )
            descriptions.append(f"{direction} {item['max_distance_m']*100:g} cm")
        elif "angle_rad" in item:
            direction = "left" if item["angular_rad_s"] > 0 else "right"
            step.update(
                direction=direction,
                nominal_angle_degrees=round(math.degrees(item["angle_rad"]), 2),
            )
            descriptions.append(f"{direction} {math.degrees(item['angle_rad']):g} deg")
        else:
            descriptions.append("stop / observe")
        steps.append(step)
    return dict(
        version=1,
        source="single_model_plan",
        frame="map_at_planning",
        collision_protection="disabled" if unprotected else "enabled",
        task_complete=complete,
        actions=list(actions),
        steps=steps,
        nominal_end_pose=list(poses[-1]),
        display="; ".join(descriptions),
    )


def first_action_labels(labels, observation):
    """Reject translations blocked by the same check the executor will use."""
    if observation.get("collision_protection") == "disabled":
        return labels
    checks = observation.get("first_motion_checks", {})
    excluded = {*observation.get("recent_stalled_actions", []), *observation.get("recent_turn_overshoots", [])}
    return [label for label in labels if LABELS[label] not in excluded
            and checks.get(LABELS[label], {}).get("allowed", True)]


def recent_stalls(history, reference, snapshot_id):
    """Suppress repeated zero-progress commands in this odom epoch and pose."""
    current = reference.get("pose")
    epoch = reference.get("epoch")
    if not current or not epoch or not reference.get("valid"):
        return []
    counts = {}
    for plan in history:
        if (plan.get("control_epoch") != epoch or not 1 <= snapshot_id - plan["plan_id"] <= 4):
            continue
        steps = plan["step_results"]
        if not steps:
            continue
        step = steps[-1]
        if step["action"] == "stop" or step["outcome"] not in {
                "trajectory_collision", "insufficient_progress_or_deviation"}:
            continue
        if step.get("clearance", {}).get("reason") in {
                "missing_scan_or_transform", "insufficient_lidar_coverage"}:
            continue
        start, end = step.get("control_start_pose"), step.get("control_end_pose")
        if not start or not end:
            continue
        if (math.dist(start[:2], current[:2]) <= .035
                and abs(wrap(start[2] - current[2])) <= math.radians(7)
                and math.dist(start[:2], end[:2]) <= .018
                and abs(wrap(start[2] - end[2])) <= math.radians(4)):
            counts[step["action"]] = counts.get(step["action"], 0) + 1
    return sorted(action for action, count in counts.items() if count >= 2)



def recent_turn_overshoots(history, reference, snapshot_id):
    """Pause observed small-turn overshoots after two failures in this odom epoch."""
    epoch = reference.get("epoch")
    if not epoch or not reference.get("valid"):
        return []
    counts = {}
    for plan in history:
        if plan.get("control_epoch") != epoch or not 1 <= snapshot_id - plan["plan_id"] <= 6:
            continue
        for step in plan["step_results"]:
            action = step["action"]
            if action not in ("left_11_25", "right_11_25"):
                continue
            counts.setdefault(action, 0)
            if step["outcome"] == "duration_complete":
                counts[action] = 0
            elif step["outcome"] == "insufficient_progress_or_deviation":
                nominal = turn_delta(action, ACTION_PLAN_SPEC)
                signed_turn = math.copysign(1, nominal) * step.get("turn_rad", 0)
                if signed_turn > abs(nominal) * 1.5:
                    counts[action] += 1
    # Opposite directions use the same small-turn controller. If both have
    # overshot, neither is a reliable alternative to the other.
    if sum(counts.values()) < 2:
        return []
    return sorted(action for action, count in counts.items() if count)
