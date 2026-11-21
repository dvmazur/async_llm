"""Swept chassis checks and short escape choices using measured lidar only."""

import math
from dataclasses import dataclass

import numpy as np


HALF_LENGTH = 0.10
HALF_WIDTH = 0.08
HARD_GAP = 0.0075
STRAIGHT_SWEEP_GAP_M = 0.002
STRATEGY4_BRAKING_S = 0.55
MAX_EDGE_OVERLAP_M = 0.006
MAX_ESCAPE_SWEEP_M = 0.055
SHORT_TRANSLATIONS = frozenset({"creep_forward", "creep_backward"})


@dataclass(frozen=True)
class Clearance:
    gap_m: float | None
    reason: str
    start_gap_m: float | None = None
    end_gap_m: float | None = None

    def public(self):
        return {"allowed": self.gap_m is not None, "reason": self.reason,
                "start_gap_m": self.start_gap_m, "end_gap_m": self.end_gap_m}


def body_clearance(points):
    outside = np.abs(points) - (HALF_LENGTH, HALF_WIDTH)
    return np.linalg.norm(np.maximum(outside, 0), axis=1) + np.minimum(outside.max(axis=1), 0)


def scan_points(scan, origin):
    ranges = np.asarray(scan.ranges)
    valid = (ranges >= scan.range_min) & (
        np.isposinf(ranges) | (ranges <= scan.range_max)
    )
    angles = scan.angle_min + np.arange(ranges.size) * scan.angle_increment
    hits = valid & np.isfinite(ranges)
    points = np.column_stack((np.cos(angles[hits]), np.sin(angles[hits]))) * ranges[hits, None]
    return points + origin, angles, valid


def trajectory_clearance(scan, origin, linear, angular, duration):
    """Return the worst clearance, or None for an unobserved or colliding sweep."""
    return check_sweep(scan, origin, linear, angular, duration).gap_m


def check_strategy4_translation(scan, origin, linear, angular, action):
    return check_sweep(
        scan, origin, linear, angular, STRATEGY4_BRAKING_S,
        allow_edge_exit=action in SHORT_TRANSLATIONS, allow_tangent=True,
        margin=STRAIGHT_SWEEP_GAP_M if angular == 0 and linear != 0 else HARD_GAP,
    )


def check_sweep(scan, origin, linear, angular, duration, *,
                allow_edge_exit=False, allow_tangent=False, margin=HARD_GAP):
    """Check each measured return in the moving chassis frame."""
    if scan is None or origin is None:
        return Clearance(None, "missing_scan_or_transform")
    points, angles, valid = scan_points(scan, origin)
    if abs(angular) > 0.01 and abs(linear) < 0.01:
        zone = np.ones(len(valid), dtype=bool)
    else:
        centre = 0 if linear >= 0 else math.pi
        zone = np.abs(np.arctan2(np.sin(angles - centre), np.cos(angles - centre))) < 0.9 + abs(angular) * duration
    if not zone.any() or np.count_nonzero(valid & zone) < 0.70 * np.count_nonzero(zone):
        return Clearance(None, "insufficient_lidar_coverage")
    if not len(points):
        return Clearance(float(scan.range_max), "clear")

    initial = body_clearance(points)
    start_gap = float(initial.min())
    overlapping = initial <= 0
    edge_exit = (start_gap <= 0 and allow_edge_exit and start_gap >= -MAX_EDGE_OVERLAP_M
                 and abs(angular) < 1e-6 and 0 < abs(linear) * duration <= MAX_ESCAPE_SWEEP_M + 1e-9)
    if start_gap <= 0 and not edge_exit:
        return Clearance(None, "occupied_start", start_gap)

    # Keep the existing criteria for recovering from an initial footprint overlap.
    margin = HARD_GAP if edge_exit else margin
    near = initial < margin
    threshold = np.minimum(initial, margin) - .001
    if margin == STRAIGHT_SWEEP_GAP_M:
        # Starting below the buffer permits tangent/outward motion only.
        # A positive buffer is checked at every sample, including the endpoint.
        threshold = np.minimum(initial, margin) - 1e-9
    worst = start_gap
    previous = initial
    best_near = initial[near]
    if allow_tangent or edge_exit:
        samples = max(2, math.ceil(duration / 0.025))
        start = duration / samples
    else:
        samples = max(2, math.ceil(duration / 0.05))
        start = 0.05
    for stamp in np.linspace(start, duration, samples):
        heading = angular * stamp
        if abs(angular) < 1e-6:
            offset = np.array([linear * stamp, 0])
        else:
            offset = linear / angular * np.array([math.sin(heading), 1 - math.cos(heading)])
        c, s = math.cos(heading), math.sin(heading)
        gaps = body_clearance((points - offset) @ np.array([[c, -s], [s, c]]))
        if np.any(gaps[~overlapping] <= 0):
            return Clearance(None, "new_chassis_overlap", start_gap, float(gaps.min()))
        if np.any(gaps < threshold):
            return Clearance(None, "swept_margin", start_gap, float(gaps.min()))
        if edge_exit and np.any(gaps[overlapping] < previous[overlapping] - 1e-6):
            return Clearance(None, "overlap_deepens", start_gap, float(gaps.min()))
        if np.any(gaps[near] < best_near - 0.001):
            return Clearance(None, "near_clearance_decreases", start_gap, float(gaps.min()))
        best_near = np.maximum(best_near, gaps[near]) if allow_tangent else gaps[near]
        previous = gaps
        worst = min(worst, float(gaps.min()))

    end_gap = float(previous.min())
    if edge_exit:
        if np.any(previous[overlapping] < HARD_GAP):
            return Clearance(None, "overlap_does_not_clear", start_gap, end_gap)
        return Clearance(worst, "edge_exit", start_gap, end_gap)
    if not allow_tangent and start_gap < HARD_GAP and end_gap < start_gap + 0.002:
        return Clearance(None, "near_clearance_does_not_improve", start_gap, end_gap)
    return Clearance(worst, "clear", start_gap, end_gap)


def escape_motion(scan, origin):
    """Choose a sensor-checked 12 cm translation before attempting another turn."""
    choices = []
    for speed in (-0.06, 0.06):
        gap = trajectory_clearance(scan, origin, speed, 0.0, 2.3)
        if gap is not None:
            # Prefer the larger corridor, using reverse to resolve a tied front blockage.
            points, _, _ = scan_points(scan, origin)
            offset = np.array([math.copysign(0.12, speed), 0])
            end_gap = float(body_clearance(points - offset).min()) if len(points) else scan.range_max
            choices.append((end_gap, gap, speed))
    return max(choices)[2] if choices else None


def escape_course(scan, origin):
    """Select a clear translation or gentle arc that increases centre clearance."""
    points, _, _ = scan_points(scan, origin)
    choices = []
    duration = 2.3
    for speed in (-0.06, 0.06):
        for angular in (0.0, -0.20, 0.20, -0.35, 0.35):
            gap = trajectory_clearance(scan, origin, speed, angular, duration)
            if gap is None:
                continue
            heading = angular * duration
            if angular:
                end = speed / angular * np.array([math.sin(heading), 1 - math.cos(heading)])
            else:
                end = np.array([speed * duration, 0])
            centre_gap = float(np.linalg.norm(points - end, axis=1).min()) if len(points) else scan.range_max
            score = centre_gap - 0.025 * abs(angular)
            choices.append((score, gap, speed, angular))
    if not choices:
        return None
    _, _, speed, angular = max(choices)
    return speed, angular
