"""Task-independent, reachable candidate proposals; no hidden target coordinates."""

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .geometry import visible_unknown


def planning_window_open(active, pose, urgent=False):
    """A wakeup from a previous goal must not prefetch from a new goal's start."""
    return (
        active is None
        or urgent
        or math.dist((active.x, active.y), pose[:2])
        < min(1.2, max(0.2, active.path_length_m * 0.5))
    )


@dataclass
class Candidate:
    id: str
    kind: str
    x: float
    y: float
    yaw: float
    path_length_m: float
    clearance_m: float
    information_m2: float = 0.0
    visits: int = 0
    failures: int = 0
    description: str = ""

    def public(self):
        return {
            key: round(value, 3) if isinstance(value, float) else value
            for key, value in asdict(self).items()
        }


class CandidateGenerator:
    def __init__(self, max_candidates=8, separation=0.75):
        self.max_candidates = max_candidates
        self.separation = separation
        self.history = []
        self.serial = 0

    def record(self, candidate, outcome):
        self.history.append((candidate.x, candidate.y, outcome, candidate.yaw))
        self.history = self.history[-100:]

    def counts(self, xy):
        nearby = [h for h in self.history if math.dist(h[:2], xy) < self.separation]
        return (
            sum(h[2] == "reached" for h in nearby),
            sum(h[2] in {"failed", "stuck", "invalidated"} for h in nearby),
        )

    def generate(self, reach, pose, objects=()):
        grid = reach.grid
        free = (grid.values >= 0) & (grid.values <= 35)
        unknown = (grid.values < 0) | ((grid.values > 35) & (grid.values < 65))
        adjacent = cv2.dilate(unknown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        frontiers = (free & adjacent).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(frontiers, 8)
        pool = []
        groups = sorted(
            range(1, count), key=lambda label: stats[label, cv2.CC_STAT_AREA], reverse=True
        )[:32]
        for label in groups:
            if stats[label, cv2.CC_STAT_AREA] < 3:
                continue
            ys, xs = np.nonzero(labels == label)
            # Sample long boundaries as several viewpoints; a component centroid
            # may lie inside a wall or in unknown space.
            stride = max(1, len(xs) // 12)
            for fx, fy in zip(xs[::stride], ys[::stride]):
                span = max(3, math.ceil(0.65 / grid.resolution))
                x0, x1 = max(0, fx - span), min(grid.values.shape[1], fx + span + 1)
                y0, y1 = max(0, fy - span), min(grid.values.shape[0], fy + span + 1)
                distances = reach.distance[y0:y1, x0:x1]
                cy, cx = np.indices(distances.shape)
                cx += x0
                cy += y0
                offset = np.hypot(cx - fx, cy - fy) * grid.resolution
                valid = (
                    np.isfinite(distances) & (distances > 0.45) & (offset > 0.20) & (offset < 0.65)
                )
                if not valid.any():
                    continue
                score = np.where(
                    valid, offset + 0.05 * distances - 0.15 * reach.clearance[y0:y1, x0:x1], np.inf
                )
                iy, ix = np.unravel_index(score.argmin(), score.shape)
                cell = (int(ix + x0), int(iy + y0))
                xy = grid.world(cell)
                front = grid.world((fx, fy))
                yaw = math.atan2(front[1] - xy[1], front[0] - xy[0])
                visits, failures = self.counts(xy)
                gain = visible_unknown(grid, cell)
                if gain < 0.015 or failures >= 2:
                    continue
                pool.append(
                    Candidate(
                        "",
                        "frontier",
                        *xy,
                        yaw,
                        float(reach.distance[cell[1], cell[0]]),
                        float(reach.clearance[cell[1], cell[0]]),
                        gain,
                        visits,
                        failures,
                        "Observe unexplored space",
                    )
                )
        # Objects are sensor-derived hypotheses from RGB + measured lidar range.
        for obj in objects:
            ox, oy = obj["xy"]
            dx, dy = pose[0] - ox, pose[1] - oy
            direction = math.atan2(dy, dx)
            viewpoints = [
                (
                    ox + 0.65 * math.cos(direction + offset),
                    oy + 0.65 * math.sin(direction + offset),
                )
                for offset in (0.0, -0.6, 0.6)
            ]
            # A visible object can lie beyond the currently traversable patch.
            # Offer measured, reachable progress toward it, rather than discard
            # it until the entire final approach happens to be mapped.
            rows, cols = np.nonzero(np.isfinite(reach.distance) & (reach.distance > 0.35))
            if len(rows):
                x = (cols + 0.5) * grid.resolution
                y = (rows + 0.5) * grid.resolution
                c, s = math.cos(grid.yaw), math.sin(grid.yaw)
                world = np.column_stack(
                    (grid.origin[0] + c * x - s * y, grid.origin[1] + s * x + c * y)
                )
                separation = np.hypot(world[:, 0] - ox, world[:, 1] - oy)
                score = np.where(
                    separation >= 0.6,
                    separation
                    + 0.03 * reach.distance[rows, cols]
                    - 0.1 * reach.clearance[rows, cols],
                    np.inf,
                )
                if np.isfinite(score).any():
                    viewpoints.append(tuple(world[int(score.argmin())]))
            for xy in viewpoints:
                cell = grid.cell(xy)
                if not grid.contains(cell) or not np.isfinite(reach.distance[cell[1], cell[0]]):
                    continue
                if reach.distance[cell[1], cell[0]] < 0.18:
                    continue
                visits, failures = self.counts(xy)
                if failures >= 2:
                    continue
                pool.append(
                    Candidate(
                        "",
                        "inspect",
                        *xy,
                        math.atan2(oy - xy[1], ox - xy[0]),
                        float(reach.distance[cell[1], cell[0]]),
                        float(reach.clearance[cell[1], cell[0]]),
                        0.0,
                        visits,
                        failures,
                        obj["description"],
                    )
                )
            # A close, previously measured object can leave the camera before
            # its approach viewpoint becomes traversable. Keep a pure rotation
            # available so the model can reacquire it instead of receiving an
            # empty allowlist. This is identical for every object color/class.
            if not obj.get("visible", False) and obj.get("range_m", math.inf) <= 2.5:
                heading = math.atan2(oy - pose[1], ox - pose[0])
                look_visits = sum(
                    math.dist(h[:2], pose[:2]) < 0.5
                    and len(h) > 3
                    and abs(math.atan2(math.sin(h[3] - heading), math.cos(h[3] - heading))) < 0.4
                    for h in self.history
                )
                if look_visits < 2 and grid.contains(reach.start):
                    pool.append(
                        Candidate(
                            "",
                            "look",
                            pose[0],
                            pose[1],
                            heading,
                            0.0,
                            float(reach.clearance[reach.start[1], reach.start[0]]),
                            visits=look_visits,
                            description="Turn toward last measured " + obj["description"],
                        )
                    )
        # Filter redundant viewpoints, retaining direction diversity. This score
        # only reduces the proposal set; it never selects the executed goal.
        pool.sort(
            key=lambda c: (
                c.kind != "inspect",
                c.failures,
                c.visits,
                -(c.information_m2 / (1 + 0.12 * c.path_length_m)),
            )
        )
        selected = []
        sectors = set()
        for diverse in (True, False):
            for item in pool:
                if len(selected) >= self.max_candidates:
                    break
                if any(
                    math.dist((item.x, item.y), (other.x, other.y)) < self.separation
                    for other in selected
                ):
                    continue
                angle = math.atan2(item.y - pose[1], item.x - pose[0])
                sector = round(angle / (math.pi / 4)) % 8
                if diverse and sector in sectors:
                    continue
                selected.append(item)
                sectors.add(sector)
        if (
            not selected
            and grid.contains(reach.start)
            and reach.allowed[reach.start[1], reach.start[0]]
        ):
            for offset, label in (
                (math.pi / 2, "left"),
                (-math.pi / 2, "right"),
                (math.pi, "behind"),
            ):
                heading = math.atan2(math.sin(pose[2] + offset), math.cos(pose[2] + offset))
                visits = sum(
                    math.dist(h[:2], pose[:2]) < 0.5
                    and len(h) > 3
                    and abs(math.atan2(math.sin(h[3] - heading), math.cos(h[3] - heading))) < 0.4
                    for h in self.history
                )
                if visits:
                    continue
                selected.append(
                    Candidate(
                        "",
                        "look",
                        pose[0],
                        pose[1],
                        heading,
                        0.0,
                        float(reach.clearance[reach.start[1], reach.start[0]]),
                        description=f"Look {label} to inspect the RGB scene",
                    )
                )
        self.serial += 1
        for index, item in enumerate(selected):
            item.id = chr(ord("A") + index)
        return selected
