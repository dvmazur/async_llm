"""Grid geometry shared by proposal generation and execution validation.

Unknown cells never become traversable. Distances use the same inflated mask
and disallow diagonal corner cutting. All coordinates belong to one snapshot.
"""

import heapq
import math
from dataclasses import dataclass

import cv2
import numpy as np

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


def colored_returns(u, v, valid, mask):
    """Associate measured lidar points only with the interior of an RGB region.

    A region's bounding rectangle can include an occluding wall. An inward
    one-pixel margin also avoids mixing foreground returns at silhouette edges.
    """
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    height, width = interior.shape
    inside = valid & np.isfinite(u) & np.isfinite(v)
    x = np.rint(np.nan_to_num(u, nan=-1)).astype(int)
    y = np.rint(np.nan_to_num(v, nan=-1)).astype(int)
    inside &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
    result = np.zeros(len(u), dtype=bool)
    result[inside] = interior[y[inside], x[inside]]
    return result


def new_nearby_evidence(current, observed, threshold=0.95):
    """Require a new model decision when a close RGB object appears en route.

    All colors/classes are treated equally. An old exploration/approach choice
    must not move the camera away before the model can inspect new evidence.
    Explicit choices made after seeing that close object remain executable.
    """
    for live in current:
        if not live.get("visible") or live["range_m"] >= threshold:
            continue
        if not any(
            old.get("visible")
            and old["range_m"] < threshold
            and old.get("color") == live.get("color")
            and math.dist(old["xy"], live["xy"]) < 0.4
            for old in observed
        ):
            return True
    return False


@dataclass
class Grid:
    values: np.ndarray
    resolution: float
    origin: tuple[float, float]
    frame: str = "map"
    version: int = 0
    stamp: float = 0.0
    yaw: float = 0.0

    def cell(self, xy):
        dx, dy = xy[0] - self.origin[0], xy[1] - self.origin[1]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (
            math.floor((c * dx + s * dy) / self.resolution),
            math.floor((-s * dx + c * dy) / self.resolution),
        )

    def world(self, cell):
        x, y = (np.asarray(cell, dtype=float) + 0.5) * self.resolution
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (self.origin[0] + c * x - s * y, self.origin[1] + s * x + c * y)

    def contains(self, cell):
        return 0 <= cell[0] < self.values.shape[1] and 0 <= cell[1] < self.values.shape[0]


@dataclass
class Reachability:
    grid: Grid
    allowed: np.ndarray
    clearance: np.ndarray
    distance: np.ndarray
    parent: np.ndarray
    start: tuple[int, int]

    def path(self, xy):
        cell = self.grid.cell(xy)
        if not self.grid.contains(cell) or not math.isfinite(
            float(self.distance[cell[1], cell[0]])
        ):
            return []
        result = []
        for _ in range(self.allowed.size):
            result.append(self.grid.world(cell))
            if cell == self.start:
                return result[::-1]
            index = int(self.parent[cell[1], cell[0]])
            if index < 0:
                return []
            cell = (index % self.allowed.shape[1], index // self.allowed.shape[1])
        return []


def reachable(grid, pose, radius=0.13, max_distance=10.0, local_blocked=None):
    free = (grid.values >= 0) & (grid.values <= 35)
    if local_blocked is not None:
        free &= ~local_blocked
    # Padding makes map boundaries unknown, rather than open space.
    clearance = (
        cv2.distanceTransform(np.pad(free.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        * grid.resolution
    )
    allowed = free & (clearance >= radius)
    distance = np.full(free.shape, np.inf, dtype=np.float32)
    parent = np.full(free.shape, -1, dtype=np.int32)
    start = grid.cell(pose)
    result = Reachability(grid, allowed, clearance, distance, parent, start)
    if not grid.contains(start) or not allowed[start[1], start[0]]:
        return result
    distance[start[1], start[0]] = 0.0
    queue = [(0.0, start[0], start[1])]
    height, width = free.shape
    while queue:
        cost, x, y = heapq.heappop(queue)
        if cost > float(distance[y, x]) + 1e-5:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height) or not allowed[ny, nx]:
                continue
            if dx and dy and not (allowed[y, nx] and allowed[ny, x]):
                continue
            new = cost + grid.resolution * (math.sqrt(2) if dx and dy else 1)
            if new > max_distance or new >= float(distance[ny, nx]) - 1e-5:
                continue
            distance[ny, nx] = new
            parent[ny, nx] = y * width + x
            heapq.heappush(queue, (new, nx, ny))
    return result


def visible_unknown(grid, cell, radius=2.5, rays=48):
    """Optimistic information proxy: stop at known walls, never see through them."""
    steps = int(radius / grid.resolution)
    angles = np.linspace(-math.pi, math.pi, rays, endpoint=False)[:, None]
    distance = np.arange(1, steps + 1)[None, :]
    x = cell[0] + np.rint(np.cos(angles) * distance).astype(int)
    y = cell[1] + np.rint(np.sin(angles) * distance).astype(int)
    height, width = grid.values.shape
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    values = grid.values[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)]
    visible = np.cumsum((values >= 65) | ~valid, axis=1) == 0
    unknown = visible & ((values < 0) | ((values > 35) & (values < 65)))
    return np.unique((y * width + x)[unknown]).size * grid.resolution**2


def overlay_obstacles(grid, dynamic, transform):
    """Reproject only currently occupied dynamic cells into the SLAM frame.

    transform is (tx, ty, yaw) from dynamic.frame to grid.frame. Missing or stale
    TF must be handled by the caller; differently framed arrays are never ORed.
    """
    result = np.zeros(grid.values.shape, bool)
    rows, cols = np.nonzero(dynamic.values >= 65)
    if not len(rows):
        return result
    local = np.array([dynamic.world((x, y)) for x, y in zip(cols, rows)])
    c, s = math.cos(transform[2]), math.sin(transform[2])
    world = local @ np.array([[c, s], [-s, c]]) + np.asarray(transform[:2])
    for xy in world:
        x, y = grid.cell(xy)
        if grid.contains((x, y)):
            result[y, x] = True
    return result


def fuse_observations(grid, dynamic, transform):
    """Fill SLAM unknown cells from the recent lidar grid in the map frame.

    SLAM may retain only sparse rays when a scan has no return before maximum
    range. The rolling grid explicitly raytraces those no-return beams. Fusing
    its measured free cells gives candidate generation and Nav2 the same
    conservative, sensor-derived local evidence without making unknown space
    traversable. Existing SLAM obstacles always win over local free evidence.
    """
    values = grid.values.copy()
    rows, cols = np.nonzero(dynamic.values >= 0)
    if not len(rows):
        return Grid(
            values,
            grid.resolution,
            grid.origin,
            grid.frame,
            grid.version,
            grid.stamp,
            grid.yaw,
        )
    x = (cols + 0.5) * dynamic.resolution
    y = (rows + 0.5) * dynamic.resolution
    dc, ds = math.cos(dynamic.yaw), math.sin(dynamic.yaw)
    local_x = dynamic.origin[0] + dc * x - ds * y
    local_y = dynamic.origin[1] + ds * x + dc * y
    tc, ts = math.cos(transform[2]), math.sin(transform[2])
    world_x = transform[0] + tc * local_x - ts * local_y
    world_y = transform[1] + ts * local_x + tc * local_y
    dx, dy = world_x - grid.origin[0], world_y - grid.origin[1]
    gc, gs = math.cos(grid.yaw), math.sin(grid.yaw)
    gx = np.floor((gc * dx + gs * dy) / grid.resolution).astype(int)
    gy = np.floor((-gs * dx + gc * dy) / grid.resolution).astype(int)
    inside = (gx >= 0) & (gx < values.shape[1]) & (gy >= 0) & (gy < values.shape[0])
    gx, gy, measured = gx[inside], gy[inside], dynamic.values[rows[inside], cols[inside]]
    free = measured <= 35
    unknown = values[gy, gx] < 0
    values[gy[free & unknown], gx[free & unknown]] = 0
    occupied = measured >= 65
    values[gy[occupied], gx[occupied]] = 100
    return Grid(
        values,
        grid.resolution,
        grid.origin,
        grid.frame,
        grid.version,
        max(grid.stamp, dynamic.stamp),
        grid.yaw,
    )
