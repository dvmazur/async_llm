"""Check the chassis on a grid, allowing it to leave a shallow start overlap."""

import math

import numpy as np


_EPS = 1e-6  # Area in units of one grid cell.


def _clip_polygon(polygon, axis, boundary, side):
    result = []
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        start_inside = side * (start[axis] - boundary) >= 0
        end_inside = side * (end[axis] - boundary) >= 0
        if start_inside:
            result.append(start)
        if start_inside != end_inside:
            fraction = (boundary - start[axis]) / (end[axis] - start[axis])
            crossing = [a + fraction * (b - a) for a, b in zip(start, end)]
            crossing[axis] = boundary
            result.append(crossing)
    return result


def _cell_overlap(polygon, x, y):
    for axis, boundary, side in ((0, x, 1), (0, x+1, -1), (1, y, 1), (1, y+1, -1)):
        polygon = _clip_polygon(polygon, axis, boundary, side)
        if len(polygon) < 3:
            return 0.0
    # Subtract the cell origin to keep the area numerically stable on large maps.
    local = [(a-x, b-y) for a, b in polygon]
    return abs(sum(a*d - b*c for (a, b), (c, d) in zip(local, local[1:] + local[:1]))) / 2


class GridMotion:
    """Known-free hulls pass normally, with a narrow exception for start egress.

    A blocked cell is a coarse estimate of an obstacle. If its centre is near
    the boundary of the physical chassis, allow the chassis to leave that cell
    without covering any additional area. A cell well inside the chassis or a
    footprint outside the grid still prevents translation.
    """

    def __init__(self, grid, pose, half_length, half_width, gap):
        self.grid = grid
        self.pose = pose
        self.resolution = grid.resolution
        self.half_length = half_length + gap
        self.half_width = half_width + gap
        cos, sin = math.cos(grid.yaw), math.sin(grid.yaw)
        self.grid_rotation = np.asarray([[cos, -sin], [sin, cos]])
        self.origin = np.asarray(grid.origin[:2])
        self.initial = self.contacts(pose)
        self.deep_overlap = False

        cos, sin = math.cos(pose[2]), math.sin(pose[2])
        world_to_body = np.asarray([[cos, -sin], [sin, cos]])
        half_cell = self.resolution / 2
        height, width = grid.values.shape
        for x, y in self.initial:
            if not (0 <= x < width and 0 <= y < height):
                self.deep_overlap = True
                break
            centre = self.origin + self.resolution * (
                np.asarray([x+0.5, y+0.5]) @ self.grid_rotation.T)
            bx, by = (centre - pose[:2]) @ world_to_body
            if abs(bx) < half_length - half_cell and abs(by) < half_width - half_cell:
                self.deep_overlap = True
                break
        self.initial_area = sum(self.initial.values())

    def contacts(self, pose, length_padding=0.0):
        x, y, yaw = pose
        cos, sin = math.cos(yaw), math.sin(yaw)
        length, width = self.half_length + length_padding, self.half_width
        polygon = np.asarray([(length, width), (-length, width),
                              (-length, -width), (length, -width)])
        polygon = polygon @ np.asarray([[cos, sin], [-sin, cos]]) + [x, y]
        polygon = ((polygon - self.origin) @ self.grid_rotation) / self.resolution
        low = np.floor(polygon.min(axis=0)).astype(int)
        high = np.floor(polygon.max(axis=0)).astype(int)
        values = self.grid.values
        points = polygon.tolist()

        if (0 <= low[0] and high[0] < values.shape[1] and
                0 <= low[1] and high[1] < values.shape[0]):
            rows = values[low[1]:high[1]+1, low[0]:high[0]+1]
            ys, xs = np.nonzero((rows < 0) | (rows > 35))
            cells = zip(xs + low[0], ys + low[1])
        else:
            cells = ((gx, gy) for gy in range(low[1], high[1]+1) for gx in range(low[0], high[0]+1)
                     if not (0 <= gx < values.shape[1] and 0 <= gy < values.shape[0])
                     or not 0 <= values[gy, gx] <= 35)
        contacts = {}
        for gx, gy in cells:
            area = _cell_overlap(points, int(gx), int(gy))
            if area > _EPS:
                contacts[(int(gx), int(gy))] = area
        return contacts

    @staticmethod
    def _no_increase(current, previous):
        return all(cell in previous and area <= previous[cell] + _EPS
                   for cell, area in current.items())

    def turn(self, delta):
        if self.deep_overlap:
            return None
        previous = self.initial
        x, y, yaw = self.pose
        max_angle = min(0.08, self.resolution / (4 * math.hypot(self.half_length, self.half_width)))
        samples = math.ceil(abs(delta) / max_angle)
        for index in range(1, samples+1):
            current = self.contacts((x, y, yaw + delta * index / samples))
            if not self._no_increase(current, previous):
                return None
            previous = current
        return previous

    def distance(self, heading, previous, direction):
        x, y, _ = self.pose
        cos, sin = direction * math.cos(heading), direction * math.sin(heading)
        spacing = self.resolution / 2
        distance = 0.0
        for index in range(1, math.floor(1.0 / spacing + _EPS) + 1):
            end = index * spacing
            midpoint = end - spacing / 2
            # A straight sweep is a longer rectangle. This also checks corners
            # between samples and forbids any new overlap inside a touched cell.
            swept = self.contacts((x+cos*midpoint, y+sin*midpoint, heading), spacing / 2)
            if not self._no_increase(swept, previous):
                break
            previous = self.contacts((x+cos*end, y+sin*end, heading)) if swept else {}
            if not self.initial or sum(previous.values()) < self.initial_area - _EPS:
                distance = end
        return distance
