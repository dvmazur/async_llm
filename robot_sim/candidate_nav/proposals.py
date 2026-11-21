"""Coordinate proposals authored by Qwen; geometry validates, never invents replacements."""

import math

import numpy as np
from PIL import ImageDraw

from .candidates import Candidate
from .geometry import visible_unknown
from .render import font, pixel, render_map


def unpixel(uv, p):
    x = (uv[0] - p["offset_x"]) / p["scale"] * p["resolution"]
    y = (p["height"] - (uv[1] - p["offset_y"]) / p["scale"]) * p["resolution"]
    c, s = math.cos(p.get("yaw", 0)), math.sin(p.get("yaw", 0))
    return (p["origin"][0] + c * x - s * y, p["origin"][1] + s * x + c * y)


def proposal_map(reach, pose, path=(), active=None, objects=()):
    # Known free but unreachable cells are gray; white cells are valid locations
    # for the rover centre under the same footprint/path rules as the baseline.
    from dataclasses import replace

    values = reach.grid.values.copy()
    values[(values >= 0) & (values <= 35) & ~np.isfinite(reach.distance)] = 45
    image, p = render_map(replace(reach.grid, values=values), pose, (), path, active, objects)
    draw = ImageDraw.Draw(image)
    for value in range(64, 512, 64):
        draw.line((value, 30, value, 490), fill=(100, 110, 120), width=1)
        draw.line((10, value, 502, value), fill=(100, 110, 120), width=1)
        draw.text((value + 2, 29), str(value), font=font(10), fill="white")
        draw.text((1, value + 2), str(value), font=font(10), fill="white")
    facts = {
        "image_size": [512, 512],
        "coordinate_convention": "image pixels: x right, y down; top-left=(0,0)",
        "heading_convention": "h=0 right,1 down-right,2 down,3 down-left,4 left,5 up-left,6 up,7 up-right",
        "rover_pixel": [round(v, 1) for v in pixel(pose[:2], p)],
        "objects_pixel": [
            {"pixel": [round(v, 1) for v in pixel(o["xy"], p)], "description": o["description"]}
            for o in objects
        ],
        "white_is_reachable_robot_center": True,
        "maximum_snap_m": 0.35,
        "projection": p,
    }
    return image, facts


def validate_proposals(raw, reach, p, pose, generator, max_snap=0.35):
    """Bounded nearest-cell snap; reject walls, duplicates and disconnected goals.

    No frontier/object coordinates are generated here. Every accepted point is
    tied to a raw model coordinate and its displacement is recorded.
    """
    accepted, records = [], []
    rows, cols = np.nonzero(np.isfinite(reach.distance))
    free_world = np.asarray([reach.grid.world((int(x), int(y))) for y, x in zip(rows, cols)])
    for index, item in enumerate(raw[:8]):
        record = {"index": index, "raw": item, "accepted": False}
        records.append(record)
        if (
            not isinstance(item, dict)
            or any(type(item.get(k)) is not int for k in ("u", "v", "heading"))
            or not 0 <= item["u"] < 512
            or not 0 <= item["v"] < 512
            or not 0 <= item["heading"] < 8
            or item.get("kind") not in {"frontier", "inspect", "look"}
        ):
            record["reason"] = "invalid_schema"
            continue
        xy = unpixel((item["u"], item["v"]), p)
        record["raw_world_xy"] = list(xy)
        # A point inside a measured obstacle or outside the image's grid is not
        # rescued by snapping; snap is only for discretisation/clearance margins.
        cell = reach.grid.cell(xy)
        if not reach.grid.contains(cell) or reach.grid.values[cell[1], cell[0]] >= 65:
            record["reason"] = "outside_or_occupied"
            continue
        if reach.grid.values[cell[1], cell[0]] < 0 or not len(free_world):
            record["reason"] = "unknown_or_no_reachable_cells"
            continue
        distances = np.linalg.norm(free_world - xy, axis=1)
        nearest = int(distances.argmin())
        if distances[nearest] > max_snap:
            record["reason"] = "unreachable_or_snap_too_far"
            continue
        snapped = tuple(free_world[nearest])
        steps = max(2, math.ceil(float(distances[nearest]) / (reach.grid.resolution * 0.25)))
        segment = [reach.grid.cell(point) for point in np.linspace(xy, snapped, steps)]
        if any(
            not reach.grid.contains(c) or not 0 <= reach.grid.values[c[1], c[0]] <= 35
            for c in segment
        ):
            record["reason"] = "snap_crosses_obstacle_or_unknown"
            continue
        xy = snapped
        cell = reach.grid.cell(xy)
        heading = p.get("yaw", 0) - item["heading"] * math.pi / 4
        heading = math.atan2(math.sin(heading), math.cos(heading))
        rotation = abs(math.atan2(math.sin(heading - pose[2]), math.cos(heading - pose[2])))
        if math.dist(xy, pose[:2]) < 0.18 and rotation < 0.35:
            record["reason"] = "no_progress"
            continue
        if any(
            math.dist(xy, (other.x, other.y)) < 0.5
            and (
                item["kind"] != "look"
                or abs(math.atan2(math.sin(heading - other.yaw), math.cos(heading - other.yaw)))
                < 0.5
            )
            for other in accepted
        ):
            record["reason"] = "duplicate"
            continue
        visits, failures = generator.counts(xy)
        if failures >= 2:
            record["reason"] = "repeated_failure"
            continue
        kind = "look" if math.dist(xy, pose[:2]) < 0.18 else item["kind"]
        candidate = Candidate(
            chr(65 + len(accepted)),
            kind,
            *xy,
            heading,
            float(reach.distance[cell[1], cell[0]]),
            float(reach.clearance[cell[1], cell[0]]),
            visible_unknown(reach.grid, cell),
            visits,
            failures,
            "Qwen-generated " + kind + " viewpoint",
        )
        accepted.append(candidate)
        record.update(accepted=True, candidate=candidate.public(), snap_m=float(distances[nearest]))
    return accepted, records
