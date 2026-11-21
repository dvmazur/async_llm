"""Privileged experiment construction. Never imported by the navigation agent."""

import copy
import json
import xml.etree.ElementTree as ET

from robot_sim.rover import add, make_world, write_xml

SCENARIOS = {
    "red_near": {
        "bounds": [-3, 4, -3, 3],
        "spawn": [0, 0, 0],
        "target_xy": [2.0, 0.0],
        "target_color": "red",
        "partitions": [],
        "distractors": [("blue", 1.5, 1.5)],
        "sim_limit": 100,
        "minimum_path_m": 0.8,
    },
    "blue_occluded": {
        "bounds": [-5, 6, -5, 5],
        "spawn": [-3, -2.5, 0],
        "target_xy": [4.0, -2.5],
        "target_color": "blue",
        "partitions": [(0.0, -1.5, 0.18, 6.8)],
        "distractors": [("red", -2.0, 2.5), ("green", 3.0, 2.5)],
        "sim_limit": 350,
        "minimum_path_m": 8.0,
    },
    "green_long": {
        "bounds": [-11, 11, -5, 5],
        "spawn": [-9, -3, 0],
        "target_xy": [9.0, 3.0],
        "target_color": "green",
        "partitions": [(-3.0, -1.5, 0.18, 6.8), (3.0, 1.5, 0.18, 6.8)],
        "distractors": [("red", -7.0, 2.5), ("blue", 0.0, -3.0)],
        "sim_limit": 650,
        "minimum_path_m": 20.0,
    },
    "absent_yellow": {
        "bounds": [-3, 4, -3, 3],
        "spawn": [0, 0, 0],
        "target_xy": [2.0, 0.0],
        "target_color": "yellow",
        "partitions": [],
        "distractors": [("red", 2.0, 0.0), ("blue", 1.0, 1.5)],
        "sim_limit": 90,
        "minimum_path_m": 0.0,
        "absent": True,
    },
    "yellow_zigzag": {
        "bounds": [-10, 10, -5, 5],
        "spawn": [-8, -3, 0],
        "target_xy": [8, 3],
        "target_color": "yellow",
        "partitions": [(-4, -1.4, 0.18, 6.8), (1, 1.4, 0.18, 6.8), (5, -1.4, 0.18, 6.8)],
        "distractors": [("red", -6, 2), ("blue", -1, -3), ("green", 3, 3)],
        "sim_limit": 750,
        "minimum_path_m": 20.0,
    },
    "red_deadends": {
        "bounds": [-7, 8, -5, 5],
        "spawn": [-5, 0, 0],
        "target_xy": [6, 3],
        "target_color": "red",
        "partitions": [(0, -1.2, 0.18, 7.2), (-2, 1, 4, 0.18), (-2, -1, 4, 0.18)],
        "distractors": [("green", -1, 0), ("blue", -5, 3), ("yellow", 3, -3)],
        "sim_limit": 500,
        "minimum_path_m": 8.0,
    },
    "blue_clutter": {
        "bounds": [-8, 8, -6, 6],
        "spawn": [-6, -4, 0],
        "target_xy": [6, 4],
        "target_color": "blue",
        "partitions": [
            (-2, -2, 0.18, 7.4),
            (3, 2, 0.18, 7.4),
            (-5, 1, 1.2, 1.2),
            (-4, 3, 1.2, 1.2),
            (0, 1, 1.0, 1.0),
            (1, -2.5, 1.0, 1.0),
            (5, 0, 1.0, 1.0),
        ],
        "distractors": [("red", -5, 4), ("green", 0, -4), ("yellow", 5, -3)],
        "sim_limit": 650,
        "minimum_path_m": 18.0,
    },
}
COLORS = {
    "red": ".90 .035 .025 1",
    "blue": ".025 .12 .95 1",
    "green": ".035 .75 .07 1",
    "yellow": ".95 .8 .025 1",
    "wall": ".78 .79 .76 1",
    "floor": ".42 .45 .48 1",
}


def box(world, name, x, y, sx, sy, sz, color):
    model = add(world, "model", name=name)
    add(model, "static", "true")
    add(model, "pose", f"{x} {y} {sz / 2} 0 0 0")
    link = add(model, "link", name="body")
    for kind in ("collision", "visual"):
        element = add(link, kind, name=kind)
        add(add(add(element, "geometry"), "box"), "size", f"{sx} {sy} {sz}")
        if kind == "visual":
            material = add(element, "material")
            add(material, "ambient", COLORS[color])
            add(material, "diffuse", COLORS[color])
    return model


def build_scene(name, output):
    scenario = copy.deepcopy(SCENARIOS[name])
    scenario["name"] = name
    make_world("warehouse", output / "world.sdf", spawn=scenario["spawn"])
    document = ET.parse(output / "world.sdf")
    world = document.find("world")
    for model in list(world.findall("model")):
        if model.get("name") != "yandex_rover":
            world.remove(model)
    robot = world.find("model[@name='yandex_rover']")
    sensor = robot.find("link[@name='base_link']/sensor[@name='overview_camera']")
    if sensor is not None:
        robot.find("link[@name='base_link']").remove(sensor)
    xmin, xmax, ymin, ymax = scenario["bounds"]
    ground = box(
        world,
        "ground",
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
        xmax - xmin + 2,
        ymax - ymin + 2,
        0.05,
        "floor",
    )
    ground.find("pose").text = f"{(xmin + xmax) / 2} {(ymin + ymax) / 2} -.025 0 0 0"
    for label, x, y, sx, sy in [
        ("south", (xmin + xmax) / 2, ymin, xmax - xmin, 0.15),
        ("north", (xmin + xmax) / 2, ymax, xmax - xmin, 0.15),
        ("west", xmin, (ymin + ymax) / 2, 0.15, ymax - ymin),
        ("east", xmax, (ymin + ymax) / 2, 0.15, ymax - ymin),
    ]:
        box(world, label, x, y, sx, sy, 1.6, "wall")
    for index, (x, y, sx, sy) in enumerate(scenario["partitions"]):
        box(world, f"partition_{index}", x, y, sx, sy, 1.5, "wall")
    objects = list(scenario["distractors"])
    if not scenario.get("absent"):
        objects.append((scenario["target_color"], *scenario["target_xy"]))
    for index, (color, x, y) in enumerate(objects):
        box(world, f"object_{index}", x, y, 0.4, 0.4, 0.45, color)
    write_xml(document.getroot(), output / "world.sdf")
    (output / "evaluation_scenario.json").write_text(json.dumps(scenario, indent=2))
    return scenario
