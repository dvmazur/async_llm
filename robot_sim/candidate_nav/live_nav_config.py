"""Configure isolated native Nav2 servers for the physical rover."""

import copy
from pathlib import Path
import sys

import yaml


NS = "/rover_live"
PRIVATE_VELOCITY = NS + "/guarded_cmd_vel"


def native_config(root=Path('/src/packages/main')):
    # Preserve the rbm-80 layout; newer rover images keep the same config
    # under config/ rather than launch/.
    candidates = [root / 'launch/params.yaml', root / 'config/params.yaml']
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('Native Nav2 parameters not found: ' + ', '.join(map(str, candidates)))


def write_config(path):
    source_path = native_config()
    source = yaml.safe_load(source_path.read_text())
    print(f'Native Nav2 parameters: {source_path}', flush=True)
    control = source["controller_server"]["ros__parameters"]
    control.update(use_sim_time=False, odom_topic="/icp/odom", failure_tolerance=0.25,
                   enable_stamped_cmd_vel=False, controller_plugins=["FollowPath", "Escape"],
                   goal_checker_plugins=["general_goal_checker", "escape_goal_checker"])
    control["progress_checker"].update(required_movement_radius=0.06, movement_time_allowance=8.0)
    control["general_goal_checker"].update(xy_goal_tolerance=0.10, yaw_goal_tolerance=0.35)
    control["escape_goal_checker"] = copy.deepcopy(control["general_goal_checker"])
    control["escape_goal_checker"]["xy_goal_tolerance"] = 0.02
    control["FollowPath"].update(
        desired_linear_vel=0.10, min_approach_linear_velocity=0.06,
        rotate_to_heading_angular_vel=0.8, max_angular_accel=32.0,
        use_collision_detection=True, max_allowed_time_to_collision_up_to_carrot=1.0,
    )
    control["Escape"] = copy.deepcopy(control["FollowPath"])
    control["Escape"].update(desired_linear_vel=0.06, allow_reversing=True,
                             use_rotate_to_heading=False, min_approach_linear_velocity=0.06)
    local = source["local_costmap"]["local_costmap"]["ros__parameters"]
    local.update(use_sim_time=False, resolution=0.01, footprint_padding=0.005,
                 footprint="[[-0.10,-0.08],[-0.10,0.08],[0.10,0.08],[0.10,-0.08]]")
    local["inflation_layer"].update(inflation_radius=0.18, cost_scaling_factor=5.0)
    local["voxel_layer"]["scan"]["expected_update_rate"] = 0.5
    global_map = {
        "use_sim_time": False, "global_frame": "map", "robot_base_frame": "base_link",
        "resolution": 0.025, "rolling_window": False, "track_unknown_space": True,
        "footprint": local["footprint"], "footprint_padding": 0.005,
        "update_frequency": 3.0, "publish_frequency": 1.0,
        "plugins": ["static_layer", "inflation_layer"],
        "static_layer": {"plugin": "nav2_costmap_2d::StaticLayer",
                         "map_topic": "/candidate_nav/planning_map",
                         "map_subscribe_transient_local": True, "subscribe_to_updates": False},
        "inflation_layer": {"plugin": "nav2_costmap_2d::InflationLayer",
                            "inflation_radius": 0.18, "cost_scaling_factor": 5.0},
        "always_send_full_costmap": True,
    }
    planner = {
        "use_sim_time": False, "expected_planner_frequency": 1.0,
        "planner_plugins": ["GridBased"],
        "GridBased": {"plugin": "nav2_navfn_planner::NavfnPlanner", "tolerance": 0.10,
                      "use_astar": True, "allow_unknown": False},
    }
    path.write_text(yaml.safe_dump({
        NS + "/controller_server": {"ros__parameters": control},
        NS + "/local_costmap/local_costmap": {"ros__parameters": local},
        NS + "/planner_server": {"ros__parameters": planner},
        NS + "/global_costmap/global_costmap": {"ros__parameters": global_map},
    }))


if __name__ == "__main__":
    write_config(Path(sys.argv[1]))
