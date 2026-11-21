"""Nav2 execution only: online SLAM owns map -> odom, no AMCL/map_server."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = LaunchConfiguration("params")
    nodes = [
        ("nav2_controller", "controller_server"),
        ("nav2_planner", "planner_server"),
        ("nav2_behaviors", "behavior_server"),
        ("nav2_bt_navigator", "bt_navigator"),
    ]
    actions = [DeclareLaunchArgument("params")]
    for package, name in nodes:
        actions.append(
            Node(
                package=package,
                executable=name,
                name=name,
                parameters=[params, {"use_sim_time": True}],
                remappings=[("cmd_vel", "/cmd_vel_requested")],
                output="screen",
            )
        )
    actions.append(
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            parameters=[
                {
                    "use_sim_time": True,
                    "autostart": True,
                    "bond_timeout": 15.0,
                    "node_names": [name for _, name in nodes],
                }
            ],
            output="screen",
        )
    )
    return LaunchDescription(actions)
