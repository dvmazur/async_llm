#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec tar -cf - \
    robot_sim/candidate_nav/{__init__,artifact_writer,wire_images,startup_health,sensor_startup,body_model,committed_plan,local_odometry,action_plan,action_plan_executor,agent,candidates,direct_agent,direct_control,geometry,grid_motion,io,live,live_action_plan,live_direct,live_link,live_mapping,live_nav_config,live_proposals,live_trajectory,live_view,live_waypoint,maneuver_memory,map_memory,memory_scope,mission_context,mission_workers,model_catalog,model_transport,motion_vision,parallel_planning,streaming_perception,prepared_perception,target_geometry,target_vision,temporal_vision,proposals,render,sensor_grounding,task_policy,trajectory_policy}.py \
    robot_sim/candidate_nav/live_view.html \
    robot_sim/mapping/core.py robot_sim/scripts/rover_live_entry.sh robot_sim/scripts/rover_live_nav_entry.sh robot_sim/scripts/rover_live_view_entry.sh robot_sim/scripts/rover_live_stop.py
