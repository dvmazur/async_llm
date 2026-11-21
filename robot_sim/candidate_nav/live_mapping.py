"""Live camera, ICP odometry and fresh lidar mapping for waypoint navigation."""
from . import artifact_writer as archive

import asyncio
from collections import deque
import math
import json
import time

import numpy as np
from rclpy.time import Time
from std_msgs.msg import String

from .agent import Agent, grid_message, msg_time, yaw
from .geometry import Grid
from .live import LiveAgent
from .render import render_map
from .task_policy import MODE
from .body_model import body_description
from .map_memory import accumulated_view
from .memory_scope import MemoryScope
from robot_sim.mapping.core import LocalGrid, filter_scan


class MappedAgent(LiveAgent):
    def __init__(self, args):
        args.completion_mode = MODE
        super().__init__(args)
        policy_file = self.output / 'policy_inputs.json'
        policy = json.loads(policy_file.read_text())
        policy['map_inputs'] = ['recent_local_map', 'accumulated_slam_map_when_available']
        policy['rover_body'] = body_description()
        policy['rover_id'] = getattr(args, 'rover_id', 80)
        policy_file.write_text(json.dumps(policy, indent=2))
        self.memory_scope = MemoryScope(getattr(args,'rover_id',80),self.output.name,
                                        enabled=not getattr(args,'no_long_term_memory',False))
        self.raw_scans = deque(maxlen=20)
        self.local_grid = LocalGrid(resolution=0.025, size=8.0, ttl=3.0)
        self.fresh_grid = LocalGrid(resolution=0.025, size=8.0, ttl=8.0)
        self.fresh_grid_active = False
        self.original_slam = None
        self.slam_updates = 0
        self.grid_versions = 0
        self.grid_dir = self.output / "grid_history"
        self.grid_dir.mkdir()
        self.previous_pose = None
        self.task_start_pose = None
        self.task_started_wall = None
        self.monitor_pose = False
        self.commands = 0
        self.active_route = []
        self.latest_decision = None
        self.telemetry_pub = self.create_publisher(String, '/rover_live/telemetry', 1)

    def receive(self, name, msg):
        if name == 'odom':
            q=msg.pose.pose.orientation;v=msg.pose.pose.position
            valid=(all(math.isfinite(x) for x in (q.x,q.y,q.z,q.w))
                   and sum(x*x for x in (q.x,q.y,q.z,q.w))>.5
                   and all(math.isfinite(msg.pose.covariance[i])
                           and 0<=msg.pose.covariance[i]<1000 for i in (0,7,35)))
            self.memory_scope.observe(msg_time(msg),[v.x,v.y,yaw(q)],valid)
        if name == "scan":
            self.raw_scans.append(msg)
        if name == "map":
            self.slam_updates += 1
            self.original_slam = grid_message(msg, self.slam_updates)
            if self.fresh_grid_active:
                self.received["map"] = time.monotonic()
                return
        super().receive(name, msg)

    def get_pose(self):
        pose = super().get_pose()
        if pose is not None and self.task_start_pose is None:
            self.task_start_pose = pose
            self.task_started_wall = time.monotonic()
        if getattr(self,"execution_frame","map")=="odom" or not self.monitor_pose or pose is None or self.finished.is_set():
            return pose
        now = time.monotonic()
        previous = self.previous_pose
        self.previous_pose = (now, pose)
        if previous is not None:
            elapsed = now - previous[0]
            displacement = math.dist(pose[:2], previous[1][:2])
            delta = pose[2] - previous[1][2]
            rotation = abs(math.atan2(math.sin(delta), math.cos(delta)))
            if elapsed <= 1.25 and (displacement > 0.16 + 0.35 * elapsed or rotation > 0.40 + 1.2 * elapsed):
                self.log("events", {"event": "pose_jump", "distance_m": displacement,
                                    "yaw_rad": rotation, "seconds": elapsed, "sim_time": self.now()})
                self.finish("pose_jump")
                return None
        return pose

    async def dynamic_mapper(self):
        last_scan = -math.inf
        next_record = -math.inf
        while not self.finished.is_set():
            now = self.now()
            for scan in reversed(self.raw_scans):
                stamp = msg_time(scan)
                if stamp <= last_scan or now - stamp >= 0.9:
                    break
                if now - stamp < 0.18:
                    continue
                try:
                    tf = self.buffer.lookup_transform("odom", scan.header.frame_id, Time.from_msg(scan.header.stamp))
                except Exception:
                    continue
                last_scan = stamp
                ranges = filter_scan(scan.ranges)
                valid = ((ranges >= scan.range_min) & (ranges <= scan.range_max)) | np.isposinf(ranges)
                cap = min(4.0, scan.range_max)
                distances = np.minimum(ranges[valid], cap)
                hits = ranges[valid] < cap - 0.01
                angle = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
                angles = angle[valid] + yaw(tf.transform.rotation)
                origin = np.array([tf.transform.translation.x, tf.transform.translation.y])
                endpoints = origin + np.column_stack((distances * np.cos(angles), distances * np.sin(angles)))
                self.local_grid.recenter(origin)
                self.local_grid.update(origin, endpoints, hits, stamp)
                values = self.local_grid.occupancy(now)
                self.grid_versions += 1
                self.dynamic = Grid(values, self.local_grid.resolution, tuple(self.local_grid.origin),
                                    "odom", self.grid_versions, stamp)
                if self.args.fresh_map:
                    try:
                        map_tf = self.buffer.lookup_transform("map", "odom", Time()).transform
                    except Exception:
                        break
                    self.fresh_grid.recenter(origin)
                    self.fresh_grid.update(origin, endpoints, hits, stamp)
                    mx, my = self.fresh_grid.origin
                    heading = yaw(map_tf.rotation)
                    c, s = math.cos(heading), math.sin(heading)
                    map_origin = (map_tf.translation.x + c * mx - s * my,
                                  map_tf.translation.y + s * mx + c * my)
                    self.version += 1
                    self.map = Grid(self.fresh_grid.occupancy(now), self.fresh_grid.resolution,
                                    map_origin, "map", self.version, stamp, heading)
                    self.fresh_grid_active = True
                # Local lidar/TF can be ready before the first SLAM /map callback.
                # Only the combined debug archive needs both grids; mapping must
                # continue while the accumulated map is still unavailable.
                global_grid = self.original_slam
                if now >= next_record and self.map is not None and global_grid is not None:
                    next_record = now + 1.0
                    np.savez_compressed(
                        self.grid_dir / f"{self.grid_versions:04d}.npz",
                        stamp=stamp, slam=global_grid.values, slam_origin=global_grid.origin,
                        slam_resolution=global_grid.resolution, slam_version=global_grid.version,
                        planning=self.map.values, planning_origin=self.map.origin,
                        planning_resolution=self.map.resolution, planning_yaw=self.map.yaw,
                        dynamic=values, dynamic_origin=self.dynamic.origin,
                        dynamic_resolution=self.dynamic.resolution,
                    )
                break
            if self.dynamic is not None and now - self.dynamic.stamp > 1.0:
                self.dynamic = None
            await asyncio.sleep(0.28)

    async def proposal_builder(self):
        await asyncio.gather(Agent.proposal_builder(self), self.dynamic_mapper())

    def snapshot(self, generation=False):
        observation, candidates, created = super().snapshot(generation)
        observation.update(physical=True, completion_mode=MODE, rover_body=body_description())
        observation['rover_id'] = getattr(self.args, 'rover_id', 80)
        observation['navigation_status'] = self.navigation_status()
        observation['task_context'] = {
            'elapsed_s': round(time.monotonic() - self.task_started_wall, 3)
                         if self.task_started_wall is not None else 0.0,
            'initial_pose': [round(v, 3) for v in self.task_start_pose]
                            if self.task_start_pose is not None else None,
            'odometry_path_m': round(self.path_m, 3),
            'current_pose': [round(v, 3) for v in self.pose] if self.pose is not None else None,
            'recent_actions': list(self.action_history),
            'recent_waypoint_outcomes': list(self.generator.history[-12:]),
            'recent_trajectory': [[round(v, 3) for v in xy] for xy in self.path[-32:]],
            'previous_analysis': self.latest_decision,
        }
        folder = 'generations' if generation else 'observations'
        snapshot_dir = self.output / folder / f"{observation['snapshot_id']:04d}"
        image, global_meta = accumulated_view(self.original_slam, observation['pose'],
                                              list(self.path), self._snapshot_grid, self.now())
        local_grid = self._snapshot_grid
        from .render import projection
        observation['maps_context'] = {
            'local': {'source': 'recent_lidar_fused_planning', 'ttl_s': 8.0 if self.args.fresh_map else None,
                      'frame': local_grid.frame, 'stamp': local_grid.stamp,
                      'projection': observation.get('generation_map',{}).get('projection',projection(local_grid))},
            'accumulated': global_meta,
            'authority': 'historical layout for route context; recent local evidence for immediate clearance'}
        if image is not None:
            from .agent import image64
            observation['accumulated_map_image'] = image64(image)
            archive.image(self,image,snapshot_dir / 'accumulated_map.png')
            archive.grid(self,snapshot_dir / 'accumulated_grid.npz',
                               occupancy=self.original_slam.values, origin=self.original_slam.origin,
                               resolution=self.original_slam.resolution, yaw=self.original_slam.yaw,
                               stamp=self.original_slam.stamp, version=self.original_slam.version)
        saved = {k: v for k, v in observation.items() if k not in {'camera', 'map_image', 'accumulated_map_image'}}
        archive.text(self,self.output / folder / f"{observation['snapshot_id']:04d}" / 'observation.json',
            json.dumps(saved, indent=2, ensure_ascii=False))
        return observation, candidates, created

    def navigation_status(self):
        return None

    def publish_telemetry(self):
        self.telemetry_pub.publish(String(data=json.dumps({
            'stamp': self.now(), 'run': self.output.name, 'state': self.state,
            'rover_id': getattr(self.args, 'rover_id', 80),
            'finished': self.finished.is_set(), 'motion_enabled': self.args.enable_motion,
            'instruction': self.task['instruction'],
            'completion_mode': MODE, 'task_complete': self.reason == 'task_complete',
            'navigation_status': self.navigation_status(),
            'sensor_startup': self.sensor_readiness() if self.started_sim is None else None,
            'control_odometry': self.local_odometry.snapshot(self.now()) if hasattr(self,'local_odometry') else None,
            'strategy': getattr(self.args, 'strategy', 1),
            'collision_protection': 'disabled' if getattr(self.args,'disable_collision_protection',False) else 'enabled',
            'control_link_ok': self.control_link_ok(),
            'control_link_age_s': self.link_watch.age if self.link_watch else None,
            'pose': self.pose, 'path_m': self.local_odometry.path_m if hasattr(self,'local_odometry') else self.path_m, 'decisions': self.decisions,
            'action': self.action_label, 'map_version': self.version,
            'candidates': [c.public() for c in self.candidates],
            'active_goal': self.active.public() if self.active else None,
            'route': self.active_route, 'decision': self.latest_decision,
            'trajectory_progress': getattr(self, 'trajectory_progress', None),
            'visual_state': getattr(self, 'motion_visual_state', None),
            'lookahead': getattr(self, 'lookahead_status', None),
            'perception_pipeline': dict(getattr(self, 'perception_status', {})),
        }, allow_nan=False)))

    def finish(self, reason):
        super().finish(reason)
        self.publish_telemetry()

    async def recorder(self):
        def save_images(camera_image, grid, pose, candidates, trail, active, camera_name, map_name):
            camera_image.save(self.output / camera_name, quality=85)
            image, projection = render_map(grid, pose, candidates, trail, active)
            image.save(self.output / map_name)
            return projection

        index = 0
        while not self.finished.is_set():
            self.publish_telemetry()
            if self.pose is not None and self.map is not None and self.camera is not None:
                now = self.now()
                camera = f"camera_frames/{index:06d}.jpg"
                map_file = f"map_frames/policy_{index:06d}.png"
                grid = self.reach.grid if self.reach is not None else self.map
                pose, active = self.pose, self.active
                candidates = [item for item in self.candidates
                              if active is None or item.id != active.id]
                if active is not None:
                    candidates.append(active)
                record = {'sim_time': now, 'camera': camera, 'map': map_file,
                          'pose': list(pose), 'state': self.state, 'path_m': self.path_m,
                          'decisions': self.decisions, 'action': self.action_label,
                          'active_goal': active.public() if active else None}
                record['projection'] = await asyncio.to_thread(
                    save_images, self.camera[0], grid, pose, candidates, list(self.path), active,
                    camera, map_file)
                self.log('frames', record)
                index += 1
            await asyncio.sleep(0.45)
