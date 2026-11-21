"""Physical strategy 1 with Qwen selection, continuous Nav2 replanning and escape."""

import argparse
import asyncio
import json
import math
from pathlib import Path
import signal
import time

from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav_msgs.msg import Path as PathMessage
import rclpy
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions

from .agent import request_model, yaw
from .geometry import fuse_observations, reachable
from .live import MotionGate, verify_backend
from .live_mapping import MappedAgent
from .live_nav_config import NS, PRIVATE_VELOCITY
from .live_trajectory import check_strategy4_translation, escape_course, trajectory_clearance
from .live_proposals import reach_status, validation_feedback
from .proposals import validate_proposals
from rclpy.time import Time


class NavRelay(MotionGate):
    def __init__(self, agent):
        self.armed = False
        self.last_command = -math.inf
        self.blocked = None
        self.valid_command_count = 0
        super().__init__(agent, agent.args.enable_motion)
        agent.create_subscription(Twist, PRIVATE_VELOCITY, self.receive, qos_profile_sensor_data)

    def rejection(self, linear, angular):
        self.collision_check = None
        if not self.agent.control_link_ok():
            return 'control_link_lost'
        if not self.armed or self.agent.finished.is_set():
            return 'not_armed'
        if self.competing_control():
            return 'other_velocity_publisher'
        if not self.agent.fresh() or self.agent.get_pose() is None:
            return 'stale_sensors_or_tf'
        if getattr(self.agent.args,'strategy',None) == 4 and getattr(self.agent.args,'disable_collision_protection',False):
            return None
        # Strategy 4 operator policy: lidar does not veto a bounded in-place
        # turn. Link, arming, competing-control and sensor checks above still apply.
        if (getattr(self.agent.args, 'strategy', None) == 4
                and linear == 0.0 and 0 < abs(angular) <= 0.8 / 1.8):
            return None
        if getattr(self.agent.args, 'strategy', None) == 4:
            self.collision_check = check_strategy4_translation(
                self.agent.scan, self.agent.scan_origin, linear * 2.0,
                angular * 1.8, self.agent.action_label,
            )
            return 'trajectory_collision' if self.collision_check.gap_m is None else None
        gap = trajectory_clearance(self.agent.scan, self.agent.scan_origin,
                                   linear * 2.0, angular * 1.8, 0.55)
        return 'trajectory_collision' if gap is None else None

    def receive(self, message):
        if not self.armed:
            return
        publishers = self.agent.get_publishers_info_by_topic(PRIVATE_VELOCITY)
        if len(publishers) != 1 or (
            publishers[0].node_name, publishers[0].node_namespace
        ) != ('controller_server', NS):
            self.blocked = 'unexpected_controller'
            self.stop()
            return
        v, w = message.linear.x, message.angular.z
        if not math.isfinite(v) or not math.isfinite(w):
            self.blocked = 'nonfinite_velocity'
            self.stop()
            return
        if abs(v) < 1e-6 and abs(w) < 1e-6:
            self.stop()
            return
        if not -0.061 <= v <= 0.101 or abs(w) > 0.801:
            self.blocked = 'velocity_bounds'
            self.stop()
            return
        if abs(v) < 0.05 and 0.1 < abs(w) < 0.799:
            self.blocked = 'hardware_turn_deadband'
            self.stop()
            return
        if abs(w) < 0.1:
            w = 0.0
        command = Twist()
        command.linear.x, command.angular.z = v / 2.0, w / 1.8
        reason = self.rejection(command.linear.x, command.angular.z)
        if reason is not None:
            self.blocked = reason
            self.stop()
            return
        self.last_command = time.monotonic()
        self.valid_command_count += 1
        if self.valid_command_count == 1 or self.agent.action_label == 'stop':
            self.agent.log('events', {'event': 'checked_nav2_velocity', 'linear_m_s': v,
                                      'angular_rad_s': w, 'motor_output': self.output is not None})
        self.publish(command)
        if self.moving:
            self.agent.commands += 1
            self.agent.action_label = 'reverse' if v < 0 else 'forward' if v else 'turn'

    def watchdog(self):
        if self.moving and (
            not self.armed or not self.agent.control_link_ok()
            or time.monotonic() - self.last_command > 0.20
            or not self.agent.fresh() or self.competing_control() or self.agent.get_pose() is None
        ):
            self.blocked = 'command_or_sensor_timeout'
            self.stop()


class WaypointAgent(MappedAgent):
    def __init__(self, args):
        super().__init__(args)
        self.stop_pub.close()
        self.stop_pub = NavRelay(self)
        self.controller = ActionClient(self, FollowPath, NS + '/follow_path')
        self.planner = ActionClient(self, ComputePathToPose, NS + '/compute_path_to_pose')
        self.cooldown = []
        self.recovery_count = 0
        self.replan_count = 0
        self.empty_recoveries = []
        self.last_generation_feedback = None
        policy_file = self.output / 'policy_inputs.json'
        policy = json.loads(policy_file.read_text())
        policy.pop('action_spec', None)
        policy.update(
            control_mode='waypoint', candidate_mode=args.candidate_mode, selector=f'qwen_strategy_{args.strategy}',
            strategy=args.strategy,
            nav2_execution=True, nav2_collision_detection=True, custom_sector_stop=False,
            duration_s=args.wall_limit, goal_hold_s=45, replan_interval_s=1.5,
            continuous_path_updates=True,
            planning_clearance_m=0.05, footprint_m=[0.20, 0.16],
            nav2_footprint_padding_m=0.005, map_source='fresh_lidar_icp',
            model_images=['front_rgb', 'online_slam_with_candidates'],
            model_coroutines=['camera_observer', 'spatial_reasoner', 'goal_selector'],
            coroutines=['ros_observer', 'proposal_builder', 'model_planner', 'nav2_executor', 'recorder'],
            completion_mode='task_completion', motion_enabled=args.enable_motion,
        )
        policy_file.write_text(json.dumps(policy, indent=2))

    def build_reach(self, grid, pose, dynamic, transform):
        planning = fuse_observations(grid, dynamic, transform) if dynamic is not None else grid
        reach = reachable(planning, pose, radius=0.145, max_distance=8)
        x, y = reach.start
        if planning.contains((x, y)) and not reach.allowed[y, x]:
            # Match Nav2's inscribed footprint when starting inside the preferred margin.
            # The controller and swept rectangle still validate the full chassis.
            return reachable(planning, pose, radius=0.085, max_distance=8)
        return reach

    def navigation_status(self):
        status = reach_status(self.reach)
        if status is not None:
            feedback = getattr(self, 'last_generation_feedback', None)
            status['generation'] = ({k: feedback[k] for k in
                                     ('raw_count', 'accepted_count', 'rejection_counts')}
                                    if feedback is not None else None)
        return status

    def snapshot(self, generation=False):
        observation, candidates, stamp = super().snapshot(generation)
        now = time.monotonic()
        self.cooldown = [(xy, until) for xy, until in self.cooldown if until > now]
        candidates = [c for c in candidates if not any(
            math.dist((c.x, c.y), xy) < 0.40 for xy, _ in self.cooldown
        )]
        observation['candidates'] = [item for item in observation['candidates']
                                     if any(item['id'] == c.id for c in candidates)]
        observation['physical'] = True
        observation['completion_mode'] = 'task_completion'
        observation['navigation_recovery'] = {
            'replans': self.replan_count, 'escapes': self.recovery_count,
            'blocked_viewpoints': len(self.cooldown),
        }
        if generation:
            observation['proposal_feedback'] = self.last_generation_feedback
        saved = {k: v for k, v in observation.items() if k not in {'camera', 'map_image', 'accumulated_map_image'}}
        folder = 'generations' if generation else 'observations'
        (self.output / folder / f"{observation['snapshot_id']:04d}" / 'observation.json').write_text(
            json.dumps(saved, indent=2, ensure_ascii=False))
        return observation, candidates, stamp

    async def generate_candidates(self):
        self.state = 'qwen_generating_candidates'
        observation, _, _ = self.snapshot(generation=True)
        response = await request_model(observation, 18766, socket_path=self.args.model_socket)
        self.generation_latencies.append(response['latency_s'])
        transform = self.buffer.lookup_transform('map', 'odom', Time()).transform
        current = (transform.translation.x, transform.translation.y, yaw(transform.rotation))
        previous = observation['map_from_odom']
        delta = current[2] - previous[2]
        valid_frame = (math.dist(current[:2], previous[:2]) <= 0.25
                       and abs(math.atan2(math.sin(delta), math.cos(delta))) <= 0.15)
        if valid_frame:
            candidates, checks = validate_proposals(response['proposals'], self.reach,
                observation['generation_map']['projection'], self.pose, self.generator)
        else:
            candidates, checks = [], [{'accepted': False, 'reason': 'map_frame_changed'}]
        self.candidates = candidates
        self.raw_proposals += len(response['proposals'])
        self.accepted_proposals += len(candidates)
        self.last_generation_id = observation['snapshot_id']
        self.last_generation_feedback = validation_feedback(observation, response['proposals'], checks)
        folder = self.output / 'generations' / f"{observation['snapshot_id']:04d}"
        (folder / 'response.json').write_text(json.dumps(response, indent=2))
        (folder / 'validation.json').write_text(json.dumps(checks, indent=2))
        self.log('events', {'event': 'llm_candidates_generated', 'raw': len(response['proposals']),
                            'accepted': len(candidates), 'sim_time': self.now(),
                            'rejection_counts': self.last_generation_feedback['rejection_counts']})

    async def model_planner(self):
        errors = 0
        while not self.finished.is_set():
            await self.need_plan.wait()
            if self.active is not None or not self.pending.empty() or self.reach is None or not self.fresh():
                await asyncio.sleep(0.15)
                continue
            self.need_plan.clear()
            if self.args.strategy == 2:
                try:
                    # An all-gray reachable map offers no valid coordinate to
                    # generate. Still ask the model about task completion below
                    # (e.g. a wait task), while mapping continues in the background.
                    if reach_status(self.reach)['start_state'] == 'ready':
                        await self.generate_candidates()
                    else:
                        self.candidates = []
                except Exception as error:
                    errors += 1
                    self.log('events', {'event': 'generation_error', 'error': repr(error)})
                    if errors >= 3:
                        self.finish('model_errors')
                    else:
                        await asyncio.sleep(1)
                        self.need_plan.set()
                    continue
            observation, candidates, created = self.snapshot()
            if not candidates and self.args.strategy == 1 and not self.args.validate_only:
                now = time.monotonic()
                self.empty_recoveries = [t for t in self.empty_recoveries if now - t < 30]
                if len(self.empty_recoveries) < 2:
                    self.empty_recoveries.append(now)
                    await self.escape()
                    await asyncio.sleep(1.2)
                    self.need_plan.set()
                    continue
            self.state = 'qwen_thinking'
            try:
                response = await request_model(observation, 18766, socket_path=self.args.model_socket)
                metadata = response.get('metadata', {})
                if metadata.get('backend') != 'native_async_llm' or metadata.get('tp_size') not in (1, 2):
                    raise ValueError('Real TP=2 Qwen inference metadata is required')
                errors = 0
                self.decisions += 1
                self.latencies.append(response['latency_s'])
                self.latest_decision = {k: response.get(k) for k in
                                        ('goal_id', 'observer', 'reasoner', 'latency_s', 'completion_probabilities')}
                self.log('decisions', {**response, 'sim_time': self.now(),
                                       'choices': [c.public() for c in candidates],
                                       'snapshot_wall_age': time.monotonic() - created})
                if self.args.validate_only:
                    candidate = next((c for c in candidates if c.id == response['goal_id']), None)
                    route = await self.plan(candidate) if candidate is not None else []
                    course = escape_course(self.scan, self.scan_origin)
                    probe = await self.escape() if course is not None else None
                    if course is not None and (not probe or self.stop_pub.valid_command_count == 0):
                        self.finish('nav2_validation_failed')
                        raise RuntimeError('Native escape did not produce a validated private command')
                    self.log('events', {'event': 'inference_verified', 'route_points': len(route),
                                        'escape_course': course, 'private_escape_probe': probe,
                                        'candidate_count': len(candidates),
                                        'checked_commands': self.stop_pub.valid_command_count,
                                        'motor_publisher': self.stop_pub.output is not None})
                    self.finish('inference_verified')
                    return
                if (not self.fresh() or time.monotonic() - created > 90
                        or math.dist(self.pose[:2], observation['pose'][:2]) > 0.20):
                    self.need_plan.set()
                    continue
                if response['goal_id'] == 'Z':
                    self.finish('task_complete')
                    return
                selected = next((c for c in candidates if c.id == response['goal_id']), None)
                if selected is None:
                    self.state = ('waiting_for_map_clearance'
                                  if reach_status(self.reach)['start_state'] != 'ready'
                                  else 'waiting_for_new_evidence')
                    await asyncio.sleep(1)
                    self.need_plan.set()
                else:
                    await self.pending.put(selected)
            except Exception as error:
                errors += 1
                self.log('events', {'event': 'model_error', 'error': repr(error)})
                if errors >= 3:
                    self.finish('model_errors')
                else:
                    await asyncio.sleep(1)
                    self.need_plan.set()

    async def plan(self, candidate):
        if not self.planner.server_is_ready():
            return []
        goal = ComputePathToPose.Goal()
        goal.goal.header.frame_id = 'map'
        goal.goal.header.stamp = self.get_clock().now().to_msg()
        goal.goal.pose.position.x, goal.goal.pose.position.y = candidate.x, candidate.y
        goal.goal.pose.orientation.z = math.sin(candidate.yaw / 2)
        goal.goal.pose.orientation.w = math.cos(candidate.yaw / 2)
        goal.planner_id = 'GridBased'
        handle = await self.ros_future(self.planner.send_goal_async(goal), 2)
        if not handle.accepted:
            return []
        result_future = handle.get_result_async()
        try:
            result = await self.ros_future(result_future, 3)
            if result.status != 4:
                return []
            route = [(p.pose.position.x, p.pose.position.y) for p in result.result.path.poses]
            return [self.pose[:2], *route] if len(route) == 1 else route
        finally:
            if not result_future.done():
                handle.cancel_goal_async()

    def make_path(self, route, heading):
        path = PathMessage()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in route:
            point = PoseStamped(header=path.header)
            point.pose.position.x, point.pose.position.y = float(x), float(y)
            point.pose.orientation.z, point.pose.orientation.w = math.sin(heading / 2), math.cos(heading / 2)
            path.poses.append(point)
        return path

    async def drive_path(self, route, heading, seconds, escape=False, candidate=None):
        def goal_for(points):
            goal = FollowPath.Goal()
            goal.path = self.make_path(points, heading)
            goal.controller_id = 'Escape' if escape else 'FollowPath'
            goal.goal_checker_id = 'escape_goal_checker' if escape else 'general_goal_checker'
            goal.progress_checker_id = 'progress_checker'
            return goal

        async def submit(points):
            future = self.controller.send_goal_async(goal_for(points))
            try:
                return await self.ros_future(future, 2)
            except BaseException:
                # A delayed acceptance must not leave an untracked goal executing.
                def cancel_late_goal(done):
                    if not done.cancelled() and done.exception() is None:
                        accepted = done.result()
                        if accepted.accepted:
                            accepted.cancel_goal_async()
                future.add_done_callback(cancel_late_goal)
                raise

        handle = await submit(route)
        if not handle.accepted:
            return 'rejected'
        self.goal_handle = handle
        self.active_route = route
        self.stop_pub.blocked = None
        self.stop_pub.armed = True
        result_future = handle.get_result_async()
        started = time.monotonic()
        next_plan = started + 1.5
        last_progress = started
        previous = self.get_pose()
        planning = None
        try:
            while time.monotonic() - started < seconds and not self.finished.is_set():
                now = time.monotonic()
                if result_future.done():
                    result = result_future.result()
                    if result.status == 4:
                        return 'reached'
                    self.log('events', {'event': 'nav2_blocked', 'code': result.result.error_code})
                    return 'blocked'
                if self.stop_pub.blocked:
                    self.log('events', {'event': 'trajectory_blocked', 'reason': self.stop_pub.blocked})
                    return 'blocked'
                if candidate is not None:
                    pose = self.get_pose()
                    if pose is None or not self.fresh():
                        return 'stale_sensors'
                    if previous is not None:
                        delta = pose[2] - previous[2]
                        turned = abs(math.atan2(math.sin(delta), math.cos(delta))) > 0.12
                        if math.dist(pose[:2], previous[:2]) > 0.04 or turned:
                            last_progress, previous = now, pose
                    if now - last_progress > 4:
                        return 'blocked'
                    if planning is None and now >= next_plan:
                        planning = asyncio.create_task(self.plan(candidate))
                        next_plan = now + 1.5
                    if planning is not None and planning.done():
                        try:
                            updated = planning.result()
                        except TimeoutError:
                            updated = []
                        planning = None
                        self.replan_count += 1
                        if len(updated) < 2:
                            return 'blocked'
                        # Nav2 accepts the pending path in its running control loop.
                        # Keep the relay armed and avoid cancelling the current goal.
                        replacement = await submit(updated)
                        if not replacement.accepted:
                            return 'blocked'
                        handle = replacement
                        result_future = handle.get_result_async()
                        self.goal_handle = handle
                        self.active_route = updated
                        self.log('events', {'event': 'path_updated', 'kind': 'nav2',
                                            'points': updated, 'candidate': candidate.public(),
                                            'sim_time': self.now()})
                await asyncio.sleep(0.03)
            return 'goal_budget' if candidate is not None else 'replan'
        finally:
            self.stop_pub.armed = False
            self.stop_pub.stop()
            self.action_label = 'stop'
            self.active_route = []
            if planning is not None:
                planning.cancel()
                await asyncio.gather(planning, return_exceptions=True)
            if not result_future.done():
                try:
                    await self.ros_future(handle.cancel_goal_async(), 1.5)
                    await self.ros_future(result_future, 1.5)
                except asyncio.CancelledError:
                    handle.cancel_goal_async()
            self.goal_handle = None

    async def escape(self):
        if not self.fresh() or self.get_pose() is None:
            return False
        course = escape_course(self.scan, self.scan_origin)
        if course is None:
            self.log('events', {'event': 'escape_unavailable'})
            return False
        speed, angular = course
        pose = self.get_pose()
        route = []
        for index in range(7):
            stamp = index / 3.0
            if angular:
                x = speed / angular * math.sin(angular * stamp)
                y = speed / angular * (1 - math.cos(angular * stamp))
            else:
                x, y = speed * stamp, 0.0
            c, sn = math.cos(pose[2]), math.sin(pose[2])
            route.append((pose[0] + c * x - sn * y, pose[1] + sn * x + c * y))
        self.recovery_count += 1
        self.state = 'escaping_obstacle'
        self.log('events', {'event': 'escape_started', 'speed': speed,
                            'angular_rad_s': angular, 'distance_m': speed * 2})
        result = await self.drive_path(route, pose[2] + angular * 2, seconds=2.5, escape=True)
        self.log('events', {'event': 'escape_finished', 'outcome': result})
        return result in {'reached', 'replan'}

    async def follow_target(self, candidate):
        deadline = time.monotonic() + 45
        failures = 0
        while time.monotonic() < deadline and not self.finished.is_set():
            self.state = 'replanning_path'
            try:
                route = await self.plan(candidate)
            except TimeoutError:
                route = []
            self.replan_count += 1
            if self.get_pose() is None or not self.fresh():
                return 'stale_sensors'
            if len(route) >= 2:
                self.log('events', {'event': 'route', 'kind': 'nav2', 'points': route,
                                    'candidate': candidate.public(), 'sim_time': self.now()})
                self.state = f'driving_{candidate.id}'
                outcome = await self.drive_path(route, candidate.yaw,
                                                seconds=max(0.1, deadline - time.monotonic()),
                                                candidate=candidate)
                if outcome in {'reached', 'goal_budget', 'stale_sensors'}:
                    return outcome
            failures += 1
            if failures > 2:
                return 'blocked_viewpoint'
            if not await self.escape():
                return 'no_safe_escape'
        return 'goal_budget'

    async def nav2_executor(self):
        while not self.finished.is_set():
            candidate = await self.pending.get()
            self.active = candidate
            self.executed += 1
            outcome = 'run_finished'
            try:
                outcome = await self.follow_target(candidate)
            finally:
                self.stop_pub.armed = False
                self.stop_pub.stop()
                self.active = None
                self.generator.record(candidate, 'reached' if outcome == 'reached' else 'failed')
                self.completed += int(outcome == 'reached')
                self.failed += int(outcome not in {'reached', 'run_finished'})
                if outcome not in {'reached', 'run_finished'}:
                    self.cooldown.append(((candidate.x, candidate.y), time.monotonic() + 30))
                self.log('events', {'event': 'candidate_result', 'candidate': candidate.id,
                                    'outcome': outcome, 'sim_time': self.now()})
                self.need_plan.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-model',default=None)
    parser.add_argument('--model-socket', type=Path, required=True)
    parser.add_argument('--wall-limit', type=float, default=180)
    parser.add_argument('--strategy', type=int, choices=[1, 2, 3, 4], default=1)
    from .action_plan import trajectory_steps
    parser.add_argument('--trajectory-steps', type=trajectory_steps,
                        help='Strategy 4: maximum actions per request (1..32, default 5; stop ends early)')
    parser.add_argument('--rover-id', type=int, choices=[80, 107], default=80)
    parser.add_argument('--imu', action='store_true', help='Require verified /hardware/imu')
    parser.add_argument('--link-socket', type=Path)
    parser.add_argument('--link-token')
    parser.add_argument('--disable-collision-protection', action='store_true')
    parser.add_argument('--no-long-term-memory', action='store_true',help='Strategy 4: disable persistent memory for A/B experiments')
    parser.add_argument('--sub-strategy', type=int, choices=[1, 2], default=None)
    parser.add_argument('--no-plan-rationale', action='store_true',
                        help='Strategy 4: skip the post-plan explanation and its review')
    parser.add_argument('--enable-motion', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.no_long_term_memory and args.strategy != 4:
        parser.error('--no-long-term-memory requires strategy 4')
    if args.sub_strategy is not None and args.strategy != 4:
        parser.error('--sub-strategy requires strategy 4')
    if args.no_plan_rationale and args.strategy != 4:
        parser.error('--no-plan-rationale requires strategy 4')
    if args.disable_collision_protection and args.strategy != 4:
        parser.error('--disable-collision-protection requires strategy 4')
    if args.trajectory_steps is not None and args.strategy != 4:
        parser.error('--trajectory-steps requires --strategy 4')
    if args.strategy == 4 and args.trajectory_steps is None:
        args.trajectory_steps = 5
    if args.enable_motion and not (args.link_socket and args.link_token):
        parser.error('Motion requires a supervised control link')
    if args.validate_only and args.enable_motion:
        parser.error('Validation requires motion to be disabled')
    if not math.isfinite(args.wall_limit) or args.wall_limit <= 0:
        parser.error('Wall limit must be positive and finite')
    ready_deadline = time.monotonic() + 75
    while True:
        try:
            verify_backend(args.model_socket, False, args.expected_model)
            break
        except (OSError, ValueError):
            if time.monotonic() >= ready_deadline:
                raise
            time.sleep(1)
    args.output.mkdir()
    args.task = args.output / 'task.json'
    args.task.write_text(json.dumps({'instruction': args.instruction}, ensure_ascii=False))
    args.physical, args.no_imu, args.wall_clock = True, not args.imu, True
    args.candidate_mode = {1: 'geometric', 2: 'qwen', 3: 'none', 4: 'none'}[args.strategy]
    args.control_mode = 'direct' if args.strategy in {3, 4} else 'waypoint'
    args.test_backend, args.fresh_map = False, True
    args.front_clearance, args.sim_limit = None, args.wall_limit
    ros_args = ['--ros-args', '-r', '__node:=rover_live_qwen', '-r', '/odom:=/icp/odom']
    if args.imu:
        ros_args += ['-r', '/imu/data:=/hardware/imu']
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    if args.strategy == 4:
        from .live_action_plan import ActionPlanLiveAgent
        agent = ActionPlanLiveAgent(args)
    elif args.strategy == 3:
        from .live_direct import DirectLiveAgent
        agent = DirectLiveAgent(args)
    else:
        agent = WaypointAgent(args)
    agent.monitor_pose = True
    def interrupted(_signal, _frame):
        agent.finish('interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        asyncio.run(agent.run())
    finally:
        agent.stop_pub.armed = False
        agent.stop_pub.stop()
        agent.stream.closed.set()
        agent.destroy_node()
        rclpy.shutdown()
    descriptions = {
        'model_found': 'Модель сообщила о выполнении задания.',
        'task_complete': 'Модель сообщила о выполнении всего задания.',
        'wall_budget': 'Прогон завершён по заданному времени.',
        'simulation_budget': 'Прогон завершён по заданному времени.',
        'inference_verified': 'Проверка модели и датчиков без движения завершена.',
        'control_link_lost': 'Остановлено. Потеряна связь управления с Mac.',
        'interrupted': 'Прогон остановлен пользователем.',
        'model_errors': 'Остановлено из-за ошибок ответов модели.',
        'context_budget_exceeded': 'Входные данные превысили бюджет контекста модели; подробности в events.jsonl.',
        'coroutine_error': 'Остановлено из-за ошибки агента. Подробности в events.jsonl.',
    }
    print('\nРезультат — ' + descriptions.get(agent.reason, str(agent.reason))
          + f' [{agent.reason}]', flush=True)
    print(f'Решений Qwen {agent.decisions}, путь по одометрии {agent.path_m:.2f} м.', flush=True)
    print(f'Журналы внутри контейнера ros — {args.output}', flush=True)
    if args.validate_only and agent.reason != 'inference_verified':
        raise SystemExit(1)
    if agent.reason in {'coroutine_error', 'model_errors', 'context_budget_exceeded', 'control_link_lost', 'sensor_startup_timeout'}:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
