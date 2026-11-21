"""Physical strategy 3: Qwen chooses bounded direction primitives."""
from . import artifact_writer as archive

import asyncio
import json
import math
import time
from types import SimpleNamespace

from geometry_msgs.msg import Twist

from .action_plan import translation_finished_at_brake
from .agent import request_model
from .direct_control import LIVE_ACTION_SPEC as ACTION_SPEC
from .geometry import fuse_observations
from .live_mapping import MappedAgent
from .live_waypoint import NavRelay


class DirectLiveAgent(MappedAgent):
    action_spec = ACTION_SPEC

    def __init__(self, args):
        super().__init__(args)
        self.stop_pub.close()
        self.stop_pub = NavRelay(self)
        policy_file = self.output / 'policy_inputs.json'
        policy = json.loads(policy_file.read_text())
        policy.update(strategy=3, control_mode='direct', candidate_mode='none',
                      selector='qwen_strategy_3', nav2_execution=False,
                      action_spec=self.action_spec, completion_mode='task_completion',
                      model_images=['front_rgb', 'online_slam_without_candidates'])
        policy_file.write_text(json.dumps(policy, indent=2))

    def build_reach(self, grid, pose, dynamic, transform):
        return SimpleNamespace(grid=fuse_observations(grid, dynamic, transform)
                               if dynamic is not None else grid)

    def snapshot(self, generation=False):
        observation, _, created = super().snapshot(generation)
        observation.update(physical=True, completion_mode='task_completion', action_spec=self.action_spec)
        saved = {k: v for k, v in observation.items() if k not in {'camera', 'map_image', 'accumulated_map_image'}}
        archive.text(self,self.output / 'observations' / f"{observation['snapshot_id']:04d}" / 'observation.json',
            json.dumps(saved, indent=2, ensure_ascii=False))
        return observation, [], created

    async def model_planner(self):
        errors = 0
        while not self.finished.is_set():
            await self.need_plan.wait()
            if (self.reach is None or self.pose is None or not self.fresh()
                    or not self.pending.empty() or getattr(self, 'executing_sequence', False)):
                await asyncio.sleep(0.1)
                continue
            self.need_plan.clear()
            self.state = 'qwen_thinking'
            try:
                observation, _, created = self.snapshot()
                reply = await request_model(observation, 18766, socket_path=self.args.model_socket)
                metadata = reply.get('metadata', {})
                if metadata.get('backend') != 'native_async_llm' or metadata.get('tp_size') not in (1, 2):
                    raise ValueError('Real TP=2 Qwen inference metadata is required')
                self.decisions += 1
                self.latencies.append(reply['latency_s'])
                self.latest_decision = {key: reply.get(key) for key in
                    ('goal_id', 'action', 'actions', 'observer', 'reasoner', 'latency_s', 'completion_probabilities')}
                self.log('decisions', {**reply, 'sim_time': self.now()})
                errors = 0
                if self.args.validate_only:
                    self.log('events', {'event': 'inference_verified', 'strategy': self.args.strategy,
                                        'action': reply['action'], 'motor_publisher': self.stop_pub.output is not None})
                    self.finish('inference_verified')
                    return
                if not self.fresh() or time.monotonic() - created > 90:
                    self.need_plan.set()
                    continue
                if reply['task_complete']:
                    self.finish('task_complete')
                    return
                await self.queue_decision(reply, observation, created)
            except Exception as error:
                errors += 1
                self.log('events', {'event': 'model_error', 'error': repr(error)})
                if errors >= 3:
                    self.finish('model_errors')
                else:
                    await asyncio.sleep(1)
                    self.need_plan.set()

    async def queue_decision(self, reply, observation, created):
        await self.pending.put((reply['action'], observation['pose']))

    async def execute_action(self, action, target_pose=None):
        odom_control=getattr(self,'execution_frame','map')=='odom'
        read_pose=self.get_control_pose if odom_control else self.get_pose
        map_start=self.get_pose() if odom_control else None
        pose = read_pose()
        if pose is None or not self.fresh():
            return {'action': action, 'outcome': 'stale_sensors'}
        spec = self.action_spec[action]
        start = time.monotonic()
        self.executed += 1
        self.stop_pub.blocked = None
        self.stop_pub.last_decision = None
        if hasattr(self.stop_pub, 'collision_check'):
            self.stop_pub.collision_check = None
        self.stop_pub.armed = True
        self.action_label = action
        self.state = 'action_' + action
        outcome = 'duration_complete'
        try:
            while not self.finished.is_set() and time.monotonic() - start < spec['max_sim_s']:
                if hasattr(self, 'collect_motion_frame'):
                    self.collect_motion_frame()
                current = read_pose()
                if current is None or not self.fresh():
                    outcome = 'stale_sensors'
                    break
                if 'max_distance_m' in spec and target_pose is not None:
                    along = (target_pose[0]-current[0])*math.cos(target_pose[2]) + (target_pose[1]-current[1])*math.sin(target_pose[2])
                    if along * (1 if spec['linear_m_s'] > 0 else -1) <= 0.008:
                        break
                if 'max_distance_m' in spec and target_pose is None and math.dist(pose[:2], current[:2]) >= spec['max_distance_m']:
                    break
                if 'angle_rad' in spec:
                    left = spec['angular_rad_s'] > 0
                    delta = current[2] - pose[2]
                    turned = math.atan2(math.sin(delta), math.cos(delta))
                    signed = turned if left else -turned
                    remaining = math.atan2(math.sin(target_pose[2]-current[2]), math.cos(target_pose[2]-current[2])) if target_pose is not None else None
                    reached_target = remaining is not None and (remaining if left else -remaining) <= 0.025
                    if reached_target or (target_pose is None and signed >= spec['angle_rad'] - 0.025):
                        break
                command = Twist()
                command.linear.x = spec.get('linear_m_s', 0.0) / 2.0
                command.angular.z = spec.get('angular_rad_s', 0.0) / 1.8
                if 'angle_rad' in spec and target_pose is not None:
                    # ICP feedback and motor response are delayed. Approach the
                    # absolute target slowly rather than braking at full speed.
                    direction = 1 if left else -1
                    speed = min(abs(command.angular.z), max(0.06, 1.2 * direction * remaining))
                    command.angular.z = direction * speed
                self.stop_pub.last_command = time.monotonic()
                self.stop_pub.publish(command)
                if self.stop_pub.last_decision not in {None, 'allowed'}:
                    outcome = self.stop_pub.last_decision
                    break
                await asyncio.sleep(0.04)
            if self.finished.is_set():
                outcome = 'run_finished'
        except asyncio.CancelledError:
            outcome = 'interrupted'
            raise
        except Exception:
            outcome = 'execution_error'
            raise
        finally:
            handoff = (getattr(self,'continuous_handoff',False) and outcome=='duration_complete'
                       and not self.finished.is_set())
            if not handoff:
                self.stop_pub.armed = False
                self.stop_pub.stop()
                self.action_label = 'stop'
            if not handoff:
                if getattr(self,'adaptive_settle',False):await self.wait_motion_settled()
                else:await asyncio.sleep(0.35)
        end = read_pose()
        record = {'action': action, 'outcome': outcome, 'duration_s': time.monotonic()-start,
                  'start_pose': list(pose), 'end_pose': list(end) if end is not None else None}
        if outcome == 'trajectory_collision':
            check = getattr(self.stop_pub, 'collision_check', None)
            if check is not None:
                record['clearance'] = check.public()
                if (odom_control and check.reason == 'swept_margin'
                        and not self.finished.is_set() and self.fresh() and self.control_link_ok()
                        and translation_finished_at_brake(pose, end, target_pose, spec)):
                    # publish() has already stopped the wheels and we have settled.
                    # The following primitive still goes through its own fresh checks.
                    outcome = 'duration_complete'
                    record.update(outcome=outcome, stopped_at_target=True)
        if odom_control:
            map_end=self.get_pose()
            record.update(control_frame='odom',control_start_pose=list(pose),
                          control_end_pose=list(end) if end is not None else None,
                          start_pose=list(map_start) if map_start is not None else None,
                          end_pose=list(map_end) if map_end is not None else None)
            if end is not None:
                dx,dy=end[0]-pose[0],end[1]-pose[1]
                record.update(distance_m=math.hypot(dx,dy),
                    signed_distance_m=dx*math.cos(pose[2])+dy*math.sin(pose[2]),
                    lateral_distance_m=-dx*math.sin(pose[2])+dy*math.cos(pose[2]))
        if odom_control and outcome=='duration_complete' and 'max_distance_m' in spec:
            sign=1 if spec['linear_m_s']>0 else -1
            if end is None or record.get('signed_distance_m',0)*sign<spec['max_distance_m']*.6:
                outcome='insufficient_progress_or_deviation';record['outcome']=outcome
        self.completed+=int(outcome=='duration_complete')
        totals = getattr(self, 'execution_totals', None)
        if totals is not None and end is not None:
            delta = math.atan2(math.sin(end[2]-pose[2]), math.cos(end[2]-pose[2]))
            totals['actions'] += 1
            totals['signed_turn_rad'] += delta
            totals['absolute_turn_rad'] += abs(delta)
            if odom_control and action in {'forward','creep_forward','backward','creep_backward'}:
                signed=record['signed_distance_m']
                totals['forward_m']+=max(0.,signed)
                totals['backward_m']=totals.get('backward_m',0)+max(0.,-signed)
            elif action in {'forward','creep_forward'}:
                totals['forward_m']+=math.dist(pose[:2],end[:2])
            elif action in {'backward','creep_backward'}:
                totals['backward_m']=totals.get('backward_m',0)+math.dist(pose[:2],end[:2])
        self.action_history.append(record)
        self.log('events', {'event': 'command_finished', **record})
        return record

    async def nav2_executor(self):
        while not self.finished.is_set():
            action, source = await self.pending.get()
            pose = self.get_pose()
            delta = pose[2] - source[2] if pose is not None else math.inf
            turn = abs(math.atan2(math.sin(delta), math.cos(delta))) if math.isfinite(delta) else math.inf
            if pose is None or math.dist(pose[:2], source[:2]) > 0.12 or turn > 0.12:
                self.rejected += 1
                self.need_plan.set()
                continue
            try:
                await self.execute_action(action)
            finally:
                self.need_plan.set()
