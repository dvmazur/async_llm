"""Sequence scheduling independent of ROS, camera and motor transport."""

import asyncio
import math
import time

from .action_plan import pose_matches, rollout, validate_response


class ActionPlanExecutor:
    """The agent supplies fresh poses, sensor checks and bounded execute_action."""

    async def execute_plan(self, reply, observation, created):
        validate_response(reply, observation)
        actions = reply['actions']
        odom_control=getattr(self,'execution_frame','map')=='odom'
        read_pose=self.get_control_pose if odom_control else self.get_pose
        reference=observation.get('control_reference') or {}
        origin=reference.get('pose') if odom_control else observation['pose']
        expected = rollout(origin, actions, self.action_spec) if origin is not None else []
        self.active_route = [list(p[:2]) for p in rollout(observation['pose'],actions,self.action_spec)]
        plan_id = observation['snapshot_id']
        start_pose = self.get_pose()
        start_pose = list(start_pose) if start_pose is not None else None
        start_control=read_pose()
        executed = 0
        completed = 0
        step_results = []
        outcome = 'completed'
        self.log('events', {'event': 'trajectory_started', 'plan_id': plan_id,
                            'actions': actions, 'nominal_poses': expected})
        try:
            if odom_control and (not reference.get('valid') or origin is None or reference.get('epoch')!=self.local_odometry.epoch):
                outcome='control_odometry_unavailable_or_reset'
            elif time.monotonic() - created > 90:
                outcome = 'stale_decision'
            else:
                for index, action in enumerate(actions):
                    self.trajectory_progress = {'plan_id': plan_id, 'step': index + 1,
                                                'total': len(actions), 'actions': actions}
                    if self.finished.is_set():
                        outcome = 'run_finished'
                        break
                    if not self.fresh() or not self.control_link_ok():
                        outcome = 'stale_sensors_or_control_link'
                        break
                    measured_pose=read_pose()
                    if measured_pose is None:
                        outcome = 'pose_unavailable'
                        break
                    if not pose_matches(measured_pose, expected[index]):
                        outcome = 'trajectory_deviation'
                        break
                    if getattr(self, 'closed_loop_targets', False) and 'max_distance_m' in self.action_spec[action]:
                        # Rebase translation to the measured start while retaining committed yaw.
                        actual = read_pose()
                        dx,dy=actual[0]-expected[index][0],actual[1]-expected[index][1]
                        expected[index:] = [(x+dx,y+dy,a) for x,y,a in expected[index:]]
                    if hasattr(self, 'collect_motion_frame'):
                        self.collect_motion_frame(force=True)
                    self.continuous_handoff = False
                    if getattr(self,'enable_continuous_steps',False) and index+1<len(actions):
                        next_spec=self.action_spec[actions[index+1]]
                        self.continuous_handoff = ('max_distance_m' in self.action_spec[action]
                            and 'max_distance_m' in next_spec
                            and self.action_spec[action]['linear_m_s']==next_spec['linear_m_s'])
                    if getattr(self, 'closed_loop_targets', False):
                        record = await self.execute_action(action, target_pose=expected[index + 1])
                    else:
                        record = await self.execute_action(action)
                    if hasattr(self, 'collect_motion_frame'):
                        self.collect_motion_frame(force=True)
                    executed += 1
                    angle_limit = 0.20
                    nominal_angle = self.action_spec[action].get('angle_rad')
                    if nominal_angle is not None and nominal_angle < angle_limit:
                        # An 11.25-degree command must not pass as a zero turn
                        # merely because the legacy tolerance is 11.46 degrees.
                        angle_limit = nominal_angle/2
                    measured_pose = read_pose()
                    target = expected[index + 1]
                    if record['outcome'] == 'duration_complete' and not pose_matches(
                            measured_pose, target, angle_limit):
                        record = {**record, 'outcome': 'insufficient_progress_or_deviation'}
                        if measured_pose is not None:
                            angle_error = measured_pose[2] - target[2]
                            record['deviation'] = {
                                'distance_m': round(math.dist(measured_pose[:2], target[:2]), 4),
                                'heading_error_rad': round(math.atan2(math.sin(angle_error),
                                                                     math.cos(angle_error)), 4),
                                'heading_limit_rad': round(angle_limit, 4),
                            }
                    measured = {key: record[key] for key in ('action', 'outcome', 'clearance', 'deviation', 'stopped_at_target')
                                if key in record}
                    start, end = ((record.get('control_start_pose'),record.get('control_end_pose')) if odom_control else (record.get('start_pose'),record.get('end_pose')))
                    if start is not None and end is not None:
                        delta = end[2] - start[2]
                        measured.update(distance_m=round(math.dist(start[:2], end[:2]), 4),
                                        turn_rad=round(math.atan2(math.sin(delta), math.cos(delta)), 4),
                                        start_pose=record.get('start_pose'), end_pose=record.get('end_pose'))
                        if odom_control:
                            signed=(end[0]-start[0])*math.cos(start[2])+(end[1]-start[1])*math.sin(start[2])
                            measured.update(control_frame='odom',control_start_pose=list(start),control_end_pose=list(end),
                                            signed_distance_m=round(signed,4))
                    step_results.append(measured)
                    if ('max_distance_m' in self.action_spec[action] and record['outcome']=='duration_complete'
                            and start is not None and end is not None
                            and ((measured.get('signed_distance_m',0)*(1 if self.action_spec[action]['linear_m_s']>0 else -1)) if odom_control else math.dist(start[:2],end[:2])) < self.action_spec[action]['max_distance_m']*.6):
                        measured['outcome'] = 'insufficient_progress_or_deviation'
                        record = {**record,'outcome':'insufficient_progress_or_deviation'}
                    self.log('events', {'event': 'trajectory_step', 'plan_id': plan_id,
                                        'step': index, **record})
                    if record['outcome'] != 'duration_complete':
                        outcome = record['outcome']
                        break
                    completed += 1
                    if hasattr(self,'after_trajectory_step'):
                        self.after_trajectory_step(plan_id,actions[index+1:],expected[-1])
                    if action == 'stop':
                        outcome = 'observation_requested'
                        break
        except asyncio.CancelledError:
            outcome = 'interrupted'
            raise
        except Exception:
            outcome = 'execution_error'
            raise
        finally:
            self.continuous_handoff = False
            self.stop_pub.armed = False
            self.stop_pub.stop()
            self.active_route = []
            self.trajectory_progress = None
            self.action_label = 'stop'
            end_pose = self.get_pose()
            self.last_plan_outcome = {'plan_id': plan_id, 'outcome': outcome,
                                      'planned_actions': list(actions), 'start_pose': start_pose,
                                      'end_pose': list(end_pose) if end_pose is not None else None,
                                      'executed_actions': executed,
                                      'completed_actions': completed, 'step_results': step_results,
                                      'discarded_actions': actions[executed:]}
            if odom_control:
                self.last_plan_outcome.update(control_frame='odom', control_start_pose=start_control,
                                              control_end_pose=read_pose(), control_epoch=reference.get('epoch'))
            if hasattr(self, 'plan_history'):
                self.plan_history.append(self.last_plan_outcome)
            self.log('events', {'event': 'trajectory_finished', **self.last_plan_outcome})

    async def nav2_executor(self):
        while not self.finished.is_set():
            reply, observation, created = await self.pending.get()
            self.executing_sequence = True
            try:
                await self.execute_plan(reply, observation, created)
            finally:
                promoted = False
                if hasattr(self,'finish_trajectory_planning'):
                    promoted = await self.finish_trajectory_planning()
                elif hasattr(self, 'flush_motion_vision') and not self.finished.is_set():
                    await self.flush_motion_vision()
                self.executing_sequence = False
                self.state = 'ready'
                if not promoted:
                    self.need_plan.set()
