"""Strategy 4: one model request produces a checked sequence of direction actions."""
from . import artifact_writer as archive

import json
import asyncio
import time
import math
from rclpy.time import Time
from .agent import request_model, image64, yaw
from .mission_workers import MissionWorkers
from .motion_vision import MotionFrames, MotionObserver
from .local_odometry import LocalOdometry
from .committed_plan import recent_stalls, recent_turn_overshoots
from .live_trajectory import check_strategy4_translation
from .agent import msg_time
from collections import deque

from .action_plan import ACTION_PLAN_SPEC, PROTOCOL, trajectory_steps
from .action_plan_executor import ActionPlanExecutor
from .streaming_perception import StreamingPerception
from .prepared_perception import prepared_pose_matches, state_usable
from . import target_vision
from .temporal_vision import task_key
from .live_direct import DirectLiveAgent
from .maneuver_memory import maneuver_memory, model_outcome, heading_options


class ActionPlanLiveAgent(StreamingPerception, MotionObserver, ActionPlanExecutor, DirectLiveAgent):
    action_spec = ACTION_PLAN_SPEC
    execution_frame = "odom"

    def __init__(self, args):
        self.trajectory_steps = trajectory_steps(args.trajectory_steps)
        self.local_odometry=LocalOdometry()
        self.active_control_epoch=None
        super().__init__(args)
        self.init_streaming_perception()
        self.mission_workers = MissionWorkers(self) if args.sub_strategy == 2 else None
        self.mission_pending = None
        self.plan_rationale_enabled = not (args.no_plan_rationale or self.mission_workers)
        self.artifact_writer=archive.ArtifactWriter()
        self.adaptive_settle=True
        self.escape_heading = None
        self.last_policy_camera = None
        self.motion_frames = MotionFrames()
        self.vision_wakeup = asyncio.Event()
        self.vision_idle = asyncio.Event()
        self.vision_idle.set()
        self.visual_memory = deque(maxlen=8)
        (self.output / 'motion_frames').mkdir(exist_ok=True)
        self.last_plan_outcome = None
        self.closed_loop_targets = True
        self.execution_totals = dict(frame="odom_body_projection", forward_m=0.0, signed_turn_rad=0.0, absolute_turn_rad=0.0, actions=0)
        self.plan_history = deque(maxlen=8)
        self.executing_sequence = False
        self.trajectory_progress = None
        file = self.output / 'policy_inputs.json'
        policy = json.loads(file.read_text())
        model_coroutines = ['motion_camera_observer', 'current_camera_observer',
                            'spatial_assessment', 'committed_plan_selector']
        if self.mission_workers:
            model_coroutines.extend(['mission_memory', 'mission_subgoal'])
        if self.plan_rationale_enabled:
            model_coroutines.append('plan_rationale')
        policy.update(strategy=4, control_mode='action_trajectory', selector='qwen_strategy_4',
                      trajectory_steps=self.trajectory_steps,
                      inference_motion='streaming perception during committed trajectory; measured-boundary action selection',
                      turn_degrees=22.5,
                      available_turn_degrees=[11.25,22.5,45.0],
                      collision_protection='disabled' if getattr(args,'disable_collision_protection',False) else 'enabled',
                      long_term_memory=not getattr(args,'no_long_term_memory',False),
                      memory_scope='rover-specific; coordinates require current run and tracking epoch',
                      execution_feedback_frame='odom',planning_map_frame='map',
                      sub_strategy=args.sub_strategy or 1,
                      plan_rationale_enabled=self.plan_rationale_enabled,
                      model_coroutines=model_coroutines)
        file.write_text(json.dumps(policy, indent=2))

    def select_snapshot_camera(self, current, pose):
        self.prepared_for_snapshot = None
        completed = getattr(self, 'motion_last_result', None)
        if self.executing_sequence or completed is None:
            return current
        frame, image, reply, digest = completed
        state = reply.get('visual_state', {})
        if (frame['frame_id'] != self.motion_frames.serial
                or state.get('task_key') != task_key(self.task['instruction'])
                or not state_usable(state, split=target_vision.is_split_state(state))
                or not prepared_pose_matches(frame, pose, self.now())):
            return current
        self.prepared_for_snapshot = dict(task_key=state['task_key'],
            camera_stamp=frame['stamp'], camera_sha256=digest,
            visual_state=state, evidence=reply['evidence'])
        # Reuse only the exact analysed endpoint image while the rover is at rest.
        return image, frame['stamp'], frame['pose']

    def receive(self,name,msg):
        super().receive(name,msg)
        if name=='odom':
            q=msg.pose.pose.orientation;v=msg.pose.pose.position
            valid=(all(math.isfinite(x) for x in (q.x,q.y,q.z,q.w))
                   and sum(x*x for x in (q.x,q.y,q.z,q.w))>.5
                   and all(math.isfinite(msg.pose.covariance[i]) and 0<=msg.pose.covariance[i]<1000 for i in (0,7,35)))
            self.local_odometry.observe(msg_time(msg),[v.x,v.y,yaw(q)],valid,
                msg.header.frame_id,msg.child_frame_id,self.memory_scope.session)

    def get_control_pose(self):
        sample=self.local_odometry.snapshot(self.now())
        if not sample['valid'] or (self.active_control_epoch is not None and sample['epoch']!=self.active_control_epoch):return None
        return sample['pose']

    def fresh(self):
        return super().fresh() and self.get_control_pose() is not None

    async def execute_plan(self, reply, observation, created):
        start_scope=self.memory_scope.export()
        self.active_control_epoch=(observation.get("control_reference") or {}).get("epoch")
        try:
            return await super().execute_plan(reply, observation, created)
        finally:
            self.active_control_epoch=None
            end_scope=self.memory_scope.export()
            if self.last_plan_outcome is not None:
                verified=(start_scope.get('tracking_valid') and end_scope.get('tracking_valid')
                          and start_scope['anchor_epoch']==end_scope['anchor_epoch']
                          and start_scope['map_session_id']==end_scope['map_session_id'])
                self.last_plan_outcome['memory_scope']={**end_scope,'tracking_valid':bool(verified)}

    def memory_packet_fields(self):
        scope=self.memory_scope.export()
        commitment=self.escape_heading
        if commitment and (not scope.get('tracking_valid')
                or commitment.get('anchor_epoch')!=scope.get('anchor_epoch')):
            if self.escape_heading is commitment:self.escape_heading=None
        return dict(physical=True,rover_id=getattr(self.args,'rover_id',80),
                    episode_id=self.output.name,memory_context=scope)

    def snapshot(self, generation=False):
        observation, _, created = super().snapshot(generation)
        self.memory_scope.observe_map_transform(observation.get('map_from_odom'))
        observation.update(self.memory_packet_fields(), visual_tracking=target_vision.MODE)
        observation['control_reference']=self.local_odometry.snapshot(self.now())
        observation['task_context']['odometry_path_m']=round(self.local_odometry.path_m,3)
        observation['memory_update'] = {'previous_plan': self.last_plan_outcome}
        if not self.plan_rationale_enabled:
            observation['no_plan_rationale'] = True
        else:
            observation['response_mode'] = 'plan_stream_v1'
        observation.update(protocol=PROTOCOL, trajectory_steps=self.trajectory_steps,
                           previous_plan_outcome=model_outcome(self.last_plan_outcome),
                           maneuver_memory=maneuver_memory(self.plan_history, observation['pose']),
                           heading_options=heading_options(self._snapshot_grid, observation['pose'], self.action_spec))
        observation['collision_protection'] = 'disabled' if getattr(self.args,'disable_collision_protection',False) else 'enabled'
        if observation['collision_protection'] == 'enabled':
            scan, origin = self.scan, self.scan_origin
            observation['first_motion_checks'] = {
                action: check_strategy4_translation(scan, origin, spec['linear_m_s'], 0, action).public()
                for action, spec in self.action_spec.items() if 'max_distance_m' in spec
            }
            observation['recent_stalled_actions'] = recent_stalls(
                self.plan_history, observation['control_reference'], observation['snapshot_id']
            )
            observation['recent_turn_overshoots'] = recent_turn_overshoots(
                self.plan_history, observation['control_reference'], observation['snapshot_id']
            )
        if self.mission_workers is None:
            observation['escape_heading_commitment'] = self.escape_heading
        try:
            tf = self.buffer.lookup_transform('map','base_link',Time(nanoseconds=int(observation['camera_stamp']*1e9))).transform
            observation['camera_pose'] = [tf.translation.x,tf.translation.y,yaw(tf.rotation)]
            observation['camera_pose_paired'] = True
        except Exception:
            observation['camera_pose'] = list(observation['pose'])
            observation['camera_pose_paired'] = False
        if self.last_policy_camera is not None:
            observation['previous_camera'] = self.last_policy_camera
        self.last_policy_camera = dict(image=observation['camera'], stamp=observation['camera_stamp'],
                                       pose=list(observation['camera_pose']), snapshot_id=observation['snapshot_id'])
        observation['temporal_visual_state'] = getattr(self,'motion_visual_state',{})
        observation['target_reference'] = getattr(self, 'target_reference', None)
        observation["motion_visual_memory"] = list(self.visual_memory)
        observation["task_context"]["execution_totals"] = dict(self.execution_totals)
        if self.prepared_for_snapshot is not None:
            observation['prepared_perception'] = self.prepared_for_snapshot
            observation['camera_evidence_age_s'] = round(self.now()-observation['camera_stamp'],3)
        saved = {k: v for k, v in observation.items() if k not in {'camera', 'map_image', 'accumulated_map_image'}}
        if saved.get('target_reference') is not None:
            saved['target_reference'] = {k: v for k, v in saved['target_reference'].items() if k != 'image'}
        if 'previous_camera' in saved:
            saved['previous_camera'] = {k:v for k,v in saved['previous_camera'].items() if k!='image'}
        archive.text(self,self.output / 'observations' / f"{observation['snapshot_id']:04d}" / 'observation.json',
            json.dumps(saved, indent=2, ensure_ascii=False))
        return observation, [], created

    def collect_motion_frame(self, force=False):
        if self.camera is None or not self.executing_sequence:
            return
        if (self.trajectory_progress or {}).get('actions') == ['stop']:
            return  # No executed motion to analyse; the next measured snapshot covers waiting.
        # A small lag lets TF catch up with camera timestamps during turns.
        candidates = [entry for entry in reversed(self.camera_buffer)
                      if .08 <= self.now()-entry[1] < .35]
        if not candidates:
            return
        image, stamp, _ = candidates[0]
        try:
            tf = self.buffer.lookup_transform("map", "base_link", Time(nanoseconds=int(stamp*1e9))).transform
            pose = [tf.translation.x, tf.translation.y, yaw(tf.rotation)]
        except Exception:
            return  # Do not attach a historical image to an unrelated current pose.
        plan_id = (self.trajectory_progress or {}).get('plan_id')
        before = self.motion_frames.dropped
        meta = self.motion_frames.add(image, stamp, pose, plan_id, force)
        if meta is None:
            return
        meta['memory_scope'] = self.memory_scope.export()
        meta['execution'] = dict(action=self.action_label,
            progress=dict(self.trajectory_progress or {}),
            measured_totals=dict(self.execution_totals),
            sensors=self.sensor_observation({}))
        self.log('events', dict(event='motion_frame', **meta, dropped=self.motion_frames.dropped-before))
        self.vision_idle.clear()
        self.vision_wakeup.set()

    async def request_motion_observation(self, packet):
        async with self.model_rpc_lock:
            return await request_model(packet,18766,timeout=120,socket_path=self.args.model_socket)

    async def queue_decision(self, reply, observation, created):
        visual = reply.get('target_visual_state') or {}
        if target_vision.is_split_state(visual) and visual['task_key'] == task_key(observation['instruction']):
            frame = target_vision.snapshot_frame(observation)
            last = (getattr(self, 'motion_visual_state', {}) or {}).get('latest_frame')
            if not last or last['stamp'] < frame['stamp']:
                self.motion_visual_state = visual
                self.target_reference = target_vision.remember_reference(
                    getattr(self, 'target_reference', None), visual, {frame['frame_id']: observation['camera']})
        plan=reply.get('selected_plan') or {}
        if plan.get('nominal_end_pose'):
            self.escape_heading=dict(map_yaw_degrees=math.degrees(plan['nominal_end_pose'][2]),
                source='committed_model_plan',anchor_epoch=(observation.get('memory_context') or {}).get('anchor_epoch'))
        if self.mission_workers:
            self.mission_workers.submit(observation, reply, self.mission_pending)
        await self.pending.put((reply, observation, created))

    async def run(self):
        tasks = [asyncio.create_task(self.motion_observer(), name='motion_visual_observer')]
        if self.mission_workers:
            for protocol in self.mission_workers.jobs:
                tasks.append(asyncio.create_task(self.mission_workers.run_worker(protocol), name=protocol))
        try:
            await super().run()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            from .model_transport import close_model_connections
            await close_model_connections()
            # Save any unprocessed frames after motors/control tasks have stopped.
            pending = self.motion_frames.take(128)
            if pending:
                await asyncio.to_thread(self.prepare_motion_frames, pending)
            await asyncio.to_thread(self.artifact_writer.close)
        file = self.output / 'agent_summary.json'
        summary = json.loads(file.read_text())
        summary.update(path_m=self.local_odometry.path_m,odometry_frame='odom',strategy=4, control_mode='action_trajectory',
                       sub_strategy=self.args.sub_strategy or 1,
                       trajectory_steps=self.trajectory_steps,
                       perception_frames_collected=self.motion_frames.serial,
                       perception_frames_processed=self.motion_processed_count,
                       perception_frames_dropped=self.motion_frames.dropped)
        file.write_text(json.dumps(summary, indent=2))
