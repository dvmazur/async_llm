"""Perception runs during committed motion; actions are chosen only at a boundary."""
import asyncio
import time

from .action_plan import ACTION_LABELS, validate_response
from .committed_plan import first_action_labels
from .model_transport import request_model


class StreamingPerception:
    def init_streaming_perception(self):
        self.model_rpc_lock = asyncio.Lock()
        self.plan_request_pending = False
        self.enable_continuous_steps = True
        self.continuous_handoff = False
        self.visual_revision = 0
        self.visual_revision_stamp = 0
        self.vision_progress = asyncio.Event()
        self.motion_processed_frame_id = 0
        self.motion_failed_frame_id = 0
        self.motion_processed_count = 0
        self.perception_status = {'state': 'idle', 'processed_frames': 0}
        self.prepared_for_snapshot = None

    async def wait_motion_settled(self):
        # Fresh ICP odometry, rather than an unconditional delay. Keep the old
        # maximum; a small rest window avoids sampling a still-moving camera.
        import math
        start=time.monotonic();anchor=None;last_stamp=None;updates=0
        while time.monotonic()-start<.35:
            await asyncio.sleep(.025)
            sample=self.local_odometry.snapshot(self.now())
            if not sample['valid']:
                anchor=None;updates=0;continue
            stamp=sample.get('stamp');pose=sample['pose']
            if stamp==last_stamp:continue
            last_stamp=stamp
            now=time.monotonic()
            if (anchor is None or sample.get('epoch')!=anchor[2]
                    or math.dist(pose[:2],anchor[1][:2])>=.001
                    or abs(math.atan2(math.sin(pose[2]-anchor[1][2]),math.cos(pose[2]-anchor[1][2])))>=.004):
                anchor=(now,pose,sample.get('epoch'));updates=0
            else:
                updates+=1
                if updates>=2 and now-anchor[0]>=.10 and now-start>=.15:break

    def after_trajectory_step(self, plan_id, remaining, anchor):
        # Wake perception without generating, replacing or cancelling any action.
        self.vision_wakeup.set()

    async def finish_trajectory_planning(self):
        """Wait only for the small unprocessed tail, never enqueue a forecast plan."""
        if self.finished.is_set():
            return False
        target = self.motion_frames.serial
        if target <= self.motion_processed_frame_id:
            return False
        self.state = 'finishing_motion_observations'
        self.perception_status.update(state='finishing_tail', boundary_frame_id=target)
        self.vision_wakeup.set()
        started = time.monotonic()
        async def drain():
            while self.motion_processed_frame_id < target:
                self.vision_progress.clear()
                if self.motion_failed_frame_id >= target or self.finished.is_set():
                    break
                await self.vision_progress.wait()
        try:
            await asyncio.wait_for(drain(), 8)
        except asyncio.TimeoutError:
            self.log('events', dict(event='motion_tail_timeout', frame_id=target))
        self.log('events', dict(event='motion_tail_finished', frame_id=target,
                               processed_frame_id=self.motion_processed_frame_id,
                               wait_s=time.monotonic()-started))
        return False

    def receive_plan_comment(self, reply):
        self.log('events', dict(event='plan_comment_ready', snapshot_id=reply['snapshot_id'],
            reasoner=reply.get('reasoner'),rationale_consistency=reply.get('rationale_consistency'),
            analysis_completed_latency_s=reply.get('analysis_completed_latency_s')))
        current=getattr(self,'latest_decision',None)
        if current is not None and current.get('snapshot_id')==reply['snapshot_id']:
            for key in ('reasoner','spatial_rationale','rationale_consistency','rationale_pending'):
                current[key]=reply.get(key)

    async def infer_plan(self, observation):
        # An already running image batch may finish; subsequent batches wait.
        self.plan_request_pending = True
        try:
            async with self.model_rpc_lock:
                reply = await request_model(observation, 18766, socket_path=self.args.model_socket, on_update=self.receive_plan_comment)
        finally:
            self.plan_request_pending = False
        validate_response(reply, observation)
        if not first_action_labels([ACTION_LABELS[reply['action']]], observation):
            raise ValueError('Model returned a blocked first action for the measured pose')
        if observation.get('collision_protection') != 'disabled' and any(
                action in observation.get('recent_turn_overshoots', []) for action in reply['actions']):
            raise ValueError('Model returned a temporarily excluded small turn')
        if observation.get('no_plan_rationale') and reply.get('rationale_enabled') is not False:
            raise ValueError('Qwen does not support --no-plan-rationale. Start with the updated model service.')
        if observation.get('mission_context') and not reply.get('mission_context'):
            raise ValueError('Qwen does not support --sub-strategy 2. Start with the updated model service.')
        metadata = reply.get('metadata', {})
        if metadata.get('backend') != 'native_async_llm' or metadata.get('tp_size') not in (1, 2):
            raise ValueError('Native Qwen metadata is required')
        self.decisions += 1
        self.latencies.append(reply['latency_s'])
        self.log('decisions', {**reply, 'sim_time': self.now(), 'speculative': False})
        return reply

    async def model_planner(self):
        errors = 0
        while not self.finished.is_set():
            await self.need_plan.wait()
            if (self.executing_sequence or not self.pending.empty() or self.reach is None
                    or self.get_pose() is None or not self.fresh() or not self.control_link_ok()):
                await asyncio.sleep(.08)
                continue
            self.need_plan.clear()
            self.state = 'qwen_thinking'
            try:
                started=time.monotonic()
                observation, _, created = await asyncio.to_thread(self.snapshot)
                snapshot_s=time.monotonic()-started
                workers = getattr(self, 'mission_workers', None)
                if workers:
                    self.mission_pending = workers.decorate(observation)
                    self.log('events', dict(event='mission_context', snapshot_id=observation['snapshot_id'],
                                            **observation['mission_context']))
                rpc_started=time.monotonic()
                reply = await self.infer_plan(observation)
                self.log('events',dict(event='control_plan_timing',snapshot_id=observation['snapshot_id'],snapshot_s=snapshot_s,rpc_s=time.monotonic()-rpc_started,total_s=time.monotonic()-started,transport=reply.get('transport_timing')))
                errors = 0
                self.latest_decision = {key: reply.get(key) for key in
                    ('snapshot_id','rationale_pending','goal_id', 'action', 'actions', 'observer', 'reasoner', 'latency_s',
                     'completion_probabilities', 'observer_reused','selected_plan','spatial_rationale',
                     'reasoner_role','rationale_enabled','rationale_consistency','mission_context')}
                if self.args.validate_only:
                    self.log('events', dict(event='inference_verified', strategy=4,
                                           action=reply['action'], motor_publisher=self.stop_pub.output is not None))
                    self.finish('inference_verified')
                    return
                if self.finished.is_set():
                    return
                if self.get_pose() is None or not self.fresh() or time.monotonic()-created > 90:
                    self.need_plan.set()
                    continue
                if reply['task_complete']:
                    self.finish('task_complete')
                    return
                await self.queue_decision(reply, observation, created)
            except Exception as error:
                errors += 1
                self.log('events', dict(event='model_error', error=repr(error)))
                if getattr(error, 'code', None) == 'context_budget_exceeded':
                    self.finish('context_budget_exceeded')
                    return
                if errors >= 3:
                    self.finish('model_errors')
                else:
                    await asyncio.sleep(1)
                    self.need_plan.set()
