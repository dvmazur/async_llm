"""One speculative continuation while physical execution and perception continue."""
import asyncio
import copy
import json
import math
import time
from .action_plan import validate_response, rollout, turn_delta
from .model_transport import request_model
from .maneuver_memory import heading_options


def remaining_seconds(actions, spec):
    total = 0.0
    for action in actions:
        item = spec[action]
        if 'max_distance_m' in item:
            total += item['max_distance_m'] / (abs(item['linear_m_s']) / 2) + .1
        elif 'angle_rad' in item:
            total += item['angle_rad'] / (abs(item['angular_rad_s']) / 1.8) + .35
        else:
            total += .35
    return total


def continuation_observation(observation, anchor, parent, remaining, spec, grid, measured_totals=None):
    result = copy.deepcopy(observation)
    measured = list(result['pose'])
    result['perception_pose'] = measured
    result['pose'] = list(anchor)
    result['pipeline'] = dict(speculative=True, parent_plan_id=parent,
                              measured_pose=measured, planned_start_pose=list(anchor),
                              committed_remaining_actions=list(remaining),
                              pending_actions_are_not_execution=True)
    projected=dict(measured_totals if measured_totals is not None else
                   result.get('task_context',{}).get('execution_totals',{}))
    for action in remaining:
        item=spec[action]
        if 'max_distance_m' in item:
            key='forward_m' if item['linear_m_s']>0 else 'backward_m'
            projected[key]=projected.get(key,0)+item['max_distance_m']
        elif 'angle_rad' in item:
            delta=turn_delta(action,spec)
            projected['signed_turn_rad']=projected.get('signed_turn_rad',0)+delta
            projected['absolute_turn_rad']=projected.get('absolute_turn_rad',0)+abs(delta)
    result['pipeline']['projected_execution_totals']=projected
    result['measured_task_context']=copy.deepcopy(result.get('task_context',{}))
    result['task_context']['current_pose']=list(anchor)
    result['task_context']['execution_totals']=projected
    result['task_context']['state_kind']='conditional_forecast_after_committed_actions'
    result['heading_options'] = heading_options(grid, anchor, spec)
    # Preserve measured facts separately from the explicitly conditional planning state.
    return result


def promotion_reason(reply, observation, actual, parent_outcome, *, age, fresh,
                     link_ok, new_evidence=False, geometry_ok=True):
    if parent_outcome.get('outcome') not in ('completed','observation_requested'):
        return 'parent_interrupted'
    if observation['pipeline']['parent_plan_id'] != parent_outcome.get('plan_id'):
        return 'parent_changed'
    if reply.get('task_complete'):
        return 'completion_needs_current_observation'
    if not fresh or not link_ok or actual is None:
        return 'pose_sensors_or_link_unavailable'
    if age > 25:
        return 'continuation_too_old'
    anchor=observation['pose']
    angle=math.atan2(math.sin(actual[2]-anchor[2]),math.cos(actual[2]-anchor[2]))
    if math.dist(actual[:2],anchor[:2]) > .08 or abs(angle) > .15:
        return 'predicted_start_mismatch'
    if new_evidence:
        return 'new_visual_evidence'
    if not geometry_ok:
        return 'changed_local_geometry'
    return None


def translation_path_free(grid, pose, actions, spec):
    """Recheck translated body volumes on the latest local map; no new turn veto."""
    import numpy as np
    from .live_trajectory import HALF_LENGTH, HALF_WIDTH, HARD_GAP
    spacing=grid.resolution/2
    xx,yy=np.meshgrid(np.linspace(-HALF_LENGTH-HARD_GAP,HALF_LENGTH+HARD_GAP,
                                 math.ceil(2*(HALF_LENGTH+HARD_GAP)/spacing)+1),
                      np.linspace(-HALF_WIDTH-HARD_GAP,HALF_WIDTH+HARD_GAP,
                                 math.ceil(2*(HALF_WIDTH+HARD_GAP)/spacing)+1))
    body=np.column_stack((xx.ravel(),yy.ravel()))
    gc,gs=math.cos(grid.yaw),math.sin(grid.yaw)
    def free(p):
        c,s=math.cos(p[2]),math.sin(p[2])
        points=body@np.array([[c,s],[-s,c]])+np.asarray(p[:2])
        dx=points[:,0]-grid.origin[0];dy=points[:,1]-grid.origin[1]
        gx=np.floor((gc*dx+gs*dy)/grid.resolution).astype(int)
        gy=np.floor((-gs*dx+gc*dy)/grid.resolution).astype(int)
        if ((gx<0)|(gy<0)|(gx>=grid.values.shape[1])|(gy>=grid.values.shape[0])).any():return False
        values=grid.values[gy,gx]
        return bool(((values>=0)&(values<=35)).all())
    poses=rollout(pose,actions,spec)
    for i,action in enumerate(actions):
        if 'max_distance_m' not in spec[action]:continue
        start,end=poses[i:i+2]
        for f in np.linspace(0,1,max(2,math.ceil(spec[action]['max_distance_m']/spacing)+1)):
            if not free((start[0]+f*(end[0]-start[0]),start[1]+f*(end[1]-start[1]),start[2])):return False
    return True


class ParallelPlanning:
    """Mixin: initial plans at rest, one bounded lookahead from a predicted boundary."""
    def init_parallel_planning(self):
        self.model_rpc_lock=asyncio.Lock()
        self.plan_request_pending=False
        self.lookahead_task=None
        self.lookahead_status={'state':'idle'}
        self.visual_revision=0
        self.visual_revision_stamp=0
        self.continuous_handoff=False
        self.enable_continuous_steps=True

    async def infer_plan(self, observation):
        self.plan_request_pending=True
        try:
            async with self.model_rpc_lock:
                reply=await request_model(observation,18766,socket_path=self.args.model_socket)
        finally:
            self.plan_request_pending=False
        validate_response(reply,observation)
        metadata=reply.get('metadata',{})
        if metadata.get('backend')!='native_async_llm' or metadata.get('tp_size') not in (1,2):
            raise ValueError('Native Qwen metadata is required')
        self.decisions+=1
        self.latencies.append(reply['latency_s'])
        self.log('decisions',{**reply,'sim_time':self.now(),
                              'speculative':bool(observation.get('pipeline',{}).get('speculative'))})
        return reply

    def display_plan(self, reply, *, prefetched=False):
        self.latest_decision={key:reply.get(key) for key in
            ('goal_id','action','actions','observer','reasoner','latency_s','completion_probabilities')}
        self.latest_decision['prepared_during_motion']=prefetched

    async def model_planner(self):
        errors=0
        while not self.finished.is_set():
            await self.need_plan.wait()
            if (self.executing_sequence or not self.pending.empty() or self.reach is None
                    or self.get_pose() is None or not self.fresh() or not self.control_link_ok()):
                await asyncio.sleep(.08)
                continue
            self.need_plan.clear()
            self.state='qwen_thinking'
            try:
                observation,_,created=await asyncio.to_thread(self.snapshot)
                reply=await self.infer_plan(observation)
                errors=0
                self.display_plan(reply)
                if self.args.validate_only:
                    self.log('events',dict(event='inference_verified',strategy=4,
                                           action=reply['action'],motor_publisher=self.stop_pub.output is not None))
                    self.finish('inference_verified');return
                if self.finished.is_set():return
                if self.get_pose() is None or not self.fresh() or time.monotonic()-created>90:
                    self.need_plan.set();continue
                if reply['task_complete']:
                    self.finish('task_complete');return
                await self.queue_decision(reply,observation,created)
            except Exception as error:
                errors+=1
                self.log('events',dict(event='model_error',error=repr(error)))
                if errors>=3:self.finish('model_errors')
                else:
                    await asyncio.sleep(1)
                    self.need_plan.set()

    def after_trajectory_step(self, plan_id, remaining, anchor):
        if self.lookahead_task is not None or self.finished.is_set() or not remaining:return
        # Wait until the remaining driving time approaches model latency, allowing one image RPC.
        latency=max(list(self.latencies)[-4:] or [8.0])
        eta=remaining_seconds(remaining,self.action_spec)
        if eta>max(4.,latency+3.):return
        self.lookahead_status=dict(state='thinking',parent_plan_id=plan_id,remaining_s=round(eta,2))
        self.lookahead_task=asyncio.create_task(self.prepare_continuation(plan_id,list(remaining),list(anchor),dict(getattr(self,'execution_totals',{}))),name='next_trajectory')

    async def prepare_continuation(self, parent, remaining, anchor, measured_totals=None):
        try:
            captured,_,created=await asyncio.to_thread(self.snapshot)
            if self.finished.is_set():return None
            # Mark visual revision after capture; later meaningful changes invalidate this proposal.
            revision=self.visual_revision
            grid=self._snapshot_grid
            observation=await asyncio.to_thread(continuation_observation,captured,anchor,parent,remaining,self.action_spec,grid,measured_totals)
            if getattr(self,'output',None) is not None:
                public={k:v for k,v in observation.items() if k not in {'camera','map_image','accumulated_map_image'}}
                if 'previous_camera' in public:
                    public['previous_camera']={k:v for k,v in public['previous_camera'].items() if k!='image'}
                file=self.output/'observations'/f"{observation['snapshot_id']:04d}"/'continuation_input.json'
                await asyncio.to_thread(file.write_text,json.dumps(public,indent=2,ensure_ascii=False))
            self.log('events',dict(event='lookahead_started',parent_plan_id=parent,snapshot_id=observation['snapshot_id'],
                                   perception_pose=captured['pose'],planned_start_pose=anchor))
            reply=await self.infer_plan(observation)
            self.lookahead_status=dict(state='ready',parent_plan_id=parent,latency_s=reply['latency_s'])
            return reply,observation,created,revision
        except asyncio.CancelledError:raise
        except Exception as error:
            self.log('events',dict(event='lookahead_error',error=repr(error)))
            return None

    async def finish_trajectory_planning(self):
        task=self.lookahead_task
        if task is None:return False
        if self.last_plan_outcome['outcome'] not in ('completed','observation_requested') or self.finished.is_set():
            task.cancel();await asyncio.gather(task,return_exceptions=True)
            self.lookahead_task=None
            self.lookahead_status=dict(state='discarded',reason='parent_interrupted')
            return False
        if not task.done():self.state='waiting_for_continuation'
        result=await task
        self.lookahead_task=None
        if result is None:return False
        reply,observation,created,revision=result
        pose=self.get_pose()
        geometry_ok=(getattr(self.args,'disable_collision_protection',False) or
                     (pose is not None and translation_path_free(self.reach.grid,pose,reply['actions'],self.action_spec)))
        reason=promotion_reason(reply,observation,pose,self.last_plan_outcome,age=time.monotonic()-created,
                                fresh=self.fresh(),link_ok=self.control_link_ok(),
                                new_evidence=(self.visual_revision!=revision and
                                              getattr(self,'visual_revision_stamp',float('inf'))>observation.get('camera_stamp',0)),geometry_ok=geometry_ok)
        self.lookahead_status=dict(state='discarded' if reason else 'promoted',reason=reason)
        self.log('events',dict(event='lookahead_discarded' if reason else 'lookahead_promoted',
                               reason=reason,snapshot_id=observation['snapshot_id'],age_s=time.monotonic()-created))
        if reason:return False
        # Rebase a validated relative sequence onto the fresh measured boundary pose.
        promoted=dict(observation,pose=list(pose))
        self.display_plan(reply,prefetched=True)
        await self.queue_decision(reply,promoted,time.monotonic())
        return True
