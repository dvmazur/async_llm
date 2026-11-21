import asyncio
import copy
import time
import unittest
import ast
from pathlib import Path
from collections import deque
from types import SimpleNamespace

from PIL import Image

from robot_sim.candidate_nav.action_plan import ACTION_PLAN_SPEC, PROTOCOL
from robot_sim.candidate_nav.action_plan_executor import ActionPlanExecutor
from robot_sim.candidate_nav.motion_vision import MotionFrames, MotionObserver
from robot_sim.candidate_nav.prepared_perception import image_digest, observer_text, prepared_pose_matches
from robot_sim.candidate_nav.streaming_perception import StreamingPerception
from robot_sim.candidate_nav.temporal_vision import initial_state


class PhysicalCameraTests(unittest.TestCase):
    def method(self,file,name,namespace):
        source=Path(__file__).parents[1]/('candidate_nav/'+file)
        tree=ast.parse(source.read_text())
        method=next(m for node in tree.body if isinstance(node,ast.ClassDef)
                    for m in node.body if isinstance(m,ast.FunctionDef) and m.name==name)
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
        return namespace[name]

    def test_physical_frame_uses_the_same_base_frame_as_motion_localisation(self):
        frame=Image.new('RGB',(8,8),'red')
        calls=[]
        def lookup(world,base,stamp):
            calls.append((world,base))
            if base!='base_link':raise RuntimeError('Unknown physical base frame')
            return SimpleNamespace(transform=SimpleNamespace(
                translation=SimpleNamespace(x=1.,y=2.),rotation=None))
        agent=SimpleNamespace(camera=(frame,30.8,'front_rgb'),executing_sequence=True,
            camera_buffer=deque([(frame,30.8,'front_rgb')]),now=lambda:31.,
            buffer=SimpleNamespace(lookup_transform=lookup),trajectory_progress=dict(plan_id=1),
            motion_frames=MotionFrames(),action_label='forward',execution_totals={},
            memory_scope=SimpleNamespace(export=lambda:dict(anchor_epoch='capture-epoch')),
            sensor_observation=lambda ranges:{},log=lambda *args:None,
            vision_idle=asyncio.Event(),vision_wakeup=asyncio.Event())
        collect=self.method('live_action_plan.py','collect_motion_frame',
                            dict(Time=lambda **kw:kw,yaw=lambda q:.2))
        collect(agent,force=True)
        self.assertEqual(calls,[('map','base_link')])
        self.assertEqual(agent.motion_frames.serial,1)
        self.assertEqual(agent.motion_frames.pending[0][0]['pose'],[1.,2.,.2])
        self.assertEqual(agent.motion_frames.pending[0][0]['memory_scope']['anchor_epoch'],'capture-epoch')
        agent.trajectory_progress['actions']=['stop']
        collect(agent,force=True)
        self.assertEqual(calls,[('map','base_link')])
        self.assertEqual(agent.motion_frames.serial,1)

    def test_mjpeg_delivery_populates_the_history_used_by_motion_perception(self):
        # Execute the actual receiver method without importing ROS into CPU tests.
        source=Path(__file__).parents[1]/'candidate_nav/live.py'
        tree=ast.parse(source.read_text())
        method=next(m for node in tree.body if isinstance(node,ast.ClassDef)
                    for m in node.body if isinstance(m,ast.FunctionDef) and m.name=='update_camera')
        namespace={}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
        image=Image.new('RGB',(8,8),'red')
        agent=SimpleNamespace(stream=SimpleNamespace(error=None,latest=(image,12.,100.)),
                              received={},camera_buffer=deque(maxlen=20))
        namespace['update_camera'](agent)
        self.assertEqual(len(agent.camera_buffer),1)
        self.assertEqual(agent.camera_buffer[0],(image,12.,'front_rgb'))
        namespace['update_camera'](agent)
        self.assertEqual(len(agent.camera_buffer),1)
        agent.stream.latest=(image,13.,101.)
        namespace['update_camera'](agent)
        self.assertEqual([f[1] for f in agent.camera_buffer],[12.,13.])


class PreparedTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB',(16,16),'red')
        visual = initial_state('Find a box.')
        visual.update(relevant_visible=True, description='Box visible ahead.',
                      latest_frame=dict(frame_id=7,stamp=4,pose=[0,0,0]))
        self.o = dict(instruction='Find a box.',pose=[0,0,0],sim_time=5,camera_stamp=4,
                      prepared_perception=dict(task_key=visual['task_key'],visual_state=visual,
                          camera_stamp=4,camera_sha256=image_digest(self.image),evidence='Box appeared.'))
    def test_exact_analysed_frame_can_skip_duplicate_observer(self):
        self.assertIn('Box visible ahead',observer_text(self.o,self.image))
    def test_changed_image_task_time_pose_and_uncertainty_need_new_analysis(self):
        self.assertIsNone(observer_text(self.o,Image.new('RGB',(16,16),'blue')))
        for key,value in [('instruction','Find a chair.'),('camera_stamp',5),
                          ('sim_time',13),('pose',[.05,0,0]),('pose',[0,0,.12])]:
            o=copy.deepcopy(self.o);o[key]=value
            with self.subTest(key=key):self.assertIsNone(observer_text(o,self.image))
        o=copy.deepcopy(self.o);o['prepared_perception']['visual_state']['relevant_visible']=None
        self.assertIsNone(observer_text(o,self.image))
    def test_yaw_wrap_and_no_future_or_old_capture(self):
        frame=dict(stamp=4,pose=[0,0,3.14])
        self.assertTrue(prepared_pose_matches(frame,[0,0,-3.14],5))
        self.assertFalse(prepared_pose_matches(frame,[0,0,-3.14],3))
        self.assertFalse(prepared_pose_matches(frame,[0,0,-3.14],13))


class Fake(StreamingPerception, MotionObserver, ActionPlanExecutor):
    action_spec=ACTION_PLAN_SPEC
    def __init__(self):
        self.init_streaming_perception()
        self.finished=asyncio.Event();self.vision_wakeup=asyncio.Event();self.vision_idle=asyncio.Event()
        self.motion_frames=MotionFrames();self.visual_memory=deque(maxlen=8)
        self.task=dict(instruction='Move forward 40 centimeters.');self.version=1
        self.pose=[0,0,0];self.executing_sequence=True;self.events=[]
        self.started=asyncio.Event();self.release=asyncio.Event();self.observed=[]
        self.stop_pub=SimpleNamespace(armed=False,stop=lambda:None)
        self.trajectory_progress=None;self.active_route=[];self.action_label='stop'
    def log(self,_,item):self.events.append(item)
    def get_pose(self):return list(self.pose)
    def fresh(self):return True
    def control_link_ok(self):return True
    def collect_motion_frame(self,force=False):
        # A distinct camera sample after each measured step.
        if force:
            self.motion_frames.add(Image.new('RGB',(8,8),'red'),self.motion_frames.serial+1,self.pose,1,True)
            self.vision_wakeup.set()
    async def request_motion_observation(self,packet):
        self.started.set();await self.release.wait();self.observed.extend(f['frame_id'] for f in packet['frames'])
        state=initial_state(self.task['instruction']);frame=packet['frames'][-1]
        state.update(relevant_visible=False,latest_frame={k:frame[k] for k in ('frame_id','stamp','pose','plan_id')})
        return dict(evidence='nothing new',visual_state=state,events=[],latency_s=.01)
    async def execute_action(self,action):
        start=list(self.pose)
        await asyncio.sleep(.01)
        if action=='forward':self.pose[0]+=.1
        return dict(action=action,outcome='duration_complete',start_pose=start,end_pose=list(self.pose))


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_perception_never_shortens_or_replaces_committed_trajectory(self):
        a=Fake();worker=asyncio.create_task(a.motion_observer())
        observation=dict(protocol=PROTOCOL,snapshot_id=1,map_version=1,pose=[0,0,0],
                         trajectory_steps=4,action_spec=ACTION_PLAN_SPEC)
        reply=dict(protocol=PROTOCOL,snapshot_id=1,map_version=1,actions=['forward']*4,
                   goal_id='F',action='forward',task_complete=False,trajectory_steps=4)
        try:
            await a.execute_plan(reply,observation,time.monotonic())
            self.assertTrue(a.started.is_set())
            self.assertEqual(a.last_plan_outcome['completed_actions'],4)
            self.assertEqual(a.last_plan_outcome['discarded_actions'],[])
            self.assertAlmostEqual(a.pose[0],.4)
            self.assertFalse(any(e['event'].startswith('lookahead') for e in a.events))
            self.assertFalse(hasattr(a,'lookahead_task'))
            a.release.set()
            self.assertFalse(await a.finish_trajectory_planning())
            self.assertEqual(a.motion_processed_frame_id,a.motion_frames.serial)
            self.assertEqual(a.observed,list(range(1,a.motion_frames.serial+1)))
            self.assertEqual(a.motion_processed_count,a.motion_frames.serial)
        finally:
            worker.cancel();await asyncio.gather(worker,return_exceptions=True)
    async def test_final_tail_wait_is_only_for_frames_not_a_new_action_plan(self):
        a=Fake();a.collect_motion_frame(force=True)
        worker=asyncio.create_task(a.motion_observer())
        try:
            await a.started.wait()
            tail=asyncio.create_task(a.finish_trajectory_planning())
            await asyncio.sleep(.01);self.assertFalse(tail.done())
            a.release.set();self.assertFalse(await tail)
            self.assertEqual(a.motion_processed_frame_id,1)
            self.assertEqual(a.motion_last_result[0]['frame_id'],1)
        finally:
            worker.cancel();await asyncio.gather(worker,return_exceptions=True)
    async def test_finished_run_does_not_wait_for_inflight_perception(self):
        a=Fake();a.collect_motion_frame(force=True);a.finished.set()
        self.assertFalse(await asyncio.wait_for(a.finish_trajectory_planning(),.1))


if __name__=='__main__':unittest.main()

class PrefetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_next_frame_is_encoded_while_previous_gpu_call_is_waiting(self):
        a=Fake();worker=asyncio.create_task(a.motion_observer())
        try:
            a.collect_motion_frame(force=True)
            await asyncio.wait_for(a.started.wait(),1)
            a.collect_motion_frame(force=True)
            for _ in range(100):
                if a.motion_prefetch.done():break
                await asyncio.sleep(.005)
            self.assertTrue(a.motion_prefetch.done())
            self.assertEqual(a.motion_prefetch.result()[0][0][0]['frame_id'],2)
            self.assertFalse(a.observed)
            a.release.set();await a.finish_trajectory_planning()
            self.assertEqual(a.observed,[1,2])
        finally:
            worker.cancel();await asyncio.gather(worker,return_exceptions=True)

    async def test_backlog_joins_prefetched_frame_without_exceeding_four(self):
        class Batches(Fake):
            def __init__(self):super().__init__();self.batches=[]
            async def request_motion_observation(self,p):
                self.batches.append([f['frame_id'] for f in p['frames']])
                return await super().request_motion_observation(p)
        a=Batches();worker=asyncio.create_task(a.motion_observer())
        try:
            a.collect_motion_frame(force=True);await a.started.wait()
            a.collect_motion_frame(force=True)
            for _ in range(100):
                if a.motion_prefetch.done():break
                await asyncio.sleep(.005)
            for _ in range(4):a.collect_motion_frame(force=True)
            a.release.set();await a.finish_trajectory_planning()
            self.assertEqual(a.batches,[[1],[2,3,4,5],[6]])
            self.assertEqual(a.observed,[1,2,3,4,5,6])
        finally:
            worker.cancel();await asyncio.gather(worker,return_exceptions=True)
