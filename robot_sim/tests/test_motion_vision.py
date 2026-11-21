import unittest, math
from robot_sim.candidate_nav.motion_vision import MotionFrames
from robot_sim.candidate_nav.action_plan import ACTION_PLAN_SPEC, rollout
from robot_sim.candidate_nav.direct_control import LIVE_ACTION_SPEC
class MotionFramesTests(unittest.TestCase):
 def test_mid_turn_sighting_is_retained_after_turn_ends(self):
  q=MotionFrames()
  q.add('before',1,[0,0,0],7,True)
  q.add('target-visible',2,[0,0,math.pi/16],7)
  q.add('after',3,[0,0,math.pi/8],7,True)
  q.add('duplicate',3,[0,0,math.pi/8],7,True)
  frames=q.take()
  self.assertEqual([img for _,img in frames],['before','target-visible','after'])
  self.assertEqual([m['frame_id'] for m,_ in frames],[1,2,3])
  self.assertFalse(q.pending)
 def test_redundant_frames_skip_and_storage_is_bounded(self):
  q=MotionFrames();q.add('a',1,[0,0,0],1)
  self.assertIsNone(q.add('same-view',2,[0,0,.01],1))
  for i in range(140):q.add(str(i),i+3,[0,0,0],1,True)
  self.assertEqual(len(q.pending),128)
  self.assertGreater(q.dropped,0)
 def test_22_5_turns_cover_180_and_preserve_strategy3(self):
  poses=rollout([0,0,0],['right']*8,ACTION_PLAN_SPEC)
  self.assertAlmostEqual(abs(poses[-1][2]),math.pi)
  self.assertAlmostEqual(LIVE_ACTION_SPEC['right']['angle_rad'],math.pi/4)

import asyncio
from collections import deque
from PIL import Image
from robot_sim.candidate_nav.motion_vision import MotionObserver
class MotionObserverTests(unittest.IsolatedAsyncioTestCase):
 async def test_inflight_perception_preserves_execution_and_drains_before_next_plan(self):
  class Agent(MotionObserver):
   def __init__(self):
    self.finished=asyncio.Event();self.vision_wakeup=asyncio.Event();self.vision_idle=asyncio.Event()
    self.motion_frames=MotionFrames();self.visual_memory=deque(maxlen=8)
    self.task={'instruction':'Find the red square and approach to 5 cm.'};self.version=2
    self.started=asyncio.Event();self.answer=asyncio.Event();self.events=[];self.state='moving'
   async def request_motion_observation(self,packet):
    self.packet=packet;self.started.set();await self.answer.wait()
    return dict(evidence='Frame 1: red square was seen.',latency_s=1,
                events=[dict(frame=dict(stamp=1),event='target_visible')])
   def log(self,_,item):self.events.append(item)
  a=Agent();a.motion_frames.add(Image.new('RGB',(16,16),'red'),1,[0,0,0],1,True)
  a.vision_wakeup.set();worker=asyncio.create_task(a.motion_observer())
  try:
   await asyncio.wait_for(a.started.wait(),1)
   executed=[]
   for step in ('left','left','forward'):
    executed.append(step);await asyncio.sleep(0)
   self.assertEqual(executed,['left','left','forward'])
   self.assertFalse(a.finished.is_set());self.assertFalse(a.vision_idle.is_set())
   a.answer.set();await asyncio.wait_for(a.vision_idle.wait(),1)
   self.assertFalse(a.visual_memory[0]['current_visibility'])
   self.assertEqual(a.visual_memory[0]['frames'][0]['pose'],[0,0,0])
   self.assertNotIn('actions',a.visual_memory[0])
   self.assertEqual(a.visual_revision,1)
   self.assertEqual(a.visual_revision_stamp,1)
  finally:
   worker.cancel();await asyncio.gather(worker,return_exceptions=True)
