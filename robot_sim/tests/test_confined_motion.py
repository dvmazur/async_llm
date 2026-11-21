import math,unittest
from robot_sim.candidate_nav.action_plan import ACTION_PLAN_SPEC,rollout,validate_response
from robot_sim.candidate_nav.trajectory_policy import translation_options,action_labels
from robot_sim.candidate_nav.maneuver_memory import maneuver_memory
class ConfinedTests(unittest.TestCase):
 def test_short_forward_and_reverse_remain_available_below_ten_cm(self):
  rows=[dict(turn_prefix='',turn_sweep_known_free=True,full_forward_steps=0,short_forward_steps=1,full_backward_steps=0,short_backward_steps=1,map_yaw_degrees=0)]
  options=translation_options(rows)
  self.assertEqual({x['translation_label'] for x in options},{'G','C'})
  self.assertEqual(action_labels([],options,options[0]),['G','S'])
 def test_reverse_and_short_steps_follow_heading(self):
  poses=rollout([0,0,0],['forward','creep_forward','backward','creep_backward'],ACTION_PLAN_SPEC)
  self.assertAlmostEqual(poses[1][0],.1);self.assertAlmostEqual(poses[2][0],.15)
  self.assertAlmostEqual(poses[-1][0],0)
 def test_spinning_without_forward_attempts_is_a_loop(self):
  history=[]
  for i,turn in enumerate((.4,.4,-.4,-.4)):
   history.append(dict(plan_id=i,start_pose=[0,0,0],end_pose=[0,0,turn],planned_actions=[],outcome='observation_requested',step_results=[dict(action='left' if turn>0 else 'right',turn_rad=turn,distance_m=0,outcome='duration_complete')]))
  self.assertTrue(maneuver_memory(history,[0,0,0])['oscillation_detected'])
 def test_new_wire_labels_are_valid_and_do_not_report_completion(self):
  for label,name in [('B','backward'),('C','creep_backward'),('G','creep_forward')]:
   r=dict(protocol='action_trajectory_v1',trajectory_steps=1,actions=[name],action=name,goal_id=label,task_complete=False)
   validate_response(r,dict(trajectory_steps=1))

import asyncio,time
from types import SimpleNamespace
from robot_sim.candidate_nav.action_plan_executor import ActionPlanExecutor
class ShortMotionExecutionTests(unittest.IsolatedAsyncioTestCase):
 async def test_stationary_five_cm_command_is_not_false_success(self):
  class Agent:
   action_spec=ACTION_PLAN_SPEC
   def __init__(self):
    self.finished=asyncio.Event();self.stop_pub=SimpleNamespace(armed=False,stop=lambda:None)
   def get_pose(self):return [0,0,0]
   def fresh(self):return True
   def control_link_ok(self):return True
   def log(self,*a):pass
   async def execute_action(self,action):
    return dict(action=action,outcome='duration_complete',start_pose=[0,0,0],end_pose=[0,0,0])
  a=Agent();r=dict(protocol='action_trajectory_v1',trajectory_steps=1,actions=['creep_forward'],action='creep_forward',goal_id='G',task_complete=False)
  await ActionPlanExecutor.execute_plan(a,r,dict(trajectory_steps=1,pose=[0,0,0],snapshot_id=1),time.monotonic())
  self.assertEqual(a.last_plan_outcome['outcome'],'insufficient_progress_or_deviation')
  self.assertEqual(a.last_plan_outcome['completed_actions'],0)
