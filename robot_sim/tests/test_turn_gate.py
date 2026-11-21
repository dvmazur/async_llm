"""Exercise the actual NavRelay rejection method without importing ROS on CPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

class TurnGateTests(unittest.TestCase):
 def setUp(self):
  path=Path(__file__).resolve().parents[1]/'candidate_nav/live_waypoint.py'
  tree=ast.parse(path.read_text())
  cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='NavRelay')
  method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='rejection')
  self.clearance=Mock(return_value=None)
  self.strategy4_clearance=Mock(return_value=SimpleNamespace(gap_m=None,reason='occupied_start'))
  scope={'trajectory_clearance':self.clearance,'check_strategy4_translation':self.strategy4_clearance}
  exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
  self.rejection=scope['rejection']
  self.agent=SimpleNamespace(args=SimpleNamespace(strategy=4),control_link_ok=lambda:True,
      finished=SimpleNamespace(is_set=lambda:False),fresh=lambda:True,get_pose=lambda:(0,0,0),
      scan=None,scan_origin=(0,0),action_label="creep_backward")
  self.gate=SimpleNamespace(agent=self.agent,armed=True,competing_control=lambda:False)
 def test_strategy4_left_and_right_ignore_lidar_collision(self):
  for angular in (-.8/1.8,.8/1.8):
   self.assertIsNone(self.rejection(self.gate,0,angular))
  self.clearance.assert_not_called()
  self.strategy4_clearance.assert_not_called()
 def test_forward_and_arcs_still_use_clearance(self):
  for v,w in ((.05,0),(.02,.3),(-.05,0),(0,.6)):
   self.assertEqual(self.rejection(self.gate,v,w),'trajectory_collision')
  self.assertEqual(self.strategy4_clearance.call_count,4)
  self.strategy4_clearance.assert_any_call(None,(0,0),-.10,0,"creep_backward")
  self.assertEqual(self.gate.collision_check.reason,'occupied_start')
  self.clearance.assert_not_called()
 def test_other_strategies_still_check_turns(self):
  for strategy in (1,2,3):
   self.agent.args.strategy=strategy
   self.assertEqual(self.rejection(self.gate,0,.8/1.8),'trajectory_collision')
 def test_link_sensors_arming_and_competing_controller_still_stop_turns(self):
  cases=[('control_link_ok',lambda:False,'control_link_lost'),
         ('fresh',lambda:False,'stale_sensors_or_tf'),
         ('get_pose',lambda:None,'stale_sensors_or_tf')]
  for name,value,reason in cases:
   original=getattr(self.agent,name);setattr(self.agent,name,value)
   self.assertEqual(self.rejection(self.gate,0,.8/1.8),reason)
   setattr(self.agent,name,original)
  self.gate.armed=False
  self.assertEqual(self.rejection(self.gate,0,.8/1.8),'not_armed')
  self.gate.armed=True;self.gate.competing_control=lambda:True
  self.assertEqual(self.rejection(self.gate,0,.8/1.8),'other_velocity_publisher')
  self.clearance.assert_not_called()
  self.strategy4_clearance.assert_not_called()

 def test_explicit_flag_bypasses_all_lidar_paths_but_not_link(self):
  self.agent.args.disable_collision_protection=True
  for v,w in ((.05,0),(-.03,0),(0,.8/1.8),(.02,.2)):
   self.assertIsNone(self.rejection(self.gate,v,w))
  self.clearance.assert_not_called()
  self.strategy4_clearance.assert_not_called()
  self.agent.control_link_ok=lambda:False
  self.assertEqual(self.rejection(self.gate,.05,0),'control_link_lost')

 def test_contact_is_recorded_without_stopping_only_under_flag(self):
  path=Path(__file__).resolve().parents[1]/'candidate_nav/agent.py'
  tree=ast.parse(path.read_text())
  cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Agent')
  method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='contact')
  scope={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
  a=SimpleNamespace(args=SimpleNamespace(strategy=4,disable_collision_protection=True),contacts=0,finish=Mock(),log=Mock())
  scope['contact'](a,SimpleNamespace(contacts=[1]));a.finish.assert_not_called()
  self.assertEqual(a.contacts,1)
  a.args.disable_collision_protection=False
  scope['contact'](a,SimpleNamespace(contacts=[1]));a.finish.assert_called_once_with('body_contact')
