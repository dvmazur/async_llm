import ast
import asyncio
import math
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robot_sim.candidate_nav.action_plan import (ACTION_PLAN_SPEC, LABELS, ACTION_LABELS,
    PROTOCOL, rollout, validate_response, turn_actions_for_angle)
from robot_sim.candidate_nav.action_plan_executor import ActionPlanExecutor
from robot_sim.candidate_nav.direct_control import LIVE_ACTION_SPEC
from robot_sim.candidate_nav.geometry import Grid
from robot_sim.candidate_nav.maneuver_memory import heading_options, maneuver_memory
from robot_sim.candidate_nav.trajectory_policy import translation_options, action_labels
from robot_sim.candidate_nav.task_policy import TRAJECTORY_EVIDENCE_SYSTEM, MOTION_REALITY_RULES


class GeometryTests(unittest.TestCase):
    def test_old_turns_and_strategy_three_are_unchanged(self):
        self.assertAlmostEqual(ACTION_PLAN_SPEC['left']['angle_rad'],math.pi/8)
        self.assertAlmostEqual(LIVE_ACTION_SPEC['left']['angle_rad'],math.pi/4)
        self.assertNotIn('left_11_25',LIVE_ACTION_SPEC)
        self.assertEqual(ACTION_PLAN_SPEC['left_11_25']['angular_rad_s'],
                         ACTION_PLAN_SPEC['left']['angular_rad_s']/2)
        self.assertEqual(ACTION_PLAN_SPEC['right_11_25']['angular_rad_s'],
                         ACTION_PLAN_SPEC['right']['angular_rad_s']/2)
    def test_all_six_turns_have_correct_sign_magnitude_and_wire_contract(self):
        for label,name in LABELS.items():
            item=ACTION_PLAN_SPEC[name]
            if 'angle_rad' not in item:continue
            with self.subTest(action=name):
                degrees=11.25 if '11_25' in name else 45 if name.endswith('_45') else 22.5
                actual=rollout([0,0,0],[name],ACTION_PLAN_SPEC)[-1][2]
                self.assertAlmostEqual(math.degrees(actual),degrees*(1 if name.startswith('left') else -1))
                reply=dict(protocol=PROTOCOL,trajectory_steps=1,actions=[name],action=name,
                           goal_id=label,task_complete=False)
                validate_response(reply,dict(trajectory_steps=1))
    def test_mixed_turns_define_subsequent_translation_heading(self):
        actions=['left_45','right','left_11_25','forward']
        pose=rollout([0,0,0],actions,ACTION_PLAN_SPEC)[-1]
        self.assertAlmostEqual(math.degrees(pose[2]),33.75)
        self.assertAlmostEqual(pose[0],.1*math.cos(math.radians(33.75)))
        self.assertAlmostEqual(pose[1],.1*math.sin(math.radians(33.75)))
    def test_heading_grid_and_menu_keep_literal_angle_sequences(self):
        grid=Grid(np.zeros((120,120),np.int16),.025,(-1.5,-1.5))
        rows=heading_options(grid,[0,0,0],ACTION_PLAN_SPEC)
        self.assertEqual(len(rows),33)
        by_angle={row['turn_degrees']:row for row in rows}
        self.assertEqual(by_angle[11.25]['turn_actions'],['left_11_25'])
        self.assertEqual(by_angle[-45]['turn_actions'],['right_45'])
        self.assertEqual(by_angle[180]['turn_actions'],['left_45']*4)
        row=by_angle[33.75]
        option=translation_options([row])[0]
        self.assertEqual(option['turn_actions'],['left','left_11_25'])
        self.assertEqual(action_labels([], [option], option),['L','S'])
        self.assertEqual(action_labels(['L'], [option], option),['Q','S'])
        self.assertEqual(action_labels(['L','Q'], [option], option),['F','S'])
    def test_shortest_exact_turn_sequences_do_not_fill_horizon(self):
        self.assertEqual(turn_actions_for_angle(math.pi,ACTION_PLAN_SPEC),['left_45']*4)
        self.assertEqual(turn_actions_for_angle(math.radians(-78.75),ACTION_PLAN_SPEC),['right_45','right','right_11_25'])
    def test_measured_history_tracks_new_actions_not_nominal_counts(self):
        p=dict(plan_id=1,start_pose=[0,0,0],end_pose=[0,0,.51],outcome='observation_requested',
               planned_actions=['left_11_25','left_45','stop'],discarded_actions=[],
               step_results=[dict(action='left_11_25',turn_rad=.09,distance_m=0,outcome='insufficient_progress_or_deviation'),
                             dict(action='left_45',turn_rad=.42,distance_m=0,outcome='duration_complete')])
        memory=maneuver_memory([p],[0,0,.51])
        self.assertAlmostEqual(memory['local_measured_turn_rad'],.51)
        self.assertEqual(memory['last_measured_turn_direction'],'left')
    def test_real_motion_limits_are_explicit_in_active_system_prompt(self):
        self.assertIn(MOTION_REALITY_RULES,TRAJECTORY_EVIDENCE_SYSTEM)
        for word in ('NOMINAL','Wheel slip','measured','11.25','22.5','45 degrees'):
            self.assertIn(word,TRAJECTORY_EVIDENCE_SYSTEM)


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_stationary_small_turn_is_not_accepted_as_success(self):
        class Agent:
            action_spec=ACTION_PLAN_SPEC
            def __init__(self):
                self.finished=asyncio.Event()
                self.stop_pub=SimpleNamespace(armed=False,stop=lambda:None)
            def get_pose(self):return [0,0,0]
            def fresh(self):return True
            def control_link_ok(self):return True
            def log(self,*args):pass
            async def execute_action(self,action):
                return dict(action=action,outcome='duration_complete',start_pose=[0,0,0],end_pose=[0,0,0])
        a=Agent();name='left_11_25'
        reply=dict(protocol=PROTOCOL,trajectory_steps=1,actions=[name],action=name,goal_id='Q',task_complete=False)
        await ActionPlanExecutor.execute_plan(a,reply,dict(trajectory_steps=1,pose=[0,0,0],snapshot_id=1),time.monotonic())
        self.assertEqual(a.last_plan_outcome['outcome'],'insufficient_progress_or_deviation')
        self.assertEqual(a.last_plan_outcome['completed_actions'],0)

    async def test_real_live_executor_handles_each_turn_and_wraps_without_ros(self):
        # Run the production method against measured deterministic feedback,
        # avoiding ROS imports and never publishing to a physical robot.
        path=Path(__file__).parents[1]/'candidate_nav/live_direct.py'
        tree=ast.parse(path.read_text())
        method=next(m for n in tree.body if isinstance(n,ast.ClassDef) for m in n.body
                    if isinstance(m,ast.AsyncFunctionDef) and m.name=='execute_action')
        class Twist:
            def __init__(self):
                self.linear=SimpleNamespace(x=0.);self.angular=SimpleNamespace(z=0.)
        for action,item in ACTION_PLAN_SPEC.items():
            if 'angle_rad' not in item:continue
            with self.subTest(action=action):
                clock=[0.];heading=[3.1];velocity=[0.]
                async def sleep(dt):
                    clock[0]+=dt;heading[0]+=velocity[0]*dt
                def publish(command):velocity[0]=command.angular.z
                agent=SimpleNamespace(get_pose=lambda:[0,0,math.atan2(math.sin(heading[0]),math.cos(heading[0]))],
                    fresh=lambda:True,action_spec=ACTION_PLAN_SPEC,executed=0,completed=0,
                    finished=asyncio.Event(),action_history=[],log=lambda *args:None,
                    stop_pub=SimpleNamespace(armed=False,last_decision='allowed',publish=publish,
                                             stop=lambda:velocity.__setitem__(0,0.)))
                ns=dict(time=SimpleNamespace(monotonic=lambda:clock[0]),asyncio=SimpleNamespace(sleep=sleep),
                        math=math,Twist=Twist)
                exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
                target=rollout(agent.get_pose(),[action],ACTION_PLAN_SPEC)[-1]
                result=await ns['execute_action'](agent,action,target_pose=target)
                self.assertEqual(result['outcome'],'duration_complete')
                angle=math.atan2(math.sin(heading[0]-3.1),math.cos(heading[0]-3.1))
                nominal=math.copysign(item['angle_rad'],item['angular_rad_s'])
                self.assertLess(abs(angle-nominal),.045)
                self.assertEqual(velocity[0],0)

if __name__=='__main__':unittest.main()
