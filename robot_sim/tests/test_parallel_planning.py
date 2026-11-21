import asyncio
import copy
import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from robot_sim.candidate_nav.action_plan import ACTION_PLAN_SPEC,PROTOCOL,rollout
from robot_sim.candidate_nav.geometry import Grid
from robot_sim.candidate_nav.parallel_planning import (ParallelPlanning,continuation_observation,
    promotion_reason,translation_path_free,remaining_seconds)
from robot_sim.candidate_nav.action_plan_executor import ActionPlanExecutor


def observation():
    return dict(protocol=PROTOCOL,snapshot_id=2,map_version=1,camera_stamp=2,pose=[0,0,0],trajectory_steps=5,
                physical=True,completion_mode='task_completion',action_spec=ACTION_PLAN_SPEC,
                task_context=dict(current_pose=[0,0,0],execution_totals={'forward_m':0}))


def reply(actions=['forward']):
    return dict(protocol=PROTOCOL,snapshot_id=2,map_version=1,actions=list(actions),action=actions[0],
                goal_id={'forward':'F','left':'L','stop':'S'}[actions[0]],task_complete=False,
                completion_mode='task_completion',latency_s=.01,
                metadata=dict(backend='native_async_llm',tp_size=1))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.grid=Grid(np.zeros((160,160),np.int16),.025,(-2,-2))
        self.o=continuation_observation(observation(),[.2,0,0],1,['forward','forward'],ACTION_PLAN_SPEC,self.grid)
    def check(self,**changes):
        values=dict(actual=[.2,0,0],parent_outcome=dict(plan_id=1,outcome='completed'),age=8,fresh=True,link_ok=True)
        values.update(changes)
        return promotion_reason(reply(),self.o,**values)
    def test_planning_origin_is_predicted_but_perception_and_totals_remain_measured(self):
        self.assertEqual(self.o['pose'],[.2,0,0])
        self.assertEqual(self.o['perception_pose'],[0,0,0])
        self.assertEqual(self.o['measured_task_context']['current_pose'],[0,0,0])
        self.assertEqual(self.o['task_context']['current_pose'],[.2,0,0])
        self.assertEqual(self.o['measured_task_context']['execution_totals']['forward_m'],0)
        self.assertAlmostEqual(self.o['task_context']['execution_totals']['forward_m'],.2)
        self.assertAlmostEqual(self.o['pipeline']['projected_execution_totals']['forward_m'],.2)
        self.assertEqual(observation()['pose'],[0,0,0])
    def test_valid_boundary_is_promoted(self):self.assertIsNone(self.check())
    def test_interruption_staleness_pose_link_and_new_evidence_reject(self):
        for change in (dict(actual=None),dict(actual=[.31,0,0]),dict(actual=[.2,0,.2]),
                       dict(parent_outcome=dict(plan_id=1,outcome='trajectory_collision')),
                       dict(parent_outcome=dict(plan_id=7,outcome='completed')),
                       dict(age=26),dict(fresh=False),dict(link_ok=False),
                       dict(new_evidence=True),dict(geometry_ok=False)):
            with self.subTest(change=change):self.assertIsNotNone(self.check(**change))
    def test_speculative_completion_is_never_promoted(self):
        r=reply(['stop']);r['task_complete']=True
        self.assertEqual(promotion_reason(r,self.o,[.2,0,0],dict(plan_id=1,outcome='completed'),age=1,fresh=True,link_ok=True),'completion_needs_current_observation')
    def test_new_obstacle_and_unknown_reject_full_body_translation(self):
        self.assertTrue(translation_path_free(self.grid,[0,0,0],['forward'],ACTION_PLAN_SPEC))
        x,y=self.grid.cell((.17,.07));self.grid.values[y,x]=100
        self.assertFalse(translation_path_free(self.grid,[0,0,0],['forward'],ACTION_PLAN_SPEC))
        self.grid.values[y,x]=-1
        self.assertFalse(translation_path_free(self.grid,[0,0,0],['forward'],ACTION_PLAN_SPEC))
        # Preserve the user's existing turn policy: no new in-place rotation veto.
        self.assertTrue(translation_path_free(self.grid,[0,0,0],['left'],ACTION_PLAN_SPEC))
    def test_reverse_volume_is_checked(self):
        x,y=self.grid.cell((-.17,.07));self.grid.values[y,x]=100
        self.assertFalse(translation_path_free(self.grid,[0,0,0],['backward'],ACTION_PLAN_SPEC))
    def test_estimate_uses_physical_command_scale(self):
        self.assertGreater(remaining_seconds(['forward'],ACTION_PLAN_SPEC),2)


class Fake(ParallelPlanning):
    action_spec=ACTION_PLAN_SPEC
    def __init__(self):
        self.init_parallel_planning();self.pose=[.2,0,0];self.finished=asyncio.Event()
        self.latencies=[.01];self.decisions=0;self.events=[];self.pending=asyncio.Queue()
        self.args=SimpleNamespace(disable_collision_protection=False,model_socket=None)
        self.grid=Grid(np.zeros((160,160),np.int16),.025,(-2,-2));self.reach=SimpleNamespace(grid=self.grid)
        self.last_plan_outcome=dict(plan_id=1,outcome='completed');self.executing_sequence=True
        self.rpc_started=asyncio.Event();self.rpc_release=asyncio.Event();self.steps_during_rpc=0
    def snapshot(self):
        self._snapshot_grid=self.grid;return observation(),[],time.monotonic()
    async def infer_plan(self,o):
        assert self.executing_sequence
        self.rpc_started.set();await self.rpc_release.wait();return reply()
    def get_pose(self):return self.pose
    def fresh(self):return True
    def control_link_ok(self):return True
    def now(self):return time.time()
    def log(self,_,item):self.events.append(item)
    async def queue_decision(self,r,o,t):await self.pending.put((r,o,t))


class OverlapTests(unittest.IsolatedAsyncioTestCase):
    async def test_planning_runs_during_motion_and_queues_validated_continuation(self):
        f=Fake();f.after_trajectory_step(1,['forward'],[.2,0,0])
        await asyncio.wait_for(f.rpc_started.wait(),2)
        f.steps_during_rpc+=1 # Motor executor can advance while the RPC awaits GPU output.
        f.rpc_release.set();await f.lookahead_task
        self.assertTrue(await f.finish_trajectory_planning())
        r,o,_=f.pending.get_nowait();self.assertEqual(o['pose'],f.pose)
        self.assertEqual(f.steps_during_rpc,1)
        self.assertTrue(f.latest_decision['prepared_during_motion'])
    async def test_only_one_pending_plan_and_parent_interrupt_cancels_it(self):
        f=Fake();f.after_trajectory_step(1,['forward'],[.2,0,0]);task=f.lookahead_task
        f.after_trajectory_step(1,['forward'],[.3,0,0]);self.assertIs(f.lookahead_task,task)
        await asyncio.wait_for(f.rpc_started.wait(),2)
        f.last_plan_outcome['outcome']='trajectory_collision'
        self.assertFalse(await f.finish_trajectory_planning())
        self.assertTrue(task.cancelled());self.assertTrue(f.pending.empty())
    async def test_new_visual_evidence_discards_ready_plan(self):
        f=Fake();f.rpc_release.set();f.after_trajectory_step(1,['forward'],[.2,0,0]);await f.lookahead_task
        f.visual_revision+=1;f.visual_revision_stamp=3
        self.assertFalse(await f.finish_trajectory_planning());self.assertTrue(f.pending.empty())
    async def test_late_report_about_an_old_frame_does_not_invalidate_newer_camera(self):
        f=Fake();f.rpc_release.set();f.after_trajectory_step(1,['forward'],[.2,0,0]);await f.lookahead_task
        f.visual_revision+=1;f.visual_revision_stamp=1
        self.assertTrue(await f.finish_trajectory_planning())
    async def test_disabled_collision_mode_keeps_pose_checks_but_skips_map_veto(self):
        f=Fake();f.args.disable_collision_protection=True;f.grid.values[:]=100
        f.rpc_release.set();f.after_trajectory_step(1,['forward'],[.2,0,0]);await f.lookahead_task
        self.assertTrue(await f.finish_trajectory_planning())


if __name__=='__main__':unittest.main()
