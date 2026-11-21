import asyncio
from collections import deque
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from robot_sim.candidate_nav.action_plan import ACTION_PLAN_SPEC
from robot_sim.candidate_nav.mission_context import (MEMORY, SUBGOAL, anchor, choose_passage, diverse_routes, passage_labels,
                                                     local_routes, measured_event, route_labels,
                                                     route_to_goal)
from robot_sim.candidate_nav.mission_workers import MissionWorkers
from robot_sim.candidate_nav.mission_policy import evaluate_mission


SCOPE = dict(version=1, enabled=True, tracking_valid=True, map_session_id='map-a', anchor_epoch='epoch-1')
ACTIONS = ACTION_PLAN_SPEC


def observation():
    rows = [dict(turn_prefix='AA', turn_actions=['left_45', 'left_45'], map_yaw_degrees=90,
                 turn_sweep_known_free=True, full_forward_steps=7),
            dict(turn_prefix='R', turn_actions=['right'], map_yaw_degrees=-22.5,
                 turn_sweep_known_free=True, full_forward_steps=1)]
    return dict(episode_id='test', rover_id=107, snapshot_id=8, map_version=10, pose=[0., 0., 0.],
                instruction='Explore', memory_context=dict(SCOPE), action_spec=ACTIONS,
                trajectory_steps=4, heading_options=rows, task_context={})


def executed(i, start, end):
    return dict(plan_id=i, memory_scope=dict(SCOPE), outcome='completed',
                start_pose=start, end_pose=end, control_start_pose=start, control_end_pose=end,
                step_results=[dict(action='forward', distance_m=.22, end_pose=end)])


class GeometryTests(TestCase):
    def test_measured_retrace_is_scoped_to_localization_epoch(self):
        old = executed(6, [0, 0, 0], [.22, 0, 0])
        current = executed(7, [.22, 0, 0], [.01, 0, 0])
        note = measured_event(current, old, SCOPE)
        self.assertEqual(note['kind'], 'returned')
        self.assertEqual(note['net_m'], .01)
        self.assertIsNone(measured_event(current, old, {**SCOPE, 'anchor_epoch': 'different'}))

    def test_passage_uses_forward_branch_before_short_recovery_and_stays_committed(self):
        obs = observation()
        obs['instruction'] = 'Leave this maze'
        scope = dict(SCOPE)
        agent = SimpleNamespace(plan_history=[], finished=asyncio.Event(),
                                memory_scope=SimpleNamespace(export=lambda: scope),
                                log=lambda *_: None)
        obs['heading_options'].append(dict(turn_prefix='', turn_actions=[], map_yaw_degrees=0,
                                         turn_sweep_known_free=True, full_backward_steps=4))
        workers = MissionWorkers(agent)
        pending = workers.decorate(obs)
        route, goal = choose_passage(obs, {})
        self.assertEqual(route['turn_actions'], ['left_45', 'left_45'])
        workers.submit(obs, dict(visual_task_scope='M', passage_goal=goal), pending)
        self.assertIs(workers.goal, goal)
        self.assertTrue(workers.jobs[SUBGOAL].empty())

        obs['snapshot_id'] += 1
        obs['pose'][2] = 1.5707963267948966
        obs['heading_options'] = [dict(turn_prefix='', turn_actions=[], map_yaw_degrees=90,
                                      turn_sweep_known_free=True, full_forward_steps=3,
                                      full_backward_steps=4)]
        workers.decorate(obs)
        again, retained = choose_passage(obs, dict(center='O'))
        self.assertEqual(retained, goal)
        self.assertEqual(again['turn_actions'], [])
        self.assertIn('F', passage_labels([], again, ACTIONS, ['F', 'G', 'S']))

        recovery, new_goal = choose_passage(obs, dict(center='B'))
        self.assertIsNone(new_goal)
        self.assertEqual(recovery['translation_label'], 'C')
        self.assertEqual(passage_labels(['C'], recovery, ACTIONS, ['C', 'S']), ['S'])
        workers.submit(obs, dict(visual_task_scope='M', passage_goal=None), pending)
        self.assertIsNone(workers.goal)

    def test_goal_survives_partial_turn_and_stays_within_verified_budget(self):
        obs = observation()
        routes = local_routes(obs, [])
        route = diverse_routes(routes)['A']
        self.assertEqual(route['turn_actions'], ['left_45', 'left_45'])
        self.assertEqual(route_labels(['A'], route, ACTIONS), {'A', 'S'})
        self.assertEqual(route_labels(['A', 'A'], route, ACTIONS), {'F', 'G', 'S'})
        self.assertEqual(route_labels(['A', 'A', 'F', 'F', 'F', 'F'], route, ACTIONS), {'S'})
        obs['pose'][2] = 1.5707963267948966
        obs['heading_options'] = [dict(turn_prefix='', turn_actions=[], map_yaw_degrees=90,
                                      turn_sweep_known_free=True, full_forward_steps=3)]
        self.assertEqual(route_to_goal(obs, local_routes(obs, []), route)['turn_actions'], [])
        obs['heading_options'] = []
        self.assertIsNone(route_to_goal(obs, local_routes(obs, []), route))


class WorkerTests(IsolatedAsyncioTestCase):
    async def test_first_measured_return_is_preserved_without_gpu(self):
        packet = dict(protocol=MEMORY, snapshot_id=9, map_version=1, notes=[],
                      event=dict(kind='returned', pose=[.01, 0, 0]))
        self.assertEqual((await evaluate_mission(None, packet))['choice'], 'K')
        packet['notes'] = [packet['event']]
        self.assertEqual((await evaluate_mission(None, packet))['choice'], 'S')

    async def test_two_workers_commit_during_execution_and_invalidate_on_epoch_change(self):
        logs = []
        scope = dict(SCOPE)
        agent = SimpleNamespace(
            plan_history=deque([executed(6, [0, 0, 0], [.22, 0, 0]),
                                executed(7, [.22, 0, 0], [.01, 0, 0])]),
            finished=asyncio.Event(), executing_sequence=False, trajectory_progress=None,
            plan_request_pending=False, model_rpc_lock=asyncio.Lock(), args=SimpleNamespace(model_socket=None),
            memory_scope=SimpleNamespace(export=lambda: scope), log=lambda _, data: logs.append(data))
        workers = MissionWorkers(agent)
        obs = observation()
        workers.submit(obs, {}, workers.decorate(obs))
        calls = []
        done = asyncio.Event()

        async def answer(packet, *_args, **_kwargs):
            calls.append(packet['protocol'])
            if len(calls) == 2:
                done.set()
            return {'choice': 'K' if packet['protocol'] == MEMORY else 'A'}

        with patch('robot_sim.candidate_nav.mission_workers.request_model', answer):
            tasks = [asyncio.create_task(workers.run_worker(kind)) for kind in (MEMORY, SUBGOAL)]
            try:
                await asyncio.sleep(.06)
                self.assertEqual(calls, [])
                agent.executing_sequence = True
                agent.trajectory_progress = {'plan_id': 8}
                await asyncio.wait_for(done.wait(), 1)
                self.assertEqual({*calls}, {MEMORY, SUBGOAL})
                self.assertEqual(len(workers.notes), 1)
                self.assertIsNotNone(workers.goal)
                scope['anchor_epoch'] = 'new'
                obs['memory_context'] = scope
                workers.decorate(obs)
                self.assertIsNone(workers.goal)
                self.assertEqual(list(workers.notes), [])
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
