"""Sequence protocol and physical executor regressions; no model or motors."""

import asyncio
import io
import json
import math
import subprocess
import tarfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from robot_sim.candidate_nav.action_plan import (
    ACTION_PLAN_SPEC,
    PROTOCOL,
    pose_matches,
    rollout,
    trajectory_steps,
    translation_finished_at_brake,
    validate_response,
)
from robot_sim.candidate_nav.action_plan_executor import ActionPlanExecutor
from robot_sim.candidate_nav.direct_control import LIVE_ACTION_SPEC as ACTION_SPEC


def observation(horizon=5):
    return {'protocol': PROTOCOL, 'trajectory_steps': horizon, 'pose': [0, 0, 0],
                'snapshot_id': 7, 'map_version': 3, 'physical': True,
                'completion_mode': 'task_completion', 'objects': []}


def response(actions, horizon=5, complete=False):
    return {'protocol': PROTOCOL, 'trajectory_steps': horizon, 'actions': actions,
                'action': actions[0], 'goal_id': 'Z' if complete else
                {'forward': 'F', 'left': 'L', 'right': 'R', 'stop': 'S'}[actions[0]],
                'snapshot_id': 7, 'map_version': 3, 'task_complete': complete,
                'completion_mode': 'task_completion'}


class PlanContractTests(unittest.TestCase):
    def test_live_sequence_timeout_does_not_change_nominal_motion_or_strategy3(self):
        for action in ('forward', 'left', 'right'):
            original = ACTION_SPEC[action]
            planned = ACTION_PLAN_SPEC[action]
            self.assertGreater(planned['max_sim_s'], original['max_sim_s'])
            self.assertEqual({k:v for k,v in planned.items() if k not in {'max_sim_s','angle_rad','max_distance_m'}},
                             {k:v for k,v in original.items() if k not in {'max_sim_s','angle_rad','max_distance_m'}})
        self.assertAlmostEqual(ACTION_PLAN_SPEC['left']['angle_rad'], math.pi/8)
        self.assertAlmostEqual(ACTION_PLAN_SPEC['right']['angle_rad'], math.pi/8)
        self.assertEqual(ACTION_SPEC['left']['max_sim_s'], 1.4)
        self.assertEqual(ACTION_SPEC['forward']['max_sim_s'], 3.5)

    def test_predictive_stop_at_measured_endpoint(self):
        spec = ACTION_PLAN_SPEC['creep_forward']
        for x in (.043, .05, .060):
            self.assertTrue(translation_finished_at_brake((0,0,0), (x,0,0), (.05,0,0), spec))
        for end in ((.026,0,0), (.07,0,0), (.05,.03,0), (.05,0,.21), None):
            self.assertFalse(translation_finished_at_brake((0,0,0), end, (.05,0,0), spec))
        self.assertFalse(translation_finished_at_brake((0,0,0), (-.05,0,0), (.05,0,0), spec))
        self.assertTrue(translation_finished_at_brake((0,0,math.pi/2), (0,-.052,math.pi/2),
                        (0,-.05,math.pi/2), ACTION_PLAN_SPEC['creep_backward']))

    def test_horizon_limits(self):
        for value in (1, '1', 32, '32'):
            self.assertEqual(trajectory_steps(value), int(value))
        for value in (0, 33, -1, '3.0', 'abc', True, '02', 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                trajectory_steps(value)

    def test_rejects_malformed_or_inconsistent_plans(self):
        valid = response(['left', 'forward', 'stop'])
        validate_response(valid, observation())
        changes = [{'protocol': 'direct_navigation_v1'}, {'trajectory_steps': 4},
                   {'trajectory_steps': True}, {'actions': []}, {'actions': 'forward'},
                   {'actions': ['forward'] * 6}, {'actions': ['back']},
                   {'actions': [True]}, {'actions': [['forward']]},
                   {'actions': ['left', 'stop', 'forward']}, {'task_complete': 1},
                   {'action': 'right'}, {'goal_id': 'R'}, {'task_complete': True}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_response({**valid, **change}, observation())

    def test_stop_is_not_task_completion(self):
        validate_response(response(['stop']), observation())
        validate_response(response(['stop'], complete=True), observation())
        with self.assertRaises(ValueError):
            validate_response(response(['stop'], complete=True),
                              {**observation(), 'completion_mode': 'visual_confirmation'})

    def test_rollout_respects_relative_heading_and_wraps(self):
        poses = rollout([0, 0, 0], ['left', 'left', 'forward', 'right', 'stop'], ACTION_SPEC)
        self.assertAlmostEqual(poses[3][0], 0)
        self.assertAlmostEqual(poses[3][1], .25)
        self.assertAlmostEqual(poses[-1][2], math.pi / 4)
        self.assertTrue(pose_matches([0, 0, -math.pi], [0, 0, math.pi]))
        self.assertFalse(pose_matches(None, [0, 0, 0]))
        self.assertFalse(pose_matches([.13, 0, 0], [0, 0, 0]))
        self.assertFalse(pose_matches([0, 0, float('nan')], [0, 0, 0]))

    def test_deployment_bundle_includes_new_protocol_and_executor(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(['bash', str(root / 'robot_sim/scripts/rover_live_bundle.sh')],
                                check=True, capture_output=True)
        with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
            names = archive.getnames()
        for name in ('action_plan', 'action_plan_executor', 'live_action_plan', 'task_policy', 'body_model', 'map_memory'):
            self.assertIn(f'robot_sim/candidate_nav/{name}.py', names)


class FakeExecutor:
    action_spec = ACTION_SPEC
    def __init__(self, *, blocked=None, progress=True):
        self.pose = [0, 0, 0]
        self.finished = asyncio.Event()
        self.need_plan = asyncio.Event()
        self.pending = asyncio.Queue()
        self.calls, self.events = [], []
        self.active_route = []
        self.trajectory_progress = None
        self.last_plan_outcome = None
        self.executing_sequence = False
        self.blocked, self.progress = blocked, progress
        self.sensors_ok, self.link_ok = True, True
        self.stop_count = 0
        self.stop_pub = SimpleNamespace(armed=False, stop=self.stop)

    def stop(self):
        self.stop_count += 1

    def get_pose(self):
        return self.pose

    def fresh(self):
        return self.sensors_ok

    def control_link_ok(self):
        return self.link_ok

    def log(self, _, item):
        self.events.append(item)

    async def execute_action(self, action):
        self.calls.append(action)
        if self.progress and action != self.blocked:
            self.pose = rollout(self.pose, [action], ACTION_SPEC)[-1]
        return {'action': action, 'outcome': 'trajectory_collision' if action == self.blocked
                    else 'duration_complete'}


class PlanExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def run_plan(self, fake, actions, *, age=0):
        await ActionPlanExecutor.execute_plan(fake, response(actions), observation(),
                                              time.monotonic() - age)
        self.assertFalse(fake.stop_pub.armed)
        self.assertGreater(fake.stop_count, 0)
        self.assertEqual(fake.active_route, [])
        return fake.last_plan_outcome

    async def test_executes_entire_sequence_and_records_outcomes(self):
        fake = FakeExecutor()
        result = await self.run_plan(fake, ['left', 'forward', 'right'])
        self.assertEqual(fake.calls, ['left', 'forward', 'right'])
        self.assertEqual(result['outcome'], 'completed')
        self.assertEqual(result['executed_actions'], 3)
        self.assertEqual(result['discarded_actions'], [])
        self.assertEqual(sum(e['event'] == 'trajectory_step' for e in fake.events), 3)

    async def test_absolute_targets_prevent_accumulated_turn_overshoot(self):
        class ClosedLoop(FakeExecutor):
            closed_loop_targets = True
            async def execute_action(self, action, target_pose=None):
                self.calls.append(action)
                self.pose = list(target_pose)
                if action in ('left', 'right'):
                    self.pose[2] += .10 if action == 'left' else -.10
                return dict(action=action, outcome='duration_complete')
        fake=ClosedLoop()
        result=await self.run_plan(fake, ['left','left','left','left','forward'])
        self.assertEqual(result['completed_actions'],5)
        self.assertEqual(result['outcome'],'completed')

    async def test_collision_discards_all_remaining_actions(self):
        fake = FakeExecutor(blocked='forward')
        result = await self.run_plan(fake, ['left', 'forward', 'right', 'forward'])
        self.assertEqual(fake.calls, ['left', 'forward'])
        self.assertEqual(result['outcome'], 'trajectory_collision')
        self.assertEqual(result['discarded_actions'], ['right', 'forward'])

    async def test_undertravel_never_executes_the_rest(self):
        fake = FakeExecutor(progress=False)
        result = await self.run_plan(fake, ['forward', 'left'])
        self.assertEqual(fake.calls, ['forward'])
        self.assertEqual(result['outcome'], 'insufficient_progress_or_deviation')
        self.assertEqual(result['discarded_actions'], ['left'])
        self.assertEqual(result['completed_actions'], 0)
        self.assertEqual(result['step_results'][0]['outcome'], 'insufficient_progress_or_deviation')

    async def test_blocked_plan_keeps_actual_pose_and_turns_for_next_request(self):
        fake = FakeExecutor(blocked='forward')
        fake.plan_history = []
        action = fake.execute_action
        async def measured(command):
            start = list(fake.pose)
            record = await action(command)
            return {**record, 'start_pose': start, 'end_pose': list(fake.pose)}
        fake.execute_action = measured
        result = await self.run_plan(fake, ['right', 'right', 'forward', 'forward', 'stop'])
        self.assertEqual(len(fake.plan_history), 1)
        self.assertEqual(result['planned_actions'], ['right', 'right', 'forward', 'forward', 'stop'])
        self.assertAlmostEqual(result['end_pose'][2], -math.pi / 2)
        self.assertEqual(result['step_results'][-1]['start_pose'], list(fake.pose))
        self.assertEqual(result['discarded_actions'], ['forward', 'stop'])

    async def test_all_step_measurements_are_returned_to_next_plan(self):
        fake = FakeExecutor()
        action = fake.execute_action
        async def measure(command):
            start = list(fake.pose)
            record = await action(command)
            return {**record, 'start_pose': start, 'end_pose': list(fake.pose)}
        fake.execute_action = measure
        result = await self.run_plan(fake, ['left', 'forward', 'right'])
        self.assertEqual(result['completed_actions'], 3)
        self.assertEqual(result['step_results'][1]['distance_m'], .25)
        self.assertAlmostEqual(result['step_results'][0]['turn_rad'], math.pi/4, places=4)

    async def test_stale_decision_or_changed_start_never_moves(self):
        fake = FakeExecutor()
        result = await self.run_plan(fake, ['forward', 'left'], age=91)
        self.assertEqual(fake.calls, [])
        self.assertEqual(result['outcome'], 'stale_decision')
        fake.pose = [0, 0, .5]
        result = await self.run_plan(fake, ['forward', 'left'])
        self.assertEqual(fake.calls, [])
        self.assertEqual(result['outcome'], 'trajectory_deviation')

    async def test_sensor_or_link_loss_before_motion(self):
        for name in ('sensors_ok', 'link_ok'):
            fake = FakeExecutor()
            setattr(fake, name, False)
            result = await self.run_plan(fake, ['forward', 'left'])
            self.assertEqual(fake.calls, [])
            self.assertEqual(result['outcome'], 'stale_sensors_or_control_link')

    async def test_sensor_loss_between_steps_aborts_sequence(self):
        fake = FakeExecutor()
        action = fake.execute_action
        async def lose_sensors(command):
            record = await action(command)
            fake.sensors_ok = False
            return record
        fake.execute_action = lose_sensors
        result = await self.run_plan(fake, ['left', 'forward'])
        self.assertEqual(fake.calls, ['left'])
        self.assertEqual(result['discarded_actions'], ['forward'])

    async def test_stop_requests_new_observation(self):
        fake = FakeExecutor()
        result = await self.run_plan(fake, ['left', 'stop'])
        self.assertEqual(result['outcome'], 'observation_requested')

    async def test_cancelled_execution_stops_and_records_interruption(self):
        fake = FakeExecutor()
        entered = asyncio.Event()
        async def wait_forever(_):
            fake.stop_pub.armed = True
            entered.set()
            await asyncio.Event().wait()
        fake.execute_action = wait_forever
        task = asyncio.create_task(self.run_plan(fake, ['forward', 'left']))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(fake.stop_pub.armed)
        self.assertGreater(fake.stop_count, 0)
        self.assertEqual(fake.last_plan_outcome['outcome'], 'interrupted')
        self.assertEqual(fake.last_plan_outcome['discarded_actions'], ['forward', 'left'])

    async def test_invalid_plan_is_rejected_before_arming(self):
        fake = FakeExecutor()
        reply = response(['forward', 'stop', 'left'])
        with self.assertRaises(ValueError):
            await ActionPlanExecutor.execute_plan(fake, reply, observation(), time.monotonic())
        self.assertEqual(fake.calls, [])
        self.assertFalse(fake.stop_pub.armed)

    async def test_executor_requests_inference_only_after_whole_sequence(self):
        fake = FakeExecutor()
        async def execute_plan(*_):
            self.assertTrue(fake.executing_sequence)
            self.assertFalse(fake.need_plan.is_set())
            for _ in range(3):
                await asyncio.sleep(0)
                self.assertFalse(fake.need_plan.is_set())
            fake.finished.set()
        fake.execute_plan = execute_plan
        await fake.pending.put((response(['left', 'forward', 'right']), observation(), time.monotonic()))
        await ActionPlanExecutor.nav2_executor(fake)
        self.assertTrue(fake.need_plan.is_set())
        self.assertFalse(fake.executing_sequence)

    async def test_socket_validates_sequence_and_snapshot_before_execution(self):
        from robot_sim.candidate_nav.model_transport import request_model
        reply = response(['left', 'forward'])
        async def handle(reader, writer):
            await reader.readline()
            writer.write((json.dumps(reply) + '\n').encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        server = await asyncio.start_server(handle, '127.0.0.1', 0)
        try:
            port = server.sockets[0].getsockname()[1]
            self.assertEqual((await request_model(observation(), port))['actions'], ['left', 'forward'])
            reply['snapshot_id'] = 8
            with self.assertRaises(ValueError):
                await request_model(observation(), port)
            reply['snapshot_id'] = 7
            reply['actions'] = ['forward'] * 6
            with self.assertRaises(ValueError):
                await request_model(observation(), port)
        finally:
            server.close()
            await server.wait_closed()



class ContinuousHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_same_direction_translations_receive_handoff(self):
        class Continuous(FakeExecutor):
            enable_continuous_steps=True
            def __init__(self):super().__init__();self.flags=[]
            async def execute_action(self,action):
                self.flags.append(self.continuous_handoff)
                return await super().execute_action(action)
        fake=Continuous()
        await ActionPlanExecutor.execute_plan(fake,response(['forward','forward','left','forward','stop']),observation(),time.monotonic())
        self.assertEqual(fake.flags,[True,False,False,False,False])
        self.assertFalse(fake.continuous_handoff)
    async def test_missing_pose_is_not_reported_as_geometric_deviation(self):
        fake=FakeExecutor();fake.pose=None
        await ActionPlanExecutor.execute_plan(fake,response(['forward']),observation(),time.monotonic())
        self.assertEqual(fake.last_plan_outcome['outcome'],'pose_unavailable')
        self.assertEqual(fake.calls,[])

if __name__ == '__main__':
    unittest.main()
