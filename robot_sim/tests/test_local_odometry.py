"""Exercise the actual primitive controller and scheduler under map corrections."""

import ast
import asyncio
import math
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from robot_sim.candidate_nav.action_plan import ACTION_PLAN_SPEC as SPEC, PROTOCOL
from robot_sim.candidate_nav.action_plan_executor import ActionPlanExecutor
from robot_sim.candidate_nav.local_odometry import LocalOdometry


class LocalStateTests(unittest.TestCase):
    def test_reset_gap_staleness_and_process_change_revoke_epoch(self):
        now = 0.0
        state = LocalOdometry(clock=lambda: now)
        state.observe(0, [0, 0, 0], session="icp1")
        original = state.epoch
        now = 0.1
        state.observe(0.1, [0.001, 0, 0], session="icp1")
        self.assertEqual(original, state.epoch)
        now = 0.2
        state.observe(0.2, [0.001, 0, 0], session="icp2")
        self.assertNotEqual(original, state.epoch)
        old = state.epoch
        now = 0.3
        state.observe(0.3, [2, 0, 0], session="icp2")
        self.assertNotEqual(old, state.epoch)
        now = 1.0
        self.assertFalse(state.snapshot()["valid"])
        self.assertLess(state.path_m, 0.01)

    def test_backward_time_and_invalid_message_cannot_preserve_reference(self):
        state = LocalOdometry()
        state.observe(3, [0, 0, 0])
        old = state.epoch
        state.observe(2, [0, 0, 0])
        self.assertNotEqual(state.epoch, old)
        state.observe(2.1, [0, 0, 0], frame="map")
        self.assertFalse(state.snapshot()["valid"])


class LocalControllerTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(
        self, actions, map_jump=0.0, freeze=False, opposite=False, reset=False
    ):
        source = Path(__file__).parents[1] / "candidate_nav/live_direct.py"
        method = next(
            m
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.ClassDef)
            for m in n.body
            if isinstance(m, ast.AsyncFunctionDef) and m.name == "execute_action"
        )

        class Twist:
            def __init__(self):
                self.linear = NS(x=0.0)
                self.angular = NS(z=0.0)

        clock = 0.0
        local = [0.0, 0.0, 0.6]
        command = Twist()
        published = []
        reset_done = False
        tracker = LocalOdometry(clock=lambda: clock)
        tracker.observe(0, local, session="icp")
        origin = tracker.snapshot()
        events = []

        def control_pose():
            sample = tracker.snapshot()
            return (
                sample["pose"]
                if sample["valid"] and sample["epoch"] == origin["epoch"]
                else None
            )

        def map_pose():
            shift = map_jump if clock > 0.5 else 0.0
            return [
                local[0] + 10 + shift,
                local[1] + 20 - shift,
                local[2] + (0.8 if shift else 0),
            ]

        def publish(value):
            nonlocal command
            command = value
            published.append(value.linear.x)

        def stop():
            nonlocal command
            command = Twist()

        async def sleep(dt):
            nonlocal clock, reset_done
            for _ in range(max(1, round(dt / 0.005))):
                velocity = 0 if freeze else command.linear.x * (-1 if opposite else 1)
                local[0] += velocity * math.cos(local[2]) * 0.005
                local[1] += velocity * math.sin(local[2]) * 0.005
                local[2] += command.angular.z * 0.005
                clock += 0.005
                if reset and clock > 0.5 and not reset_done:
                    local[0] += 2
                    reset_done = True
                tracker.observe(clock, local, session="icp")

        ns = dict(
            Twist=Twist,
            math=math,
            time=NS(monotonic=lambda: clock),
            asyncio=NS(sleep=sleep),
        )
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), ns
        )
        agent = NS(
            execution_frame="odom",
            get_control_pose=control_pose,
            get_pose=map_pose,
            local_odometry=tracker,
            action_spec=SPEC,
            closed_loop_targets=True,
            fresh=lambda: control_pose() is not None,
            control_link_ok=lambda: True,
            stop_pub=NS(
                armed=False, last_decision="allowed", publish=publish, stop=stop
            ),
            finished=asyncio.Event(),
            executed=0,
            completed=0,
            action_history=[],
            execution_totals=dict(
                actions=0, forward_m=0, signed_turn_rad=0, absolute_turn_rad=0
            ),
            log=lambda name, item: events.append(item),
        )

        async def execute(action, target_pose=None):
            return await ns["execute_action"](agent, action, target_pose)

        agent.execute_action = execute
        observation = dict(
            protocol=PROTOCOL,
            snapshot_id=1,
            map_version=1,
            pose=map_pose(),
            trajectory_steps=len(actions),
            control_reference=origin,
        )
        first = actions[0]
        codes = {
            v: k
            for k, v in __import__(
                "robot_sim.candidate_nav.action_plan", fromlist=["LABELS"]
            ).LABELS.items()
        }
        reply = dict(
            protocol=PROTOCOL,
            goal_id=codes[first],
            action=first,
            actions=actions,
            task_complete=False,
            trajectory_steps=len(actions),
        )
        await ActionPlanExecutor.execute_plan(
            agent, reply, observation, time.monotonic()
        )
        self.assertFalse(agent.stop_pub.armed)
        return agent, published

    async def test_large_map_correction_does_not_change_continuous_forward_targets(
        self,
    ):
        agent, commands = await self.run_case(
            ["forward", "forward", "stop"], map_jump=0.4
        )
        plan = agent.last_plan_outcome
        self.assertEqual(plan["completed_actions"], 3)
        self.assertAlmostEqual(agent.execution_totals["forward_m"], 0.2, delta=0.025)
        self.assertTrue(all(x >= 0 for x in commands))
        self.assertTrue(
            all(s.get("control_frame") == "odom" for s in plan["step_results"])
        )
        self.assertGreater(math.dist(plan["start_pose"][:2], plan["end_pose"][:2]), 0.4)

    async def test_reverse_progress_remains_reverse_despite_map_jump(self):
        agent, commands = await self.run_case(["backward", "stop"], map_jump=-0.4)
        self.assertEqual(agent.last_plan_outcome["completed_actions"], 2)
        self.assertAlmostEqual(agent.execution_totals["backward_m"], 0.1, delta=0.02)
        self.assertEqual(agent.execution_totals["forward_m"], 0)
        self.assertTrue(all(x <= 0 for x in commands))
        self.assertLess(
            agent.last_plan_outcome["step_results"][0]["signed_distance_m"], 0
        )

    async def test_odometry_reset_stops_remaining_sequence(self):
        agent, _ = await self.run_case(["forward", "forward"], reset=True)
        self.assertEqual(agent.last_plan_outcome["executed_actions"], 1)
        self.assertNotEqual(agent.last_plan_outcome["outcome"], "completed")
        self.assertEqual(agent.last_plan_outcome["discarded_actions"], ["forward"])

    async def test_no_progress_and_opposite_motion_do_not_count_as_forward(self):
        for frozen, opposite in [(True, False), (False, True)]:
            agent, _ = await self.run_case(
                ["forward", "forward"], freeze=frozen, opposite=opposite
            )
            self.assertEqual(agent.execution_totals["forward_m"], 0)
            self.assertEqual(
                agent.last_plan_outcome["outcome"], "insufficient_progress_or_deviation"
            )
            self.assertEqual(agent.last_plan_outcome["executed_actions"], 1)

    async def test_turn_and_forward_sequence_uses_local_heading_after_map_rotation(self):
        agent,_=await self.run_case(['forward','right','forward'],map_jump=.4)
        plan=agent.last_plan_outcome
        self.assertEqual(plan['completed_actions'],3)
        self.assertAlmostEqual(agent.execution_totals['forward_m'],.2,delta=.025)
        self.assertLess(plan['step_results'][1]['turn_rad'],0)
        self.assertAlmostEqual(plan['control_end_pose'][2],.6-math.pi/8,delta=.04)


class MapPoseGuardTests(unittest.TestCase):
    def test_map_corrections_are_not_odom_discontinuities_but_legacy_guard_remains(
        self,
    ):
        source = Path(__file__).parents[1] / "candidate_nav/live_mapping.py"
        tree = ast.parse(source.read_text())
        method = next(
            m
            for c in tree.body
            if isinstance(c, ast.ClassDef)
            for m in c.body
            if isinstance(m, ast.FunctionDef) and m.name == "get_pose"
        )

        class Base:
            def get_pose(self):
                return [5, 0, 0]

        cls = ast.ClassDef(
            name="Mapped",
            bases=[ast.Name(id="Base", ctx=ast.Load())],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        ns = dict(Base=Base, math=math, time=NS(monotonic=lambda: 1.0))
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                str(source),
                "exec",
            ),
            ns,
        )
        for frame in ("odom", "map"):
            finished = []
            agent = ns["Mapped"]()
            agent.execution_frame = frame
            agent.task_start_pose = [0, 0, 0]
            agent.monitor_pose = True
            agent.finished = asyncio.Event()
            agent.previous_pose = (0.99, [0, 0, 0])
            agent.log = lambda *args: None
            agent.now = lambda: 1.0
            agent.finish = lambda reason: finished.append(reason)
            value = agent.get_pose()
            self.assertEqual(finished, [] if frame == "odom" else ["pose_jump"])
            self.assertEqual(value, [5, 0, 0] if frame == "odom" else None)


class HeadingScopeTests(unittest.TestCase):
    def test_map_correction_expires_old_heading_without_resetting_local_control(self):
        source=Path(__file__).parents[1]/'candidate_nav/live_action_plan.py'
        tree=ast.parse(source.read_text());method=next(m for c in tree.body if isinstance(c,ast.ClassDef)
            for m in c.body if isinstance(m,ast.FunctionDef) and m.name=='memory_packet_fields')
        ns={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),ns)
        for epoch,valid,kept in [('one',True,True),('two',True,False),('one',False,False)]:
            local=object();scope=dict(anchor_epoch=epoch,tracking_valid=valid)
            agent=NS(args=NS(rover_id=107),output=NS(name='run'),local_odometry=local,
                memory_scope=NS(export=lambda:scope),escape_heading=dict(anchor_epoch='one',map_yaw_degrees=90))
            ns['memory_packet_fields'](agent)
            self.assertEqual(agent.escape_heading is not None,kept)
            self.assertIs(agent.local_odometry,local)
