"""Durability, retrieval, provenance and fail-open tests; no ROS/GPU/motors."""

import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from robot_sim.candidate_nav.long_term_memory import (
    EpisodeMemory,
    MemoryCoordinator,
    dumps,
)
from robot_sim.candidate_nav.memory_scope import MemoryScope


def packet(snapshot=1, **scope):
    return dict(
        protocol="action_trajectory_v1",
        physical=True,
        rover_id=107,
        episode_id="test-episode",
        snapshot_id=snapshot,
        instruction="Find the metal bucket.",
        pose=[0, 0, 0],
        sim_time=snapshot,
        camera_stamp=snapshot,
        memory_context=dict(
            version=1,
            enabled=True,
            tracking_valid=True,
            map_session_id="test-map",
            anchor_epoch="first",
            replay=True,
            **scope
        ),
        task_context=dict(recent_actions=[]),
    )


def motion(
    frame=1, label="A", description="Metal bucket beside a white appliance.", **changes
):
    p = packet(frame)
    p.update(protocol="motion_visual_memory_v1", **changes)
    p["frames"] = [dict(frame_id=frame, stamp=frame, pose=[0.3, 0, 0], plan_id=1)]
    r = dict(
        events=[dict(frame=p["frames"][0], description=description)],
        probes=[dict(frame_id=frame, event=label)],
    )
    return p, r


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = EpisodeMemory(self.temp.name)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def evidence(self, p=None):
        p = p or packet()
        return self.store.prepare(p, lambda s: len(s) // 4)["long_term_memory"][
            "evidence"
        ]

    def test_appearance_survives_disappearance_and_hundreds_of_new_events(self):
        p, r = motion()
        self.store.record_reply(p, r)
        p, r = motion(2, "D", "Bucket left the camera frame.")
        self.store.record_reply(p, r)
        for i in range(3, 220):
            p, r = motion(i, "U", "Uncertain visual change.")
            self.store.record_reply(p, r)
        e = self.evidence()
        self.assertEqual(e[0]["kind"], "sighting")
        self.assertIn("Metal bucket", e[0]["report"])
        self.assertGreaterEqual(
            self.store.db.execute("SELECT count(*) FROM events").fetchone()[0], 219
        )

    def test_old_cross_task_lexical_retrieval_is_not_just_fixed_recent_window(self):
        p, r = motion(description="A distinctive volleyball under the chair.")
        self.store.record_reply(p, r)
        for i in range(2, 220):
            p, r = motion(i, "U", "Uncertain room.")
            self.store.record_reply(p, r)
        p = packet(300)
        p["instruction"] = "Find the volleyball."
        self.assertTrue(any("volleyball" in e["report"] for e in self.evidence(p)))

    def test_fallible_sightings_are_not_merged_into_an_invented_object_position(self):
        for i, text in [
            (1, "Maybe a bucket on the left."),
            (2, "Possibly a different bin on the right."),
        ]:
            p, r = motion(i, description=text)
            self.store.record_reply(p, r)
        e = self.evidence()
        self.assertEqual(len([x for x in e if x["kind"] == "sighting"]), 2)
        self.assertTrue(all(x["report_is_model_interpretation"] for x in e))
        self.assertTrue(all("object_position" not in x for x in e))

    def test_scope_changes_retain_semantics_but_hide_coordinates(self):
        p, r = motion()
        self.store.record_reply(p, r)
        for field, value in [
            ("episode_id", "new-run"),
            ("memory_context", dict(packet()["memory_context"], anchor_epoch="second")),
            (
                "memory_context",
                dict(packet()["memory_context"], map_session_id="new-map"),
            ),
            ("memory_context", dict(packet()["memory_context"], tracking_valid=False)),
        ]:
            q = packet()
            q[field] = value
            e = self.evidence(q)
            self.assertTrue(e)
            self.assertTrue(all("camera_viewpoint" not in x for x in e))
            self.assertTrue(
                all(x["spatial_status"] == "unverified_previous_scope" for x in e)
            )
        self.assertIn("camera_viewpoint", self.evidence()[0])

    def test_rover_and_replay_namespaces_never_mix(self):
        p, r = motion()
        self.store.record_reply(p, r)
        q = packet()
        q["rover_id"] = 80
        self.assertFalse(self.evidence(q))
        q = packet()
        q["memory_context"]["replay"] = False
        self.assertFalse(self.evidence(q))

    def test_queued_frame_uses_capture_epoch(self):
        p, r = motion()
        p["frames"][0]["memory_scope"] = dict(
            p["memory_context"], anchor_epoch="old-frame"
        )
        self.store.record_reply(p, r)
        self.assertNotIn("camera_viewpoint", self.evidence()[0])

    def test_retry_is_idempotent_and_restart_preserves_journal(self):
        p, r = motion()
        self.store.record_reply(p, r)
        self.store.record_reply(p, r)
        self.assertEqual(
            self.store.db.execute("SELECT count(*) FROM events").fetchone()[0], 1
        )
        self.store.close()
        self.store = EpisodeMemory(self.temp.name)
        self.assertTrue(self.evidence())
        self.assertEqual(
            self.store.db.execute("SELECT count(*) FROM journal").fetchone()[0], 2
        )

    def test_keyframe_persists_by_content_hash(self):
        import base64

        p, r = motion()
        p["frames"][0]["image"] = base64.b64encode(b"\x89PNG\r\n\x1a\nfixture").decode()
        self.store.record_reply(p, r)
        self.assertTrue(self.evidence()[0]["has_keyframe"])
        self.assertEqual(len(list(self.store.images.glob("*.png"))), 1)

    def test_only_measured_executed_step_endpoints_build_routes(self):
        p = packet()
        scope = p["memory_context"]
        plan = dict(
            plan_id=1,
            start_pose=[0, 0, 0],
            end_pose=[0.5, 0, 0],
            planned_actions=["forward"] * 10,
            discarded_actions=["forward"] * 8,
            step_results=[
                dict(
                    action="forward", outcome="duration_complete", end_pose=[0.25, 0, 0]
                ),
                dict(
                    action="forward",
                    outcome="lidar_brake",
                    start_pose=[0.25, 0, 0],
                    end_pose=[0.5, 0, 0],
                ),
            ],
            memory_scope=scope,
        )
        p["memory_update"] = dict(previous_plan=plan)
        self.store.prepare(p, lambda s: len(s) // 4)
        self.store.prepare(p, lambda s: len(s) // 4)
        self.assertEqual(
            self.store.db.execute("SELECT count(*) FROM places").fetchone()[0], 3
        )
        self.assertEqual(
            self.store.db.execute("SELECT sum(traversals) FROM edges").fetchone()[0], 2
        )
        self.assertEqual(
            self.store.db.execute(
                'SELECT count(*) FROM events WHERE kind="blocked_attempt"'
            ).fetchone()[0],
            1,
        )
        p = packet(2)
        p["memory_update"] = dict(
            previous_plan=dict(
                plan_id=2,
                planned_actions=["forward"] * 5,
                discarded_actions=["forward"] * 5,
            )
        )
        self.store.prepare(p, lambda s: len(s) // 4)
        self.assertEqual(
            self.store.db.execute("SELECT count(*) FROM edges").fetchone()[0], 2
        )

    def test_plan_across_tracking_reset_does_not_reanchor_old_motion(self):
        p = packet()
        p["memory_update"] = dict(
            previous_plan=dict(
                plan_id=1,
                start_pose=[0, 0, 0],
                step_results=[dict(end_pose=[2, 0, 0])],
                memory_scope=dict(p["memory_context"], tracking_valid=False),
            )
        )
        self.store.prepare(p, lambda s: len(s) // 4)
        self.assertEqual(
            self.store.db.execute("SELECT count(*) FROM edges").fetchone()[0], 0
        )
        self.assertIsNone(
            self.store.db.execute(
                'SELECT anchor FROM events WHERE kind="execution"'
            ).fetchone()[0]
        )

    def test_recorded_before_inference_and_prompt_compaction_is_not_data_loss(self):
        p = packet()
        p["motion_visual_memory"] = [dict(evidence=str(x)) for x in range(20)]
        p["task_context"]["recent_actions"] = [dict(outcome=str(x)) for x in range(30)]
        p["maneuver_memory"] = dict(recent_plans=list(range(8)))
        original = copy.deepcopy(p)
        q = self.store.prepare(p, lambda s: len(s) // 4)
        self.assertEqual(p, original)
        self.assertEqual(len(q["motion_visual_memory"]), 2)
        saved = json.loads(
            self.store.db.execute("SELECT packet FROM journal").fetchone()[0]
        )
        self.assertEqual(len(saved["motion_visual_memory"]), 20)
        self.assertEqual(len(saved["task_context"]["recent_actions"]), 30)
        self.assertIsNone(
            self.store.db.execute("SELECT reply FROM journal").fetchone()[0]
        )
        self.assertLessEqual(q["_memory_status"]["memory_tokens"], 1000)

    def test_disabled_and_simulator_requests_are_unchanged(self):
        p = packet()
        p["memory_context"]["enabled"] = False
        self.assertIs(self.store.prepare(p, len), p)
        p = packet()
        p["physical"] = False
        self.assertIs(self.store.prepare(p, len), p)


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.session = "map1"
        self.scope = MemoryScope(
            107, "run", identity=lambda: self.session, clock=lambda: self.now
        )

    def observe(self, t, p=None, valid=True):
        self.now = t
        self.scope.observe(t, p or [0, 0, 0], valid)
        return self.scope.export()

    def test_stale_and_invalid_odometry_revoke_spatial_anchor(self):
        old = self.observe(0)
        self.assertTrue(old["tracking_valid"])
        self.now = 2
        new = self.scope.export()
        self.assertFalse(new["tracking_valid"])
        self.assertNotEqual(new["anchor_epoch"], old["anchor_epoch"])
        self.assertFalse(self.observe(3, valid=False)["tracking_valid"])

    def test_process_restart_and_jump_change_epoch(self):
        old = self.observe(0)
        self.session = "map2"
        new = self.observe(3)
        self.assertNotEqual(old["map_session_id"], new["map_session_id"])
        self.assertNotEqual(old["anchor_epoch"], new["anchor_epoch"])
        jump = self.observe(3.1, [2, 0, 0])
        self.assertNotEqual(new["anchor_epoch"], jump["anchor_epoch"])

    def test_no_mapping_identity_means_semantic_memory_only(self):
        self.session = None
        self.assertFalse(self.observe(0)["tracking_valid"])

    def test_backwards_ros_time_creates_new_epoch(self):
        old = self.observe(2)
        new = self.observe(1)
        self.assertNotEqual(old["anchor_epoch"], new["anchor_epoch"])


class CoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_storage_failure_preserves_entire_original_history_and_does_not_raise(
        self,
    ):
        def fail(_):
            raise OSError("disk unavailable")

        c = MemoryCoordinator("/unused", len, factory=fail)
        p = packet()
        p["motion_visual_memory"] = list(range(20))
        q = await c.prepare(p)
        self.assertEqual(q["motion_visual_memory"], p["motion_visual_memory"])
        self.assertEqual(q["_memory_status"]["status"], "unavailable")
        await c.record(p, {})
        await c.close()

    async def test_client_cannot_supply_stale_memory_and_errors_are_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            c = MemoryCoordinator(d, lambda s: len(s) // 4)
            p = packet()
            p["long_term_memory"] = {"injected": True}
            q = await c.prepare(p)
            self.assertNotIn("injected", q["long_term_memory"])
            await c.record(p, dict(error="model fault"))
            self.assertIn(
                "model fault",
                c.store.db.execute("SELECT reply FROM journal").fetchone()[0],
            )
            self.assertEqual(q["_memory_status"]["extra_llm_calls"], 0)
            await c.close()


class AdmissionTests(unittest.TestCase):
    def test_memory_yields_to_fresh_sensors_without_mutating_record(self):
        from robot_sim.candidate_nav.prompt_state import budgeted_inputs

        class IDs:
            def __init__(self, n):
                self.n = n

            def numel(self):
                return self.n

        def build(text):
            return dict(input_ids=IDs(len(text)))

        p = dict(
            instruction="keep the complete task",
            sensors=dict(front_range=0.12),
            long_term_memory=dict(
                rules="historical",
                places=dict(huge="x" * 500),
                evidence=[dict(report="x" * 400) for _ in range(8)],
            ),
        )
        original = copy.deepcopy(p)
        result, stats = budgeted_inputs(build, p, True, 1000, 12288)
        self.assertLessEqual(result["input_ids"].numel(), 1000)
        self.assertTrue(stats["memory_reduced_for_context"])
        self.assertEqual(p, original)
        self.assertLess(stats["memory_evidence_included"], 8)

    def test_metadata_is_never_repeated_as_model_history(self):
        from robot_sim.candidate_nav.prompt_state import model_state

        p = dict(
            memory_context={"secret_scope": "metadata"},
            memory_update={"huge": "plan"},
            motion_visual_memory=[
                dict(memory_scope={"anchor": "metadata"}, evidence="visible")
            ],
        )
        packed = dumps(model_state(p))
        self.assertNotIn("metadata", packed)
        self.assertIn("visible", packed)


class SpatialCorrectionTests(unittest.TestCase):
    def test_accumulated_slam_correction_revokes_old_viewpoints(self):
        scope = MemoryScope(107, "run", identity=lambda: "map", clock=lambda: 0)
        scope.observe(0, [0, 0, 0], True)
        scope.observe_map_transform([0, 0, 0])
        first = scope.export()["anchor_epoch"]
        scope.observe_map_transform([0.04, 0, 0])
        self.assertEqual(scope.export()["anchor_epoch"], first)
        scope.observe_map_transform([0.09, 0, 0])
        self.assertNotEqual(scope.export()["anchor_epoch"], first)


class PlanScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_executor_attaches_capture_scope_and_revokes_cross_epoch_plan(self):
        import ast
        from types import SimpleNamespace

        path = Path(__file__).parents[1] / "candidate_nav/live_action_plan.py"
        tree = ast.parse(path.read_text())
        method = next(
            m
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            for m in node.body
            if isinstance(m, ast.AsyncFunctionDef) and m.name == "execute_plan"
        )

        class Executor:
            async def execute_plan(self, *args):
                self.last_plan_outcome = {"plan_id": 1}

        cls = ast.ClassDef(
            name="Captured",
            bases=[ast.Name(id="Executor", ctx=ast.Load())],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
        ns = {"Executor": Executor}
        exec(compile(module, str(path), "exec"), ns)
        for changed in (False, True):
            scopes = iter(
                [
                    dict(tracking_valid=True, anchor_epoch="one", map_session_id="map"),
                    dict(
                        tracking_valid=True,
                        anchor_epoch="two" if changed else "one",
                        map_session_id="map",
                    ),
                ]
            )
            agent = ns["Captured"]()
            agent.memory_scope = SimpleNamespace(export=lambda: next(scopes))
            await agent.execute_plan({}, {}, 0)
            self.assertEqual(
                agent.last_plan_outcome["memory_scope"]["tracking_valid"], not changed
            )


class InitialCameraTests(unittest.TestCase):
    def test_stationary_first_camera_report_survives_the_recent_window(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeMemory(directory)
            p = packet()
            store.record_reply(p, dict(observer="A metal container under the shelf."))
            with store.db:
                ident = store.identity(p)
                for i in range(250):
                    store.event(
                        ident,
                        "visual_change",
                        "Uncertain view.",
                        i,
                        key="noise" + str(i),
                    )
            selected = store.prepare(packet(300), lambda text: len(text) // 4)[
                "long_term_memory"
            ]["evidence"]
            self.assertTrue(any("under the shelf" in e["report"] for e in selected))
            store.close()


class DispatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_rank_zero_broadcasts_prepared_packet_and_publishes_memory_with_early_reply(
        self,
    ):
        """Execute the actual socket dispatcher with CPU model/TP stand-ins."""
        import ast
        from types import SimpleNamespace
        from unittest.mock import patch
        import os

        source = Path(__file__).parents[1] / "candidate_nav/llm.py"
        tree = ast.parse(source.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "run"
        )
        broadcasts = []
        ready = asyncio.Event()
        servers = []
        policies = []

        class CPUModel:
            def __init__(self, args):
                self.rank = 0
                self.world = 2
                self.metadata = {}
                self.closed = False
                self.tokens = SimpleNamespace(
                    encode=lambda text, **kw: list(range(len(text) // 4))
                )
                self.llm = SimpleNamespace(engine=SimpleNamespace(tp_cpu_group=None))
                policies.append(self)

            async def decide(self, p, on_selection):
                self.packet = p
                reply = dict(
                    snapshot_id=p["snapshot_id"],
                    map_version=p["map_version"],
                    observer="A container visible.",
                )
                on_selection(reply)
                return reply

            async def close(self):
                self.closed = True

        class AsyncProxy:
            def __getattr__(self, name):
                return getattr(asyncio, name)

            async def start_server(self, *args, **kw):
                server = await asyncio.start_server(*args, **kw)
                servers.append(server)
                ready.set()
                return server

        ns = dict(
            torch=SimpleNamespace(set_num_threads=lambda n: None),
            asyncio=AsyncProxy(),
            AsyncCandidatePolicy=CPUModel,
            MemoryCoordinator=MemoryCoordinator,
            json=json,
            os=os,
            dist=SimpleNamespace(
                broadcast_object_list=lambda payload, **kw: broadcasts.append(
                    json.loads(json.dumps(payload))
                )
            ),
            time=__import__("time"),
            __file__=str(source),
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"ROBOT_LLM_MEMORY_DIR": directory}
        ):
            job = asyncio.create_task(
                ns["run"](SimpleNamespace(hardware_profile="h100", port=0))
            )
            try:
                await asyncio.wait_for(ready.wait(), 2)
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", servers[0].sockets[0].getsockname()[1]
                )
                p = packet()
                p["map_version"] = 1
                p["long_term_memory"] = {"stale_client_fact": True}
                writer.write((json.dumps(p) + "\n").encode())
                await writer.drain()
                reply = json.loads(await asyncio.wait_for(reader.readline(), 2))
                writer.close()
                await writer.wait_closed()
                self.assertEqual(reply["memory_status"]["status"], "ready")
                self.assertEqual(reply["memory_status"]["extra_llm_calls"], 0)
                self.assertEqual(broadcasts[0][0], policies[0].packet)
                self.assertIn("long_term_memory", broadcasts[0][0])
                self.assertNotIn(
                    "stale_client_fact", broadcasts[0][0]["long_term_memory"]
                )
            finally:
                job.cancel()
                await asyncio.gather(job, return_exceptions=True)
            self.assertTrue(policies[0].closed)


class PlanarOdometryTests(unittest.TestCase):
    def test_unused_3d_covariance_does_not_disable_planar_memory(self):
        import ast, math
        from types import SimpleNamespace as NS

        source = Path(__file__).parents[1] / "candidate_nav/live_mapping.py"
        tree = ast.parse(source.read_text())
        method = next(
            m
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            for m in node.body
            if isinstance(m, ast.FunctionDef) and m.name == "receive"
        )

        class Base:
            def receive(self, name, msg):
                pass

        cls = ast.ClassDef(
            name="Mapped",
            bases=[ast.Name(id="Base", ctx=ast.Load())],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        ns = {"Base": Base, "math": math, "msg_time": lambda msg: 1, "yaw": lambda q: 0}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                str(source),
                "exec",
            ),
            ns,
        )
        observed = []
        agent = ns["Mapped"]()
        agent.memory_scope = NS(
            observe=lambda stamp, pose, valid: observed.append(valid)
        )
        cov = [0.0] * 36
        for i in (0, 7, 35):
            cov[i] = 0.001
        for i in (14, 21, 28):
            cov[i] = 9999.0
        msg = NS(
            pose=NS(
                pose=NS(position=NS(x=0, y=0), orientation=NS(x=0, y=0, z=0, w=1)),
                covariance=cov,
            )
        )
        agent.receive("odom", msg)
        self.assertTrue(observed[-1])
        cov[0] = 9999.0
        agent.receive("odom", msg)
        self.assertFalse(observed[-1])
