"""Direction protocol boundaries and live lidar stop behavior."""

import asyncio
import base64
import io
import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from robot_sim.candidate_nav.direct_control import (
    camera_clock_ready,
    scan_clearance,
    signed_turn,
    validate_response,
)


def response(label="F", action="forward", complete=False):
    return {
        "protocol": "direct_navigation_v1",
        "goal_id": label,
        "action": action,
        "task_complete": complete,
        "snapshot_id": 1,
        "map_version": 3,
    }


class SafetyTests(unittest.TestCase):
    def test_camera_before_ros_clock_cannot_start_inference(self):
        self.assertFalse(camera_clock_ready(0, 6.2))
        self.assertFalse(camera_clock_ready(7, 7.2))
        self.assertFalse(camera_clock_ready(8.5, 7.2))
        self.assertFalse(camera_clock_ready(math.nan, 7.2))
        self.assertTrue(camera_clock_ready(7.3, 7.2))

    def scan(self, ranges, angle_min=-math.pi / 2, increment=math.pi / 2):
        return SimpleNamespace(
            ranges=ranges,
            angle_min=angle_min,
            angle_increment=increment,
            range_min=0.12,
            range_max=6,
        )

    def test_missing_lidar_coverage_cannot_drive(self):
        self.assertEqual(
            scan_clearance(self.scan([math.nan] * 3), "forward"), "missing_scan_coverage"
        )
        self.assertEqual(
            scan_clearance(self.scan([2, 2], 1, 0.1), "forward"), "missing_scan_coverage"
        )

    def test_forward_brakes_without_changing_direction(self):
        self.assertEqual(scan_clearance(self.scan([4, 0.3, 4]), "forward"), "lidar_brake")
        self.assertIsNone(scan_clearance(self.scan([0.2, 4, 0.2]), "forward"))

    def test_rotational_footprint_and_open_scan(self):
        self.assertEqual(scan_clearance(self.scan([0.14, 4, 4]), "left"), "lidar_brake")
        self.assertEqual(scan_clearance(self.scan([4, 4, 0.14]), "right"), "lidar_brake")
        self.assertIsNone(scan_clearance(self.scan([math.inf] * 3), "forward"))

    def test_front_corner_obstacle_outside_narrow_cone_stops_forward(self):
        scan = self.scan(
            [math.hypot(0.12, 0.09), 4], math.atan2(0.09, 0.12), -math.atan2(0.09, 0.12)
        )
        self.assertEqual(scan_clearance(scan, "forward"), "lidar_brake")

    def test_relative_turn_crosses_pi_correctly(self):
        self.assertAlmostEqual(signed_turn(3, -2.5, "left"), 2 * math.pi - 5.5)
        self.assertAlmostEqual(signed_turn(-3, 2.5, "right"), 2 * math.pi - 5.5)
        self.assertLess(signed_turn(0, -0.5, "left"), 0)

    def test_completion_needs_stop_and_live_close_evidence(self):
        for objects in (
            [],
            [{"visible": False, "range_m": 0.4}],
            [{"visible": True, "range_m": 1.2}],
        ):
            with self.assertRaises(ValueError):
                validate_response(response("Z", "stop", True), {"objects": objects})
        validate_response(
            response("Z", "stop", True), {"objects": [{"visible": True, "range_m": 0.7}]}
        )

    def test_unknown_or_inconsistent_actions_cannot_execute(self):
        for item in (
            response("F", "reverse"),
            response("L", "forward"),
            response("F", "forward", True),
            response("Z", "stop", False),
        ):
            with self.assertRaises(ValueError):
                validate_response(item, {"objects": []})


class DirectProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def check(self, result):
        from robot_sim.candidate_nav.model_transport import request_model

        async def handle(reader, writer):
            await reader.readline()
            writer.write((json.dumps(result) + "\n").encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        try:
            return await request_model(
                {
                    "protocol": "direct_navigation_v1",
                    "snapshot_id": 1,
                    "map_version": 3,
                    "objects": [],
                    "candidates": [],
                },
                server.sockets[0].getsockname()[1],
            )
        finally:
            server.close()
            await server.wait_closed()

    async def test_direct_action_does_not_require_waypoints(self):
        self.assertEqual((await self.check(response()))["action"], "forward")

    async def test_wrong_snapshot_or_map_is_rejected(self):
        for key in ("snapshot_id", "map_version"):
            item = response()
            item[key] += 1
            with self.assertRaises(ValueError):
                await self.check(item)


class SharedCachePolicyTests(unittest.IsolatedAsyncioTestCase):
    """Exercise real coroutine/selection logic with a CPU fake engine, not model quality."""

    async def policy_decision(self, selected="R", complete=False):
        import torch
        from PIL import Image

        from robot_sim.candidate_nav.llm import AsyncCandidatePolicy

        class Tokens:
            eos_token_id = 0

            def __init__(self):
                self.words = {c: ord(c) for c in "FLRSZ"}
                self.words.update(
                    {"yes": 100, "Yes": 101, "YES": 102, "no": 103, "No": 104, "NO": 105}
                )
                self.prompts = {}

            def encode(self, text, **_):
                if text in self.words:
                    return [self.words[text]]
                key = 200 + len(self.prompts)
                self.prompts[key] = text
                return [key]

            def decode(self, tokens, **_):
                return "CPU test trace"

        class Engine:
            def __init__(self):
                self.tokenizer = Tokens()
                self.processor = self
                self.engine = SimpleNamespace(tp_cpu_group=None)
                self.blocks, self.freed = [], []

            def apply_chat_template(self, messages, **_):
                self.messages = messages
                return {"input_ids": torch.tensor([[1, 2]])}

            async def create_block(self):
                block = len(self.blocks)
                self.blocks.append(block)
                return block

            async def forward(self, tokens=None, cache_view=None, **_):
                await asyncio.sleep(0)
                scores = torch.full((512,), -10.0)
                token = 65  # Continue both analysis streams for their full budget.
                if cache_view and cache_view[-1] == 3:
                    prompt = "".join(self.tokenizer.prompts.get(t, "") for t in (tokens or []))
                    token = (
                        (101 if complete else 104)
                        if "Answer exactly Yes or No" in prompt
                        else ord(selected)
                    )
                scores[token] = 10
                return SimpleNamespace(logits=scores)

            async def free_block(self, block):
                self.freed.append(block)

        engine = Engine()
        policy = AsyncCandidatePolicy.__new__(AsyncCandidatePolicy)
        policy.llm, policy.tokens = engine, engine.tokenizer
        policy.args = SimpleNamespace(analysis_tokens=24)
        policy.rank, policy.world, policy.metadata = 0, 1, {"test_only": True}
        stream = io.BytesIO()
        Image.new("RGB", (16, 16)).save(stream, format="PNG")
        encoded = base64.b64encode(stream.getvalue()).decode()
        observation = {
            "protocol": "direct_navigation_v1",
            "snapshot_id": 1,
            "map_version": 3,
            "episode_id": "CPU_LOGIC_TEST",
            "camera": encoded,
            "map_image": encoded,
            "candidates": [],
            "objects": [{"visible": True, "range_m": 0.7}] if complete else [],
        }
        early = []

        def publish(item):
            early.append(dict(item))

        def gather(peaks, value, **_):
            peaks[0] = value

        with (
            patch("torch.cuda.max_memory_allocated", return_value=0),
            patch("torch.distributed.all_gather_object", side_effect=gather),
            patch("builtins.print"),
        ):
            result = await policy.decide(observation, publish)
        self.assertEqual(sorted(engine.freed), [0, 1, 2, 3])
        self.assertTrue(early[0]["analysis_in_progress"])
        self.assertFalse(result["analysis_in_progress"])
        return result

    async def test_direction_selector_runs_concurrently_without_candidates(self):
        item = await self.policy_decision()
        self.assertEqual(item["action"], "right")
        self.assertFalse(item["task_complete"])

    async def test_semantic_completion_is_published_as_stop(self):
        item = await self.policy_decision(complete=True)
        self.assertEqual(item["action"], "stop")
        self.assertTrue(item["task_complete"])


if __name__ == "__main__":
    unittest.main()
