"""Full-frame camera geometry and faithful execution evidence, without ROS/motors."""

import copy
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from robot_sim.candidate_nav.sensor_grounding import (
    camera_image,
    frame_size,
    sensor_context,
)


class CameraGeometryTests(unittest.TestCase):
    def test_widescreen_and_portrait_preserve_received_field_of_view(self):
        self.assertEqual(frame_size(1920, 1080), (320, 180))
        self.assertEqual(frame_size(640, 480), (320, 240))
        self.assertEqual(frame_size(1080, 1920), (135, 240))
        self.assertEqual(frame_size(160, 120), (160, 120))

    @unittest.skipUnless(importlib.util.find_spec("av"), "Optional PyAV codec dependency")
    def test_actual_decoder_keeps_borders_and_legacy_4by3_pixels(self):
        import av
        import numpy as np

        for width, height in [(640, 360), (640, 480)]:
            pixels = np.full((height, width, 3), 100, np.uint8)
            pixels[:12, :] = [255, 0, 0]
            pixels[-12:, :] = [0, 255, 0]
            pixels[:, :12] = [0, 0, 255]
            pixels[:, -12:] = [255, 255, 0]
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            image = camera_image(frame)
            out = np.asarray(image)
            self.assertGreater(out[0, out.shape[1] // 2, 0], 200)
            self.assertGreater(out[-1, out.shape[1] // 2, 1], 200)
            self.assertGreater(out[out.shape[0] // 2, 0, 2], 200)
            self.assertGreater(out[out.shape[0] // 2, -1, 0], 200)
            if height == 480:
                legacy = frame.reformat(
                    width=320, height=240, format="rgb24"
                ).to_ndarray()
                self.assertTrue(np.array_equal(out, legacy))
            else:
                self.assertEqual(image.size, (320, 180))

    @unittest.skipUnless(importlib.util.find_spec("av"), "Optional PyAV codec dependency")
    def test_rtsp_receiver_uses_full_frame_helper(self):
        import ast, math, time, threading
        import av
        import numpy as np

        source = Path(__file__).parents[1] / "candidate_nav/live.py"
        tree = ast.parse(source.read_text())
        method = next(
            m
            for c in tree.body
            if isinstance(c, ast.ClassDef) and c.name == "Camera"
            for m in c.body
            if isinstance(m, ast.FunctionDef) and m.name == "read"
        )
        stopped = threading.Event()
        frame = av.VideoFrame.from_ndarray(
            np.zeros((360, 640, 3), np.uint8), format="rgb24"
        )

        class Stream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def decode(self, **kwargs):
                yield frame
                stopped.set()

        ns = dict(
            av=SimpleNamespace(
                open=lambda *a, **kw: Stream(), FFmpegError=av.FFmpegError
            ),
            math=math,
            time=time,
            camera_image=camera_image,
        )
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), ns
        )
        receiver = SimpleNamespace(closed=stopped, latest=None, error=None)
        ns["read"](receiver)
        self.assertEqual(receiver.latest[0].size, (320, 180))
        self.assertIsNone(receiver.error)


class SensorProvenanceTests(unittest.TestCase):
    def observation(self, steps=None):
        return dict(
            action_spec={
                "forward": dict(max_distance_m=0.1),
                "backward": dict(max_distance_m=0.1),
                "left": dict(angle_rad=0.4),
                "stop": {},
            },
            previous_plan_outcome=dict(
                plan_id=7,
                step_results=steps or [],
                planned_actions=["forward"] * 5,
                discarded_actions=["forward"] * 5,
            ),
            sensors=dict(lidar_min_m=dict(front=0.35)),
        )

    def test_clear_lidar_does_not_erase_measured_undertravel_or_prove_its_cause(self):
        p = self.observation(
            [
                dict(
                    action="forward",
                    outcome="insufficient_progress_or_deviation",
                    distance_m=0.001,
                )
            ]
        )
        original = copy.deepcopy(p)
        e = sensor_context(p, (320, 240))
        step = e["execution"]["translation_attempts"][0]
        self.assertEqual(
            step["interpretation"], "undertravel_or_deviation_cause_unknown"
        )
        self.assertEqual(step["measured_displacement_m"], 0.001)
        self.assertEqual(step["nominal_target_m"], 0.1)
        self.assertFalse(e["lidar"]["sector_minimum_is_target_range"])
        self.assertEqual(p, original)
        self.assertNotIn("object_range", str(e))
        self.assertNotIn("collision_detected", str(e))

    def test_guards_and_no_translation_are_not_physical_contact_claims(self):
        for outcome in ("lidar_brake", "trajectory_collision"):
            e = sensor_context(
                self.observation(
                    [dict(action="backward", outcome=outcome, distance_m=0)]
                ),
                (320, 240),
            )
            self.assertEqual(
                e["execution"]["translation_attempts"][0]["interpretation"],
                "software_geometry_stop",
            )
        for steps in (
            [],
            [dict(action="left", outcome="duration_complete")],
            [dict(action="stop", outcome="duration_complete")],
        ):
            e = sensor_context(self.observation(steps), (320, 240))
            self.assertEqual(
                e["execution"]["last_plan_status"], "no_translation_attempted"
            )
            self.assertEqual(e["execution"]["translation_attempts"], [])

    def test_missing_range_and_missing_history_stay_unknown(self):
        self.assertEqual(
            sensor_context({}, (320, 240))["execution"]["last_plan_status"],
            "unavailable",
        )
        e = sensor_context(
            self.observation([dict(action="forward", outcome="interrupted")]),
            (320, 240),
        )
        self.assertIsNone(
            e["execution"]["translation_attempts"][0]["measured_displacement_m"]
        )
