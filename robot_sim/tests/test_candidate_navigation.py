import math
import unittest

import numpy as np

from robot_sim.candidate_nav.candidates import CandidateGenerator
from robot_sim.candidate_nav.geometry import Grid, fuse_observations, overlay_obstacles, reachable
from robot_sim.mapping.core import LocalGrid


class CandidateNavigationTests(unittest.TestCase):
    def test_unknown_and_disconnected_free_space_are_unreachable(self):
        values = np.zeros((30, 40), np.int16)
        values[:, 19:21] = 100
        grid = Grid(values, 0.1, (0.0, 0.0))
        reach = reachable(grid, (0.6, 1.5), radius=0.1)
        self.assertTrue(reach.path((1.5, 1.5)))
        self.assertFalse(reach.path((3.0, 1.5)))
        values[:, 19:21] = -1
        self.assertFalse(reachable(grid, (0.6, 1.5), radius=0.1).path((3.0, 1.5)))

    def test_entire_robot_must_fit_through_passage(self):
        values = np.zeros((40, 40), np.int16)
        values[:, 19:21] = 100
        values[19:21, 19:21] = 0
        grid = Grid(values, 0.1, (0.0, 0.0))
        self.assertFalse(reachable(grid, (1.0, 2.0), radius=0.16).path((3.0, 2.0)))
        values[16:24, 19:21] = 0
        self.assertTrue(reachable(grid, (1.0, 2.0), radius=0.16).path((3.0, 2.0)))

    def test_diagonal_cannot_cut_obstacle_corners(self):
        values = np.array([[0, 100], [100, 0]], np.int16)
        reach = reachable(Grid(values, 1.0, (0.0, 0.0)), (0.5, 0.5), radius=0)
        self.assertFalse(reach.path((1.5, 1.5)))

    def test_rotated_grid_coordinate_roundtrip(self):
        grid = Grid(np.zeros((20, 30)), 0.04, (-2.0, 3.0), yaw=math.pi / 3)
        for cell in ((0, 0), (10, 15), (29, 19)):
            self.assertEqual(grid.cell(grid.world(cell)), cell)

    def test_frontier_goals_have_known_free_collision_checked_paths(self):
        values = np.full((60, 80), -1, np.int16)
        values[5:55, 5:45] = 0
        values[10:30, 25:29] = 100
        grid = Grid(values, 0.1, (0.0, 0.0), version=17)
        reach = reachable(grid, (1.0, 3.0), radius=0.15)
        candidates = CandidateGenerator().generate(reach, (1.0, 3.0, 0.0))
        self.assertGreater(len(candidates), 0)
        for candidate in candidates:
            self.assertTrue(reach.path((candidate.x, candidate.y)))
            for point in reach.path((candidate.x, candidate.y)):
                x, y = grid.cell(point)
                self.assertLessEqual(values[y, x], 35)
                self.assertGreaterEqual(values[y, x], 0)
            self.assertGreaterEqual(candidate.clearance_m, 0.15)

    def test_failed_frontiers_are_suppressed(self):
        values = np.full((60, 80), -1, np.int16)
        values[5:55, 5:45] = 0
        reach = reachable(Grid(values, 0.1, (0.0, 0.0)), (1.0, 3.0), radius=0.15)
        generator = CandidateGenerator(separation=1.0)
        first = generator.generate(reach, (1.0, 3.0, 0.0))[0]
        generator.record(first, "failed")
        generator.record(first, "failed")
        for candidate in generator.generate(reach, (1.0, 3.0, 0.0)):
            self.assertGreaterEqual(math.dist((first.x, first.y), (candidate.x, candidate.y)), 1.0)

    def test_dynamic_obstacles_are_transformed_before_overlay(self):
        grid = Grid(np.zeros((40, 40)), 0.1, (0.0, 0.0))
        dynamic = Grid(np.array([[100]], np.int16), 0.1, (0.0, 0.0), frame="odom")
        blocked = overlay_obstacles(grid, dynamic, (1.0, 2.0, 0.0))
        self.assertTrue(blocked[20, 10])
        self.assertEqual(blocked.sum(), 1)

    def test_recent_lidar_free_space_fills_only_slam_unknown(self):
        slam = Grid(np.full((20, 20), -1, np.int16), 0.1, (0.0, 0.0), version=7)
        slam.values[10, 10] = 100
        local = Grid(np.zeros((10, 10), np.int16), 0.1, (0.0, 0.0), frame="odom")
        local.values[5, 5] = 100
        fused = fuse_observations(slam, local, (0.5, 0.5, 0.0))
        self.assertEqual(fused.values[7, 7], 0)
        self.assertEqual(fused.values[10, 10], 100)
        self.assertEqual(fused.values[10, 10], slam.values[10, 10])
        self.assertEqual(fused.version, 7)

    def test_lidar_fusion_makes_sparse_slam_start_traversable(self):
        slam = Grid(np.full((50, 50), -1, np.int16), 0.1, (-2.5, -2.5))
        slam.values[25, 25] = 0
        local = Grid(np.full((30, 30), -1, np.int16), 0.1, (-1.5, -1.5), frame="odom")
        local.values[5:25, 5:25] = 0
        fused = fuse_observations(slam, local, (0.0, 0.0, 0.0))
        self.assertTrue(reachable(fused, (0.05, 0.05), radius=0.13).path((0.65, 0.05)))

    def test_rolling_grid_preserves_measurements_in_world_coordinates(self):
        grid = LocalGrid(0.1, 4.0)
        grid.update((0, 0), ((1.0, 0),), (True,), 1.0)
        for center in ((1.5, 0.0), (-0.2, 1.6), (1.3, -0.1)):
            grid.recenter(center)
            cell, valid = grid.index(np.array([[1.0, 0.0]]))
            self.assertTrue(valid[0])
            x, y = cell[0]
            self.assertGreaterEqual(grid.occupancy(1.0)[y, x], 65)

    def test_rolling_grid_new_area_is_unknown(self):
        grid = LocalGrid(0.1, 4.0)
        grid.seen[:] = 1.0
        grid.odds[:] = -3.0
        grid.recenter((2.0, 0.0))
        self.assertTrue(np.all(grid.occupancy(1.0)[:, 20:] == -1))
        grid.recenter((20.0, 0.0))
        self.assertTrue(np.all(grid.occupancy(1.0) == -1))

    def test_boundary_is_not_treated_as_infinite_free_space(self):
        grid = Grid(np.zeros((10, 10)), 0.1, (0.0, 0.0))
        reach = reachable(grid, (0.5, 0.5), radius=0.2)
        self.assertFalse(reach.path((0.05, 0.05)))
        self.assertTrue(reach.path((0.65, 0.65)))


class ObjectProposalTests(unittest.TestCase):
    def test_foreground_wall_inside_bbox_cannot_supply_object_range(self):
        from robot_sim.candidate_nav.geometry import colored_returns

        # The colored object's actual silhouette is an L. All four returns
        # fall inside its bounding rectangle; only one hits colored interior.
        mask = np.zeros((30, 30), np.uint8)
        mask[5:25, 5:12] = 1
        mask[18:25, 5:25] = 1
        u = np.array([8.0, 20.0, 5.0, np.nan, -4.0])
        v = np.array([10.0, 10.0, 5.0, 8.0, 10.0])
        result = colored_returns(u, v, np.ones(5, bool), mask)
        np.testing.assert_array_equal(result, [True, False, False, False, False])

    def test_visible_object_beyond_known_patch_gets_reachable_progress_goal(self):
        values = np.full((40, 60), -1, np.int16)
        values[5:35, 5:25] = 0
        grid = Grid(values, 0.1, (0.0, 0.0))
        reach = reachable(grid, (1.0, 2.0), radius=0.13)
        candidates = CandidateGenerator().generate(
            reach,
            (1.0, 2.0, 0.0),
            [{"xy": [4.0, 2.0], "description": "blue region; class unconfirmed"}],
        )
        inspection = [c for c in candidates if c.kind == "inspect"]
        self.assertTrue(inspection)
        self.assertTrue(all(c.x < 2.5 and reach.path((c.x, c.y)) for c in inspection))

    def test_close_remembered_object_gets_rotation_recovery(self):
        values = np.zeros((50, 50), np.int16)
        grid = Grid(values, 0.1, (0.0, 0.0))
        reach = reachable(grid, (2.5, 2.5), radius=0.13)
        candidates = CandidateGenerator().generate(
            reach,
            (2.5, 2.5, 0.0),
            [
                {
                    "xy": [2.5, 4.0],
                    "range_m": 1.5,
                    "visible": False,
                    "description": "generic RGB region; class unconfirmed",
                }
            ],
        )
        recovery = [c for c in candidates if c.kind == "look"]
        self.assertTrue(recovery)
        self.assertAlmostEqual(recovery[0].x, 2.5)
        self.assertAlmostEqual(recovery[0].y, 2.5)
        self.assertAlmostEqual(recovery[0].yaw, math.pi / 2)


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def check_response(self, response, objects=None):
        import asyncio
        import json

        from robot_sim.candidate_nav.model_transport import request_model

        async def handle(reader, writer):
            await reader.readline()
            writer.write((json.dumps(response) + "\n").encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            return await request_model(
                {
                    "snapshot_id": 1,
                    "map_version": 3,
                    "candidates": [{"id": "A"}],
                    "objects": objects or [],
                },
                port,
            )
        finally:
            server.close()
            await server.wait_closed()

    async def test_matching_candidate_response_is_accepted(self):
        result = await self.check_response({"snapshot_id": 1, "map_version": 3, "goal_id": "A"})
        self.assertEqual(result["goal_id"], "A")

    async def test_wrong_snapshot_cannot_execute(self):
        with self.assertRaises(ValueError):
            await self.check_response({"snapshot_id": 2, "map_version": 3, "goal_id": "A"})

    async def test_wrong_map_version_cannot_execute(self):
        with self.assertRaises(ValueError):
            await self.check_response({"snapshot_id": 1, "map_version": 4, "goal_id": "A"})

    async def test_hallucinated_candidate_is_rejected(self):
        with self.assertRaises(ValueError):
            await self.check_response({"snapshot_id": 1, "map_version": 3, "goal_id": "H"})

    async def test_old_or_far_object_cannot_allow_completion(self):
        for obj in ({"visible": False, "range_m": 0.5}, {"visible": True, "range_m": 2.0}):
            with self.assertRaises(ValueError):
                await self.check_response(
                    {"snapshot_id": 1, "map_version": 3, "goal_id": "Z"}, [obj]
                )


class SemanticFreshnessTests(unittest.TestCase):
    def test_old_goal_wakeup_cannot_prefetch_from_distant_new_goal(self):
        from robot_sim.candidate_nav.candidates import Candidate, planning_window_open

        active = Candidate("A", "frontier", 5.0, 0.0, 0.0, 5.0, 0.3)
        self.assertFalse(planning_window_open(active, (0.0, 0.0, 0.0)))
        self.assertTrue(planning_window_open(active, (4.0, 0.0, 0.0)))
        self.assertTrue(planning_window_open(active, (0.0, 0.0, 0.0), urgent=True))
        self.assertTrue(planning_window_open(None, (0.0, 0.0, 0.0)))

    def test_new_close_object_invalidates_a_prefetched_goal(self):
        from robot_sim.candidate_nav.geometry import new_nearby_evidence

        live = [{"color": "blue", "xy": [2.0, 1.0], "range_m": 0.7, "visible": True}]
        old = [{"color": "blue", "xy": [2.0, 1.0], "range_m": 1.5, "visible": True}]
        self.assertTrue(new_nearby_evidence(live, old))

    def test_explicit_choice_after_inspection_is_preserved(self):
        from robot_sim.candidate_nav.geometry import new_nearby_evidence

        live = [{"color": "red", "xy": [2.0, 1.0], "range_m": 0.7, "visible": True}]
        self.assertFalse(new_nearby_evidence(live, [dict(live[0])]))

    def test_distant_or_nonvisible_objects_do_not_interrupt_execution(self):
        from robot_sim.candidate_nav.geometry import new_nearby_evidence

        self.assertFalse(new_nearby_evidence([{"range_m": 2.0, "visible": True}], []))
        self.assertFalse(new_nearby_evidence([{"range_m": 0.5, "visible": False}], []))

    def test_information_gain_does_not_see_through_measured_wall(self):
        from robot_sim.candidate_nav.geometry import visible_unknown

        values = np.zeros((80, 80), np.int16)
        values[:, 41:] = -1
        values[:, 40] = 100
        grid = Grid(values, 0.1, (0.0, 0.0))
        self.assertEqual(visible_unknown(grid, (20, 40), radius=6.0), 0.0)
        values[:, 40] = 0
        self.assertGreater(visible_unknown(grid, (20, 40), radius=6.0), 0.0)


class BinaryLabelTests(unittest.TestCase):
    def test_yes_capitalization_is_not_misclassified_as_no(self):
        import torch

        from robot_sim.candidate_nav.llm import binary_probabilities

        class Tokenizer:
            def encode(self, text, **kwargs):
                return [{"yes": 0, "Yes": 1, "YES": 2, "no": 3, "No": 4, "NO": 5}[text]]

        logits = torch.tensor([-9.0, 8.0, -9.0, -6.0, -7.0, -9.0])
        probabilities, tokens = binary_probabilities(logits, Tokenizer())
        self.assertGreater(probabilities[0], 0.999)
        self.assertEqual(tokens[0], 1)


if __name__ == "__main__":
    unittest.main()
