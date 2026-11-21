import math
import unittest

import numpy as np

from robot_sim.candidate_nav.candidates import CandidateGenerator
from robot_sim.candidate_nav.geometry import Grid, reachable
from robot_sim.candidate_nav.proposals import proposal_map, unpixel, validate_proposals
from robot_sim.candidate_nav.render import pixel
from robot_sim.candidate_nav.scenes import SCENARIOS


class LLMProposalTests(unittest.TestCase):
    def test_decimal_grammar_covers_exactly_512_image_coordinates(self):
        from robot_sim.candidate_nav.llm import coordinate_digits

        prefixes = [""]
        for _ in range(3):
            prefixes = [
                prefix + digit for prefix in prefixes for digit in coordinate_digits(prefix)
            ]
        self.assertEqual([int(value) for value in prefixes], list(range(512)))

    def setUp(self):
        values = np.zeros((80, 100), np.int16)
        values[:, 48:52] = 100
        values[:8] = -1
        self.grid = Grid(values, 0.1, (-5.0, -4.0), yaw=0.2)
        self.pose = (*self.grid.world((20, 40)), 0.2)
        self.reach = reachable(self.grid, self.pose, radius=0.13)
        _, self.facts = proposal_map(self.reach, self.pose)
        self.p = self.facts["projection"]

    def item(self, cell, heading=0):
        u, v = pixel(self.grid.world(cell), self.p)
        return {"u": round(u), "v": round(v), "heading": heading, "kind": "frontier"}

    def check(self, raw):
        return validate_proposals(raw, self.reach, self.p, self.pose, CandidateGenerator())

    def test_rotated_projection_roundtrip(self):
        for xy in ((-3.0, 1.0), (1.0, 2.0)):
            np.testing.assert_allclose(unpixel(pixel(xy, self.p), self.p), xy)

    def test_generated_coordinate_is_reachable_and_snap_is_bounded(self):
        candidates, checks = self.check([self.item((30, 40))])
        self.assertEqual(len(candidates), 1)
        self.assertTrue(self.reach.path((candidates[0].x, candidates[0].y)))
        self.assertLessEqual(checks[0]["snap_m"], 0.35)
        self.assertAlmostEqual(candidates[0].yaw, 0.2)

    def test_wall_unknown_and_disconnected_goals_are_rejected(self):
        candidates, checks = self.check(
            [self.item((50, 40)), self.item((20, 3)), self.item((70, 40))]
        )
        self.assertFalse(candidates)
        self.assertTrue(all(not c["accepted"] for c in checks))

    def test_empty_or_bad_proposals_never_invoke_geometric_fallback(self):
        self.assertEqual(self.check([]), ([], []))
        for raw in ({"u": float("nan")}, {"u": True, "v": 200, "heading": 0, "kind": "look"}):
            self.assertFalse(self.check([raw])[0])

    def test_duplicates_rejected_and_rotation_allowed(self):
        point = self.item((30, 40))
        candidates, _ = self.check([point, point])
        self.assertEqual(len(candidates), 1)
        look = self.item((20, 40), heading=2)
        look["kind"] = "look"
        candidates, _ = self.check([look])
        self.assertEqual(candidates[0].kind, "look")
        self.assertAlmostEqual(candidates[0].yaw, 0.2 - math.pi / 2)


class ExperimentSceneTests(unittest.TestCase):
    def test_new_scene_targets_are_physically_reachable(self):
        from robot_sim.candidate_nav.comparison import reference_distance

        for name in ("yellow_zigzag", "red_deadends", "blue_clutter"):
            distance = reference_distance(SCENARIOS[name])
            self.assertTrue(math.isfinite(distance), name)
            self.assertGreater(distance, SCENARIOS[name]["minimum_path_m"], name)


if __name__ == "__main__":
    unittest.main()
