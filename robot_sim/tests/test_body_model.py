"""Body information reaches live planning without changing geometric protections."""
import math
import unittest
from robot_sim.candidate_nav.body_model import body_description, BODY_RULES
from robot_sim.candidate_nav.live_trajectory import HALF_LENGTH, HALF_WIDTH
from robot_sim.candidate_nav import task_policy


class BodyModelTests(unittest.TestCase):
    def test_operator_dimensions_are_separate_from_existing_planning_geometry(self):
        body = body_description()
        self.assertEqual(body['approximate_physical_size_m'], {'length': .20, 'width': .12})
        self.assertEqual(body['planning_footprint_m'], {'length': 2*HALF_LENGTH, 'width': 2*HALF_WIDTH})
        self.assertEqual(body['planning_footprint_m']['width'], .16)
        self.assertAlmostEqual(body['planning_corner_radius_m'], math.hypot(.10, .08), places=4)
        body['planning_half_extents_m']['rear'] = 999
        self.assertEqual(body_description()['planning_half_extents_m']['rear'], .10)

    def test_all_live_planning_modes_receive_body_rules(self):
        for system in (task_policy.WAYPOINT_SYSTEM, task_policy.DIRECT_SYSTEM,
                       task_policy.GENERATION_SYSTEM, task_policy.TRAJECTORY_SYSTEM,
                       task_policy.TRAJECTORY_EVIDENCE_SYSTEM):
            self.assertIn(BODY_RULES, system)


if __name__ == '__main__':
    unittest.main()
