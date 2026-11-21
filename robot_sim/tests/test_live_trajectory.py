"""Physical navigation regressions, without ROS or a motor publisher."""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from robot_sim.candidate_nav.live_trajectory import check_strategy4_translation, escape_course, escape_motion, trajectory_clearance


def scan(*points):
    angles = np.linspace(-math.pi, math.pi, 720, endpoint=False)
    ranges = np.full(720, np.inf)
    for x, y in points:
        angle = math.atan2(y, x)
        index = int(round((angle + math.pi) / (2 * math.pi) * 720)) % 720
        ranges[index] = math.hypot(x, y)
    return SimpleNamespace(ranges=ranges, angle_min=-math.pi,
                           angle_increment=2 * math.pi / 720, range_min=0.02, range_max=5.0)


class SweptChassis(unittest.TestCase):
    def test_close_side_point_allows_translation_but_blocks_turn_into_it(self):
        measurement = scan((0, -0.106))
        self.assertIsNotNone(trajectory_clearance(measurement, (0, 0), 0.10, 0, 0.55))
        self.assertIsNone(trajectory_clearance(measurement, (0, 0), 0, 0.8, 1.0))

    def test_front_blockage_has_reverse_escape(self):
        measurement = scan((0.15, 0))
        self.assertIsNone(trajectory_clearance(measurement, (0, 0), 0.10, 0, 0.55))
        self.assertEqual(escape_motion(measurement, (0, 0)), -0.06)

    def test_rear_blockage_has_forward_escape(self):
        self.assertEqual(escape_motion(scan((-0.15, 0)), (0, 0)), 0.06)

    def test_dead_end_does_not_blindly_reverse(self):
        self.assertIsNone(escape_motion(scan((0.15, 0), (-0.15, 0)), (0, 0)))

    def test_missing_rear_observations_prevent_reverse(self):
        measurement = scan((0.15, 0))
        measurement.ranges[:100] = np.nan
        measurement.ranges[-100:] = np.nan
        self.assertIsNone(escape_motion(measurement, (0, 0)))

    def test_leave_safety_margin_only_towards_greater_clearance(self):
        measurement = scan((0.104, 0))
        self.assertIsNone(trajectory_clearance(measurement, (0, 0), 0.06, 0, 0.55))
        self.assertIsNotNone(trajectory_clearance(measurement, (0, 0), -0.06, 0, 0.55))

    def test_shallow_corner_overlap_allows_only_short_exit(self):
        measurement = scan((.087, -.075))
        original = trajectory_clearance(measurement, (0, 0), -.06, 0, .55)
        self.assertIsNone(original)
        exit = check_strategy4_translation(measurement, (0, 0), -.06, 0, "creep_backward")
        self.assertEqual(exit.reason, "edge_exit")
        self.assertGreaterEqual(exit.end_gap_m, .0075)
        for speed, action in ((-.06, "backward"), (.10, "creep_forward")):
            self.assertIsNone(check_strategy4_translation(measurement, (0, 0), speed, 0, action).gap_m)

    def test_reduced_translation_margin_checks_the_entire_projection(self):
        for v, action in ((.10, 'forward'), (-.06, 'backward')):
            end = math.copysign(.10 + abs(v) * .55, v)
            for clearance, allowed in ((.004, True), (.001, False), (-.003, False)):
                point = end + math.copysign(1, v) * clearance
                measurement = scan((point, 0))
                with self.subTest(action=action, clearance=clearance):
                    result = check_strategy4_translation(measurement, (0,0), v, 0., action)
                    self.assertEqual(result.gap_m is not None, allowed)
                    self.assertIsNone(trajectory_clearance(measurement, (0,0), v, 0., .55))

    def test_egress_keeps_rear_coverage_and_obstacles(self):
        blocked = scan((.087, -.075), (-.13, .01))
        self.assertIsNone(check_strategy4_translation(blocked, (0, 0), -.06, 0, "creep_backward").gap_m)
        unknown = scan((.087, -.075))
        unknown.ranges[:100] = np.nan
        unknown.ranges[-100:] = np.nan
        self.assertEqual(check_strategy4_translation(unknown, (0, 0), -.06, 0, "creep_backward").reason,
                         "insufficient_lidar_coverage")
        deep = scan((.087, -.069))
        self.assertEqual(check_strategy4_translation(deep, (0, 0), -.06, 0, "creep_backward").reason,
                         "occupied_start")

    def test_translation_can_hold_clearance_along_the_side(self):
        parallel = scan((.03, -.084))
        self.assertIsNotNone(check_strategy4_translation(parallel, (0, 0), .10, 0, "forward").gap_m)

    def test_escape_arc_leaves_side_wall(self):
        measurement = scan(*[(float(x), -0.12) for x in np.linspace(-0.6, 0.6, 90)])
        course = escape_course(measurement, (0, 0))
        self.assertIsNotNone(course)
        speed, angular = course
        self.assertGreater(speed * angular, 0)
        self.assertIsNotNone(trajectory_clearance(measurement, (0, 0), speed, angular, 2.3))

    def test_actual_body_intersection_requires_inspection(self):
        self.assertIsNone(escape_motion(scan((0.05, 0)), (0, 0)))


if __name__ == '__main__':
    unittest.main()
