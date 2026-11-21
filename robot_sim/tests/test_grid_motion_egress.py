"""Map geometry regressions for maneuvering out of shallow raster overlap."""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from robot_sim.candidate_nav.grid_motion import GridMotion
from robot_sim.candidate_nav.maneuver_memory import heading_options
from robot_sim.candidate_nav.action_plan import ACTION_PLAN_SPEC
from robot_sim.candidate_nav.live_trajectory import HARD_GAP


def room():
    return SimpleNamespace(values=np.zeros((160, 160), dtype=np.int16),
                           resolution=0.025, origin=(-2.0, -2.0), yaw=0.0)


def place_cell(grid, x, y, value):
    ix = math.floor((x-grid.origin[0]) / grid.resolution)
    iy = math.floor((y-grid.origin[1]) / grid.resolution)
    grid.values[iy, ix] = value


def motion(grid, pose=(0.001, 0.0, 0.0)):
    return GridMotion(grid, pose, half_length=0.1, half_width=0.08, gap=0.002)


class GridEgressTests(unittest.TestCase):
    def test_trailing_cell_can_be_left_but_not_entered_more_deeply(self):
        grid = room()
        place_cell(grid, -0.11, -0.04, 100)
        before = grid.values.copy()
        chassis = motion(grid)
        self.assertTrue(chassis.initial)
        self.assertFalse(chassis.deep_overlap)
        self.assertGreaterEqual(chassis.distance(0, chassis.turn(0), 1), 0.5)
        self.assertEqual(chassis.distance(0, chassis.turn(0), -1), 0.0)
        np.testing.assert_array_equal(grid.values, before)

    def test_new_occupied_and_unknown_cells_still_limit_the_sweep(self):
        for value in (100, 49, -1):
            with self.subTest(value=value):
                grid = room()
                place_cell(grid, -0.11, -0.04, 100)
                place_cell(grid, 0.20, 0, value)
                chassis = motion(grid)
                travel = chassis.distance(0, chassis.turn(0), 1)
                self.assertGreaterEqual(travel, 0.050)
                self.assertLessEqual(travel, 0.10)
                self.assertFalse(chassis.contacts((chassis.pose[0]+travel, 0, 0)))

    def test_inner_overlap_and_map_boundary_remain_blocked(self):
        grid = room()
        place_cell(grid, 0, 0, 100)
        self.assertIsNone(motion(grid).turn(0))
        self.assertIsNone(motion(room(), (-1.99, 0, 0)).turn(0))

    def test_egress_is_invariant_under_map_rotation(self):
        grid = room()
        place_cell(grid, -0.11, -0.04, 100)
        base = motion(grid)
        angle = 0.71
        c, s = math.cos(angle), math.sin(angle)
        ox, oy = grid.origin
        grid.origin = (c*ox-s*oy, s*ox+c*oy)
        grid.yaw = angle
        rotated = motion(grid, (c*0.001, s*0.001, angle))
        self.assertTrue(rotated.initial)
        self.assertFalse(rotated.deep_overlap)
        self.assertAlmostEqual(rotated.distance(angle, rotated.turn(0), 1),
                               base.distance(0, base.turn(0), 1))

    def test_reduced_map_margin_keeps_narrow_known_corridor_available(self):
        grid = room()
        grid.origin = (-2, -2.01)
        # Allowed lateral grid band is [-8.5 cm, +9 cm] at these poses.
        ys = grid.origin[1] + (np.arange(grid.values.shape[0])+0.5) * grid.resolution
        grid.values[(ys < -0.085) | (ys > 0.09)] = 100
        old = GridMotion(grid, (0, 0, 0), 0.10, 0.08, gap=HARD_GAP)
        self.assertTrue(old.initial)
        options = heading_options(grid, (0, 0, 0), ACTION_PLAN_SPEC)
        straight = next(option for option in options if not option['turn_actions'])
        self.assertEqual(straight['initial_overlap_kind'], 'none')
        self.assertGreaterEqual(straight['full_forward_steps'], 3)
        self.assertEqual(HARD_GAP, 0.0075)
