from robot_sim.candidate_nav.committed_plan import first_action_labels, recent_stalls, recent_turn_overshoots
import math
import unittest
from robot_sim.candidate_nav.action_plan import (
    ACTION_PLAN_SPEC as SPEC,
    validate_response,
    PROTOCOL,
)
from robot_sim.candidate_nav.committed_plan import admissible, make_plan, orientation

TURNS = ["L", "R", "Q", "E", "A", "D"]


def option(turns=(), label="F", steps=2):
    return dict(
        turn_actions=list(turns), translation_label=label, then_forward_steps=steps
    )


class CommittedPlanTests(unittest.TestCase):
    def test_forward_and_reverse_admitted_from_current_geometry_not_catalog(self):
        allowed = admissible([], [option(label="C", steps=1)], SPEC, TURNS)
        self.assertIn("C", allowed)
        self.assertNotIn("B", allowed)
        self.assertNotIn("F", allowed)
        self.assertNotIn("G", allowed)
        self.assertEqual(admissible([], [], SPEC, TURNS), TURNS + ["S"])

    def test_short_steps_share_metric_clearance_with_full_steps(self):
        options = [option(label="F", steps=2)]
        self.assertIn("G", admissible([], options, SPEC, TURNS))
        self.assertIn("F", admissible(["G", "G"], options, SPEC, TURNS))
        self.assertNotIn("F", admissible(["F", "G"], options, SPEC, TURNS))
        self.assertIn("G", admissible(["F", "G"], options, SPEC, TURNS))
        self.assertNotIn("G", admissible(["F", "G", "G"], options, SPEC, TURNS))

    def test_equivalent_turns_are_not_mistaken_for_different_heading(self):
        options = [option(turns=["left_45"])]
        self.assertIn("F", admissible(["L", "L"], options, SPEC, TURNS))
        self.assertNotIn("F", admissible(["R", "R"], options, SPEC, TURNS))

    def test_new_translation_after_moving_then_turning_requires_new_geometry(self):
        options = [option(steps=4), option(label="B", steps=4)]
        self.assertNotIn("F", admissible(["F", "L", "R"], options, SPEC, TURNS))
        self.assertNotIn("B", admissible(["F"], options, SPEC, TURNS))

    def test_reverse_heading_and_travel_bearing_are_distinct(self):
        p = make_plan(["backward"], [0, 0, math.pi / 2], SPEC, False)
        self.assertAlmostEqual(p["nominal_end_pose"][2], math.pi / 2)
        self.assertAlmostEqual(p["steps"][0]["travel_map_bearing_degrees"], -90)
        self.assertIn("backward 10 cm", p["display"])

    def test_orientation_includes_rotated_map_pixels(self):
        p = orientation(
            [0, 0, math.pi], dict(local=dict(projection=dict(yaw=math.pi / 2)))
        )
        self.assertEqual(p["forward_xy"], [-1.0, 0.0])
        self.assertEqual(p["backward_xy"], [1.0, -0.0])
        self.assertEqual(p["local_forward_image_xy"], [0.0, -1.0])

    def test_executor_block_removes_first_primitive_from_model_menu(self):
        observation = dict(first_motion_checks={"creep_backward": {"allowed": False}},
                           recent_stalled_actions=["forward"])
        self.assertEqual(first_action_labels(["F", "C", "L", "S"], observation), ["L", "S"])
        self.assertEqual(first_action_labels(["F", "C", "S"],
                                             {**observation, "collision_protection": "disabled"}),
                         ["F", "C", "S"])

    def test_repeated_cancels_expire_and_do_not_follow_pose_or_epoch(self):
        blocked = dict(action="creep_backward", outcome="trajectory_collision",
                       control_start_pose=[0., 0., 0.], control_end_pose=[.002, 0., 0.])
        history = [dict(plan_id=i, control_epoch="e", step_results=[blocked]) for i in (4, 5)]
        reference = dict(epoch="e", valid=True, pose=[.003, 0., .01])
        self.assertEqual(recent_stalls(history, reference, 6), ["creep_backward"])
        self.assertEqual(recent_stalls(history, reference, 10), [])
        self.assertEqual(recent_stalls(history, {**reference, "pose": [.10, 0., 0.]}, 6), [])
        self.assertEqual(recent_stalls(history, {**reference, "epoch": "e2"}, 6), [])
        self.assertEqual(recent_stalls(history[:1], reference, 6), [])
        blocked["clearance"] = {"reason": "insufficient_lidar_coverage"}
        self.assertEqual(recent_stalls(history, reference, 6), [])

    def test_repeated_small_turn_overshoot_temporarily_excludes_only_that_angle(self):
        step = dict(action="left_11_25", outcome="insufficient_progress_or_deviation", turn_rad=.34)
        history = [dict(plan_id=i, control_epoch="e", step_results=[step]) for i in (3, 5)]
        reference = dict(epoch="e", valid=True)
        unreliable = recent_turn_overshoots(history, reference, 6)
        self.assertEqual(unreliable, ["left_11_25"])
        self.assertEqual(first_action_labels(["Q", "L", "E", "S"], {"recent_turn_overshoots": unreliable}),
                         ["L", "E", "S"])
        self.assertEqual(recent_turn_overshoots(history, reference, 10), [])
        history.append(dict(plan_id=6, control_epoch="e", step_results=[{**step, "outcome": "duration_complete"}]))
        self.assertEqual(recent_turn_overshoots(history, reference, 7), [])

    def test_short_turn_overruns_in_both_directions_do_not_trigger_ping_pong(self):
        history = [
            dict(plan_id=i, control_epoch="e", step_results=[
                dict(action=action, outcome="insufficient_progress_or_deviation", turn_rad=turn)])
            for i, action, turn in ((1, "left_11_25", .34), (2, "right_11_25", -.36))
        ]
        reference = dict(epoch="e", valid=True)
        self.assertEqual(recent_turn_overshoots(history, reference, 3), ["left_11_25", "right_11_25"])

    def test_wire_rejects_command_that_differs_from_committed_plan(self):
        observation = dict(
            protocol=PROTOCOL, snapshot_id=4, map_version=3, trajectory_steps=1
        )
        plan = make_plan(["backward"], [0, 0, 0], SPEC, False)
        plan.update(snapshot_id=4, map_version=3)
        response = dict(
            protocol=PROTOCOL,
            snapshot_id=4,
            map_version=3,
            trajectory_steps=1,
            action="backward",
            goal_id="B",
            actions=["backward"],
            task_complete=False,
            selected_plan=plan,
        )
        validate_response(response, observation)
        response["selected_plan"]["actions"] = ["forward"]
        with self.assertRaises(ValueError):
            validate_response(response, observation)
