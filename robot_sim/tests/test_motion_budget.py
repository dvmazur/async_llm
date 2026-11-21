import unittest
from robot_sim.candidate_nav.action_plan import ACTION_PLAN_SPEC
from robot_sim.candidate_nav.trajectory_policy import immediate_mobility,action_labels

class MotionBudgetTests(unittest.TestCase):
    def test_open_corridor_exposes_normal_step_and_horizon_budgets(self):
        rows=[dict(turn_actions=[],translation_label='F',then_forward_steps=10)]
        item=immediate_mobility(rows,ACTION_PLAN_SPEC,5)[0]
        self.assertEqual(item['measured_steps'],10)
        self.assertEqual(item['max_steps_this_sequence'],5)
        self.assertEqual(item['known_translation_m'],1.)
        self.assertEqual(item['max_translation_this_sequence_m'],.5)
    def test_short_clearance_keeps_creep_and_does_not_extend_it(self):
        row=dict(turn_actions=[],translation_label='G',then_forward_steps=1)
        item=immediate_mobility([row],ACTION_PLAN_SPEC,5)[0]
        self.assertEqual(item['nominal_step_m'],.05)
        self.assertEqual(item['max_steps_this_sequence'],1)
        self.assertEqual(item['max_translation_this_sequence_m'],.05)
        self.assertEqual(action_labels(['G'],[row],row),['S'])
    def test_reverse_distance_uses_spec_and_turn_routes_are_not_immediate(self):
        rows=[dict(turn_actions=[],translation_label='C',then_forward_steps=2),
              dict(turn_actions=['left'],translation_label='F',then_forward_steps=10)]
        items=immediate_mobility(rows,ACTION_PLAN_SPEC,5)
        self.assertEqual(len(items),1);self.assertEqual(items[0]['max_translation_this_sequence_m'],.1)
    def test_missing_distance_is_not_invented_and_input_not_modified(self):
        rows=[dict(turn_actions=[],then_forward_steps=10)]
        self.assertEqual(immediate_mobility(rows,{},5),[dict(action='F',measured_steps=10)])
        self.assertEqual(rows,[dict(turn_actions=[],then_forward_steps=10)])
