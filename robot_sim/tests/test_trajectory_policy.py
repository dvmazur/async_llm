import unittest
from robot_sim.candidate_nav.trajectory_policy import action_labels
class GeometryGrammarTests(unittest.TestCase):
 def test_known_corridor_budget_and_new_frame_require_refresh(self):
  options=[dict(turn_actions=['right','right'],then_forward_steps=2)]
  self.assertNotIn('F',action_labels([],options))
  self.assertIn('F',action_labels(['R','R'],options))
  self.assertIn('F',action_labels(['R','R','F'],options))
  self.assertNotIn('F',action_labels(['R','R','F','F'],options))
  self.assertNotIn('F',action_labels(['R','R','F','L'],options))
  self.assertEqual(action_labels([],[]),['L','R','S'])
 def test_mirrored_openings_are_symmetric(self):
  for turn,name in [('L','left'),('R','right')]:
   self.assertIn('F',action_labels([turn,turn],[dict(turn_actions=[name,name],then_forward_steps=1)]))

 def test_selected_route_cannot_be_rewritten_and_inspection_does_not_translate(self):
  route=dict(turn_actions=['left','left'],then_forward_steps=2)
  self.assertEqual(action_labels([],[],route),['L','S'])
  self.assertEqual(action_labels(['L','L'],[],route),['F','S'])
  self.assertEqual(action_labels(['L','L','F','F'],[],route),['S'])
  self.assertNotIn('F',action_labels([],[],inspect_only=True))
