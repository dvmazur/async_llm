import unittest
from robot_sim.candidate_nav.temporal_vision import initial_state,event_labels,update_state
class TemporalStateTests(unittest.TestCase):
 def frame(self,i):return dict(frame_id=i,stamp=i,pose=[0,0,i*.2],plan_id=1)
 def test_appearance_loss_then_unchanged_preserves_last_seen_pose(self):
  s=initial_state('find red square')
  update_state(s,'N','',self.frame(1),False)
  update_state(s,'A','red square appeared left',self.frame(2),True)
  self.assertTrue(s['relevant_visible'])
  update_state(s,'D','red square left view during turn',self.frame(3),True)
  update_state(s,'N','',self.frame(4),True)
  self.assertFalse(s['relevant_visible'])
  self.assertEqual(s['last_seen'],self.frame(2))
  self.assertEqual(s['latest_frame'],self.frame(4))
 def test_task_change_clears_reference_and_state_copy_does_not_mutate(self):
  old=initial_state('red square');update_state(old,'A','visible',self.frame(1),False)
  other=initial_state('exit room',old)
  self.assertIsNone(other['last_seen'])
  same=initial_state('red square',old);update_state(same,'D','gone',self.frame(2),True)
  self.assertTrue(old['relevant_visible'])
 def test_unchanged_visible_frame_updates_anchor_without_duplicate_appearance(self):
  s=initial_state('red square');update_state(s,'A','visible',self.frame(1),False)
  self.assertNotIn('A',event_labels(s,True))
  update_state(s,'N','',self.frame(2),True)
  self.assertEqual(s['last_seen'],self.frame(2))
 def test_uncertainty_is_not_absence_and_does_not_erase_anchor(self):
  s=initial_state('red square');update_state(s,'A','visible',self.frame(1),False)
  update_state(s,'U','',self.frame(2),True)
  self.assertIsNone(s['relevant_visible']);self.assertEqual(s['last_seen'],self.frame(1))
