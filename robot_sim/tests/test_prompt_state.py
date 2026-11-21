import copy
import json
import unittest

from robot_sim.candidate_nav.prompt_state import model_state, state_text


class PromptStateTests(unittest.TestCase):
    def test_unique_history_preserved_once_without_mutating_logs(self):
        actions = [{'action': 'left', 'outcome': 'duration_complete',
                    'start_pose': [1.123456789012345, -.032455555555, 1.57079632679],
                    'end_pose': [1.123456789012345, -.032455555555, 2.35619449019]}]
        state = dict(instruction='Найди коробку и подъедь на 5 сантиметров.',
                     recent_outcomes=actions,
                     task_context=dict(recent_actions=actions,
                                       initial_pose=[1, 2, 3], odometry_path_m=4.123456),
                     previous_plan_outcome=dict(discarded_actions=['forward']),
                     maneuver_memory=dict(local_blocked_forward_attempts=[{'yaw_rad': 2.356194}]))
        original = copy.deepcopy(state)
        packed = json.loads(state_text(state, True))
        self.assertNotIn('recent_outcomes', packed)
        self.assertEqual(len(packed['task_context']['recent_actions']), 1)
        self.assertEqual(packed['task_context']['recent_actions'][0]['end_pose'], [1.123, -.032, 2.356])
        self.assertEqual(packed['instruction'], state['instruction'])
        self.assertEqual(packed['previous_plan_outcome'], state['previous_plan_outcome'])
        self.assertEqual(packed['maneuver_memory']['local_blocked_forward_attempts'][0]['yaw_rad'], 2.356)
        self.assertEqual(state, original)
        self.assertIn('Найди', state_text(state, True))
        self.assertLess(len(state_text(state, True)), len(state_text(state, False)))

    def test_distinct_outcomes_are_not_dropped_even_if_rounding_matches(self):
        x = dict(recent_outcomes=[{'distance_m': .10001}],
                 task_context=dict(recent_actions=[{'distance_m': .10002}]))
        self.assertIn('recent_outcomes', model_state(x))

    def test_waypoint_duplicate_and_missing_context(self):
        x = dict(recent_outcomes=[['A', 'blocked']],
                 task_context=dict(recent_waypoint_outcomes=[['A', 'blocked']]))
        self.assertNotIn('recent_outcomes', model_state(x))
        self.assertIn('recent_outcomes', model_state(dict(recent_outcomes=['blocked'])))

    def test_simulation_unchanged_and_nonfinite_still_rejected(self):
        state = dict(instruction='Найди', pose=[1.123456789], recent_outcomes=[])
        self.assertEqual(state_text(state, False), json.dumps(state, allow_nan=False))
        for live in (True, False):
            with self.assertRaises(ValueError):
                state_text(dict(pose=[float('nan')]), live)
