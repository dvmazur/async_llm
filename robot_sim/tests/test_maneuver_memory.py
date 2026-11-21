"""Measured turn progress must survive a blocked forward step and another plan."""
import math
import unittest
from robot_sim.candidate_nav.maneuver_memory import maneuver_memory, model_outcome


def plan(pid, start_heading, turn, *, outcome='trajectory_collision', xy=(0,0)):
    end_heading=math.atan2(math.sin(start_heading+turn),math.cos(start_heading+turn))
    start=[*xy,start_heading];end=[*xy,end_heading]
    direction='right' if turn<0 else 'left'
    return {'plan_id':pid,'start_pose':start,'end_pose':end,
            'planned_actions':[direction,direction,'forward','forward','stop'],
            'outcome':outcome,'step_results':[
                {'action':direction,'outcome':'duration_complete','turn_rad':turn/2,'distance_m':0},
                {'action':direction,'outcome':'duration_complete','turn_rad':turn/2,'distance_m':0},
                {'action':'forward','outcome':outcome,'distance_m':0,
                 'start_pose':end,'end_pose':end}],
            'discarded_actions':['forward','stop']}

class ManeuverMemoryTests(unittest.TestCase):
    def test_blocked_forward_preserves_two_executed_right_turns(self):
        p=plan(1,math.pi/2,-math.pi/2)
        m=maneuver_memory([p],p['end_pose'])
        self.assertAlmostEqual(m['local_measured_turn_rad'],-math.pi/2,places=3)
        self.assertEqual(m['last_measured_turn_direction'],'right')
        self.assertEqual(m['local_forward_progress_m'],0)
        self.assertEqual(m['local_blocked_forward_attempts'][0]['yaw_rad'],0)
        self.assertEqual(m['recent_plans'][0]['discarded'],['forward','stop'])
    def test_opposite_turns_and_failed_forwards_expose_the_loop(self):
        right=plan(1,math.pi/2,-math.pi/2);left=plan(2,0,math.pi/2)
        m=maneuver_memory([right,left],left['end_pose'])
        self.assertTrue(m['oscillation_detected'])
        self.assertEqual(m['turn_reversals'],1)
        self.assertEqual(m['local_measured_turn_rad'],0)
        self.assertEqual(len(m['local_blocked_forward_attempts']),2)
        self.assertAlmostEqual(m['local_blocked_forward_attempts'][0]['relative_yaw_rad'],-math.pi/2,places=3)
    def test_unexecuted_turns_never_count_as_progress(self):
        p=plan(1,0,math.pi/2);p['step_results']=[];p['end_pose']=p['start_pose']
        m=maneuver_memory([p],p['end_pose'])
        self.assertEqual(m['local_measured_turn_rad'],0)
        self.assertEqual(m['local_blocked_forward_attempts'],[])
    def test_undertravel_is_not_reported_as_an_obstacle(self):
        p=plan(1,0,math.pi/2,outcome='insufficient_progress_or_deviation')
        self.assertEqual(maneuver_memory([p],p['end_pose'])['local_blocked_forward_attempts'],[])
    def test_previous_location_is_not_a_current_blocked_heading(self):
        p=plan(1,0,math.pi/2)
        m=maneuver_memory([p],[2,0,0])
        self.assertEqual(m['local_blocked_forward_attempts'],[])
        self.assertFalse(m['oscillation_detected'])
    def test_history_is_bounded_and_missing_pose_never_fabricates_progress(self):
        history=[plan(i,0,math.pi/2) for i in range(20)]
        m=maneuver_memory(history,None)
        self.assertEqual(len(m['recent_plans']),4)
        self.assertEqual(m['recent_plans'][0]['plan_id'],16)
        self.assertIsNone(m['current_pose'])
        self.assertEqual(m['local_measured_turn_rad'],0)

    def test_small_back_and_forth_progress_does_not_hide_a_turn_loop(self):
        history=[plan(i,math.pi/2 if i%2==0 else 0, -math.pi/2 if i%2==0 else math.pi/2)
                 for i in range(8)]
        for p in history:p['step_results'][-1]['distance_m']=.04
        m=maneuver_memory(history,[0,0,math.pi/2])
        self.assertGreater(m['local_forward_progress_m'],.15)
        self.assertTrue(m['oscillation_detected'])

    def test_prompt_outcome_keeps_steps_without_duplicating_full_precision_poses(self):
        p=plan(1,math.pi/2,-math.pi/2)
        compact=model_outcome(p)
        self.assertEqual(len(compact['step_results']),3)
        self.assertEqual(compact['end_pose'],[0,0,0])
        self.assertNotIn('start_pose',compact['step_results'][-1])
        self.assertIn('start_pose',p['step_results'][-1])

class HeadingFactsTests(unittest.TestCase):
    def room(self):
        import numpy as np
        from robot_sim.candidate_nav.geometry import Grid
        v=np.full((120,120),100,np.int16);r,c=np.indices(v.shape)
        x=-1.5+(c+.5)*.025;y=-1.5+(r+.5)*.025
        v[((abs(x)<.2)&(abs(y)<.2))|((abs(x)<.2)&(y<=-.2))]=0
        return Grid(v,.025,(-1.5,-1.5))
    def test_blocked_after_ninety_degrees_but_continuing_turn_reaches_exit(self):
        from robot_sim.candidate_nav.maneuver_memory import heading_options
        from robot_sim.candidate_nav.action_plan import ACTION_PLAN_SPEC
        facts={x['turn_prefix']:x for x in heading_options(self.room(),[0,0,0],ACTION_PLAN_SPEC)}
        self.assertEqual(facts['']['full_forward_steps'],0)
        self.assertEqual(facts['A']['full_forward_steps'],0)
        self.assertGreaterEqual(facts['DD']['full_forward_steps'],1)
        self.assertTrue(facts['DD']['turn_sweep_known_free'])
    def test_mirrored_pose_allows_left_continuation(self):
        from robot_sim.candidate_nav.maneuver_memory import heading_options
        from robot_sim.candidate_nav.action_plan import ACTION_PLAN_SPEC
        facts={x['turn_prefix']:x for x in heading_options(self.room(),[0,0,math.pi],ACTION_PLAN_SPEC)}
        self.assertEqual(facts['D']['full_forward_steps'],0)
        self.assertGreaterEqual(facts['AA']['full_forward_steps'],1)
    def test_unknown_cells_do_not_become_an_affordance(self):
        from robot_sim.candidate_nav.maneuver_memory import heading_options
        from robot_sim.candidate_nav.action_plan import ACTION_PLAN_SPEC
        grid=self.room();grid.values[:]=-1
        facts=heading_options(grid,[0,0,0],ACTION_PLAN_SPEC)
        self.assertTrue(all(not x['turn_sweep_known_free'] and x['full_forward_steps']==0 for x in facts))

if __name__=='__main__':unittest.main()
