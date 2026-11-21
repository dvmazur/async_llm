"""A blocked start must be distinguishable from Qwen authoring invalid points."""
import unittest
import numpy as np
from robot_sim.candidate_nav.geometry import Grid,reachable
from robot_sim.candidate_nav.live_proposals import reach_status,validation_feedback

class CandidateDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.grid=Grid(np.zeros((40,40),np.int16),.025,(-.5,-.5))
        self.pose=(0,0,0)
    def test_clear_start_reports_reachable_space(self):
        state=reach_status(reachable(self.grid,self.pose,radius=.085))
        self.assertEqual(state['start_state'],'ready')
        self.assertGreater(state['reachable_cells'],0)
    def test_close_occupied_cell_preserves_map_and_reports_blocked_start(self):
        x,y=self.grid.cell((.04,.06));self.grid.values[y,x]=100
        state=reach_status(reachable(self.grid,self.pose,radius=.085))
        self.assertEqual(state['start_state'],'insufficient_clearance')
        self.assertEqual(state['reachable_cells'],0)
        self.assertLess(state['start_clearance_m'],.085)
        self.assertEqual(self.grid.values[y,x],100)
    def test_unknown_and_occupied_start_are_not_conflated(self):
        x,y=self.grid.cell(self.pose)
        for value,expected in [(-1,'unknown_start'),(100,'occupied_start')]:
            self.grid.values[y,x]=value
            self.assertEqual(reach_status(reachable(self.grid,self.pose))['start_state'],expected)
    def test_rejections_return_the_models_own_coordinates(self):
        raw=[dict(u=256,v=256,heading=0,kind='frontier'),dict(u=100,v=120,heading=2,kind='look')]
        checks=[dict(raw=raw[0],accepted=False,reason='unknown_or_no_reachable_cells'),
                dict(raw=raw[1],accepted=True)]
        feedback=validation_feedback(dict(snapshot_id=7,map_version=10),raw,checks)
        self.assertEqual(feedback['raw_count'],2)
        self.assertEqual(feedback['accepted_count'],1)
        self.assertEqual(feedback['checks'][0]['proposal'],raw[0])
        self.assertEqual(feedback['rejection_counts'],{'unknown_or_no_reachable_cells':1})

if __name__=='__main__':unittest.main()
