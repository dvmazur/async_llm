"""Historical map context never replaces local clearance or proposal coordinates."""
import asyncio
import base64
import io
import json
import math
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image
from robot_sim.candidate_nav.geometry import Grid
from robot_sim.candidate_nav.map_memory import accumulated_view, MAP_MEMORY_RULES
from robot_sim.candidate_nav.render import pixel
from robot_sim.candidate_nav import task_policy


class MapMemoryTests(unittest.TestCase):
    def setUp(self):
        self.global_grid=Grid(np.zeros((160,200),np.int16), .05, (-5,-4), version=7,stamp=1,yaw=.2)
        self.global_grid.values[40:45,:]=100
        self.local=Grid(np.zeros((80,80),np.int16),.05,(-2,-2),stamp=20,yaw=-.1)

    def test_entire_old_map_is_retained_and_scale_and_local_extent_are_explicit(self):
        before=self.global_grid.values.copy()
        image,meta=accumulated_view(self.global_grid,[0,0,.4],[(-2,0),(0,0)],self.local,20)
        self.assertEqual(image.size,(512,512))
        self.assertEqual(meta['extent_m'],[10,8])
        self.assertEqual(meta['known_cells'],32000)
        self.assertEqual(meta['age_s'],19)
        self.assertFalse(meta['recent_update'])
        self.assertIsNone(meta['ttl_s'])
        self.assertEqual(meta['projection']['yaw'],.2)
        np.testing.assert_allclose(meta['rover_pixel'],pixel([0,0],meta['projection']))
        np.testing.assert_allclose(meta['local_extent_world'][1],[-2+4*math.cos(-.1),-2+4*math.sin(-.1)])
        np.testing.assert_array_equal(before,self.global_grid.values)

    def test_no_fake_global_map_if_missing_or_frame_mismatched(self):
        self.assertFalse(accumulated_view(None,[0,0,0],[],self.local,20)[1]['available'])
        self.global_grid.frame='other_map'
        image,meta=accumulated_view(self.global_grid,[0,0,0],[],self.local,20)
        self.assertIsNone(image)
        self.assertEqual(meta['reason'],'map_frame_mismatch')

    def test_live_rules_separate_map_roles_in_all_strategies(self):
        for system in (task_policy.WAYPOINT_SYSTEM,task_policy.GENERATION_SYSTEM,
                       task_policy.DIRECT_SYSTEM,task_policy.TRAJECTORY_EVIDENCE_SYSTEM):
            self.assertIn(MAP_MEMORY_RULES,system)

    def test_proposal_input_has_three_images_and_no_base64_in_json(self):
        from robot_sim.candidate_nav.llm import AsyncCandidatePolicy
        class Packed(Exception): pass
        class Capture:
            def apply_chat_template(self,messages,**_):
                self.messages=messages
                raise Packed()
        processor=Capture()
        policy=AsyncCandidatePolicy.__new__(AsyncCandidatePolicy)
        policy.args=SimpleNamespace(analysis_tokens=24)
        policy.llm=SimpleNamespace(processor=processor)
        data=io.BytesIO();Image.new('RGB',(16,16)).save(data,format='PNG')
        image=base64.b64encode(data.getvalue()).decode()
        observation=dict(physical=True,completion_mode=task_policy.MODE,camera=image,map_image=image,
                         accumulated_map_image=image,maps_context={'accumulated':{'available':True}})
        with self.assertRaises(Packed):asyncio.run(policy.propose(observation))
        content=processor.messages[1]['content']
        self.assertEqual(sum(x['type']=='image' for x in content),3)
        state=json.loads(content[-1]['text'])
        self.assertEqual(state['image_roles']['accumulated_map'],3)
        self.assertNotIn('accumulated_map_image',state)
        self.assertIn('Proposal coordinates still refer to IMAGE 2.',content[-3]['text'])


if __name__=='__main__':unittest.main()
