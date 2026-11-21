import asyncio
import base64
import io
import unittest
from types import SimpleNamespace
from PIL import Image
from robot_sim.candidate_nav.motion_vision_policy import inspect

class Captured(Exception):pass
class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_visual_observer_receives_measured_motion_without_plan_stop_labels(self):
        saved=[]
        class Processor:
            def apply_chat_template(self,messages,**kw):saved.extend(messages);raise Captured()
        llm=SimpleNamespace(processor=Processor())
        policy=SimpleNamespace(llm=llm)
        image=io.BytesIO();Image.new('RGB',(16,16)).save(image,format='PNG')
        frame=dict(frame_id=1,stamp=1.,pose=[0,0,0],plan_id=1,
                   image=base64.b64encode(image.getvalue()).decode(),
                   execution=dict(action='stop',progress=dict(actions=['forward','stop']),
                                  measured_totals=dict(forward_m=.4),sensors=dict(imu_available=False)))
        with self.assertRaises(Captured):
            await inspect(policy,dict(instruction='Approach the object.',frames=[frame]))
        import json
        text=saved[1]['content'][1]['text']
        context=json.loads(text.removeprefix('NEW '))
        self.assertNotIn('execution',context)
        self.assertNotIn('action',context)
        self.assertEqual(context['measured_motion'],dict(measured_totals=dict(forward_m=.4),sensors=dict(imu_available=False)))
        self.assertEqual(context['pose'],[0,0,0])
