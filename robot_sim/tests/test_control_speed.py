import asyncio, json, tempfile, unittest, threading, time
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from robot_sim.candidate_nav.model_transport import request_model,close_model_connections
from robot_sim.candidate_nav.artifact_writer import ArtifactWriter,image,text,grid
from robot_sim.candidate_nav.action_plan import ACTION_PLAN_SPEC,PROTOCOL
from robot_sim.candidate_nav.streaming_perception import StreamingPerception

class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.release=asyncio.Event();self.connections=0;self.seen=[];self.changed=False;self.legacy=False
        self.o=dict(physical=True,snapshot_id=1,map_version=2,protocol=PROTOCOL,trajectory_steps=1,action_spec=ACTION_PLAN_SPEC)
        self.server=await asyncio.start_server(self.handle,'127.0.0.1',0)
        self.port=self.server.sockets[0].getsockname()[1]
    async def asyncTearDown(self):
        self.release.set();await close_model_connections();self.server.close();await self.server.wait_closed()
    async def handle(self,reader,writer):
        self.connections+=1
        from robot_sim.candidate_nav.wire_images import WireImages
        image_cache=WireImages()
        try:
            while line:=await reader.readline():
                o=image_cache.decode(json.loads(line));self.seen.append(o['snapshot_id'])
                response=dict(protocol=PROTOCOL,snapshot_id=o['snapshot_id'],map_version=2,trajectory_steps=1,
                    action='forward',actions=['forward'],goal_id='F',task_complete=False,transport_keepalive=not self.legacy)
                if o.get('response_mode'):
                    writer.write((json.dumps(dict(response,message_kind='plan_ready'))+'\n').encode());await writer.drain()
                    await self.release.wait()
                    if self.changed:response.update(actions=['backward'],action='backward',goal_id='B')
                    response.update(message_kind='rationale_ready',reasoner='Forward by 10 cm.')
                writer.write((json.dumps(response)+'\n').encode());await writer.drain()
                if self.legacy:break
        finally:
            writer.close();await writer.wait_closed()
    async def test_returns_plan_before_comment_reuses_connection_after_final(self):
        comments=[];self.o['response_mode']='plan_stream_v1'
        r=await asyncio.wait_for(request_model(self.o,self.port,on_update=comments.append),.5)
        self.assertEqual(r['message_kind'],'plan_ready');self.assertFalse(comments)
        self.release.set()
        for _ in range(20):
            if comments:break
            await asyncio.sleep(.01)
        self.assertEqual(len(comments),1)
        r=await request_model(dict(self.o,snapshot_id=2,response_mode=None),self.port)
        self.assertTrue(r['transport_timing']['connection_reused']);self.assertEqual(self.connections,1)
    async def test_inflight_comment_cannot_be_read_as_another_plan(self):
        await request_model(dict(self.o,response_mode='plan_stream_v1'),self.port)
        r=await request_model(dict(self.o,snapshot_id=2),self.port)
        self.assertEqual(r['snapshot_id'],2);self.assertEqual(self.connections,2)
    async def test_changed_late_plan_is_rejected_without_callback(self):
        self.changed=True;comments=[]
        r=await request_model(dict(self.o,response_mode='plan_stream_v1'),self.port,on_update=comments.append)
        self.release.set();await asyncio.sleep(.05)
        self.assertEqual(r['actions'],['forward']);self.assertEqual(len(comments),1)
        self.assertEqual(comments[0]['actions'],['forward'])
        self.assertEqual(comments[0]['message_kind'],'rationale_error')
        self.assertFalse(comments[0]['rationale_pending'])
    async def test_old_single_response_server_still_works(self):
        self.legacy=True
        await request_model(self.o,self.port);await request_model(dict(self.o,snapshot_id=2),self.port)
        self.assertEqual(self.connections,2)

class ArchiveTests(unittest.TestCase):
    def test_order_and_immutable_camera_copy_and_bounded_backpressure(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            a=SimpleNamespace(artifact_writer=ArtifactWriter(capacity=2));p=Path(folder)
            original=Image.new('RGB',(8,8),'red');arr=np.zeros((8,8))
            image(a,original,p/'a.png');original.paste('blue',(0,0,8,8))
            grid(a,p/'a.npz',occupancy=arr);arr[:]=4
            for i in range(10):text(a,p/'x',str(i))
            a.artifact_writer.close()
            self.assertEqual(Image.open(p/'a.png').getpixel((0,0)),(255,0,0))
            self.assertFalse(np.load(p/'a.npz')['occupancy'].any());self.assertEqual((p/'x').read_text(),'9')
    def test_disk_errors_are_reported_on_close(self):
        with tempfile.TemporaryDirectory() as folder:
            a=SimpleNamespace(artifact_writer=ArtifactWriter())
            text(a,Path(folder)/'missing'/'x','x')
            with self.assertRaises(FileNotFoundError):a.artifact_writer.close()

class StopTests(unittest.IsolatedAsyncioTestCase):
    async def test_frozen_odom_does_not_qualify_as_settled(self):
        a=StreamingPerception();a.now=lambda:0
        a.local_odometry=SimpleNamespace(snapshot=lambda now:dict(valid=True,stamp=1,pose=[0,0,0]))
        t=time.monotonic();await a.wait_motion_settled();self.assertGreaterEqual(time.monotonic()-t,.34)
    async def test_small_but_continuing_movement_waits_full_interval(self):
        a=StreamingPerception();start=time.monotonic();a.now=lambda:time.monotonic()
        a.local_odometry=SimpleNamespace(snapshot=lambda now:dict(valid=True,stamp=now,pose=[(now-start)*.03,0,0]))
        t=time.monotonic();await a.wait_motion_settled();self.assertGreaterEqual(time.monotonic()-t,.34)
    async def test_fresh_stationary_samples_allow_shorter_settle(self):
        a=StreamingPerception();a.now=lambda:time.monotonic()
        a.local_odometry=SimpleNamespace(snapshot=lambda now:dict(valid=True,stamp=now,pose=[0,0,0]))
        t=time.monotonic();await a.wait_motion_settled();self.assertLess(time.monotonic()-t,.3)

class WireTests(unittest.TestCase):
    def test_reference_bytes_eviction_and_legacy_inputs(self):
        from robot_sim.candidate_nav.wire_images import WireImages
        encoder=WireImages(2);decoder=WireImages(2)
        for i in [1,2,2,3,1,3]:
            payload={'frames':[{'image':str(i)*1000}],'previous_frame':{'image':'reference'*100},'instruction':'Unchanged.'}
            wire=encoder.encode(payload,True)
            self.assertEqual(decoder.decode(wire),payload)
        plain={'camera':'a','map_image':'b'}
        self.assertEqual(decoder.decode(encoder.encode(plain,False)),plain)
        wire=encoder.encode(plain,True)
        self.assertEqual(wire['_wire_images'],{})
        self.assertEqual(decoder.decode(wire),plain)
    def test_corrupt_and_unknown_reference_rejected(self):
        from robot_sim.candidate_nav.wire_images import WireImages
        with self.assertRaises(ValueError):WireImages().decode({'camera':{'_wire_image':'x'}})

class RealDispatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_server_streams_before_late_compute_and_keeps_image_references(self):
        import ast,os
        from unittest.mock import patch
        source=Path(__file__).parents[1]/'candidate_nav/llm.py'
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.AsyncFunctionDef) and n.name=='run')
        ready=asyncio.Event();release=asyncio.Event();servers=[];packets=[];broadcasts=[]
        class Model:
            rank=0;world=2
            def __init__(self,args):
                self.metadata={};self.tokens=SimpleNamespace(encode=lambda x,**kw:[])
                self.llm=SimpleNamespace(engine=SimpleNamespace(tp_cpu_group=None))
            async def decide(self,p,publish):
                packets.append(p)
                r=dict(protocol=PROTOCOL,snapshot_id=p['snapshot_id'],map_version=2,trajectory_steps=1,action='forward',actions=['forward'],goal_id='F',task_complete=False)
                publish(dict(r,message_kind='plan_ready',rationale_pending=True))
                await release.wait()
                return dict(r,message_kind='rationale_ready',rationale_pending=False,reasoner='Go forward.')
            async def close(self):pass
        class Memory:
            def __init__(self,*a):pass
            async def prepare(self,p):return p
            async def record(self,*a):pass
            async def close(self):pass
        class Proxy:
            def __getattr__(self,key):return getattr(asyncio,key)
            async def start_server(self,*args,**kw):
                s=await asyncio.start_server(*args,**kw);servers.append(s);ready.set();return s
        ns=dict(torch=SimpleNamespace(set_num_threads=lambda x:None),AsyncCandidatePolicy=Model,MemoryCoordinator=Memory,asyncio=Proxy(),json=json,os=os,time=time,__file__=str(source),dist=SimpleNamespace(broadcast_object_list=lambda p,**kw:broadcasts.append(p)))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'ROBOT_LLM_MEMORY_DIR':d}):
            job=asyncio.create_task(ns['run'](SimpleNamespace(port=0,hardware_profile='h100')))
            try:
                await asyncio.wait_for(ready.wait(),1);port=servers[0].sockets[0].getsockname()[1]
                o=dict(physical=True,protocol=PROTOCOL,snapshot_id=1,map_version=2,trajectory_steps=1,action_spec=ACTION_PLAN_SPEC,response_mode='plan_stream_v1',camera='exact pixels')
                comments=[];r=await asyncio.wait_for(request_model(o,port,on_update=comments.append),1)
                self.assertEqual(r['message_kind'],'plan_ready');self.assertFalse(comments)
                release.set()
                for _ in range(30):
                    if comments:break
                    await asyncio.sleep(.01)
                self.assertEqual(len(comments),1)
                r=await request_model(dict(o,snapshot_id=2),port)
                self.assertTrue(r['transport_timing']['connection_reused'])
                self.assertEqual(packets[-1]['camera'],'exact pixels');self.assertEqual(len(broadcasts),2)
            finally:
                release.set();await close_model_connections();job.cancel();await asyncio.gather(job,return_exceptions=True)

class PrefillAdmissionTests(unittest.TestCase):
    def test_large_requests_fall_back_without_allocating_graph_state(self):
        import ast
        source=Path(__file__).parents[2]/'python/minisgl/shared_cache/graph.py'
        tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SharedGraphIO')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='select_prefill')
        ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        a=SimpleNamespace(prefill_rows=[64,256,512])
        job=SimpleNamespace(input_ids=SimpleNamespace(numel=lambda:1024))
        self.assertIsNone(ns['select_prefill'](a,[job]))

class RecurrentPrecisionTests(unittest.TestCase):
    def test_graph_core_uses_separate_fp32_affine_capture_like_eager(self):
        import ast
        import torch
        source=Path(__file__).parents[2]/'python/minisgl/shared_cache/gdn_prefill.py'
        tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GDNPrefillBuffers')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='core_and_capture')
        ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        calls=[];result=object()
        a=SimpleNamespace(core=lambda *x:calls.append(('core',x)) or result,capture_prefill=lambda *x:calls.append(('fp32_affines',x)))
        g=torch.tensor([-.5]);out=ns['core_and_capture'](a,1,None,None,None,g,None,None,True,None)
        self.assertIs(out,result);self.assertEqual([x[0] for x in calls],['core','fp32_affines'])
        self.assertTrue(torch.equal(calls[1][1][3],g.exp()))
