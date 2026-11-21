"""Run the real mapper loop with controlled scan/SLAM arrival order; no motors."""
import ast,asyncio,math,tempfile,unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from robot_sim.mapping.core import LocalGrid,filter_scan
from robot_sim.candidate_nav.geometry import Grid

SOURCE=Path(__file__).resolve().parents[1]/'candidate_nav/live_mapping.py'

def mapper_method():
 tree=ast.parse(SOURCE.read_text())
 cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='MappedAgent')
 method=next(x for x in cls.body if isinstance(x,ast.AsyncFunctionDef) and x.name=='dynamic_mapper')
 namespace=dict(math=math,np=np,Grid=Grid,filter_scan=filter_scan,
                msg_time=lambda s:s.header.stamp.sec+s.header.stamp.nanosec/1e9,
                yaw=lambda q:0.,Time=type('StampTime',(),{'from_msg':staticmethod(lambda s:s)}))
 exec(compile(ast.Module(body=[method],type_ignores=[]),str(SOURCE),'exec'),namespace)
 return namespace,namespace['dynamic_mapper']

def scan(stamp):
 return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=int(stamp),nanosec=round((stamp-int(stamp))*1e9)),frame_id='lidar'),
                        ranges=[.3,.4,.5],range_min=.02,range_max=5.,angle_min=0.,angle_increment=.1)

class MappingStartupTests(unittest.IsolatedAsyncioTestCase):
 async def exercise(self, arrival):
  with tempfile.TemporaryDirectory() as folder:
   now=[100.3];finished=asyncio.Event();global_grid=Grid(np.full((8,8),35,np.int16),.1,(-.4,-.4),'map',7,100.)
   tf=SimpleNamespace(transform=SimpleNamespace(translation=SimpleNamespace(x=0.,y=0.),rotation=SimpleNamespace(w=1.)))
   agent=SimpleNamespace(finished=finished,now=lambda:now[0],raw_scans=deque([scan(100.)]),
       buffer=SimpleNamespace(lookup_transform=lambda *a:tf),local_grid=LocalGrid(.1,.8,3.),fresh_grid=LocalGrid(.1,.8,8.),
       grid_versions=0,args=SimpleNamespace(fresh_map=True),version=0,map=None,dynamic=None,
       fresh_grid_active=False,original_slam=global_grid if arrival=='first' else None,grid_dir=Path(folder))
   calls=[];namespace,method=mapper_method()
   async def tick(_):
    calls.append(agent.grid_versions)
    self.assertIsNotNone(agent.map);self.assertIsNotNone(agent.dynamic)
    if len(calls)==1 and arrival!='first':
     self.assertEqual(list(agent.grid_dir.iterdir()),[])
     if arrival=='delayed':agent.original_slam=global_grid
     now[0]=100.7;agent.raw_scans.append(scan(100.4))
    else:finished.set()
    await asyncio.sleep(0)
   namespace['asyncio']=SimpleNamespace(sleep=tick)
   await method(agent)
   files=list(agent.grid_dir.glob('*.npz'))
   if arrival=='never':self.assertEqual(files,[])
   else:
    self.assertEqual(len(files),1)
    with np.load(files[0],allow_pickle=False) as saved:
     np.testing.assert_array_equal(saved['slam'],global_grid.values)
     self.assertEqual(int(saved['slam_version']),7)
     self.assertGreater(saved['planning'].size,0);self.assertGreater(saved['dynamic'].size,0)
   self.assertEqual(agent.grid_versions,1 if arrival=='first' else 2)
 async def test_local_mapping_survives_delayed_first_slam_message(self):await self.exercise('delayed')
 async def test_mapping_continues_without_accumulated_map(self):await self.exercise('never')
 async def test_existing_combined_archive_is_preserved_when_slam_is_ready(self):await self.exercise('first')
