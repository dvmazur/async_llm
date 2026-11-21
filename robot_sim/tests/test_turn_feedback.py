"""Exercise the real executor with delayed, sampled, inertial virtual feedback."""
import ast
import asyncio
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from robot_sim.candidate_nav.action_plan import ACTION_PLAN_SPEC,rollout

class FeedbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_turn_sizes_under_sensor_delay_and_motor_inertia(self):
        path=Path(__file__).parents[1]/'candidate_nav/live_direct.py'
        method=next(m for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef)
                    for m in n.body if isinstance(m,ast.AsyncFunctionDef) and m.name=='execute_action')
        class Twist:
            def __init__(self):self.linear=SimpleNamespace(x=0.);self.angular=SimpleNamespace(z=0.)
        for delay in (.10,.18):
            for gain in (.8,1.5):
                for action,spec in ACTION_PLAN_SPEC.items():
                    if 'angle_rad' not in spec:continue
                    with self.subTest(delay=delay,gain=gain,action=action):
                        clock=0.;heading=3.10;velocity=0.;commanded=0.;samples=[(0.,heading)]
                        async def sleep(dt):
                            nonlocal clock,heading,velocity
                            for _ in range(round(dt/.005)):
                                velocity+=(commanded*gain-velocity)*(1-math.exp(-.005/.08))
                                heading+=velocity*.005;clock+=.005;samples.append((clock,heading))
                        def pose():
                            sample=max(0,int((clock-delay)/.1))*.1
                            value=min(samples,key=lambda row:abs(row[0]-sample))[1]
                            return [0.,0.,math.atan2(math.sin(value),math.cos(value))]
                        def publish(cmd):
                            nonlocal commanded
                            commanded=cmd.angular.z
                            self.assertLessEqual(abs(commanded),abs(spec['angular_rad_s'])/1.8+1e-8)
                        def stop():
                            nonlocal commanded
                            commanded=0.
                        a=SimpleNamespace(get_pose=pose,fresh=lambda:True,action_spec=ACTION_PLAN_SPEC,
                            executed=0,completed=0,finished=asyncio.Event(),action_history=[],log=lambda *args:None,
                            stop_pub=SimpleNamespace(armed=False,last_decision='allowed',publish=publish,stop=stop))
                        ns=dict(math=math,time=SimpleNamespace(monotonic=lambda:clock),asyncio=SimpleNamespace(sleep=sleep),Twist=Twist)
                        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
                        target=rollout(pose(),[action],ACTION_PLAN_SPEC)[-1]
                        result=await ns['execute_action'](a,action,target_pose=target)
                        error=abs(math.atan2(math.sin(heading-target[2]),math.cos(heading-target[2])))
                        self.assertLess(error,math.radians(5.625))
                        self.assertEqual(commanded,0.)
                        self.assertEqual(result['outcome'],'duration_complete')
