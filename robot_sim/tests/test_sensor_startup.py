"""Startup policy/recovery tests: no ROS, network or motor connections."""
import copy
import fcntl
import os
from pathlib import Path
import signal
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from robot_sim.candidate_nav import startup_health as health


def good():
    return dict(ages=dict(camera=.1,scan=.1,odom=.1,map=.3,pose=.1),
                lidar_tf=True,odom_valid=True,invalid_odom_streak=0,valid_scan_rays=450,external_motion=False)


def lost():
    result=good();result.update(odom_valid=False,invalid_odom_streak=100)
    result['ages'].update(pose=None,map=40.)
    return result


class FakeProbe:
    def __init__(self,before=None,after=None):
        self.now=0.;self.before=before or good();self.after=after or good()
        self.recoveries=0;self.clears=0;self.events=[]
    def poll(self,seconds):self.now+=seconds
    def snapshot(self):return copy.deepcopy(self.after if self.clears else self.before)
    def recover(self):self.recoveries+=1
    def clear_mapping(self):self.clears+=1
    def status(self,state,snapshot,attempted=False):self.events.append((state,attempted))
    def run(self,**kwargs):return health.preflight(self,clock=lambda:self.now,**kwargs)


class StartupPolicyTests(unittest.TestCase):
    def test_healthy_sensors_need_stable_samples_and_no_reset(self):
        p=FakeProbe();p.run()
        self.assertGreaterEqual(p.now,1.);self.assertLess(p.now,2.)
        self.assertEqual(p.recoveries,0);self.assertEqual(p.events[-1],('sensors_ready',False))
    def test_latched_invalid_icp_recovers_once_before_mission(self):
        p=FakeProbe(lost());p.run()
        self.assertEqual((p.recoveries,p.clears),(1,1));self.assertGreaterEqual(p.now,9.)
        self.assertEqual(p.events[-1],('sensors_ready',True))
    def test_persistent_tracking_failure_is_bounded_not_a_reset_loop(self):
        p=FakeProbe(lost(),lost())
        with self.assertRaisesRegex(RuntimeError,'ICP'):p.run(timeout=12.)
        self.assertEqual(p.recoveries,1);self.assertLess(p.now,12.1)
    def test_missing_or_stale_other_sensors_never_triggers_mapping_restart(self):
        for change in ('camera','scan','lidar_tf','external_motion','future_scan','few_returns','missing_odom'):
            s=lost()
            if change in ('camera','scan'):s['ages'][change]=None
            elif change=='lidar_tf':s[change]=False
            elif change=='external_motion':s[change]=True
            elif change=='future_scan':s['ages']['scan']=-1.
            elif change=='few_returns':s['valid_scan_rays']=5
            else:s.update(invalid_odom_streak=0);s['ages']['odom']=None
            with self.subTest(change=change):
                p=FakeProbe(s)
                with self.assertRaises(RuntimeError):p.run(timeout=10.)
                self.assertEqual(p.recoveries,0)
    def test_single_invalid_packet_does_not_restart(self):
        s=lost();s['invalid_odom_streak']=1
        self.assertFalse(health.recovery_allowed(s,20))
    def test_recovery_failure_propagates_without_retry(self):
        p=FakeProbe(lost())
        with patch.object(p,'recover',side_effect=RuntimeError('archive failed')) as recover:
            with self.assertRaisesRegex(RuntimeError,'archive failed'):p.run()
            self.assertEqual(recover.call_count,1);self.assertEqual(p.clears,0)
    def test_stability_window_restarts_on_a_bad_packet(self):
        p=FakeProbe()
        p.snapshot=lambda:lost() if .9<p.now<1.1 else good()
        p.run();self.assertGreaterEqual(p.now,2.1);self.assertEqual(p.recoveries,0)
    def test_invalid_covariance_quaternion_and_nonfinite_position_are_rejected(self):
        q=SimpleNamespace(x=0.,y=0.,z=0.,w=1.)
        v=SimpleNamespace(x=0.,y=0.,z=0.)
        msg=SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(position=v,orientation=q),covariance=[0.]*36))
        self.assertTrue(health.odom_valid(msg))
        for name in ('covariance','quaternion','position'):
            sample=copy.deepcopy(msg)
            if name=='covariance':sample.pose.covariance[35]=9999.
            elif name=='quaternion':sample.pose.pose.orientation.w=0.
            else:sample.pose.pose.position.x=float('nan')
            self.assertFalse(health.odom_valid(sample))


class RestartTests(unittest.TestCase):
    def test_inherited_lock_descriptor_and_wrong_file(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'run.lock';p.touch()
            with p.open('a') as lock:
                health.lock_run(lock.fileno(),p)
                with p.open('a') as other:
                    with self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
                wrong=Path(directory)/'wrong';wrong.touch()
                with wrong.open('a') as other:
                    with self.assertRaisesRegex(RuntimeError,'descriptor'):health.lock_run(other.fileno(),p)
    def test_restart_signals_only_exact_mapping_workers(self):
        workers={101:(sorted(health.EXECUTABLES)[0],'11'),102:(sorted(health.EXECUTABLES)[1],'12')}
        remaining=dict(workers);calls=[]
        def kill(pid,sig):
            self.assertEqual(sig,signal.SIGTERM);calls.append(pid);remaining.pop(pid)
        with patch.object(health,'lock_run'),patch.object(Path,'is_file',return_value=True),patch.object(health,'mapping_workers',side_effect=lambda:dict(remaining)),patch.object(health.os,'kill',side_effect=kill),patch.object(health.subprocess,'run',return_value=SimpleNamespace(returncode=1)) as run:
            health.restart_mapping(9)
            self.assertEqual(calls,[101,102])
            self.assertEqual(run.call_args.args[0][:4],['tmux','new-session','-d','-s'])
            self.assertFalse(any('send-keys' in c.args[0] for c in run.call_args_list))
    def test_ambiguous_processes_and_missing_launcher_do_not_stop_anything(self):
        with patch.object(health,'lock_run'),patch.object(Path,'is_file',return_value=False),patch.object(health.os,'kill') as kill:
            with self.assertRaisesRegex(RuntimeError,'missing'):health.restart_mapping(9)
            kill.assert_not_called()
        with patch.object(health,'lock_run'),patch.object(Path,'is_file',return_value=True),patch.object(health,'mapping_workers',return_value={}),patch.object(health.os,'kill') as kill:
            with self.assertRaisesRegex(RuntimeError,'exactly one'):health.restart_mapping(9)
            kill.assert_not_called()
    def test_workers_that_do_not_exit_are_not_force_killed_or_duplicated(self):
        workers={101:(sorted(health.EXECUTABLES)[0],'11'),102:(sorted(health.EXECUTABLES)[1],'12')};clock=[0.]
        def sleep(seconds):clock[0]+=seconds
        with patch.object(health,'lock_run'),patch.object(Path,'is_file',return_value=True),patch.object(health,'mapping_workers',return_value=workers),patch.object(health.os,'kill') as kill,patch.object(health.subprocess,'run',return_value=SimpleNamespace(returncode=1)) as run:
            with self.assertRaisesRegex(RuntimeError,'did not stop'):
                health.restart_mapping(9,clock=lambda:clock[0],sleep=sleep)
            self.assertEqual(kill.call_count,2)
            self.assertFalse(any('new-session' in c.args[0] for c in run.call_args_list))


if __name__=='__main__':unittest.main()


class ProbeAdapterTests(unittest.TestCase):
    def adapter(self,folder,*,available=True,result=0,map_seen=True):
        import ast
        from concurrent.futures import Future
        source=Path(health.__file__).with_name('sensor_startup.py')
        tree=ast.parse(source.read_text());main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
        cls=next(x for x in main.body if isinstance(x,ast.ClassDef) and x.name=='Probe')
        request_type=SimpleNamespace(Request=lambda:SimpleNamespace())
        events=[];future=Future();future.set_result(SimpleNamespace(result=result))
        client=SimpleNamespace(wait_for_service=lambda **kw:available,
                call_async=lambda req:events.append('archive') or future)
        ns=dict(Node=object,Path=Path,time=__import__('time'),math=__import__('math'),json=__import__('json'),
                evidence=Path(folder),SerializePoseGraph=request_type,args=SimpleNamespace(lock_fd=9),
                rclpy=SimpleNamespace(spin_until_future_complete=lambda *a,**kw:None),
                restart_mapping=lambda fd:events.append('restart'),odom_valid=health.odom_valid,
                Buffer=lambda:events.append('new-buffer') or object(),
                TransformListener=lambda *a:events.append('new-listener') or object())
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),ns)
        probe=object.__new__(ns['Probe']);probe.map_message=object() if map_seen else None
        probe.create_client=lambda *a:client;probe.destroy_client=lambda c:events.append('destroy-client')
        return probe,events,ns
    def test_observed_map_is_archived_before_workers_are_stopped(self):
        with tempfile.TemporaryDirectory() as folder:
            p,events,_=self.adapter(folder);p.recover()
            self.assertLess(events.index('archive'),events.index('restart'))
            self.assertTrue(__import__('json').loads((Path(folder)/'recovery.json').read_text())['graph_saved'])
    def test_failed_archive_of_existing_map_prevents_restart(self):
        for available,result in [(False,0),(True,255)]:
            with self.subTest(available=available,result=result),tempfile.TemporaryDirectory() as folder:
                p,events,_=self.adapter(folder,available=available,result=result)
                with self.assertRaises(RuntimeError):p.recover()
                self.assertNotIn('restart',events)
    def test_cache_is_cleared_after_mapping_process_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            p,events,_=self.adapter(folder);p.latest={'map':object(),'odom':object(),'scan':object()}
            p.get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=100*10**9))
            p.invalid_streak=99;p.listener=SimpleNamespace(unregister=lambda:events.append('unregister'))
            p.clear_mapping()
            self.assertEqual(set(p.latest),{'scan'});self.assertEqual(p.invalid_streak,0)
            self.assertIsNone(p.map_message);self.assertIn('new-buffer',events)
    def test_recent_transport_does_not_hide_old_measurement_stamp(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            p,_,ns=self.adapter(folder)
            header=lambda stamp:SimpleNamespace(stamp=SimpleNamespace(sec=int(stamp),nanosec=round((stamp-int(stamp))*1e9)))
            p.latest={'scan':(SimpleNamespace(header=header(90.)),9.9)}
            p.lidar_frame='lidar';p.rays=450;p.motion_at=-100;p.invalid_streak=3
            p.camera=SimpleNamespace(latest=(None,99.9,9.9))
            p.buffer=SimpleNamespace(lookup_transform=lambda *a:SimpleNamespace(header=header(99.9)))
            p.get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=100*10**9))
            ns['Time']=lambda:None
            with patch.object(ns['time'],'monotonic',return_value=10.):snapshot=p.snapshot()
            self.assertEqual(snapshot['ages']['scan'],10.)
            self.assertIn('scan',health.blockers(snapshot))

    def test_old_queued_mapping_message_is_rejected_after_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            p,_,_=self.adapter(folder);p.mapping_after_stamp=100.;p.latest={}
            msg=SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=99,nanosec=0)))
            p.receive('map',msg);self.assertNotIn('map',p.latest)
            msg.header.stamp.sec=101
            p.receive('map',msg);self.assertIs(p.latest['map'][0],msg)
