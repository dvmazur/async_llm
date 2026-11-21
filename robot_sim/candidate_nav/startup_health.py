"""Startup diagnosis and bounded mapping recovery. No motor interfaces."""
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import time

LIMITS = dict(camera=2.5, scan=.30, odom=.40, map=12., pose=1.)
LABELS = dict(camera='камера', scan='лидар', odom='ICP-одометрия', map='SLAM-карта',
              pose='TF map → base_link', lidar_tf='TF base_link → lidar',
              odom_valid='ICP потерял сопоставление сканов', external_motion='другое управление движением')


def odom_valid(message):
    q=message.pose.pose.orientation;v=message.pose.pose.position
    values=[v.x,v.y,v.z,q.x,q.y,q.z,q.w]
    return (all(math.isfinite(x) for x in values)
            and .5 < sum(x*x for x in (q.x,q.y,q.z,q.w)) < 1.5
            and all(math.isfinite(message.pose.covariance[i])
                    and 0 <= message.pose.covariance[i] < 1000 for i in (0,7,35)))


def blockers(snapshot):
    result=[]
    for name,limit in LIMITS.items():
        age=snapshot.get('ages',{}).get(name)
        if age is None or not math.isfinite(age) or not -.15 <= age < limit:result.append(name)
    if not snapshot.get('lidar_tf'):result.append('lidar_tf')
    if not snapshot.get('odom_valid'):result.append('odom_valid')
    if snapshot.get('external_motion'):result.append('external_motion')
    return result


def recovery_allowed(snapshot,elapsed):
    """Only a sustained invalid ICP stream, with a live, usable scan and camera."""
    missing=blockers(snapshot)
    return (elapsed >= 8 and snapshot.get('invalid_odom_streak',0) >= 15
            and snapshot.get('valid_scan_rays',0) >= 30
            and all(x not in missing for x in ('scan','camera','lidar_tf','external_motion'))
            and 'odom_valid' in missing)


def preflight(probe, *, clock=time.monotonic, timeout=75.):
    started=clock();deadline=started+timeout;stable=None;attempted=False;next_status=0
    while clock()<deadline:
        probe.poll(.05)
        snapshot=probe.snapshot();missing=blockers(snapshot);now=clock()
        if not missing:
            if stable is None:stable=now
            if now-stable>=1.:
                probe.status('sensors_ready',snapshot,attempted=attempted)
                return snapshot
        else:stable=None
        if now>=next_status:
            probe.status('waiting_for_sensors',snapshot,attempted=attempted)
            next_status=now+2.
        if not attempted and recovery_allowed(snapshot,now-started):
            attempted=True
            probe.status('recovering_mapping',snapshot,attempted=True)
            probe.recover()
            # Discard cached TF/map and require new samples from the new workers.
            probe.clear_mapping();stable=None
    snapshot=probe.snapshot()
    probe.status('sensor_startup_failed',snapshot,attempted=attempted)
    raise RuntimeError('Sensor startup failed: '+', '.join(LABELS[x] for x in blockers(snapshot)))


def lock_run(fd, path=Path('/tmp/rbm-live-qwen.lock')):
    # Reuse the entrypoint's lock, rather than opening a racing second lock.
    expected=path.stat();actual=os.fstat(fd)
    if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
        raise RuntimeError('Invalid rover run lock descriptor')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)


EXECUTABLES = {'/opt/ros/jazzy/lib/rtabmap_odom/icp_odometry',
               '/opt/ros/jazzy/lib/slam_toolbox/async_slam_toolbox_node'}
SESSION='rover-mapping-quality'


def mapping_workers(proc=Path('/proc')):
    workers={}
    for item in proc.glob('[0-9]*'):
        try:
            executable=(item/'cmdline').read_bytes().split(b'\0')[0].decode()
            if executable not in EXECUTABLES:continue
            start=(item/'stat').read_text().split(') ',1)[1].split()[19]
            workers[int(item.name)]=(executable,start)
        except (OSError,IndexError,UnicodeError):continue
    return workers


def restart_mapping(fd, *, clock=time.monotonic, sleep=time.sleep):
    lock_run(fd)
    script=Path('/opt/rover-mapping-quality/run_mapping.sh')
    if not script.is_file():raise RuntimeError('Mapping recovery launcher is missing; no process stopped')
    workers=mapping_workers()
    if len(workers)!=2 or {v[0] for v in workers.values()}!=EXECUTABLES:
        raise RuntimeError('Expected exactly one ICP and one SLAM process; no process stopped')
    def run(args,**kw):return subprocess.run(args,timeout=5,check=True,**kw)
    exists=subprocess.run(['tmux','has-session','-t','='+SESSION],capture_output=True,timeout=5).returncode==0
    if mapping_workers()!=workers:
        raise RuntimeError('Mapping workers changed before restart; no process stopped')
    # Signal only exact mapping executables, never an entire tmux session/tree.
    for pid,identity in workers.items():
        if mapping_workers().get(pid)==identity:
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
    deadline=clock()+12
    while clock()<deadline:
        remaining=mapping_workers()
        if not remaining:break
        if any(pid not in workers or workers[pid]!=identity for pid,identity in remaining.items()):
            raise RuntimeError('Mapping workers changed concurrently; replacement not started')
        sleep(.1)
    else:raise RuntimeError('Previous mapping workers did not stop; no forced kill or duplicate started')
    # The old tmux parent may need a moment to reap both workers.
    if exists:
        deadline=clock()+3
        while subprocess.run(['tmux','has-session','-t','='+SESSION],capture_output=True,timeout=5).returncode==0:
            if clock()>=deadline:raise RuntimeError('Previous mapping session did not exit')
            sleep(.1)
    run(['tmux','new-session','-d','-s',SESSION,f'exec bash {script}'])
