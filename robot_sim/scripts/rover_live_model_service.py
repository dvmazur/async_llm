"""Reuse or replace our managed model, leaving foreign GPU jobs untouched."""
import argparse,contextlib,fcntl,json,os,shlex,shutil,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from robot_sim.candidate_nav.task_policy import source_hash
from robot_sim.candidate_nav.hardware_profile import settings,discover,choose_devices,PROFILES
from robot_sim.candidate_nav.model_catalog import MODELS,model_key,model_path,validate_model,model_tp,select_devices,gpu_matches

def health(config):
    try:
        with socket.create_connection(('127.0.0.1',18766),timeout=3) as c:
            c.sendall(b'{"health":true}\n')
            with c.makefile('rb') as f:data=json.loads(f.readline(65536))
    except (OSError,ValueError):return None
    if (data.get('ready') is True and data.get('backend')=='native_async_llm'
            and data.get('model') in [str(model_path(ROOT,k)) for k in MODELS]
            and data.get('tp_size')==model_tp(config['profile'],model_key(Path(data['model']).name))):return data
    raise RuntimeError('Port 18766 serves an incompatible backend; no process stopped')

def session_exists(config):
    return subprocess.run(['tmux','has-session','-t',config['session']],capture_output=True).returncode==0

def owned_pids(config):
    if not session_exists(config):return set()
    result=subprocess.run(['tmux','display-message','-p','-t',config['session'],'#{pane_pid}'],capture_output=True,text=True,check=True)
    pending=[int(result.stdout.strip())];found=set()
    while pending:
        pid=pending.pop()
        if pid in found:continue
        found.add(pid)
        try:pending.extend(map(int,Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
        except OSError:pass
    return found

def stop(config):
    pids=owned_pids(config)
    if not pids:return
    subprocess.run(['tmux','send-keys','-t',config['session'],'C-c'],check=True)
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        living=[]
        for pid in pids:
            try:
                if Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z':living.append(pid)
            except OSError:pass
        if not living and not session_exists(config):return
        time.sleep(.25)
    raise RuntimeError('Our previous model has not fully exited; no second model loaded or forced kill issued')

@contextlib.contextmanager
def exclusive_use(runtime):
    with (runtime/'model-use.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A rover run is using this model. Stop/finish that run before switching or stopping the model.')
        yield

def reusable(data,config,key,gpu):
    return (data is not None and data.get('model')==str(model_path(ROOT,key))
            and data.get('policy_source_sha256')==source_hash() and data.get('max_context_tokens')==12288
            and data.get('trajectory_policy')=='evidence'
            and (config['profile']!='h100' or (data.get('hardware_profile')=='h100'
                and gpu_matches(key,gpu,data.get('gpu_physical_ids'))
                and (data.get('fp8_backend_requested',data.get('fp8_backend'))==config['fp8_backend']))))

STARTUP_TIMEOUT_S = 1800

def start(config,key,gpu):
    selected=select_devices(config,key,gpu,discover())
    chosen=str(selected[0]['index']) if config['profile']=='h100' else 'auto'
    command=f"cd {shlex.quote(str(ROOT))} && exec bash robot_sim/scripts/run_rover_live_model.sh --gpu {chosen} --model {key} > robot_sim/.runtime/live_rover/model.log 2>&1"
    subprocess.run(['tmux','new-session','-d','-s',config['session'],command],check=True)
    print(f"Loading {MODELS[key]} on GPU {','.join(str(d['index']) for d in selected)}…",flush=True)
    started=time.monotonic();deadline=started+STARTUP_TIMEOUT_S;next_status=0
    while time.monotonic()<deadline:
        data=health(config)
        if data is not None:
            if not reusable(data,config,key,gpu):raise RuntimeError('Started model does not match requested configuration')
            return data
        if not session_exists(config):raise RuntimeError('Model startup failed; inspect robot_sim/.runtime/live_rover/model.log')
        if time.monotonic()>=next_status:print(f'Waiting for Qwen initialization… {int(time.monotonic()-started)} / {STARTUP_TIMEOUT_S} s',flush=True);next_status=time.monotonic()+10
        time.sleep(1)
    raise RuntimeError(f'Model startup timeout after {STARTUP_TIMEOUT_S} s; inspect robot_sim/.runtime/live_rover/model.log')

def archive_startup_failure(key):
    """Keep the failed worker's diagnostic log before rollback starts a new one."""
    runtime=ROOT/'robot_sim/.runtime/live_rover';log=runtime/'model.log'
    if not log.exists():return None
    saved=runtime/'startup-failures'/f'{time.time_ns()}-{key}.log'
    saved.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(log,saved)
    return saved

def ready(config,key):print(f"QWEN READY · {MODELS[key]} · TP={model_tp(config['profile'],key)}",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=model_key,default='27b')
    p.add_argument('--gpu',choices=('auto','0','1'));p.add_argument('--stop',action='store_true');p.add_argument('--check-only',action='store_true');args=p.parse_args()
    config=settings(ROOT);gpu=args.gpu or config['gpu'];key=args.model
    if args.check_only:
        raise SystemExit(subprocess.run([sys.executable,str(ROOT/'robot_sim/scripts/rover_live_launch.py'),'--check-only','--gpu',gpu,'--model',key]).returncode)
    runtime=ROOT/'robot_sim/.runtime/live_rover';runtime.mkdir(parents=True,exist_ok=True)
    with (runtime/'model-service.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.stop:
            with exclusive_use(runtime):stop(config)
            print('Our Qwen service stopped');return
        validate_model(ROOT,key) # Never unload a working model for incomplete weights.
        data=health(config)
        if reusable(data,config,key,gpu):ready(config,key);return
        if data is not None and not session_exists(config):raise RuntimeError('Model runs outside our managed session; not stopped')
        # Estimate post-stop availability using only PIDs in our managed tree.
        # A busy foreign GPU rejects the switch while the old model still runs.
        devices=discover()
        if data:
            pids=owned_pids(config)
            rows=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
            jobs=[line.split(',') for line in rows.splitlines() if line.strip()]
            for d in devices:
                own=[j for j in jobs if j[1].strip()==d['uuid'] and int(j[0]) in pids]
                d['free_mib']+=sum(int(j[2]) for j in own)
                d['busy']=any(j[1].strip()==d['uuid'] and int(j[0]) not in pids for j in jobs)
        select_devices(config,key,gpu,devices)
        with exclusive_use(runtime):
            old_key=model_key(Path(data['model']).name) if data else None
            old_gpu=('0' if data.get('gpu_physical_ids')=='0,1' else data.get('gpu_physical_ids','auto')) if data and config['profile']=='h100' else 'auto'
            if session_exists(config):
                print('Unloading our previous model…',flush=True);stop(config)
            try:start(config,key,gpu)
            except Exception as error:
                try:
                    saved=archive_startup_failure(key)
                    print(f'Model startup failed: {error}. Preserved diagnostic log: {saved}',file=sys.stderr,flush=True)
                except OSError as logging_error:
                    print(f'Could not preserve startup log: {logging_error}',file=sys.stderr,flush=True)
                # Do not start the rover with the wrong model if replacement fails.
                stop(config)
                if old_key:
                    print('Restoring previous working model after startup failure…',flush=True)
                    try:start(config,old_key,old_gpu)
                    except Exception as rollback:print(f'Restore failed: {rollback}',file=sys.stderr)
                raise
        ready(config,key)
if __name__=='__main__':
    try:main()
    except (RuntimeError,ValueError) as e:raise SystemExit(str(e))
