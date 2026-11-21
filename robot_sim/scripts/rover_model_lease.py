"""Hold a shared model-use lock until the supervising SSH channel closes."""
import argparse,fcntl,json,socket,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from robot_sim.candidate_nav.model_catalog import MODELS,model_key
p=argparse.ArgumentParser();p.add_argument('--model',type=model_key,default='27b');a=p.parse_args()
folder=ROOT/'robot_sim/.runtime/live_rover';folder.mkdir(parents=True,exist_ok=True)
with (folder/'model-use.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_SH)
    with socket.create_connection(('127.0.0.1',18766),timeout=5) as connection:
        connection.sendall(b'{"health":true}\n')
        with connection.makefile('rb') as stream:data=json.loads(stream.readline())
    if data.get('ready') is not True or Path(data.get('model','')).name!=MODELS[a.model]:
        raise SystemExit('Selected model changed before run; retry start. No rover command issued.')
    print('MODEL LEASE READY',flush=True)
    for line in sys.stdin:pass
