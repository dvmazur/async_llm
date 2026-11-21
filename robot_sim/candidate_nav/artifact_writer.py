"""Bounded ordered archival; immutable copies keep sensor/control work independent."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import copy
import json
import threading

class ArtifactWriter:
    def __init__(self, capacity=8):
        self.worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='rover-archive')
        self.pending=deque();self.capacity=capacity;self.lock=threading.Lock()
    def submit(self, function, *args, **kwargs):
        with self.lock:
            while self.pending and self.pending[0].done():self.pending.popleft().result()
            # Backpressure preserves all recordings and their order, including
            # several versions of observation.json. Never overwrite out of order.
            if len(self.pending)>=self.capacity:self.pending.popleft().result()
            self.pending.append(self.worker.submit(function,*args,**kwargs))
    def close(self):
        self.worker.shutdown(wait=True)
        for future in self.pending:future.result()
        self.pending.clear()

def image(agent,value,path,**kwargs):
    writer=getattr(agent,'artifact_writer',None)
    if writer is None:value.save(path,**kwargs)
    else:writer.submit(value.copy().save,path,**kwargs)

def text(agent,path,value):
    writer=getattr(agent,'artifact_writer',None)
    if writer is None:path.write_text(value)
    else:writer.submit(path.write_text,value)

def grid(agent,path,**values):
    import numpy as np
    writer=getattr(agent,'artifact_writer',None)
    if writer is None:np.savez_compressed(path,**values)
    else:writer.submit(np.savez_compressed,path,**{k:copy.deepcopy(v) for k,v in values.items()})
