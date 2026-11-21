"""Bounded per-model vision-tower cache, keyed by exact processed pixels/grid.

Language KV is never reused across contexts. Missing images retain their order
and share one tower invocation. A model reload creates a fresh owner/cache.
"""
from collections import OrderedDict
import hashlib
import os
import torch


class VisionEmbeddingCache:
    def __init__(self, capacity_bytes):
        self.capacity=max(0,int(capacity_bytes));self.entries=OrderedDict();self.bytes=0
        self.hits=0;self.misses=0;self.tower_calls=0

    def forward(self, visual, pixels, grid, merge):
        if not self.capacity:return visual.forward(pixels,grid)
        cpu=pixels.detach().contiguous().cpu();shapes=grid.detach().cpu().tolist()
        sizes=[int(t*h*w) for t,h,w in shapes]
        if sum(sizes)!=len(cpu):raise ValueError('Vision grid does not match processed pixels')
        parts=list(cpu.split(sizes));keys=[]
        for part,shape in zip(parts,shapes):
            key=(str(pixels.device),str(pixels.dtype),tuple(part.shape),tuple(shape),int(merge),
                 hashlib.sha256(part.view(torch.uint8).numpy().tobytes()).digest())
            keys.append(key)
        outputs=[None]*len(keys);missing=[]
        for i,key in enumerate(keys):
            if key in self.entries:
                self.hits+=1;value=self.entries.pop(key);self.entries[key]=value;outputs[i]=value
            else:self.misses+=1;missing.append(i)
        if missing:
            # All misses keep the original tensor and batch layout exactly.
            tower_pixels=pixels if len(missing)==len(keys) else torch.cat([parts[i] for i in missing]).to(pixels.device)
            tower_grid=grid if len(missing)==len(keys) else grid[missing]
            self.tower_calls+=1
            encoded=visual.forward(tower_pixels,tower_grid)
            counts=[sizes[i]//(merge*merge) for i in missing]
            if sum(counts)!=len(encoded):raise ValueError('Vision embedding/grid mismatch')
            for i,value in zip(missing,encoded.split(counts)):
                outputs[i]=value
                n=value.numel()*value.element_size();key=keys[i]
                if n<=self.capacity:
                    if key in self.entries:self.bytes-=self.entries.pop(key).numel()*value.element_size()
                    while self.entries and self.bytes+n>self.capacity:
                        _,old=self.entries.popitem(last=False);self.bytes-=old.numel()*old.element_size()
                    self.entries[key]=value.detach().clone();self.bytes+=n
        return torch.cat(outputs,dim=0)


def cached_vision(visual,pixels,grid,merge):
    capacity=int(os.environ.get('MINISGL_VISION_CACHE_MIB','0'))*1024*1024
    if not capacity:return visual.forward(pixels,grid)
    cache=getattr(visual,'_exact_embedding_cache',None)
    if cache is None or cache.capacity!=capacity:
        cache=VisionEmbeddingCache(capacity);visual._exact_embedding_cache=cache
    return cache.forward(visual,pixels,grid,merge)
