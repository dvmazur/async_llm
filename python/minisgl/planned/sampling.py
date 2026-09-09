"""Ready-only sampling batches outside the decoder graph.

Probabilities/temperature/top-k/top-p use the unchanged FlashInfer sampler.
Batching changes RNG consumption order, not the requested distributions.
No wait timer or dependency on another client reaching a future stage.
"""
import asyncio
from collections import OrderedDict,Counter

import torch

from minisgl.engine.sample import Sampler, BatchSamplingArgs


class CachedSampler(Sampler):
    """Bounded immutable parameter tensors, reusable across different streams."""
    def __init__(self,device,vocab_size,*,cache_entries=64):
        super().__init__(torch.device(device),vocab_size)
        if cache_entries<1:raise ValueError("sampling cache must be bounded and nonempty")
        self.cache_entries=cache_entries
        self._parameters=OrderedDict()
        self.parameter_builds=0

    def prepare_params(self,params):
        if all(p.is_greedy for p in params):return BatchSamplingArgs(None)
        key=tuple((p.temperature,p.top_k,p.top_p) for p in params)
        cached=self._parameters.get(key)
        if cached is None:
            args=super().prepare_params(params)
            ready=None
            if self.device.type=='cuda':
                ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(self.device))
            cached=args,ready
            self._parameters[key]=cached
            self.parameter_builds+=1
            if len(self._parameters)>self.cache_entries:self._parameters.popitem(last=False)
        self._parameters.move_to_end(key)
        args,ready=cached
        if ready is not None:
            stream=torch.cuda.current_stream(self.device)
            stream.wait_event(ready)
            # An evicted tensor can still be read on this consumer stream.
            for t in (args.temperatures,args.top_k,args.top_p):
                if t is not None:t.record_stream(stream)
        return args

    @torch.inference_mode()
    def warmup(self):
        """Compile/load actual stochastic sampling before rollout, preserving RNG."""
        from minisgl.core import SamplingParams
        device=torch.cuda.current_device() if self.device.index is None else self.device.index
        with torch.random.fork_rng(devices=[device]):
            logits=torch.zeros(1,self.vocab_size,device=self.device)
            params=self.prepare_params([SamplingParams(temperature=.8,top_k=min(20,self.vocab_size),top_p=.9)])
            self.sample(logits,params)
            torch.cuda.current_stream(self.device).synchronize()


class SamplingBatcher:
    def __init__(self,sampler,*,max_batch=128):
        if max_batch<1:raise ValueError("max sampling batch must be positive")
        self.sampler,self.max_batch=sampler,max_batch
        self.pending=[]
        self.callback=None
        self.closed=False
        self.batch_sizes=Counter()
        self.epoch=0

    async def sample(self,logits,params):
        """Input logits must remain unchanged until this await completes.

        Strong tensor references retain ownership; producer events handle
        callers using another CUDA stream. Returned tokens own their storage.
        """
        if self.closed:raise RuntimeError("sampling batcher is closed")
        if (not isinstance(logits,torch.Tensor) or logits.ndim<1
                or logits.shape[-1]!=self.sampler.vocab_size or logits.numel()!=self.sampler.vocab_size
                or not logits.is_floating_point()):
            raise ValueError("sample expects exactly one floating-point vocab row")
        loop=asyncio.get_running_loop()
        future=loop.create_future()
        ready=None
        if logits.is_cuda:
            ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(logits.device))
        self.pending.append((future,logits,params,ready))
        if self.callback is None:self.callback=loop.call_soon(self._flush)
        return await future

    @torch.no_grad()
    def _flush(self):
        self.callback=None
        self.epoch+=1
        pending,self.pending=self.pending,[]
        groups={}
        for req in pending:
            future,logits,params,_=req
            if not future.cancelled():
                # Never approximate greedy rows by tiny-temperature sampling.
                groups.setdefault((logits.device,logits.dtype,params.is_greedy),[]).append(req)
        for requests in groups.values():
            for offset in range(0,len(requests),self.max_batch):
                group=requests[offset:offset+self.max_batch]
                try:
                    device=group[0][1].device
                    if device.type=='cuda':
                        stream=torch.cuda.current_stream(device)
                        for _,row,_,ready in group:
                            stream.wait_event(ready);row.record_stream(stream)
                    logits=torch.stack([row.reshape(-1) for _,row,_,_ in group])
                    params=self.sampler.prepare_params([p for _,_,p,_ in group])
                    tokens=self.sampler.sample(logits,params)
                    self.batch_sizes[len(group)]+=1
                    for i,(future,row,_,_) in enumerate(group):
                        if not future.done():future.set_result(tokens[i].reshape(row.shape[:-1]))
                except Exception as exc:
                    for future,_,_,_ in group:
                        if not future.done():future.set_exception(exc)

    def close(self):
        self.closed=True
        if self.callback is not None:self.callback.cancel();self.callback=None
        pending,self.pending=self.pending,[]
        for future,_,_,_ in pending:
            if not future.done():future.set_exception(RuntimeError("sampling batcher is closed"))
