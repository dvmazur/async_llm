import asyncio
from types import SimpleNamespace

import pytest
import torch

from minisgl.core import SamplingParams
from minisgl.planned.sampling import SamplingBatcher,CachedSampler


class FakeSampler:
    vocab_size=7
    def __init__(self):self.calls=[];self.params=[];self.failure=False
    def prepare_params(self,params):self.params.append(params);return params
    def sample(self,logits,params):
        if self.failure:raise RuntimeError('injected sampler failure')
        self.calls.append(logits.clone());return logits.argmax(-1)


def test_ready_batch_routes_outputs_owns_tokens_and_preserves_params():
    async def run():
        sampler=FakeSampler();batcher=SamplingBatcher(sampler,max_batch=4)
        rows=torch.eye(7)
        params=[SamplingParams(temperature=.2+i/10,top_k=2+i%3,top_p=.8) for i in range(7)]
        tokens=await asyncio.gather(*(batcher.sample(rows[i],params[i]) for i in range(7)))
        assert [t.item() for t in tokens]==list(range(7))
        assert [len(c) for c in sampler.calls]==[4,3]
        assert [p for ps in sampler.params for p in ps]==params
        saved=[t.clone() for t in tokens]
        await batcher.sample(rows[6].reshape(1,7),params[0])
        for a,b in zip(tokens,saved):torch.testing.assert_close(a,b)
        batcher.close()
        with pytest.raises(RuntimeError,match='closed'):await batcher.sample(rows[0],params[0])
    asyncio.run(run())


def test_cancel_close_exception_and_greedy_do_not_strand_clients():
    async def run():
        sampler=FakeSampler();b=SamplingBatcher(sampler)
        greedy=SamplingParams();stochastic=SamplingParams(temperature=.8,top_k=3,top_p=.9)
        logits=torch.arange(7.).flip(0)
        result=await asyncio.gather(b.sample(logits,greedy),b.sample(logits,stochastic))
        assert len(sampler.calls)==2 and all(x.item()==0 for x in result)
        task=asyncio.create_task(b.sample(logits,stochastic))
        await asyncio.sleep(0);task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        await asyncio.sleep(0)
        assert not b.pending
        sampler.failure=True
        results=await asyncio.gather(b.sample(logits,greedy),b.sample(logits,stochastic),return_exceptions=True)
        assert all(isinstance(e,RuntimeError) for e in results)
        sampler.failure=False
        task=asyncio.create_task(b.sample(logits,greedy));await asyncio.sleep(0);b.close()
        with pytest.raises(RuntimeError,match='closed'):await task
    asyncio.run(run())


def test_planned_frontend_uses_queue_and_per_request_overrides():
    from minisgl.planned.frontend import PlannedAsyncLLM
    async def run():
        sampler=FakeSampler()
        config=SimpleNamespace(get_default_sampling_params=lambda:SamplingParams(temperature=.8,top_k=5,top_p=.9))
        llm=PlannedAsyncLLM(engine=SimpleNamespace(sampler=sampler,config=config),async_engine=object())
        tokens=await asyncio.gather(*(llm.sample(SimpleNamespace(logits=torch.eye(7)[i]),temperature=.5+i/10)
                                      for i in range(7)))
        assert [x.item() for x in tokens]==list(range(7)) and len(sampler.calls)==1
        assert [p.temperature for p in sampler.params[0]]==[.5+i/10 for i in range(7)]
        tokens[0].fill_(4)  # public token outputs are ordinary owned tensors
        assert tokens[1].item()==1
        await llm.close()
    asyncio.run(run())


def test_sampling_flush_preserves_model_batch_coalescing():
    from minisgl.planned.frontend import PlannedAsyncLLM
    class Engine:
        def __init__(self):self.pending=[];self.batches=[]
        @property
        def has_work(self):return bool(self.pending)
        def tick(self):
            pending,self.pending=self.pending,[]
            if not pending:return
            self.batches.append(len(pending))
            for f in pending:f.set_result(torch.arange(7.))
    async def run():
        ae=Engine();sampler=FakeSampler()
        config=SimpleNamespace(get_default_sampling_params=lambda:SamplingParams(temperature=.8,top_k=5,top_p=.9))
        llm=PlannedAsyncLLM(engine=SimpleNamespace(sampler=sampler,config=config),async_engine=ae)
        async def request():
            llm._ensure_loop();future=asyncio.get_running_loop().create_future()
            ae.pending.append(future);llm._work_event.set();return await future
        async def worker(i):
            for _ in range(8):
                logits=await request()
                await llm.sample(logits)
                # Different coroutine depths, as with thinker/descriptor tails.
                if i%2:await asyncio.sleep(0)
        await asyncio.wait_for(asyncio.gather(*(worker(i) for i in range(30))),timeout=5)
        await llm.close()
        assert ae.batches==[30]*8
    asyncio.run(run())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@torch.inference_mode()
def test_cached_params_are_exact_bounded_and_reused_across_streams():
    from minisgl.engine.sample import Sampler
    c=CachedSampler('cuda',32,cache_entries=2);old=Sampler(torch.device('cuda'),32)
    params=[SamplingParams(temperature=.7,top_k=5,top_p=.8),SamplingParams(temperature=1.1,top_k=-1,top_p=1)]
    actual=c.prepare_params(params);expected=old.prepare_params(params)
    for name in ('temperatures','top_k','top_p'):
        torch.testing.assert_close(getattr(actual,name),getattr(expected,name),rtol=0,atol=0)
    assert c.prepare_params(params) is actual and c.parameter_builds==1
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        again=c.prepare_params(params)
        retained=again.temperatures.clone()
    for t in (1.,2.,3.):c.prepare_params([SamplingParams(temperature=t,top_k=5)])
    stream.synchronize()
    torch.testing.assert_close(retained,expected.temperatures)
    assert len(c._parameters)==2


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_gpu_stochastic_batch_distribution_and_producer_stream():
    async def run():
        sampler=CachedSampler('cuda',7)
        rng=torch.cuda.get_rng_state().clone();sampler.warmup()
        assert torch.equal(rng,torch.cuda.get_rng_state())
        b=SamplingBatcher(sampler,max_batch=128)
        params=SamplingParams(temperature=1.,top_k=2,top_p=1.)
        # Two surviving tokens have equal probability; all others excluded.
        logits=torch.tensor([0.,0.,-40.,-40.,-40.,-40.,-40.],device='cuda')
        tokens=await asyncio.gather(*(b.sample(logits,params) for _ in range(2048)))
        got=torch.stack(tokens).cpu()
        assert ((got==0)|(got==1)).all()
        assert .44<float((got==0).float().mean())<.56
        assert sum(b.batch_sizes.values())==16
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            row=torch.zeros(7,device='cuda');row[5]=100
            task=asyncio.create_task(b.sample(row,SamplingParams()))
            await asyncio.sleep(0)
        assert (await task).item()==5
        b.close()
    asyncio.run(run())
