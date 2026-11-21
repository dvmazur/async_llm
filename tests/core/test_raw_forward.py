"""Raw forwards must preserve logits without an unused sampler/host round-trip."""
import asyncio
import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.llm import AsyncLLM
from minisgl.scheduler import AsyncCacheEngine
from minisgl.shared_cache import AsyncContext
from test_async_cache_engine import StubSession,VOCAB,CPU

class CountingSampler:
    def __init__(self):self.rows=[]
    def prepare_params(self,params):self.rows.append(list(params));return None
    def sample(self,logits,args):return logits.argmax(-1)

def engine():
    sampler=CountingSampler();session=StubSession()
    return AsyncCacheEngine(session=session,sampler=sampler),session,sampler

def context(e):
    b=e.create_block();e.submit_prefill(torch.tensor([1,2]),write_to=b);e.tick()
    return AsyncContext(cache_view=[b])

@pytest.mark.parametrize('return_logits',[False,True])
def test_raw_forward_skips_sampler_and_returns_unchanged_logits(return_logits):
    e,s,sampler=engine();ctx=context(e)
    f=e.submit_decode(ctx,5,return_logits=return_logits,sample_token=False);e.tick()
    assert sampler.rows==[] and len(s.decode_calls)==1
    assert e.statistics==dict(prefill_requests=1,prefill_input_tokens=2,decode_rows=1,
                             raw_forward_rows=1,sampler_rows=0)
    if return_logits:
        token,logits=f.result();assert token is None and int(logits.argmax())==7
        logits.fill_(1) # The caller still owns an ordinary, mutable tensor.
    else:assert f.result() is None
    assert ctx.output_block.token_ids==[1,2,5]

def test_mixed_greedy_batch_samples_only_consumers_and_preserves_raw_rows():
    e,s,sampler=engine();raw=context(e);sampled=context(e)
    a=e.submit_decode(raw,5,return_logits=True,sample_token=False)
    b=e.submit_decode(sampled,9,forbid_ids=[11]);e.tick()
    assert len(s.decode_calls)==1 and len(sampler.rows)==1 and len(sampler.rows[0])==1
    assert a.result()[0] is None and int(a.result()[1].argmax())==7
    assert b.result()==10

def test_mixed_stochastic_batch_preserves_original_sampler_row_order():
    e,s,sampler=engine();raw=context(e);sampled=context(e)
    a=e.submit_decode(raw,5,return_logits=True,sample_token=False)
    params=SamplingParams(temperature=.7,max_tokens=1)
    b=e.submit_decode(sampled,9,sampling_params=params);e.tick()
    assert len(sampler.rows)==1 and len(sampler.rows[0])==2
    assert sampler.rows[0][0].is_greedy and sampler.rows[0][1] is params
    assert a.result()[0] is None and b.result()==11

def test_raw_forward_rejects_meaningless_sampling_constraints():
    e,_,_=engine();ctx=context(e)
    with pytest.raises(ValueError,match='Raw forward'):
        e.submit_decode(ctx,5,forbid_ids=[7],sample_token=False)
    with pytest.raises(ValueError,match='Raw forward'):
        e.submit_decode(ctx,5,sampling_params=SamplingParams(temperature=0,max_tokens=1),sample_token=False)

def test_unified_forward_skips_sampling_but_async_generate_keeps_it():
    async def run():
        e,s,sampler=engine();llm=AsyncLLM(async_engine=e)
        try:
            block=await llm.create_block();await llm.forward([1,2],write_to=block)
            output=await llm.forward([5],cache_view=[block])
            assert int(output.logits.argmax())==7
            quiet=await llm.forward([7],cache_view=[block],return_logits=False)
            assert quiet.logits is None and sampler.rows==[]
            ctx=AsyncContext(cache_view=[block])
            assert [t async for t in llm.async_generate(ctx,first_token_id=3,max_steps=2)]==[5,7]
            assert len(sampler.rows)==2
        finally:await llm.close()
    asyncio.run(run())

def test_reference_switch_keeps_the_old_forward_sampling_for_comparisons():
    async def run():
        e,_,sampler=engine();llm=AsyncLLM(async_engine=e,sample_forward_token=True)
        try:
            block=await llm.create_block();await llm.forward([1,2],write_to=block)
            output=await llm.forward([5],cache_view=[block])
            assert int(output.logits.argmax())==7 and len(sampler.rows)==1
        finally:await llm.close()
    asyncio.run(run())
