"""Reuse the unchanged asyncio policy/frontend without allocating an Engine."""
from types import SimpleNamespace
from dataclasses import replace
import asyncio

from minisgl.llm.async_llm import AsyncLLM,logger
from .sampling import CachedSampler,SamplingBatcher


class PlannedAsyncLLM(AsyncLLM):
    """Keep model scheduling intact; coalesce ready manual sampling requests."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.sampling_batcher=SamplingBatcher(self.engine.sampler)

    async def sample(self,logits,**param_overrides):
        logits=getattr(logits,'logits',logits)
        params=replace(self.engine.config.get_default_sampling_params(),**param_overrides)
        return await self.sampling_batcher.sample(logits,params)

    async def close(self):
        self.sampling_batcher.close()
        await super().close()

    async def _engine_loop(self):
        while not self._closed:
            if not self.async_engine.has_work:
                self._work_event.clear()
                await self._work_event.wait()
                if self._closed:return
            # Manual sample() used to complete without suspending a client.
            # Give its consumers the SAME coalescing window after a new flush,
            # not a shortened window that splits one ready model batch in two.
            # The bound prevents independent resampling loops starving inference.
            epoch=self.sampling_batcher.epoch
            quiet=0
            for _ in range(3*self.batching_yield_rounds):
                await asyncio.sleep(0)
                now=self.sampling_batcher.epoch
                quiet=quiet+1 if now==epoch else 0
                epoch=now
                if quiet>=self.batching_yield_rounds and not self.sampling_batcher.pending:break
            try:self.async_engine.tick()
            except Exception:logger.exception('async-cache tick failed')


def async_llm(session,model_path,*,batching_yield_rounds=3,enable_mixed_batch=True,generation_config=None):
    from minisgl.scheduler.async_engine import AsyncCacheEngine
    from minisgl.engine import EngineConfig
    from minisgl.distributed import DistributedInfo
    from transformers import GenerationConfig
    if generation_config is None:generation_config=GenerationConfig.from_pretrained(model_path,local_files_only=True)
    config=EngineConfig(model_path=str(model_path),dtype=session.dtype,tp_info=DistributedInfo(0,1),
                        page_size=session.page_size,generation_config=generation_config)
    sampler=CachedSampler(session.device,session.model.model.embed_tokens.num_embeddings)
    sampler.warmup()
    # AsyncLLM's injected-engine contract uses config/model/sampler metadata for
    # tokenization/default sampling only. No Engine constructor/pools/forward.
    facade=SimpleNamespace(model=session.model,device=session.device,dtype=session.dtype,config=config,sampler=sampler)
    scheduler=AsyncCacheEngine(session=session,sampler=sampler,enable_mixed_batch=enable_mixed_batch)
    return PlannedAsyncLLM(str(model_path),engine=facade,async_engine=scheduler,batching_yield_rounds=batching_yield_rounds)
