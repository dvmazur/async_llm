"""Build the unchanged legacy session on shared weights, independent storage.

No new planner/evaluator/pool is used to compute reference results. A tiny
engine shell avoids loading a second checkpoint or allocating an unused serving
GDN pool; shared-cache AR uses block-owned state, not that serving pool.
"""
from types import SimpleNamespace
import torch


def legacy_session(model,*,page_size=4,pages=256,max_seq_len=2048,workers=16):
    from minisgl.core import Context,SamplingParams
    from minisgl.kvcache import create_kvcache_pool,PageAllocator
    from minisgl.moe import create_moe_backend
    from minisgl.shared_cache.session import SharedCacheSession
    config=model.model.config
    device=model.model.embed_tokens.weight.device
    dtype=model.model.embed_tokens.weight.dtype
    ctx=Context(page_size)
    ctx.kv_cache=create_kvcache_pool(config,pages,page_size,device=device,dtype=dtype)
    ctx.gdn_state=SimpleNamespace(storage_bytes=0)
    ctx.moe_backend=create_moe_backend("fused")
    ctx.page_table=torch.zeros(workers+1,max_seq_len,device=device,dtype=torch.int32)
    engine=SimpleNamespace(model=model,device=device,ctx=ctx,kv_cache=ctx.kv_cache,
        page_table=ctx.page_table,attn_backend=None,max_seq_len=max_seq_len,
        config=SimpleNamespace(model_config=config,max_prefill_rows=None,
                               get_default_sampling_params=lambda:SamplingParams()),
        page_allocator=PageAllocator(pages,page_size,device))
    result=SharedCacheSession(engine)
    result.sc_gdn.configure_compose_cache(0)
    result.sc_gdn.configure_successor_cache(0)
    return result
