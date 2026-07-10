"""Construction helpers for the shared-cache attention op.

Canonical construction of the query-rotation attention op (arXiv:2512.10931)
bound to an engine's KV pool and rotary cache, consumed by the scheduler's
``SharedCacheService``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from .attention import SharedCacheAttention

if TYPE_CHECKING:
    from minisgl.engine import Engine


def extract_cos_sin_cache(engine: Engine) -> torch.Tensor:
    """
    Extract the ``cos_sin_cache`` tensor from the model's ``RotaryEmbedding``.

    Works for Llama / Qwen / Mistral model families in mini-sglang.
    """
    layers = engine.model.model.layers.op_list
    return layers[0].self_attn.attn.rotary._cos_sin_cache


def build_shared_cache_attention(
    engine: Engine, cos_sin_cache: Optional[torch.Tensor] = None
) -> SharedCacheAttention:
    """
    Build a ``SharedCacheAttention`` bound to *engine*'s KV pool, head config,
    page size, and rotary cache.

    Pass *cos_sin_cache* to override the one extracted from the model.
    """
    cs = cos_sin_cache if cos_sin_cache is not None else extract_cos_sin_cache(engine)
    cs = cs.to(engine.device)
    attn0 = engine.model.model.layers.op_list[0].self_attn.attn
    return SharedCacheAttention(
        kv_cache=engine.kv_cache,
        cos_sin_cache=cs,
        num_qo_heads=attn0.num_qo_heads,
        num_kv_heads=attn0.num_kv_heads,
        head_dim=attn0.head_dim,
        page_size=engine.ctx.page_size,
        dtype=engine.kv_cache.dtype,
        device=engine.device,
    )
