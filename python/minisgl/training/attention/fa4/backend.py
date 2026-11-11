"""Optional Hopper replay using masked FA4 block sequences and a CuTe merge."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from minisgl.training.attention.fa4.mask import historical_mask
from minisgl.training.attention.fa4.merge import merge_attention_parts
from minisgl.training.attention.fa4.plan import FlashAttentionPlan, build_flash_attention_plan
from minisgl.training.mrope import rotate

if TYPE_CHECKING:
    from minisgl.training.trace import Trace


@lru_cache(maxsize=1)
def _flash_attention_function() -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    try:
        from flash_attn.cute import flash_attn_varlen_func
    except ImportError as exc:
        raise ImportError(
            "The fa4 replay backend requires the optional FlashAttention dependencies; "
            "see python/minisgl/training/README.md for installation."
        ) from exc
    return flash_attn_varlen_func


def _forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    inv_freq: torch.Tensor,
    plan: FlashAttentionPlan,
    scaling: float,
    mrope_section: Sequence[int] | None,
) -> torch.Tensor:
    flash = _flash_attention_function()
    query = rotate(q[plan.q_indices], plan.q_positions, inv_freq, mrope_section)
    key = rotate(k[plan.k_indices], plan.k_positions, inv_freq, mrope_section)
    value = v[plan.k_indices]
    parts, lse = flash(
        query,
        key,
        value,
        cu_seqlens_q=plan.cu_q,
        cu_seqlens_k=plan.cu_k,
        max_seqlen_q=plan.max_q,
        max_seqlen_k=plan.max_k,
        softmax_scale=scaling,
        causal=False,
        mask_mod=historical_mask(),
        aux_tensors=[plan.q_starts, plan.q_ends],
        pack_gqa=False,
        return_lse=True,
    )
    return merge_attention_parts(parts, lse, plan.merge_indptr, plan.merge_indices).flatten(1)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    replay: Trace,
    inv_freq: torch.Tensor,
    scaling: float,
    mrope_section: Sequence[int] | None = None,
) -> torch.Tensor:
    """Replay one masked sequence per KV version and merge softmax normalizers.

    Inputs and output have the same layouts as ``attention.torch.attention``.  Only
    trace metadata is cached; packed activations and rotations are rebuilt each
    forward.  The existing torch backend remains the reference and default.
    """
    if any(x.ndim != 3 for x in (q, k, v)):
        raise ValueError("Replay Q/K/V must have layout [tokens, heads, head_dim]")
    if q.device.type != "cuda" or any(x.device != q.device for x in (k, v)):
        raise ValueError("The fa4 replay backend requires Q/K/V on one CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(x.dtype != q.dtype for x in (k, v)):
        raise ValueError("The fa4 replay backend requires matching FP16 or BF16 Q/K/V")
    if torch.cuda.get_device_capability(q.device)[0] != 9:
        raise NotImplementedError("FA4 replay requires Hopper (SM90)")
    if (
        len(q) != len(k)
        or k.shape != v.shape
        or q.shape[-1] != k.shape[-1]
        or not 8 <= q.shape[-1] <= 256
        or q.shape[-1] % 8
        or not k.shape[1]
        or q.shape[1] % k.shape[1]
    ):
        raise ValueError("Unsupported replay Q/K/V dimensions for Hopper FA4")
    replay._prepare()
    if len(q) != len(replay.reads):
        raise ValueError("Q/K/V rows must match the prepared trace")
    cache_key = ("fa4", q.device)
    if cache_key not in replay._attention:
        replay._attention[cache_key] = build_flash_attention_plan(replay, q.device)
    args = (q, k, v, inv_freq, replay._attention[cache_key], scaling, mrope_section)
    if (
        replay.attention_checkpointing
        and torch.is_grad_enabled()
        and any(x.requires_grad for x in (q, k, v))
    ):
        return checkpoint(_forward, *args, use_reentrant=False)
    return _forward(*args)
