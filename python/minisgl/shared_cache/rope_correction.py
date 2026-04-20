"""
Vectorised RoPE correction for cached KV pages.

When a SharedBlock is placed at a different logical position for a particular
worker, the keys in the KV cache have incorrect RoPE.  The correction is a
per-token rotation by ``delta = target_position - stored_position``, which is
equivalent to applying RoPE at offset ``delta`` to the already-rotated keys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import torch

if TYPE_CHECKING:
    from minisgl.kvcache import BaseKVCachePool

    from .shared_block import SharedBlock

CorrectionKey = Tuple[int, int]  # (block_id, target_start)


def apply_rope_correction(
    keys: torch.Tensor,
    corrections: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    """
    Apply per-token RoPE rotation correction to cached keys.

    Args:
        keys: ``[N, num_heads, head_dim]``
        corrections: ``[N]`` integer rotation offsets (may be negative)
        cos_sin_cache: ``[max_pos, head_dim]`` — first half cols are cos,
            second half are sin (the format used by
            ``minisgl.layers.rotary.RotaryEmbedding``).

    Returns:
        Corrected keys with the same shape.
    """
    half_dim = keys.shape[-1] // 2
    orig_dtype = keys.dtype

    abs_corr = corrections.abs().clamp(max=cos_sin_cache.shape[0] - 1)
    cos = cos_sin_cache[abs_corr, :half_dim]  # [N, half_dim]
    sin = cos_sin_cache[abs_corr, half_dim:]   # [N, half_dim]

    # sin(−θ) = −sin(θ)
    neg_mask = (corrections < 0).unsqueeze(-1)
    sin = torch.where(neg_mask, -sin, sin)

    cos = cos.unsqueeze(1)  # [N, 1, half_dim]  → broadcast over heads
    sin = sin.unsqueeze(1)

    # Compute rotation in float32 (cos_sin_cache dtype) for numerical stability
    k1 = keys[..., :half_dim].to(cos.dtype)
    k2 = keys[..., half_dim:].to(cos.dtype)
    out = torch.cat(
        [cos * k1 - sin * k2, cos * k2 + sin * k1],
        dim=-1,
    )
    return out.to(orig_dtype)


def correct_kv_pages(
    kv_cache: BaseKVCachePool,
    source_pages: torch.Tensor,
    dest_pages: torch.Tensor,
    corrections: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> None:
    """
    Copy KV from *source_pages* to *dest_pages*, applying a per-token RoPE
    correction to keys.  Values are copied verbatim.

    All tensors must be on the same device as *kv_cache*.

    Args:
        kv_cache: the MHAKVCache pool backing the Engine.
        source_pages: ``[N]`` physical page indices to read from.
        dest_pages: ``[N]`` physical page indices to write to.
        corrections: ``[N]`` per-token RoPE offsets.
        cos_sin_cache: ``[max_pos, head_dim]``.
    """
    all_zero = bool(torch.all(corrections == 0).item())

    for layer_idx in range(kv_cache.num_layers):
        k_layer = kv_cache.k_cache(layer_idx)  # [pages, page_size, heads, dim]
        v_layer = kv_cache.v_cache(layer_idx)

        k_flat = k_layer.reshape(-1, *k_layer.shape[2:])  # [pages*ps, heads, dim]
        v_flat = v_layer.reshape(-1, *v_layer.shape[2:])

        src_k = k_flat[source_pages]
        src_v = v_flat[source_pages]

        if not all_zero:
            dst_k = apply_rope_correction(src_k, corrections, cos_sin_cache)
        else:
            dst_k = src_k

        k_flat[dest_pages] = dst_k
        v_flat[dest_pages] = src_v


def build_correction_plan(
    blocks_with_targets: List[Tuple[SharedBlock, int]],
) -> Dict[CorrectionKey, Tuple[SharedBlock, int, torch.Tensor]]:
    """
    Deduplicate ``(block, target_start)`` pairs and compute per-token
    correction vectors.

    Returns:
        ``{(block_id, target_start): (block, target_start, corrections)}``
        Only entries where at least one correction is non-zero are included.
    """
    plan: Dict[CorrectionKey, Tuple[SharedBlock, int, torch.Tensor]] = {}
    for block, target_start in blocks_with_targets:
        key = (block.block_id, target_start)
        if key in plan:
            continue
        if block.num_tokens == 0 or not block.needs_correction(target_start):
            continue
        plan[key] = (block, target_start, block.compute_corrections(target_start))
    return plan
