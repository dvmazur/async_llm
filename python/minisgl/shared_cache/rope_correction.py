"""
Vectorised RoPE rotation by an arbitrary per-token offset.

The query-rotation decode (see ``shared_cache.attention``) stores keys at
block-relative RoPE positions and, at attention time, rotates each query copy
by ``delta = query_position - segment_start`` against those block-relative
keys.  ``apply_rope_correction`` is the kernel that applies that rotation.
"""

from __future__ import annotations

import torch


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
    sin = cos_sin_cache[abs_corr, half_dim:]  # [N, half_dim]

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
