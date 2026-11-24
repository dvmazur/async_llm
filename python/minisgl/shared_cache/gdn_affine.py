"""
Affine-summary math for Gated DeltaNet (GDN) linear-attention states.

For a fixed token trajectory the GDN recurrent update is *affine* in the prior
state, so a whole block of tokens compresses to a matrix pair ``(A_hat, B_hat)``::

    S_out = S_in @ A_hat + B_hat

Per token ``t`` (with ``alpha = exp(g)``, ``beta = sigmoid(b)``)::

    S_t = S_{t-1} @ A_t + B_t
    A_t = alpha_t I - alpha_t beta_t k_t k_t^T      (Qwen/HF ordering: decay before erase)
    B_t = beta_t v_t k_t^T

and two blocks compose as ``A = A1 @ A2``, ``B = B1 @ A2 + B2``.  This is the
recurrent-linear-attention analogue of composing KV-cache blocks; it lets a
worker's chain of shared blocks fold into a single initial recurrent state.

State convention here is the **block convention** ``[B, H, d_v, d_k]`` (the
transpose of the HF kernel state ``[B, H, d_k, d_v]``), chosen so right-multiply
by ``A`` is a plain ``matmul``. Default storage/math is float32; optional BF16
composition accumulates in FP32 and rounds at the stored-summary boundary.

Ported from AsyncReasoning ``shared_cache/gdn_cache_block.py`` (ar_on_gdn).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

Tensor = torch.Tensor


def init_gdn_affine(
    *,
    batch_size: int,
    num_heads: int,
    d_k: int,
    d_v: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str,
) -> Tuple[Tensor, Tensor]:
    """Identity affine: ``A_hat = I [B,H,d_k,d_k]``, ``B_hat = 0 [B,H,d_v,d_k]``."""
    if d_v is None:
        d_v = d_k
    eye = torch.eye(d_k, dtype=dtype, device=device)
    A_hat = eye.view(1, 1, d_k, d_k).expand(batch_size, num_heads, d_k, d_k).clone()
    B_hat = torch.zeros(batch_size, num_heads, d_v, d_k, dtype=dtype, device=device)
    return A_hat, B_hat


def _as_head_scalar(x: Tensor, *, reference: Tensor) -> Tensor:
    """Broadcast a per-head gate to ``[B, H, 1]`` (reference is k: ``[B, H, d_k]``)."""
    b, h = reference.shape[:2]
    if x.ndim == 3 and x.shape == (b, h, 1):
        return x
    if x.ndim == 2:
        return x.reshape(b, h).unsqueeze(-1)
    if x.ndim == 1:
        return (
            x.reshape(1, h).expand(b, h).unsqueeze(-1)
            if x.shape[0] == h
            else x.reshape(b, 1).expand(b, h).unsqueeze(-1)
        )
    return x.reshape(b, h).unsqueeze(-1)


def update_affine_summary(
    *,
    A_hat: Tensor,
    B_hat: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Append one token's affine ``A_t = alpha I - alpha beta k k^T``, ``B_t = beta v k^T``.

    Rank-1 form (O(H d^2), not O(d^3)):
        A_hat <- alpha A_hat - alpha beta (A_hat k) k^T
        B_hat <- alpha B_hat - alpha beta (B_hat k) k^T + beta v k^T

    Shapes: ``A_hat [B,H,d_k,d_k]``, ``B_hat [B,H,d_v,d_k]``, ``k [B,H,d_k]``,
    ``v [B,H,d_v]``, ``alpha/beta`` broadcastable to ``[B,H]``.
    """
    alpha_b = _as_head_scalar(alpha, reference=k)  # [B,H,1]
    beta_b = _as_head_scalar(beta, reference=k)

    A_k = torch.matmul(A_hat, k.unsqueeze(-1)).squeeze(-1)  # [B,H,d_k]
    A_hat_new = alpha_b.unsqueeze(-1) * A_hat - (
        alpha_b.unsqueeze(-1) * beta_b.unsqueeze(-1) * A_k.unsqueeze(-1) * k.unsqueeze(-2)
    )

    B_k = torch.matmul(B_hat, k.unsqueeze(-1)).squeeze(-1)  # [B,H,d_v]
    B_hat_new = alpha_b.unsqueeze(-1) * B_hat - (
        alpha_b.unsqueeze(-1) * beta_b.unsqueeze(-1) * B_k.unsqueeze(-1) * k.unsqueeze(-2)
    )
    B_t = beta_b.unsqueeze(-1) * v.unsqueeze(-1) * k.unsqueeze(-2)  # [B,H,d_v,d_k]
    return A_hat_new, B_hat_new + B_t


def compose_gdn_affines(
    *,
    A_first: Tensor,
    B_first: Tensor,
    A_second: Tensor,
    B_second: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Compose ``S_mid = S_in A1 + B1`` then ``S_out = S_mid A2 + B2``:
    ``A = A1 A2``, ``B = B1 A2 + B2``."""
    if A_first.dtype == torch.bfloat16:
        # CUDA bmm uses BF16 operands with FP32 output/accumulation. Add B in
        # FP32 and round only the final stored summary, just like compose_level.
        def mm(x, y):
            if x.is_cuda:
                return torch.bmm(x.flatten(0, 1), y.flatten(0, 1),
                                 out_dtype=torch.float32).reshape(*x.shape[:-1], y.shape[-1])
            return torch.matmul(x.float(), y.float())
        A = mm(A_first, A_second).to(torch.bfloat16)
        B = (mm(B_first, A_second) + B_second.float()).to(torch.bfloat16)
        return A, B
    A = torch.matmul(A_first, A_second)
    B = torch.matmul(B_first, A_second)
    B.add_(B_second)
    return A, B


def apply_gdn_affine(state: Tensor, A_hat: Tensor, B_hat: Tensor) -> Tensor:
    """``S_out = S_in @ A_hat + B_hat`` (all block convention ``[B,H,d_v,d_k]``)."""
    return torch.matmul(state, A_hat) + B_hat


__all__ = [
    "init_gdn_affine",
    "update_affine_summary",
    "compose_gdn_affines",
    "apply_gdn_affine",
]
