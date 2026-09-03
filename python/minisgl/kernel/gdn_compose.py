"""Launch helpers for fragmented GDN affine composition."""

from __future__ import annotations

from typing import Sequence

import torch


def apply_gdn_affine_pointer_frontier(
    parent_states: torch.Tensor,
    parent_indices: Sequence[int],
    A_blocks: Sequence[torch.Tensor],
    B_blocks: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Apply a homogeneous GDN affine frontier without packing A/B tensors.

    ``A_blocks`` and ``B_blocks`` are single-row FP32 CUDA tensors which may
    belong to arbitrary allocations.  A small device metadata table carries
    their addresses to one Triton launch.
    """

    if not parent_states.is_cuda:
        raise ValueError("GDN pointer compose requires CUDA parent states")
    if parent_states.dtype != torch.float32 or not parent_states.is_contiguous():
        raise ValueError("GDN pointer compose requires contiguous FP32 parent states")
    if not A_blocks or len(A_blocks) != len(B_blocks) or len(A_blocks) != len(parent_indices):
        raise ValueError("parent_indices, A_blocks and B_blocks must have the same non-zero length")

    num_parents = parent_states.shape[0]
    parent_blocks = []
    for parent_index in parent_indices:
        if not 0 <= parent_index < num_parents:
            raise IndexError(f"parent row {parent_index} is outside [0, {num_parents})")
        parent_blocks.append(parent_states[parent_index : parent_index + 1])
    return apply_gdn_affine_pointer_nodes(parent_blocks, A_blocks, B_blocks)


def apply_gdn_affine_pointer_nodes(
    parent_blocks: Sequence[torch.Tensor],
    A_blocks: Sequence[torch.Tensor],
    B_blocks: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Apply homogeneous affines whose parent states may be fragmented."""

    import triton

    from .triton.gdn_compose import gdn_compose_pointer_kernel

    if (
        not parent_blocks
        or len(parent_blocks) != len(A_blocks)
        or len(parent_blocks) != len(B_blocks)
    ):
        raise ValueError("parent_blocks, A_blocks and B_blocks must have equal non-zero length")
    first_parent = parent_blocks[0]
    if not first_parent.is_cuda:
        raise ValueError("GDN pointer compose requires CUDA parent states")
    if first_parent.ndim != 4 or first_parent.shape[0] != 1:
        raise ValueError("each GDN parent state must be a single-row rank-4 tensor")
    _, num_heads, d_v, d_k = first_parent.shape
    expected_A = (1, num_heads, d_k, d_k)
    expected_B = (1, num_heads, d_v, d_k)
    metadata_rows = []
    for parent, A, B in zip(parent_blocks, A_blocks, B_blocks):
        if A.shape != expected_A or B.shape != expected_B:
            raise ValueError(
                f"invalid GDN affine shapes: expected {expected_A} and {expected_B}, "
                f"got {tuple(A.shape)} and {tuple(B.shape)}"
            )
        if (
            parent.shape != first_parent.shape
            or parent.device != first_parent.device
            or parent.dtype != torch.float32
            or not parent.is_contiguous()
            or A.device != first_parent.device
            or B.device != first_parent.device
            or A.dtype != torch.float32
            or B.dtype != torch.float32
            or not A.is_contiguous()
            or not B.is_contiguous()
        ):
            raise ValueError(
                "GDN pointer compose requires contiguous FP32 A/B on the same CUDA device"
            )
        metadata_rows.append((parent.data_ptr(), A.data_ptr(), B.data_ptr()))

    metadata = torch.tensor(metadata_rows, dtype=torch.int64, device=first_parent.device)
    output = torch.empty(
        len(A_blocks), num_heads, d_v, d_k, dtype=torch.float32, device=first_parent.device
    )

    # Fixed Qwen dimensions make a stable hand-tuned launch preferable to
    # runtime autotuning in the first request.  Masks retain correctness for
    # smaller test/compatible dimensions.
    # GB10 tuning for the production 128x128 state favored 64x32 tiles, K=16,
    # four warps and one pipeline stage.  Clamp small dimensions to Triton's
    # minimum dot tile while masks handle their non-power-of-two tails.
    block_m = 64 if d_v >= 64 else max(16, triton.next_power_of_2(d_v))
    block_n = 32 if d_k >= 32 else max(16, triton.next_power_of_2(d_k))
    block_k = 16
    grid = (
        len(A_blocks) * num_heads,
        triton.cdiv(d_v, block_m) * triton.cdiv(d_k, block_n),
    )
    gdn_compose_pointer_kernel[grid](
        first_parent,
        metadata,
        output,
        num_heads=num_heads,
        d_v=d_v,
        d_k=d_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["apply_gdn_affine_pointer_frontier", "apply_gdn_affine_pointer_nodes"]
