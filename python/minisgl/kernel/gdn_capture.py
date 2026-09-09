"""Launch helper for fused one-token GDN affine capture."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from .metadata import device_metadata

AffinePair = Tuple[torch.Tensor, torch.Tensor]


def capture_gdn_affine_pointer_update(
    previous: Sequence[Optional[AffinePair]],
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    output_pairs: Optional[Sequence[AffinePair]] = None,
) -> Optional[AffinePair]:
    """Update fragmented/fresh block summaries into one packed output pair."""

    import triton

    from .triton.gdn_capture import gdn_capture_pointer_update_kernel

    if not key.is_cuda or key.dtype != torch.float32 or not key.is_contiguous():
        raise ValueError("GDN capture requires contiguous FP32 CUDA key")
    workers, num_heads, d_k = key.shape
    d_v = value.shape[-1]
    expected_value = (workers, num_heads, d_v)
    expected_gate = (workers, num_heads)
    if (
        value.shape != expected_value
        or alpha.shape != expected_gate
        or beta.shape != expected_gate
        or value.dtype != torch.float32
        or alpha.dtype != torch.float32
        or beta.dtype != torch.float32
        or value.device != key.device
        or alpha.device != key.device
        or beta.device != key.device
        or not value.is_contiguous()
        or not alpha.is_contiguous()
        or not beta.is_contiguous()
    ):
        raise ValueError("GDN capture value/gates must be contiguous FP32 CUDA tensors")
    if len(previous) != workers:
        raise ValueError(f"expected {workers} previous pairs, got {len(previous)}")

    expected_A = (1, num_heads, d_k, d_k)
    expected_B = (1, num_heads, d_v, d_k)
    pointer_rows = []
    for pair in previous:
        if pair is None:
            pointer_rows.append((0, 0))
            continue
        A, B = pair
        if A.shape != expected_A or B.shape != expected_B:
            raise ValueError(
                f"invalid GDN affine shapes: expected {expected_A}/{expected_B}, "
                f"got {tuple(A.shape)}/{tuple(B.shape)}"
            )
        if (
            A.device != key.device
            or B.device != key.device
            or A.dtype != torch.float32
            or B.dtype != torch.float32
            or not A.is_contiguous()
            or not B.is_contiguous()
        ):
            raise ValueError("previous GDN affines must be contiguous FP32 CUDA tensors")
        pointer_rows.append((A.data_ptr(), B.data_ptr()))

    source_ptrs = device_metadata(pointer_rows, device=key.device)
    owns_output = output_pairs is None
    if output_pairs is None:
        output_A = torch.empty(
            workers, num_heads, d_k, d_k, dtype=torch.float32, device=key.device
        )
        output_B = torch.empty(
            workers, num_heads, d_v, d_k, dtype=torch.float32, device=key.device
        )
        output_pairs = [
            (output_A[w : w + 1], output_B[w : w + 1]) for w in range(workers)
        ]
    if len(output_pairs) != workers:
        raise ValueError(f"expected {workers} output pairs, got {len(output_pairs)}")
    output_pointer_rows = []
    for A, B in output_pairs:
        if (
            A.shape != expected_A
            or B.shape != expected_B
            or A.device != key.device
            or B.device != key.device
            or A.dtype != torch.float32
            or B.dtype != torch.float32
            or not A.is_contiguous()
            or not B.is_contiguous()
        ):
            raise ValueError("output GDN affines must be contiguous FP32 CUDA tensors")
        output_pointer_rows.append((A.data_ptr(), B.data_ptr()))
    output_ptrs = device_metadata(output_pointer_rows, device=key.device)
    block_d = triton.next_power_of_2(d_k)
    block_rows = 2
    grid = (
        workers * num_heads,
        triton.cdiv(max(d_k, d_v), block_rows),
    )
    gdn_capture_pointer_update_kernel[grid](
        source_ptrs,
        key,
        value,
        alpha,
        beta,
        output_ptrs,
        num_heads=num_heads,
        d_k=d_k,
        d_v=d_v,
        BLOCK_D=block_d,
        BLOCK_ROWS=block_rows,
        num_warps=8,
        num_stages=1,
    )
    return (output_A, output_B) if owns_output else None


def store_gdn_affine_pointer(
    state: torch.Tensor,
    output_pairs: Sequence[AffinePair],
    *,
    d_k: int,
) -> None:
    """Scatter one augmented FLA final state into fragmented block slabs."""

    import triton

    from .triton.gdn_capture import gdn_store_affine_pointer_kernel

    workers, num_heads, rows, state_d_k = state.shape
    d_v = rows - d_k
    if (
        not state.is_cuda
        or state.dtype != torch.float32
        or state_d_k != d_k
        or d_v <= 0
        or len(output_pairs) != workers
    ):
        raise ValueError("invalid augmented GDN scan state or output-pair count")
    expected_A = (1, num_heads, d_k, d_k)
    expected_B = (1, num_heads, d_v, d_k)
    pointers = []
    for A, B in output_pairs:
        if (
            A.shape != expected_A
            or B.shape != expected_B
            or A.device != state.device
            or B.device != state.device
            or A.dtype != torch.float32
            or B.dtype != torch.float32
            or not A.is_contiguous()
            or not B.is_contiguous()
        ):
            raise ValueError("output GDN affines must be contiguous FP32 CUDA tensors")
        pointers.append((A.data_ptr(), B.data_ptr()))
    destination_ptrs = device_metadata(pointers, device=state.device)
    block = 256
    total = num_heads * (d_k + d_v) * d_k
    gdn_store_affine_pointer_kernel[(workers, triton.cdiv(total, block))](
        state,
        destination_ptrs,
        *state.stride(),
        num_heads=num_heads,
        d_k=d_k,
        d_v=d_v,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )


__all__ = ["capture_gdn_affine_pointer_update", "store_gdn_affine_pointer"]
