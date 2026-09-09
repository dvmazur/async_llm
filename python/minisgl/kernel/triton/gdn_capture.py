"""Triton kernels for one-token GDN affine capture."""

import triton
import triton.language as tl


@triton.jit
def gdn_store_affine_pointer_kernel(
    source_ptr,
    # Per worker: [destination_A data_ptr, destination_B data_ptr].
    destination_ptrs,
    source_stride_w,
    source_stride_h,
    source_stride_m,
    source_stride_k,
    num_heads: tl.constexpr,
    d_k: tl.constexpr,
    d_v: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Scatter augmented ``[A; B]`` scan states into block-owned slabs."""

    worker = tl.program_id(axis=0)
    offsets = tl.program_id(axis=1) * BLOCK + tl.arange(0, BLOCK)
    rows_per_head = (d_k + d_v) * d_k
    total = num_heads * rows_per_head
    mask = offsets < total
    head = offsets // rows_per_head
    within_head = offsets % rows_per_head
    row = within_head // d_k
    col = within_head % d_k
    source = source_ptr + (
        worker * source_stride_w
        + head * source_stride_h
        + row * source_stride_m
        + col * source_stride_k
    )
    value = tl.load(source, mask=mask)

    metadata = destination_ptrs + worker * 2
    destination_A = tl.load(metadata).to(tl.pointer_type(tl.float32))
    destination_B = tl.load(metadata + 1).to(tl.pointer_type(tl.float32))
    is_A = row < d_k
    destination_row = tl.where(is_A, row, row - d_k)
    destination = tl.where(is_A, destination_A, destination_B)
    destination += (head * tl.where(is_A, d_k, d_v) + destination_row) * d_k + col
    tl.store(destination, value, mask=mask)


@triton.jit
def gdn_capture_pointer_update_kernel(
    # Per worker: [old_A data_ptr, old_B data_ptr]. Zero means a fresh block.
    source_ptrs,
    key_ptr,
    value_ptr,
    alpha_ptr,
    beta_ptr,
    # Per worker: [new_A data_ptr, new_B data_ptr].
    output_ptrs,
    num_heads: tl.constexpr,
    d_k: tl.constexpr,
    d_v: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Copy-on-write update from fragmented A/B sources into packed outputs."""

    worker_head = tl.program_id(axis=0)
    row_block = tl.program_id(axis=1)
    worker = worker_head // num_heads
    head = worker_head % num_heads
    rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < d_k

    metadata = source_ptrs + worker * 2
    old_A_address = tl.load(metadata)
    old_B_address = tl.load(metadata + 1)
    has_old = old_A_address != 0
    old_A = old_A_address.to(tl.pointer_type(tl.float32))
    old_B = old_B_address.to(tl.pointer_type(tl.float32))
    output_metadata = output_ptrs + worker * 2
    output_A = tl.load(output_metadata).to(tl.pointer_type(tl.float32))
    output_B = tl.load(output_metadata + 1).to(tl.pointer_type(tl.float32))

    key = tl.load(key_ptr + worker_head * d_k + cols, mask=col_mask, other=0.0)
    alpha = tl.load(alpha_ptr + worker_head)
    beta = tl.load(beta_ptr + worker_head)

    # A rows: a fresh block starts from identity, not from a materialized eye.
    valid_A_row = rows < d_k
    A_offsets = (head * d_k + rows[:, None]) * d_k + cols[None, :]
    A_mask = has_old & valid_A_row[:, None] & col_mask[None, :]
    A = tl.load(old_A + A_offsets, mask=A_mask, other=0.0)
    fresh_identity = (~has_old) & valid_A_row[:, None] & (rows[:, None] == cols[None, :])
    A += tl.where(fresh_identity, 1.0, 0.0)
    A_k = tl.sum(A * key[None, :], axis=1)
    A_new = alpha * A - alpha * beta * A_k[:, None] * key[None, :]
    output_A_offsets = (head * d_k + rows[:, None]) * d_k + cols[None, :]
    tl.store(
        output_A + output_A_offsets,
        A_new,
        mask=valid_A_row[:, None] & col_mask[None, :],
    )

    # B rows: a fresh block starts from zero. The token write is fused with the
    # decay/erase update, so neither B_k nor a temporary B_t is materialized.
    valid_B_row = rows < d_v
    B_offsets = (head * d_v + rows[:, None]) * d_k + cols[None, :]
    B_mask = has_old & valid_B_row[:, None] & col_mask[None, :]
    B = tl.load(old_B + B_offsets, mask=B_mask, other=0.0)
    B_k = tl.sum(B * key[None, :], axis=1)
    value = tl.load(
        value_ptr + worker_head * d_v + rows,
        mask=valid_B_row,
        other=0.0,
    )
    B_new = (
        alpha * B
        - alpha * beta * B_k[:, None] * key[None, :]
        + beta * value[:, None] * key[None, :]
    )
    output_B_offsets = (head * d_v + rows[:, None]) * d_k + cols[None, :]
    tl.store(
        output_B + output_B_offsets,
        B_new,
        mask=valid_B_row[:, None] & col_mask[None, :],
    )
