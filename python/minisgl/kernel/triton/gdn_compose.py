"""Triton kernels for composing fragmented GDN affine summaries."""

import triton
import triton.language as tl


@triton.jit
def gdn_compose_pointer_kernel(
    parent_states_ptr,
    # Per frontier node: [parent data_ptr, A data_ptr, B data_ptr].  All three
    # operands may live in unrelated allocations.
    node_metadata_ptr,
    output_ptr,
    num_heads: tl.constexpr,
    d_v: tl.constexpr,
    d_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute ``child = parent @ A + B`` for one trie frontier.

    One logical GEMM group is a ``(frontier node, GDN head)`` pair.  All groups
    have the Qwen GDN shape ``[d_v, d_k] @ [d_k, d_k]`` but their A/B operands
    may be fragmented across different CUDA allocations.
    """

    group_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)
    node_id = group_id // num_heads
    head_id = group_id % num_heads

    tiles_n = tl.cdiv(d_k, BLOCK_N)
    tile_m = tile_id // tiles_n
    tile_n = tile_id % tiles_n
    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    metadata = node_metadata_ptr + node_id * 3
    parent_address = tl.load(metadata)
    parent_base = parent_address.to(tl.pointer_type(tl.float32))
    a_base = tl.load(metadata + 1).to(tl.pointer_type(tl.float32))
    b_base = tl.load(metadata + 2).to(tl.pointer_type(tl.float32))

    parent_head = parent_base + head_id * d_v * d_k
    a_head = a_base + head_id * d_k * d_k
    b_head = b_base + head_id * d_v * d_k

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, d_k, BLOCK_K):
        k = k_start + offs_k
        parent_ptrs = parent_head + offs_m[:, None] * d_k + k[None, :]
        a_ptrs = a_head + k[:, None] * d_k + offs_n[None, :]
        parent = tl.load(
            parent_ptrs,
            mask=(offs_m[:, None] < d_v) & (k[None, :] < d_k),
            other=0.0,
        )
        a = tl.load(
            a_ptrs,
            mask=(k[:, None] < d_k) & (offs_n[None, :] < d_k),
            other=0.0,
        )
        # GDN affine state is intentionally accumulated in IEEE FP32.  Letting
        # Triton use its default TF32 mode would change recurrent-state math.
        accumulator += tl.dot(parent, a, input_precision="ieee")

    b_ptrs = b_head + offs_m[:, None] * d_k + offs_n[None, :]
    b = tl.load(
        b_ptrs,
        mask=(offs_m[:, None] < d_v) & (offs_n[None, :] < d_k),
        other=0.0,
    )
    output_head = output_ptr + (node_id * num_heads + head_id) * d_v * d_k
    output_ptrs = output_head + offs_m[:, None] * d_k + offs_n[None, :]
    tl.store(
        output_ptrs,
        accumulator + b,
        mask=(offs_m[:, None] < d_v) & (offs_n[None, :] < d_k),
    )
