import functools
import warnings
from typing import Dict, Tuple

import torch
from minisgl.moe import BaseMoeBackend
from minisgl.utils import div_ceil


@functools.cache
def _use_torch_moe_fallback(device: torch.device) -> bool:
    if device.type != "cuda":
        warnings.warn(f"MoE routing/alignment uses Torch fallback on {device}: CUDA backend unavailable. "
                      "Expert GEMM backend is unchanged.", RuntimeWarning, stacklevel=2)
        return True
    if torch.cuda.get_device_capability(device) == (12, 1):
        warnings.warn("MoE routing/alignment uses Torch fallback on SM121 (GB10): intentional "
                      "sgl_kernel compatibility guard. Triton expert GEMMs remain enabled.",
                      RuntimeWarning, stacklevel=2)
        return True
    try:
        import sgl_kernel
    except (ImportError, OSError) as exc:
        warnings.warn(f"MoE routing/alignment uses Torch fallback because sgl_kernel failed to import: "
                      f"{type(exc).__name__}: {exc}. This can reduce throughput. Install the "
                      "Torch/CUDA-compatible sglang-kernel from the project lockfile; "
                      "Triton expert GEMMs are unchanged.", RuntimeWarning, stacklevel=2)
        return True
    return False


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if _use_torch_moe_fallback(hidden_states.device):
        assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
        # sgl_kernel's prebuilt SM100 extension is not compatible with SM121
        # (GB10): its topk_softmax launch completes without writing its outputs.
        # Keep routing correct and portable by using native Torch operations.  The
        # expensive expert GEMMs below still run through the Triton fused kernel.
        routing_weights = torch.softmax(gating_output.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
        topk_ids = topk_ids.to(torch.int32)
        if renormalize:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
        if num_token_non_padded is not None:
            indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
            topk_ids[indices >= num_token_non_padded, :] = -1
        return topk_weights, topk_ids

    from sgl_kernel import topk_softmax

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    M, _ = hidden_states.shape
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    topk_softmax(topk_weights, topk_ids, gating_output.float(), renormalize)
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
    if num_token_non_padded is not None:
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
        topk_ids[indices >= num_token_non_padded, :] = -1
    return topk_weights, topk_ids


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    if _use_torch_moe_fallback(topk_ids.device):
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
        sentinel = topk_ids.numel()
        sorted_ids = torch.full(
            (max_num_tokens_padded,), sentinel, dtype=torch.int32, device=topk_ids.device
        )
        max_num_m_blocks = div_ceil(max_num_tokens_padded, block_size)
        expert_ids = torch.zeros(
            (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
        )

        flat_experts = topk_ids.flatten().to(torch.int64)
        token_ids = torch.arange(sentinel, device=topk_ids.device, dtype=torch.int64)
        order = torch.argsort(flat_experts, stable=True)
        sorted_experts = flat_experts[order]
        # bincount reads the largest GPU id to determine its output size, even
        # with minlength. Counts have a known expert capacity here: no host read.
        counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)
        counts.scatter_add_(0, flat_experts, torch.ones_like(flat_experts))
        padded_counts = ((counts + block_size - 1) // block_size) * block_size
        input_starts = torch.cumsum(counts, dim=0) - counts
        padded_starts = torch.cumsum(padded_counts, dim=0) - padded_counts
        destinations = padded_starts[sorted_experts] + (
            torch.arange(sentinel, device=topk_ids.device) - input_starts[sorted_experts]
        )
        sorted_ids[destinations] = token_ids[order].to(torch.int32)

        blocks_per_expert = padded_counts // block_size
        # Invert the cumulative counts at fixed capacity instead of creating a
        # data-dependent-length repeat_interleave result (not graph-capturable).
        block_index = torch.arange(max_num_m_blocks, device=topk_ids.device)
        cumulative_blocks = blocks_per_expert.cumsum(0)
        block_experts = torch.searchsorted(cumulative_blocks, block_index, right=True)
        expert_ids.copy_(torch.where(block_experts < num_experts, block_experts, 0).int())
        num_tokens_post_pad = padded_counts.sum().reshape(1).to(torch.int32)
        return sorted_ids, expert_ids, num_tokens_post_pad

    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = div_ceil(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)
    sgl_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
) -> Dict[str, int]:

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    if M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    M: int,
) -> Dict[str, int]:
    E, _, N = w2_shape
    config = get_default_config(M, E, N, w1_shape[2], top_k)
    return config


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    from minisgl.kernel import fused_moe_kernel_triton, moe_sum_reduce_triton
    from minisgl.layers import gelu_and_mul, silu_and_mul

    padded_size = 0
    assert hidden_states.shape[1] == w1.shape[2] - padded_size, "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    M = num_tokens
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
    )
    config = get_config_func(M)

    cache = torch.empty(
        M * topk_ids.shape[1] * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(
        (M, topk_ids.shape[1], N),
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk_ids.shape[1] * w2.shape[1]].view(
        (M, topk_ids.shape[1], w2.shape[1]),
    )
    compute_type = hidden_states.dtype
    use_fp8 = w1.dtype == torch.float8_e4m3fn
    if use_fp8:
        from minisgl.kernel.fp8 import quantize_fp8_groups, validate_weight
        validate_weight(w1, w1_scale)
        validate_weight(w2, w2_scale)
        if num_tokens == 0:
            return hidden_states
    elif w1_scale is not None or w2_scale is not None:
        raise ValueError("FP8 scales supplied for non-FP8 experts")

    out_hidden_states = hidden_states
    curr_hidden_states = hidden_states
    tokens_num, _ = curr_hidden_states.shape
    begin_token_idx, end_token_idx = 0, num_tokens

    intermediate_cache1 = intermediate_cache1[:tokens_num]
    intermediate_cache2 = intermediate_cache2[: tokens_num * topk_ids.shape[1]]
    intermediate_cache3 = intermediate_cache3[:tokens_num]
    config = get_config_func(tokens_num)
    if use_fp8:
        # One dot contribution per 128-wide quantization block.
        config = {**config, "BLOCK_SIZE_K": 128}

    curr_topk_ids = topk_ids[begin_token_idx:end_token_idx]
    curr_topk_weights = topk_weights[begin_token_idx:end_token_idx]

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        curr_topk_ids, config["BLOCK_SIZE_M"], E
    )

    gemm_input, input_scale = (quantize_fp8_groups(curr_hidden_states) if use_fp8
                              else (curr_hidden_states, None))
    fused_moe_kernel_triton(
        gemm_input,
        w1,
        intermediate_cache1,
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk_ids.shape[1],
        config,
        compute_type=compute_type,
        a_scale=input_scale,
        b_scale=w1_scale,
    )
    FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
    FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)
    gemm_input, input_scale = (quantize_fp8_groups(intermediate_cache2) if use_fp8
                              else (intermediate_cache2, None))
    fused_moe_kernel_triton(
        gemm_input,
        w2,
        (intermediate_cache3),
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        config,
        compute_type=compute_type,
        a_scale=input_scale,
        b_scale=w2_scale,
    )

    moe_sum_reduce_triton(
        intermediate_cache3,
        out_hidden_states[begin_token_idx:end_token_idx],
    )
    return out_hidden_states


class FusedMoe(BaseMoeBackend):
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        w1_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
        )
        return fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )
