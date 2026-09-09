"""Correctness checks for the optional SM121 vLLM MoE routing fast path."""

from __future__ import annotations

import pytest
import torch

import minisgl.moe.fused as fused


def _validate_alignment(
    topk_ids: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    block_size: int,
) -> None:
    flat = topk_ids.flatten().cpu()
    sentinel = flat.numel()
    valid = int(num_tokens_post_pad.item())
    sorted_cpu = sorted_ids[:valid].cpu()
    experts_cpu = expert_ids[: valid // block_size].cpu()
    observed = []
    for block, expert in enumerate(experts_cpu.tolist()):
        for token_index in sorted_cpu[
            block * block_size : (block + 1) * block_size
        ].tolist():
            if token_index == sentinel:
                continue
            observed.append(token_index)
            assert flat[token_index].item() == expert
    assert sorted(observed) == list(range(sentinel))


def test_native_torch_router_fallback_remains_available(monkeypatch):
    """The optional vLLM fast path must never replace the portable fallback."""

    monkeypatch.setattr(fused, "_use_torch_moe_fallback", lambda _device: True)
    monkeypatch.setattr(fused, "_vllm_custom_moe_ops", lambda: None)
    generator = torch.Generator().manual_seed(729)
    rows, experts, topk, block_size = 9, 16, 4, 8
    hidden = torch.randn(rows, 32, dtype=torch.bfloat16, generator=generator)
    gating = torch.randn(rows, experts, dtype=torch.bfloat16, generator=generator)

    actual_weights, actual_ids = fused.fused_topk(
        hidden, gating, topk, renormalize=True
    )
    expected_weights, expected_ids = torch.topk(
        torch.softmax(gating.float(), dim=-1), topk, dim=-1
    )
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True) + 1e-8
    torch.testing.assert_close(actual_weights, expected_weights)
    assert torch.equal(actual_ids, expected_ids.to(torch.int32))

    sorted_ids, expert_ids, num_tokens = fused.moe_align_block_size(
        actual_ids, block_size, experts
    )
    _validate_alignment(actual_ids, sorted_ids, expert_ids, num_tokens, block_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_optional_vllm_router_matches_torch_topk_and_valid_alignment():
    ops = fused._vllm_custom_moe_ops()
    if ops is None:
        pytest.skip("vLLM custom MoE ops are unavailable")

    rows, experts, topk, block_size = 15, 256, 8, 16
    generator = torch.Generator(device="cuda").manual_seed(730)
    hidden = torch.randn(
        rows, 64, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    gating = torch.randn(
        rows, experts, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    actual_weights, actual_ids = fused.fused_topk(
        hidden, gating, topk, renormalize=True
    )
    expected_weights, expected_ids = torch.topk(
        torch.softmax(gating.float(), dim=-1), topk, dim=-1
    )
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True) + 1e-8
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-6, atol=1e-7)
    # BF16 gate logits can tie exactly; the two kernels may return tied expert
    # ids in a different (semantically equivalent) order.
    assert torch.equal(
        actual_ids.sort(dim=-1).values,
        expected_ids.to(torch.int32).sort(dim=-1).values,
    )

    sorted_ids, expert_ids, num_tokens = fused.moe_align_block_size(
        actual_ids, block_size, experts
    )
    _validate_alignment(
        actual_ids, sorted_ids, expert_ids, num_tokens, block_size
    )
