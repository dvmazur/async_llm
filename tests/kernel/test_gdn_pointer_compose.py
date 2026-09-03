"""Direct correctness tests for the fragmented-pointer GDN Triton kernel."""

from __future__ import annotations

import pytest
import torch
from minisgl.kernel import (
    apply_gdn_affine_pointer_frontier,
    apply_gdn_affine_pointer_nodes,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("packed", [False, True], ids=["fragmented", "packed-views"])
@pytest.mark.parametrize("shared_parent", [False, True], ids=["different-parents", "shared-parent"])
def test_pointer_frontier_matches_torch_for_both_storage_layouts(packed, shared_parent):
    generator = torch.Generator(device="cuda").manual_seed(321)
    nodes, parents, heads, d_v, d_k = 7, 3, 4, 17, 19
    parent_states = torch.randn(
        parents, heads, d_v, d_k, device="cuda", dtype=torch.float32, generator=generator
    )
    parent_indices = [0] * nodes if shared_parent else [row % parents for row in range(nodes)]

    A_storage = torch.randn(
        nodes, heads, d_k, d_k, device="cuda", dtype=torch.float32, generator=generator
    )
    B_storage = torch.randn(
        nodes, heads, d_v, d_k, device="cuda", dtype=torch.float32, generator=generator
    )
    if packed:
        A_blocks = list(A_storage.split(1))
        B_blocks = list(B_storage.split(1))
    else:
        A_blocks = [row.clone() for row in A_storage.split(1)]
        B_blocks = [row.clone() for row in B_storage.split(1)]

    before_A = [row.clone() for row in A_blocks]
    before_B = [row.clone() for row in B_blocks]
    actual = apply_gdn_affine_pointer_frontier(parent_states, parent_indices, A_blocks, B_blocks)
    expected = torch.cat(
        [
            torch.matmul(parent_states[parent : parent + 1], A).add(B)
            for parent, A, B in zip(parent_indices, A_blocks, B_blocks)
        ],
        dim=0,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    for actual_A, expected_A, actual_B, expected_B in zip(A_blocks, before_A, B_blocks, before_B):
        assert torch.equal(actual_A, expected_A)
        assert torch.equal(actual_B, expected_B)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pointer_frontier_matches_production_qwen_shape():
    generator = torch.Generator(device="cuda").manual_seed(777)
    nodes, parents, heads, d_k = 64, 8, 32, 128
    parent_states = torch.randn(
        parents, heads, d_k, d_k, device="cuda", dtype=torch.float32, generator=generator
    )
    parent_indices = [row % parents for row in range(nodes)]
    A_blocks = [
        torch.randn(1, heads, d_k, d_k, device="cuda", generator=generator) for _ in range(nodes)
    ]
    B_blocks = [
        torch.randn(1, heads, d_k, d_k, device="cuda", generator=generator) for _ in range(nodes)
    ]

    actual = apply_gdn_affine_pointer_frontier(parent_states, parent_indices, A_blocks, B_blocks)
    expected = torch.cat(
        [
            torch.matmul(parent_states[parent : parent + 1], A).add(B)
            for parent, A, B in zip(parent_indices, A_blocks, B_blocks)
        ],
        dim=0,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pointer_nodes_reads_fragmented_parent_states():
    generator = torch.Generator(device="cuda").manual_seed(778)
    nodes, heads, d_v, d_k = 5, 4, 13, 19
    parents = [
        torch.randn(1, heads, d_v, d_k, device="cuda", generator=generator) for _ in range(nodes)
    ]
    A_blocks = [
        torch.randn(1, heads, d_k, d_k, device="cuda", generator=generator) for _ in range(nodes)
    ]
    B_blocks = [
        torch.randn(1, heads, d_v, d_k, device="cuda", generator=generator) for _ in range(nodes)
    ]

    actual = apply_gdn_affine_pointer_nodes(parents, A_blocks, B_blocks)
    expected = torch.cat(
        [torch.matmul(parent, A).add(B) for parent, A, B in zip(parents, A_blocks, B_blocks)]
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
