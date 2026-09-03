"""Integration tests for SharedCacheGDN affine capture dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
from minisgl.shared_cache.gdn import SharedCacheGDN
from minisgl.shared_cache.gdn_affine import init_gdn_affine


@dataclass(eq=False)
class _Block:
    linear_affine: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


def _gdn(targets, *, heads, d_k, d_v, device):
    result = SharedCacheGDN(
        num_heads=heads,
        head_k_dim=d_k,
        head_v_dim=d_v,
        conv_dim=1,
        conv_kernel=1,
        device=torch.device(device),
    )
    result.set_context([[] for _ in targets], targets)
    return result


def _reference(previous, key, value, alpha, beta, eps=1e-6):
    workers, sequence, heads, d_k = key.shape
    d_v = value.shape[-1]
    key = key.float()
    key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + eps)
    value, alpha, beta = value.float(), alpha.float(), beta.float()
    A, B = init_gdn_affine(
        batch_size=workers,
        num_heads=heads,
        d_k=d_k,
        d_v=d_v,
        dtype=torch.float32,
        device=key.device,
    )
    for worker, pair in enumerate(previous):
        if pair is not None:
            A[worker] = pair[0][0]
            B[worker] = pair[1][0]
    for token in range(sequence):
        a = alpha[:, token].unsqueeze(-1).unsqueeze(-1)
        b = beta[:, token].unsqueeze(-1).unsqueeze(-1)
        k = key[:, token]
        v = value[:, token]
        A_k = torch.matmul(A, k.unsqueeze(-1)).squeeze(-1)
        B_k = torch.matmul(B, k.unsqueeze(-1)).squeeze(-1)
        A = a * A - a * b * A_k.unsqueeze(-1) * k.unsqueeze(-2)
        B = a * B - a * b * B_k.unsqueeze(-1) * k.unsqueeze(-2)
        B += b * v.unsqueeze(-1) * k.unsqueeze(-2)
    return A, B


def test_multitoken_cpu_fallback_and_worker_mapping():
    heads, d_k, d_v = 2, 7, 5
    generator = torch.Generator().manual_seed(610)
    targets = [_Block() for _ in range(3)]
    existing_A = torch.randn(1, heads, d_k, d_k, generator=generator)
    existing_B = torch.randn(1, heads, d_v, d_k, generator=generator)
    targets[2].linear_affine[0] = (existing_A, existing_B)
    untouched = targets[1]
    gdn = _gdn(targets, heads=heads, d_k=d_k, d_v=d_v, device="cpu")
    key = torch.randn(2, 3, heads, d_k, generator=generator)
    value = torch.randn(2, 3, heads, d_v, generator=generator)
    alpha = torch.rand(2, 3, heads, generator=generator)
    beta = torch.rand(2, 3, heads, generator=generator)
    previous = [targets[2].linear_affine[0], None]
    expected = _reference(previous, key, value, alpha, beta)

    gdn.capture_token_affines(0, key, value, alpha, beta, workers=[2, 0])
    assert 0 not in untouched.linear_affine
    torch.testing.assert_close(targets[2].linear_affine[0][0], expected[0][0:1])
    torch.testing.assert_close(targets[2].linear_affine[0][1], expected[1][0:1])
    torch.testing.assert_close(targets[0].linear_affine[0][0], expected[0][1:2])
    torch.testing.assert_close(targets[0].linear_affine[0][1], expected[1][1:2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_one_token_cuda_fast_path_mixed_history_and_packed_output():
    workers, heads, d_k, d_v = 7, 4, 19, 13
    generator = torch.Generator(device="cuda").manual_seed(611)
    targets = [_Block() for _ in range(workers)]
    A = torch.randn(workers, heads, d_k, d_k, device="cuda", generator=generator)
    B = torch.randn(workers, heads, d_v, d_k, device="cuda", generator=generator)
    for worker in range(workers):
        if worker % 3:
            targets[worker].linear_affine[0] = (
                A[worker : worker + 1].clone(),
                B[worker : worker + 1].clone(),
            )
    previous = [target.linear_affine.get(0) for target in targets]
    saved = [None if pair is None else tuple(t.clone() for t in pair) for pair in previous]
    key = torch.randn(
        workers, 1, heads, d_k, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    value = torch.randn(
        workers, 1, heads, d_v, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    alpha = torch.rand(workers, 1, heads, device="cuda", generator=generator)
    beta = torch.rand_like(alpha)
    expected = _reference(previous, key, value, alpha, beta)
    gdn = _gdn(targets, heads=heads, d_k=d_k, d_v=d_v, device="cuda")

    gdn.capture_token_affines(0, key, value, alpha, beta)
    for worker, target in enumerate(targets):
        actual = target.linear_affine[0]
        torch.testing.assert_close(
            actual[0], expected[0][worker : worker + 1], rtol=2e-5, atol=3.1e-5
        )
        torch.testing.assert_close(
            actual[1], expected[1][worker : worker + 1], rtol=2e-5, atol=3.1e-5
        )
        if saved[worker] is not None and previous[worker] is not None:
            assert torch.equal(previous[worker][0], saved[worker][0])
            assert torch.equal(previous[worker][1], saved[worker][1])

    A_storage = targets[0].linear_affine[0][0].untyped_storage().data_ptr()
    B_storage = targets[0].linear_affine[0][1].untyped_storage().data_ptr()
    assert all(
        target.linear_affine[0][0].untyped_storage().data_ptr() == A_storage for target in targets
    )
    assert all(
        target.linear_affine[0][1].untyped_storage().data_ptr() == B_storage for target in targets
    )
