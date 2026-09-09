"""Direct tests for the pointer-aware one-token GDN capture kernel."""

from __future__ import annotations

import pytest
import torch
from minisgl.kernel import capture_gdn_affine_pointer_update
from minisgl.shared_cache.gdn_affine import init_gdn_affine


def _reference(previous, key, value, alpha, beta):
    workers, heads, d_k = key.shape
    d_v = value.shape[-1]
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
    alpha_2d = alpha.unsqueeze(-1).unsqueeze(-1)
    beta_2d = beta.unsqueeze(-1).unsqueeze(-1)
    A_k = torch.matmul(A, key.unsqueeze(-1)).squeeze(-1)
    B_k = torch.matmul(B, key.unsqueeze(-1)).squeeze(-1)
    A_new = alpha_2d * A - alpha_2d * beta_2d * A_k.unsqueeze(-1) * key.unsqueeze(-2)
    B_new = alpha_2d * B - alpha_2d * beta_2d * B_k.unsqueeze(-1) * key.unsqueeze(-2)
    B_new += beta_2d * value.unsqueeze(-1) * key.unsqueeze(-2)
    return A_new, B_new


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("layout", ["fragmented", "packed"])
@pytest.mark.parametrize("history", ["fresh", "existing", "mixed"])
def test_pointer_capture_matches_reference(layout, history):
    workers, heads, d_k, d_v = 7, 4, 19, 13
    generator = torch.Generator(device="cuda").manual_seed(510)
    key = torch.randn(workers, heads, d_k, device="cuda", generator=generator)
    value = torch.randn(workers, heads, d_v, device="cuda", generator=generator)
    alpha = torch.rand(workers, heads, device="cuda", generator=generator)
    beta = torch.rand(workers, heads, device="cuda", generator=generator)
    A_storage = torch.randn(workers, heads, d_k, d_k, device="cuda", generator=generator)
    B_storage = torch.randn(workers, heads, d_v, d_k, device="cuda", generator=generator)
    pairs = []
    for worker in range(workers):
        present = history == "existing" or (history == "mixed" and worker % 3 != 0)
        if not present:
            pairs.append(None)
        elif layout == "packed":
            pairs.append((A_storage[worker : worker + 1], B_storage[worker : worker + 1]))
        else:
            pairs.append(
                (
                    A_storage[worker : worker + 1].clone(),
                    B_storage[worker : worker + 1].clone(),
                )
            )
    before = [None if pair is None else tuple(t.clone() for t in pair) for pair in pairs]

    expected = _reference(pairs, key, value, alpha, beta)
    actual = capture_gdn_affine_pointer_update(pairs, key, value, alpha, beta)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-5, atol=3.1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-5, atol=3.1e-5)
    assert actual[0].is_contiguous() and actual[1].is_contiguous()
    for pair, saved in zip(pairs, before):
        if pair is not None and saved is not None:
            assert torch.equal(pair[0], saved[0])
            assert torch.equal(pair[1], saved[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pointer_capture_production_shape():
    workers, heads, d_k = 64, 32, 128
    generator = torch.Generator(device="cuda").manual_seed(511)
    key = torch.randn(workers, heads, d_k, device="cuda", generator=generator)
    value = torch.randn_like(key)
    alpha = torch.rand(workers, heads, device="cuda", generator=generator)
    beta = torch.rand_like(alpha)
    A = torch.randn(workers, heads, d_k, d_k, device="cuda", generator=generator)
    B = torch.randn_like(A)
    previous = [(A[worker : worker + 1], B[worker : worker + 1]) for worker in range(workers)]
    expected = _reference(previous, key, value, alpha, beta)
    actual = capture_gdn_affine_pointer_update(previous, key, value, alpha, beta)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-5, atol=3.1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-5, atol=3.1e-5)
