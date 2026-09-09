"""Integration tests for SharedCacheGDN affine capture dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
import minisgl.models.qwen3_5_delta as delta_module
from minisgl.models.qwen3_5_delta import (
    _capture_affine_summary_fla,
    _core_and_capture_affine_fla,
)
from minisgl.shared_cache.gdn import SharedCacheGDN
from minisgl.shared_cache.gdn_affine import init_gdn_affine
from minisgl.shared_cache.shared_block import CacheBlock


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


def test_identity_padding_batches_ragged_affine_capture():
    """alpha=1,beta=0 padding must leave a shorter worker's summary unchanged."""
    heads, d_k, d_v = 2, 7, 5
    lengths = [2, 4]
    max_length = max(lengths)
    generator = torch.Generator().manual_seed(612)
    targets = [_Block() for _ in lengths]
    gdn = _gdn(targets, heads=heads, d_k=d_k, d_v=d_v, device="cpu")
    key = torch.randn(2, max_length, heads, d_k, generator=generator)
    value = torch.randn(2, max_length, heads, d_v, generator=generator)
    alpha = torch.rand(2, max_length, heads, generator=generator)
    beta = torch.rand(2, max_length, heads, generator=generator)
    alpha[0, lengths[0] :] = 1
    beta[0, lengths[0] :] = 0

    expected = []
    for worker, length in enumerate(lengths):
        expected.append(
            _reference(
                [None],
                key[worker : worker + 1, :length],
                value[worker : worker + 1, :length],
                alpha[worker : worker + 1, :length],
                beta[worker : worker + 1, :length],
            )
        )

    gdn.capture_token_affines(0, key, value, alpha, beta)
    for worker, target in enumerate(targets):
        torch.testing.assert_close(target.linear_affine[0][0], expected[worker][0])
        torch.testing.assert_close(target.linear_affine[0][1], expected[worker][1])


@pytest.mark.skipif(
    not torch.cuda.is_available() or delta_module._fla_chunk is None,
    reason="CUDA and flash-linear-attention are required",
)
def test_fla_multitoken_capture_matches_token_loop_with_existing_summary():
    workers, sequence, heads, d_k, d_v = 2, 17, 4, 16, 16
    generator = torch.Generator(device="cuda").manual_seed(613)
    targets = [_Block() for _ in range(workers)]
    previous_A = torch.randn(
        workers, heads, d_k, d_k, device="cuda", generator=generator
    ) * 0.03
    previous_B = torch.randn(
        workers, heads, d_v, d_k, device="cuda", generator=generator
    ) * 0.03
    for worker, target in enumerate(targets):
        target.linear_affine[0] = (
            previous_A[worker : worker + 1].clone(),
            previous_B[worker : worker + 1].clone(),
        )
    previous = [target.linear_affine[0] for target in targets]
    key = torch.randn(
        workers,
        sequence,
        heads,
        d_k,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    value = torch.randn(
        workers,
        sequence,
        heads,
        d_v,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    g = -torch.rand(
        workers, sequence, heads, device="cuda", dtype=torch.float32, generator=generator
    )
    beta = torch.rand(
        workers,
        sequence,
        heads,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    expected_A, expected_B = _reference(previous, key, value, g.exp(), beta)
    gdn = _gdn(targets, heads=heads, d_k=d_k, d_v=d_v, device="cuda")

    assert _capture_affine_summary_fla(gdn, 0, key, value, g, beta)
    actual_A = torch.cat([target.linear_affine[0][0] for target in targets])
    actual_B = torch.cat([target.linear_affine[0][1] for target in targets])
    for actual, expected in ((actual_A, expected_A), (actual_B, expected_B)):
        relative_error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
            expected
        )
        assert relative_error.item() < 1e-2

    # All worker A/B views are backed by the one packed final-state allocation.
    storage = targets[0].linear_affine[0][0].untyped_storage().data_ptr()
    assert all(
        tensor.untyped_storage().data_ptr() == storage
        for target in targets
        for tensor in target.linear_affine[0]
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or delta_module._fla_chunk is None,
    reason="CUDA and flash-linear-attention are required",
)
def test_fused_core_and_affine_varlen_matches_old_two_scan_reference():
    lengths = [17, 9, 23]
    workers, total, heads, d_k, d_v = len(lengths), sum(lengths), 4, 16, 16
    generator = torch.Generator(device="cuda").manual_seed(614)
    expected_targets = [_Block() for _ in lengths]
    actual_targets = [_Block() for _ in lengths]
    previous_A = torch.randn(
        workers, heads, d_k, d_k, device="cuda", generator=generator
    ) * 0.03
    previous_B = torch.randn(
        workers, heads, d_v, d_k, device="cuda", generator=generator
    ) * 0.03
    for worker in range(workers):
        for targets in (expected_targets, actual_targets):
            targets[worker].linear_affine[0] = (
                previous_A[worker : worker + 1].clone(),
                previous_B[worker : worker + 1].clone(),
            )

    query = torch.randn(
        1, total, heads, d_k, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    g = -torch.rand(
        1, total, heads, device="cuda", dtype=torch.float32, generator=generator
    )
    beta = torch.rand(
        1, total, heads, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    core_initial = torch.randn(
        workers, heads, d_v, d_k, device="cuda", generator=generator
    ) * 0.03
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu_cpu = torch.tensor(offsets, dtype=torch.long)
    cu = cu_cpu.cuda()

    expected_core, _ = delta_module._chunk_delta(
        query,
        key,
        value,
        g,
        beta,
        initial_state=core_initial,
        state_v_first=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
    )
    expected_gdn = _gdn(
        expected_targets, heads=heads, d_k=d_k, d_v=d_v, device="cuda"
    )
    assert _capture_affine_summary_fla(
        expected_gdn,
        0,
        key,
        value,
        g,
        beta,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
    )

    actual_gdn = _gdn(
        actual_targets, heads=heads, d_k=d_k, d_v=d_v, device="cuda"
    )
    actual_core = _core_and_capture_affine_fla(
        actual_gdn,
        0,
        query,
        key,
        value,
        g,
        beta,
        core_initial_state=core_initial,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
    )
    assert actual_core is not None
    torch.testing.assert_close(actual_core, expected_core, rtol=1e-2, atol=2e-2)
    for actual_target, expected_target in zip(actual_targets, expected_targets):
        for actual, expected in zip(
            actual_target.linear_affine[0], expected_target.linear_affine[0]
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_block_owned_slab_scatter_and_inplace_decode_capture():
    workers, layers, heads, dim = 3, 4, 2, 16
    targets = [CacheBlock(torch.device("cuda")) for _ in range(workers)]
    gdn = SharedCacheGDN(
        num_heads=heads,
        head_k_dim=dim,
        head_v_dim=dim,
        conv_dim=1,
        conv_kernel=1,
        device=torch.device("cuda"),
        num_linear_layers=layers,
    )
    gdn.set_context([[] for _ in targets], targets)
    generator = torch.Generator(device="cuda").manual_seed(615)
    scan_state = torch.randn(
        workers,
        heads,
        2 * dim,
        dim,
        device="cuda",
        generator=generator,
    )
    gdn.store_affine_scan_state(1, scan_state, d_k=dim)
    for worker, target in enumerate(targets):
        assert target.linear_affine_storage is not None
        assert target.linear_affine_storage.shape == (layers, 2, heads, dim, dim)
        A, B = target.linear_affine[1]
        torch.testing.assert_close(A[0], scan_state[worker, :, :dim])
        torch.testing.assert_close(B[0], scan_state[worker, :, dim:])
        assert A.untyped_storage().data_ptr() == B.untyped_storage().data_ptr()
    assert len(
        {target.linear_affine_storage.untyped_storage().data_ptr() for target in targets}
    ) == workers

    previous = [target.linear_affine[1] for target in targets]
    key = torch.randn(
        workers, 1, heads, dim, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    value = torch.randn_like(key)
    alpha = torch.rand(workers, 1, heads, device="cuda", generator=generator)
    beta = torch.rand_like(alpha)
    expected_A, expected_B = _reference(previous, key, value, alpha, beta)
    storage_before = [target.linear_affine_storage.data_ptr() for target in targets]
    gdn.capture_token_affines(1, key, value, alpha, beta)
    for worker, target in enumerate(targets):
        assert target.linear_affine_storage.data_ptr() == storage_before[worker]
        A, B = target.linear_affine[1]
        torch.testing.assert_close(
            A, expected_A[worker : worker + 1], rtol=2e-5, atol=3.1e-5
        )
        torch.testing.assert_close(
            B, expected_B[worker : worker + 1], rtol=2e-5, atol=3.1e-5
        )
