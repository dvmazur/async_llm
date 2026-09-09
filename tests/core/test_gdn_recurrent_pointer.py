"""Deferred state reads against installed FLA and the unchanged no-FLA path."""
import gc

import pytest
import torch

import minisgl.models.qwen3_5_delta as delta
from minisgl.shared_cache.gdn import _materialize_state_parts
from test_gdn_successor import world, save


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('dims', [(7, 5, 2, 2), (32, 17, 2, 4), (128, 128, 4, 4)])
@pytest.mark.parametrize('layout', ['fragmented', 'permuted', 'duplicate'])
@torch.inference_mode()
def test_pointer_matches_fla_exact_over_steps(dtype, dims, layout):
    if not torch.cuda.is_available() or delta._fla_recurrent is None:
        pytest.skip('FLA CUDA required')
    from minisgl.kernel.gdn_recurrent import recurrent_gdn_pointer

    torch.manual_seed(811)
    dk, dv, h, hv = dims
    b = 5
    original = torch.randn(b, hv, dv, dk, device='cuda') * .1
    rows = [original[i:i+1].clone() for i in range(b)]
    if layout == 'permuted':
        rows = [original[i:i+1] for i in [4, 1, 3, 0, 2]]
    elif layout == 'duplicate':
        rows = [rows[i] for i in [2, 0, 2, 3, 0]]
    before = [row.clone() for row in rows]
    state = torch.cat(rows)
    for step in range(24):
        # Non-contiguous q/k/v exercise FLA's input_guard equivalent.
        q, k = [torch.randn(b, 1, h, dk * 2, device='cuda', dtype=dtype)[..., ::2]
                for _ in range(2)]
        v = torch.randn(b, 1, hv, dv * 2, device='cuda', dtype=dtype)[..., ::2]
        g = -torch.rand(b, 1, hv, device='cuda') * .1
        beta = torch.rand(b, 1, hv, device='cuda', dtype=dtype)
        expected, state = delta._fla_recurrent(q, k, v, g=g, beta=beta,
            initial_state=state, output_final_state=True,
            use_qk_l2norm_in_kernel=True, state_v_first=True)
        actual, new_state = recurrent_gdn_pointer(q, k, v, g, beta, rows)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(new_state, state, rtol=0, atol=0)
        if step == 0:
            for row, saved in zip(rows, before):
                torch.testing.assert_close(row, saved, rtol=0, atol=0)
        rows = [new_state[i:i+1].clone() for i in range(b)]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('hits', ['all', 'partial', 'none', 'disabled'])
@torch.inference_mode()
def test_deferred_begin_matches_dense_without_terminal_cat(monkeypatch, device, hits):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    gdn, common, tails = world(4, device, dk=16, dv=11)
    save(gdn)
    if hits == 'disabled':
        gdn.configure_successor_cache(0)
    elif hits != 'all':
        for target in (tails if hits == 'none' else tails[:2]):
            target.set_linear_affine(0, target.linear_affine[0])
    order = [3, 0, 2, 1]
    gdn.set_context([gdn.cache_structure[i] for i in order], [tails[i] for i in order])
    expected, old_ticket = gdn.begin_decode_state(0)
    original = torch.cat
    calls = []
    def cat(parts, *a, **kw):
        import sys
        if sys._getframe(1).f_code.co_name in ('begin_decode_state', 'assemble_state_rows',
                '_materialize_state_parts', '_terminal_states'):
            calls.append(True)
        return original(parts, *a, **kw)
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'cat', cat)
        parts, ticket = gdn.begin_decode_state(0, materialize=False)
    assert not calls
    assert old_ticket == ticket
    gdn.configure_successor_cache(0)
    gdn.configure_compose_cache(0)
    common.clear()
    for tail in tails:
        tail.clear()
    del gdn, common, tails
    gc.collect()
    torch.testing.assert_close(_materialize_state_parts(parts), expected, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_deferred_no_fla_fallback_is_unchanged(monkeypatch, dtype):
    monkeypatch.setattr(delta, '_fla_recurrent', None)
    q, k, v = [torch.randn(3, 1, 2, 7, dtype=dtype) for _ in range(3)]
    g, beta = -torch.rand(3, 1, 2), torch.rand(3, 1, 2, dtype=dtype)
    rows = [torch.randn(1, 2, 7, 7) for _ in range(3)]
    expected = delta._recurrent_delta(q, k, v, g, beta, torch.cat(rows), state_v_first=True)
    actual = delta._recurrent_delta(q, k, v, g, beta, rows, state_v_first=True)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
