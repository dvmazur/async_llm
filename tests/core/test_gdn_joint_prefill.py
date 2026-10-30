"""Joint prefill versus installed FLA, including pointer ownership and replay.

BF16 affine capture is intentionally not bitwise-equivalent to the legacy FP32
scan. Compare the IO/algorithm with an independent public-FLA construction;
learned SGLang/Transformers tests retain the existing numerical quality limits.
"""

import pytest
import torch

from minisgl.kernel.gdn_prefill_io import pack_initial, store_affines
from minisgl.shared_cache.gdn import SharedCacheGDN
from minisgl.shared_cache.gdn_prefill import GDNPrefillBuffers
from minisgl.shared_cache.shared_block import CacheBlock

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Joint FLA prefill')


@pytest.mark.parametrize('dk,dv', [(64, 32), (128, 128)])
@pytest.mark.parametrize('state_dtype', [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_joint_state_pack_and_store_are_exact(dk, dv, state_dtype):
    torch.manual_seed(804)
    workers, heads = 3, 2
    initial = torch.randn(workers, heads, dk, dv, device='cuda', dtype=state_dtype)
    a = torch.randn(1, heads, dk, dk, device='cuda', dtype=state_dtype)  # nonsymmetric
    b = torch.randn(1, heads, dv, dk, device='cuda', dtype=state_dtype)
    read = torch.tensor([[a.data_ptr(), b.data_ptr(), 0], [0, 0, 0],
                         [a.data_ptr(), 0, 0]], device='cuda', dtype=torch.int64)
    joint = torch.empty(workers, heads, dk, dv + dk + dv, device='cuda', dtype=state_dtype)
    pack_initial(read, initial, joint)
    identity = torch.eye(dk, device='cuda', dtype=state_dtype).expand(1, heads, dk, dk)
    expected_a = torch.cat([a, identity, a]).transpose(-1, -2)
    expected_b = torch.cat([b, torch.zeros_like(b), torch.zeros_like(b)]).transpose(-1, -2)
    expected = torch.cat([initial, expected_a, expected_b], -1)
    torch.testing.assert_close(joint, expected, atol=0, rtol=0)
    cu = torch.tensor([0, 1, 1, 3], device='cuda', dtype=torch.int32)
    joint.fill_(42)
    pack_initial(read, initial, joint, cu)
    torch.testing.assert_close(joint[[0, 2]], expected[[0, 2]], atol=0, rtol=0)
    assert torch.all(joint[1] == 42), 'do not repack an inactive slot'
    out_a, out_b = torch.empty_like(a), torch.empty_like(b)
    # Inactive workers and independently missing output pointers are legal.
    write = torch.tensor([[out_a.data_ptr(), 0, 0], [0, 0, 0],
                          [0, out_b.data_ptr(), 0]], device='cuda', dtype=torch.int64)
    final = torch.randn_like(joint)
    store_affines(write, final, dv)
    torch.testing.assert_close(out_a[0], final[0, ..., dv:dv+dk].transpose(-1, -2), atol=0, rtol=0)
    torch.testing.assert_close(out_b[0], final[2, ..., dv+dk:].transpose(-1, -2), atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('dk,dv', [(64, 32), (128, 128)])
@pytest.mark.parametrize('state_dtype', [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_one_joint_fla_pass_replays_ragged_lengths_and_new_allocations(dtype, dk, dv, state_dtype, monkeypatch):
    fla = pytest.importorskip('fla.ops.gated_delta_rule')
    from minisgl.models.qwen3_5_delta import _chunk_gated_delta_rule
    import minisgl.shared_cache.gdn_prefill as module

    torch.manual_seed(808)
    rows, workers, heads = 256, 3, 2
    ar = SharedCacheGDN(num_heads=heads, head_k_dim=dk, head_v_dim=dv,
                        conv_dim=heads*(2*dk+dv), conv_kernel=4, device=torch.device('cuda'), state_dtype=state_dtype)
    buf = GDNPrefillBuffers(ar, 1, workers, 2, dtype, rows)
    common = CacheBlock(ar.device)
    common.linear_affine[0] = (torch.randn(1, heads, dk, dk, device='cuda')*.01,
                                torch.randn(1, heads, dv, dk, device='cuda')*.02)
    # Noncontiguous channel-major inputs, as produced by the prepared conv.
    q, k = [torch.randn(heads, dk, rows, device='cuda', dtype=dtype).permute(2, 0, 1)*.1
             for _ in range(2)]
    v = torch.randn(heads, dv, rows, device='cuda', dtype=dtype).permute(2, 0, 1)*.1
    g = -torch.rand(rows, heads, device='cuda')*.03
    beta = torch.rand(rows, heads, device='cuda', dtype=dtype)
    conv = torch.zeros(workers, ar.conv_dim, 4, device='cuda', dtype=dtype)
    targets = [CacheBlock(ar.device) for _ in range(workers)]
    # Start from nonempty/non-symmetric A/B as well as absent identity/zero.
    targets[0].linear_affine[0] = (torch.randn(1, heads, dk, dk, device='cuda')*.1,
                                   torch.randn(1, heads, dv, dk, device='cuda')*.1)
    calls = []
    chunk = module.chunk_gdn

    def observed(*args, **kwargs):
        calls.append((args[2].dtype, args[2].shape[-1], args[5].dtype, kwargs))
        return chunk(*args, **kwargs)

    monkeypatch.setattr(module, 'chunk_gdn', observed)
    monkeypatch.setattr(buf, 'capture_prefill', lambda *a, **k: pytest.fail('second capture pass'))

    def body():
        out = buf.core_and_capture(0, q, k, v, g, beta, buf.compose(0), True, _chunk_gated_delta_rule)
        buf.store_conv(0, conv)
        return out

    buf.prepare([[common, target] for target in targets], targets, [64, 64, 64])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        body()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = body()
    assert calls == [(dtype, dv+dk+dv, state_dtype,
                      {'output_final_state': True, 'skip_empty_states': True})]*2
    buf.publish(False)
    scratch_ptrs = (buf.joint_initial.data_ptr(), buf.joint_values.data_ptr())

    for lengths in ([63, 65], [1], [7, 5, 9], [138, 91], []):
        active = targets[:len(lengths)]
        buf.chunk_indices.fill_(2**30)
        buf.prepare([[common, t] for t in active], active, lengths)
        actual_chunks = sum((length+63)//64 for length in lengths)
        assert torch.all(buf.chunk_indices[actual_chunks:] == 2**30)
        buf.joint_initial[len(active):].fill_(float('nan'))
        initial = buf.compose(0).clone()
        references, retained = [], []
        start = 0
        for worker, (target, length) in enumerate(zip(active, lengths)):
            a, b = target.linear_affine.get(0, (
                torch.eye(dk, device='cuda').expand(1, heads, dk, dk),
                torch.zeros(1, heads, dv, dk, device='cuda')))
            retained.extend((x, x.clone()) for x in target.linear_affine.get(0, ()))
            a, b = a.to(state_dtype), b.to(state_dtype)
            state = torch.cat([initial[worker:worker+1], a.transpose(-1, -2), b.transpose(-1, -2)], -1)
            section = slice(start, start+length)
            values = torch.cat([v[None, section], torch.zeros(1, length, heads, dk, device='cuda', dtype=dtype),
                                v[None, section]], -1)
            result, final = fla.chunk_gated_delta_rule(
                q[None, section], k[None, section], values, g=g[None, section], beta=beta[None, section].float(),
                initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
            core, _ = fla.chunk_gated_delta_rule(
                q[None, section], k[None, section], v[None, section], g=g[None, section],
                beta=beta[None, section].float(), initial_state=initial[worker:worker+1],
                output_final_state=True, use_qk_l2norm_in_kernel=True)
            references.append((section, result[0, ..., :dv], final, core[0]))
            start += length
        graph.replay()
        actual = output.clone()
        buf.publish()
        for target, (section, want, final, core) in zip(active, references):
            torch.testing.assert_close(actual[section], want, atol=2e-4, rtol=2e-3)
            torch.testing.assert_close(actual[section], core, atol=2e-4, rtol=2e-3)
            for got, ref in zip(target.linear_affine[0], (final[..., dv:dv+dk].transpose(-1, -2),
                                                          final[..., dv+dk:].transpose(-1, -2))):
                assert got.dtype == state_dtype and got.is_contiguous()
                # Same BF16 initial values; public FLA returns FP32 final state.
                # Compare at the explicit storage-rounding boundary.
                torch.testing.assert_close(got, ref.to(state_dtype),
                    atol=2e-5 if state_dtype==torch.float32 else 2e-3,
                    rtol=2e-4 if state_dtype==torch.float32 else 8e-3)
                assert got.untyped_storage().data_ptr() != buf.joint_initial.untyped_storage().data_ptr()
        assert torch.count_nonzero(actual[start:]) == 0
        for old, saved in retained:
            torch.testing.assert_close(old, saved, atol=0, rtol=0)
        assert scratch_ptrs == (buf.joint_initial.data_ptr(), buf.joint_values.data_ptr())
    assert len(calls) == 2  # Replay never re-enters Python/FLA dispatch.


@torch.inference_mode()
def test_joint_entry_keeps_no_fla_core_and_fp32_capture(monkeypatch):
    import minisgl.shared_cache.gdn_prefill as module
    from minisgl.models.qwen3_5_delta import _chunk_gated_delta_rule
    torch.manual_seed(811)
    ar = SharedCacheGDN(num_heads=2, head_k_dim=64, head_v_dim=32,
                        conv_dim=320, conv_kernel=4, device=torch.device('cuda'))
    buf = GDNPrefillBuffers(ar, 1, 1, 1, torch.bfloat16, 17)
    target = CacheBlock(ar.device)
    buf.prepare([[target]], [target], [17])
    q, k = [torch.randn(17, 2, 64, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    v = torch.randn(17, 2, 32, device='cuda', dtype=torch.bfloat16)
    g, beta = -torch.rand(17, 2, device='cuda')*.03, torch.rand(17, 2, device='cuda')
    initial = buf.compose(0)
    expected = buf.core(q, k, v, g, beta, initial, False, _chunk_gated_delta_rule).clone()
    buf.capture_prefill(0, k, v, g.exp(), beta)
    expected_states = [x[0].clone() for x in buf._current[1:3]]
    monkeypatch.setattr(module, 'chunk_gdn', lambda *a, **k: pytest.fail('FLA in no-FLA mode'))
    actual = buf.core_and_capture(0, q, k, v, g, beta, initial, False, _chunk_gated_delta_rule)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for got, want in zip(buf._current[1:3], expected_states):
        torch.testing.assert_close(got[0], want, atol=0, rtol=0)
    buf.publish(False)
