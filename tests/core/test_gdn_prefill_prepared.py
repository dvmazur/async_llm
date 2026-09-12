"""Ragged prefill state, conv and recurrence gates, independent of a checkpoint."""

import pytest
import torch

from minisgl.shared_cache.gdn import SharedCacheGDN
from minisgl.shared_cache.gdn_affine import init_gdn_affine
from minisgl.shared_cache.gdn_prefill import GDNPrefillBuffers
from minisgl.shared_cache.shared_block import CacheBlock
from _gdn_reference import SharedCacheGDN as ReferenceGDN

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GDN prefill')


def backend(heads=2):
    return SharedCacheGDN(num_heads=heads, head_k_dim=128, head_v_dim=128,
                          conv_dim=3*heads*128, conv_kernel=4, device=torch.device('cuda'))


@pytest.mark.parametrize('channel_major', [False, True])
@torch.inference_mode()
def test_no_fla_pack_preserves_qk_normalization_layout(channel_major):
    from minisgl.models.qwen3_5_delta import _chunk_gated_delta_rule
    torch.manual_seed(1903)
    ar = backend(16)
    rows = 150
    values = [torch.randn(rows, 16, 128, device='cuda', dtype=torch.bfloat16)
              for _ in range(3)]
    if channel_major:
        values = [x.permute(1, 2, 0).contiguous().permute(2, 0, 1) for x in values]
    q, k, v = values
    g = -torch.rand(rows, 16, device='cuda')*.03
    beta = torch.rand(rows, 16, device='cuda', dtype=torch.bfloat16)
    initial = torch.zeros(1, 16, 128, 128, device='cuda')
    target = CacheBlock(ar.device)
    buf = GDNPrefillBuffers(ar, 1, 1, 1, torch.bfloat16, rows)
    buf.prepare([[target]], [target], [rows])
    seen = []

    def observe_layout(pq, pk, *args, **kwargs):
        seen.extend([pq.stride(-1) == 1, pk.stride(-1) == 1])
        return _chunk_gated_delta_rule(pq, pk, *args, **kwargs)

    try:
        actual = buf.core(q, k, v, g, beta, initial, False, observe_layout)
        expected, _ = _chunk_gated_delta_rule(q[None], k[None], v[None], g[None],
                                              beta[None], initial_state=initial)
        # BF16 q/k normalization happens before the fallback's FP32 conversion.
        # Preserve its reduction layout, including repeated-head token-major q/k.
        assert seen == [q.stride(-1) == 1, k.stride(-1) == 1]
        torch.testing.assert_close(actual, expected[0], atol=0, rtol=0)
    finally:
        buf.publish(False)


@pytest.mark.parametrize('lengths', [(150,), (138, 91, 1)])
@torch.inference_mode()
def test_fp32_summary_scan_against_original(lengths):
    torch.manual_seed(925)
    ar = backend(16)
    total = sum(lengths)
    key = torch.randn(16, 128, total, device='cuda', dtype=torch.bfloat16).permute(2, 0, 1)
    value = torch.randn_like(key)
    alpha = torch.rand(total, 16, device='cuda')*.1+.9
    beta = torch.rand(total, 16, device='cuda', dtype=torch.bfloat16)
    targets = [CacheBlock(ar.device) for _ in lengths]
    refs = [CacheBlock(ar.device) for _ in lengths]
    buf = GDNPrefillBuffers(ar, 1, len(lengths)+1, 1, torch.bfloat16, total)
    buf.prepare([[t] for t in targets], targets, lengths)
    buf.capture_prefill(0, key, value, alpha, beta)
    buf.publish()
    ref = ReferenceGDN(num_heads=16, head_k_dim=128, head_v_dim=128,
                        conv_dim=6144, conv_kernel=4, device=ar.device)
    ref.set_context([[b] for b in refs], refs)
    start = 0
    for w, length in enumerate(lengths):
        section = slice(start, start + length)
        ref.capture_token_affines(0, key[None, section], value[None, section],
                                   alpha[None, section], beta[None, section], workers=[w])
        start += length
        for actual, expected in zip(targets[w].linear_affine[0], refs[w].linear_affine[0]):
            # Single-request FP32 capture must retain the original accumulation
            # order; ragged slicing may alter the key normalization's reduction.
            tol = 0 if len(lengths) == 1 else 5e-6
            torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize('use_fla', [True, False])
@torch.inference_mode()
def test_ragged_prefill_replay_and_updated_targets(use_fla):
    from minisgl.models.qwen3_5_delta import _chunk_delta, _chunk_gated_delta_rule, _prefill_conv_silu
    if use_fla:
        pytest.importorskip('fla.ops.gated_delta_rule')
    torch.manual_seed(122)
    ar = backend()
    buf = GDNPrefillBuffers(ar, 1, 3, 2, torch.bfloat16, 256)
    common = CacheBlock(ar.device)
    common.linear_affine[0] = init_gdn_affine(batch_size=1, num_heads=2, d_k=128, device='cuda')
    common.linear_affine[0][1].normal_(0, .01)
    common.linear_conv_state[0] = torch.randn(ar.conv_dim, 4, device='cuda', dtype=torch.bfloat16)*.1
    targets = [CacheBlock(ar.device) for _ in range(3)]
    qkv = torch.randn(256, ar.conv_dim, device='cuda', dtype=torch.bfloat16)*.1
    weight = torch.randn(ar.conv_dim, 1, 4, device='cuda', dtype=torch.bfloat16)*.1
    g = -torch.rand(256, 2, device='cuda')*.03
    beta = torch.rand(256, 2, device='cuda', dtype=torch.bfloat16)
    output = torch.empty(256, 2, 128, device='cuda', dtype=torch.bfloat16)

    def body():
        conv = buf.convolve(0, qkv, weight)
        q, k, v = [t.reshape(256, 2, 128) for t in conv.split(256, -1)]
        output.copy_(buf.core(q, k, v, g, beta, buf.compose(0), use_fla, _chunk_gated_delta_rule))
        buf.capture_prefill(0, k, v, g.exp(), beta, g=g if use_fla else None)

    dummy = [CacheBlock(ar.device) for _ in range(3)]
    buf.prepare([[b] for b in dummy], dummy, [32, 32, 32])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        body()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    buf.publish(False)

    for lengths in ([63, 65], [1], [7, 5, 9], [138, 91]):
        n = len(lengths)
        ref_targets = [CacheBlock(ar.device) for _ in lengths]
        for actual, ref_target in zip(targets, ref_targets):
            ref_target.linear_affine = dict(actual.linear_affine)
            ref_target.linear_conv_state = dict(actual.linear_conv_state)
        ref = ReferenceGDN(num_heads=2, head_k_dim=128, head_v_dim=128,
                            conv_dim=ar.conv_dim, conv_kernel=4, device=ar.device)
        ref.set_context([[common, b] for b in ref_targets], ref_targets)
        prior = ref.prior_conv_states(0)
        initial = ref.compose_initial_recurrent_state(0, torch.float32)
        buf.prepare([[common, b] for b in targets[:n]], targets[:n], lengths)
        graph.replay()
        actual_output = output.clone()
        buf.publish()
        start = 0
        for w, length in enumerate(lengths):
            section = slice(start, start+length)
            inputs = torch.cat([prior[w:w+1, :, -3:], qkv[section].t()[None]], -1)
            conv = _prefill_conv_silu(inputs, weight, 3)[0, :, 3:3+length].t()
            q, k, v = [t.reshape(1, length, 2, 128) for t in conv.split(256, -1)]
            fn = _chunk_delta if use_fla else _chunk_gated_delta_rule
            expected, _ = fn(q, k, v, g[None, section], beta[None, section], initial_state=initial[w:w+1])
            torch.testing.assert_close(actual_output[section], expected[0], atol=2e-4, rtol=2e-3)
            ref.capture_token_affines(0, k, v, g[None, section].exp(), beta[None, section], workers=[w])
            ref.set_conv_states(0, inputs[..., -4:], workers=[w])
            for actual, expected in zip(targets[w].linear_affine[0], ref_targets[w].linear_affine[0]):
                torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)
            torch.testing.assert_close(targets[w].linear_conv_state[0], ref_targets[w].linear_conv_state[0], atol=0, rtol=0)
            start += length
        assert torch.count_nonzero(actual_output[start:]) == 0
