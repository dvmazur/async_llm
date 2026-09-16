"""Conv-only parity: literal old Torch body, pointers, layouts and graph replay."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from minisgl.shared_cache.gdn import SharedCacheGDN
from minisgl.shared_cache.gdn_prefill import GDNPrefillBuffers
from minisgl.shared_cache.shared_block import CacheBlock

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA convolution')


def old_convolve(buf, layer, qkv, weight):
    # Literal pre-optimization body, retained only as a test reference.
    prior = buf.conv(layer).transpose(1, 2).reshape(-1, buf.conv_shape[0])
    inputs = torch.cat([prior, qkv], dim=0)
    total = torch.zeros(qkv.shape[1], qkv.shape[0], device=qkv.device, dtype=torch.float32)
    for tap in range(buf.conv_shape[1]):
        total += (inputs.index_select(0, buf.conv_indices[:, tap]).t() * weight[:, 0, tap, None]).float()
    new_state = inputs[buf.state_indices].transpose(1, 2)
    buf.store_conv(layer, new_state)
    return F.silu(total).to(qkv.dtype).t()


def reference(buf, lengths):
    width = buf.conv_shape[1]
    base, start, conv, store = buf.workers * width, 0, [], []
    for worker, length in enumerate(list(lengths) + [0] * (buf.workers-len(lengths))):
        for position in range(length):
            conv.append([base+start+t if t >= 0 else worker*width+width+t
                         for t in range(position-width+1, position+1)])
        store.append([base+start+t if t >= 0 else worker*width+width+t
                      for t in range(length-width, length)])
        start += length
    conv += [[0]*width] * (buf.rows-start)
    states = {}
    obj = SimpleNamespace(conv=buf.conv, conv_shape=buf.conv_shape,
        conv_indices=torch.tensor(conv, device=buf.device),
        state_indices=torch.tensor(store, device=buf.device),
        store_conv=lambda layer, x: states.update({layer: x.clone()}))
    return obj, states


def make_buffers(channels, width, dtype, rows, workers=4):
    ar = SharedCacheGDN(num_heads=1, head_k_dim=128, head_v_dim=128,
        conv_dim=channels, conv_kernel=width, device=torch.device('cuda'))
    return GDNPrefillBuffers(ar, 2, workers, 3, dtype, rows)


def sources(buf, n):
    common = CacheBlock(buf.device)
    parents = [CacheBlock(buf.device) for _ in range(n)]
    targets = [CacheBlock(buf.device) for _ in range(n)]
    # Same shared parent, per-worker overrides, absent layer, and a noncontiguous
    # source tensor: prepare must retain its converted storage for pointer loads.
    common.linear_conv_state[0] = torch.randn(*buf.conv_shape, device=buf.device, dtype=buf.dtype)
    for i, parent in enumerate(parents):
        if i % 2:
            parent.linear_conv_state[1] = torch.randn(*buf.conv_shape[::-1], device=buf.device,
                                                     dtype=buf.dtype).t()
    return [[common, p, t] for p,t in zip(parents,targets)], targets


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('lengths,channels,width', [
    ((1,), 67, 4), ((2, 7, 1), 129, 4), ((63, 65), 97, 4),
    ((138, 91), 513, 4), ((5, 3), 67, 1), ((3, 1, 9), 65, 7), ((), 67, 4),
])
@torch.inference_mode()
def test_prefill_conv_matches_literal_torch(dtype, lengths, channels, width):
    torch.manual_seed(381)
    buf = make_buffers(channels, width, dtype, sum(lengths)+19)
    chains, targets = sources(buf, len(lengths))
    preserved = [(t, t.clone()) for chain in chains for b in chain for t in b.linear_conv_state.values()]
    qkv = torch.randn(buf.rows, channels, device='cuda', dtype=dtype)
    weight = torch.randn(channels, 1, width, device='cuda', dtype=dtype)
    buf.prepare(chains, targets, lengths)
    try:
        ref, states = reference(buf, lengths)
        for layer in range(2):
            expected = old_convolve(ref, layer, qkv, weight)
            actual = buf.convolve(layer, qkv, weight)
            assert actual.stride() == expected.stride() == (1, buf.rows)
            torch.testing.assert_close(actual[:sum(lengths)], expected[:sum(lengths)], rtol=0, atol=0)
            assert torch.count_nonzero(actual[sum(lengths):]) == 0
            for worker in range(len(lengths)):
                torch.testing.assert_close(buf._current[3][worker][layer], states[layer][worker], rtol=0, atol=0)
        for tensor, snapshot in preserved:
            torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
    finally:
        buf.publish(False)


@torch.inference_mode()
def test_prefill_conv_graph_dynamic_shapes_and_pointers():
    torch.manual_seed(582)
    buf = make_buffers(8192, 4, torch.bfloat16, 256)
    qkv = torch.randn(256, 8192, device='cuda', dtype=torch.bfloat16)*.1
    weight = torch.randn(8192, 1, 4, device='cuda', dtype=torch.bfloat16)*.1
    chains, targets = sources(buf, 3)
    buf.prepare(chains, targets, [32, 32, 32])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        buf.convolve(0, qkv, weight)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        buf.convolve(0, qkv, weight)
    buf.publish(False)
    retained = []
    for lengths in ([138, 91], [1], [63, 65, 7], [3, 2], [256], []):
        chains, targets = sources(buf, len(lengths))
        qkv.normal_(0, .2)
        buf.prepare(chains, targets, lengths)
        try:
            ref, states = reference(buf, lengths)
            expected = old_convolve(ref, 0, qkv, weight)
            graph.replay()
            torch.testing.assert_close(buf.conv_output[:sum(lengths)], expected[:sum(lengths)], atol=0, rtol=0)
            assert torch.count_nonzero(buf.conv_output[sum(lengths):]) == 0
            for i in range(len(lengths)):
                actual_state = buf._current[3][i][0]
                torch.testing.assert_close(actual_state, states[0][i], atol=0, rtol=0)
                retained.append((actual_state, actual_state.clone()))
        finally:
            buf.publish(False)
    # Later replays must not overwrite earlier worker-owned output allocations.
    for tensor, snapshot in retained:
        torch.testing.assert_close(tensor, snapshot, atol=0, rtol=0)
