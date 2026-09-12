"""Pointer compose vs frozen dense execution and the pre-graph affine oracle."""

import pytest
import torch

from minisgl.kernel.gdn_compose import compose_first, compose_level
from minisgl.kernel.gdn_io import collect_states, gather_rows
from minisgl.shared_cache.gdn import SharedCacheGDN
from minisgl.shared_cache.gdn_decode import GDNDecodeBuffers, prefix_links
from minisgl.shared_cache.shared_block import CacheBlock
from test_gdn_decode_prepared import legacy_compose

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA pointer compose')


@pytest.fixture(autouse=True)
def ieee_reference():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def dense_compose(self, layer):
    # Verbatim GDNDecodeBuffers.compose at 280a426. Retained ONLY in tests;
    # its scratch and pointers are independent of the new ping-pong buffers.
    self.compose_count += 1  # capture counts host execution; replay counted by runner
    self.initial.zero_()
    state = None
    for level in range(self.depth):
        gather_rows(self.affine_ptrs[layer, level, :, 1], self.b)
        if level == 0:
            state = self.b.clone()
        else:
            gather_rows(self.affine_ptrs[layer, level, :, 0], self.a, self.dk)
            state = torch.matmul(state.index_select(0, self.parents[level]), self.a)
            state.add_(self.b)
        collect_states(state, self.terminals[level], self.initial)
    return self.initial


def capture(body):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        body()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    return graph


@pytest.mark.parametrize('dk,dv', [(8, 6), (37, 23), (128, 128), (128, 96)])
@pytest.mark.parametrize('captured', [False, True])
@torch.inference_mode()
def test_compose_changing_trie_matches_both_references(dk, dv, captured):
    torch.manual_seed(1783)
    ar = SharedCacheGDN(num_heads=2, head_k_dim=dk, head_v_dim=dv,
                        conv_dim=7, conv_kernel=4, device=torch.device('cuda'))
    buf = GDNDecodeBuffers(ar, 2, 8, 7, torch.bfloat16)
    dense = GDNDecodeBuffers(ar, 2, 8, 7, torch.bfloat16)
    # Only the frozen dense reference needs an A staging allocation.
    dense.a = torch.empty(8, 2, dk, dk, device='cuda', dtype=torch.float32)
    blocks = [CacheBlock(ar.device) for _ in range(7)]
    targets = [CacheBlock(ar.device) for _ in range(8)]
    for i, block in enumerate(blocks):
        for layer in range(2):
            if (i + layer) % 3:  # missing states, including roots/interior/terminal
                a = torch.eye(dk, device='cuda').repeat(1, 2, 1, 1)
                a += torch.randn_like(a) * .01
                block.linear_affine[layer] = (a, torch.randn(1, 2, dv, dk, device='cuda')*.1)
    a, b, c, d, e, f, g = blocks
    # Shared roots, shared AND distinct parents, duplicate complete chains,
    # one chain ending inside another, unequal lengths and repeated block IDs.
    cases = [
        [[a, b, c], [a, b, d], [a, b], [a, b, c], [e, b, c], [], [f], [g, g, b, d]],
        [[a, b, e, f, g, c, d]],
        [[] for _ in range(8)],
        [[g, b], [a, c, d], [g, b, c], [a]],
    ]
    actual = [torch.empty_like(buf.initial) for _ in range(2)]

    def body():
        for layer in range(2):
            actual[layer].copy_(buf.compose(layer))

    buf.prepare([[]], targets[:1])
    graph = capture(body) if captured else None
    buf.publish(False)
    stable_addresses = (buf.level_counts.data_ptr(), buf.b.data_ptr(), buf.frontier.data_ptr())
    for step, chains in enumerate(cases):
        if step == 3:
            # New allocations/layouts must be seen on replay, not capture-time pointers.
            blocks[0].linear_affine[1] = (
                torch.eye(dk, device='cuda').repeat(1, 2, 1, 1).transpose(-1, -2),
                torch.randn(1, 2, dk, dv, device='cuda').transpose(-1, -2)*.01)
            blocks[1].linear_affine.clear()
        old_states = [(t, t.clone()) for block in blocks
                      for pair in block.linear_affine.values() for t in pair]
        n = len(chains)
        buf.prepare(chains, targets[:n])
        dense.prepare(chains, targets[:n])
        levels, _, _ = prefix_links(chains, buf.workers, buf.depth)
        assert buf.level_counts.tolist() == [len(nodes) for nodes in levels]
        # Stale scratch must not leak through newly absent nodes/shorter chains.
        buf.b.fill_(float('nan'))
        buf.frontier.fill_(float('nan'))
        if graph is None:
            body()
        else:
            graph.replay()
        for layer in range(2):
            expected = dense_compose(dense, layer)
            original = legacy_compose(chains, layer, 2, dk, dv, ar.device)
            torch.testing.assert_close(actual[layer], expected, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(actual[layer][:n], original, atol=2e-6, rtol=2e-6)
            assert torch.count_nonzero(actual[layer][n:]) == 0
        for tensor, saved in old_states:
            torch.testing.assert_close(tensor, saved, atol=0, rtol=0)
        assert stable_addresses == (buf.level_counts.data_ptr(), buf.b.data_ptr(), buf.frontier.data_ptr())
        buf.publish(False)
        dense.publish(False)


@pytest.mark.parametrize('captured', [False, True])
@torch.inference_mode()
def test_inactive_nodes_never_load_invalid_pointers_or_write_scratch(captured):
    # Check actual GPU side effects, not timing or launch-count assumptions.
    # Capacity stays 48 like the real sparse prefill profile. Inactive pointers
    # and parent indices are intentionally invalid; their outputs must stay NaN.
    torch.manual_seed(917)
    w, h, dk, dv = 48, 2, 37, 23
    previous = torch.randn(w, h, dv, dk, device='cuda')
    out = torch.empty_like(previous)
    root = torch.empty_like(previous)
    aa = [torch.eye(dk, device='cuda').repeat(h, 1, 1) +
          torch.randn(h, dk, dk, device='cuda')*.01 for _ in range(4)]
    bb = [torch.randn(h, dv, dk, device='cuda')*.1 for _ in range(4)]
    pointers = torch.zeros(w, 2, device='cuda', dtype=torch.int64)
    parents = torch.zeros(w, device='cuda', dtype=torch.int64)
    counts = torch.zeros(2, device='cuda', dtype=torch.int32)

    def body():
        compose_first(pointers, counts, root)
        compose_level(pointers, parents, counts, 1, previous, out)

    graph = capture(body) if captured else None
    for count in (4, 1, 0, 3):
        # Active rows cover all null combinations: present A/B, identity A,
        # zero B, and a completely missing affine state (pass parent through).
        pairs = [(aa[0].data_ptr(), bb[0].data_ptr()), (0, bb[1].data_ptr()),
                 (aa[2].data_ptr(), 0), (0, 0)]
        pointers.copy_(torch.tensor(pairs[:count] + [(1, 1)]*(w-count), device='cuda'))
        parents.copy_(torch.tensor([2, 2, 0, 1][:count] + [10**6]*(w-count), device='cuda'))
        counts.fill_(count)
        root.fill_(float('nan'))
        out.fill_(float('nan'))
        if graph is None:
            body()
        else:
            graph.replay()
        for i in range(count):
            expected = previous[[2, 2, 0, 1][i]]
            if pairs[i][0]:
                expected = expected @ aa[i]
            if pairs[i][1]:
                expected = expected + bb[i]
            torch.testing.assert_close(out[i], expected, atol=2e-6, rtol=2e-6)
            expected_root = bb[i] if pairs[i][1] else torch.zeros_like(bb[i])
            torch.testing.assert_close(root[i], expected_root, atol=0, rtol=0)
        assert torch.isnan(root[count:]).all()
        assert torch.isnan(out[count:]).all()


@torch.inference_mode()
def test_compose_has_no_dense_staging_or_gemm(monkeypatch):
    import minisgl.shared_cache.gdn_decode as module
    ar = SharedCacheGDN(num_heads=2, head_k_dim=8, head_v_dim=6,
                        conv_dim=7, conv_kernel=4, device=torch.device('cuda'))
    buf = GDNDecodeBuffers(ar, 1, 4, 3, torch.bfloat16)
    blocks = [CacheBlock(ar.device) for _ in range(2)]
    for block in blocks:
        block.linear_affine[0] = (torch.eye(8, device='cuda').repeat(1, 2, 1, 1),
                                  torch.ones(1, 2, 6, 8, device='cuda'))
    buf.prepare([blocks], [blocks[-1]])

    def forbidden(*args, **kwargs):
        raise AssertionError('Compose must not stage dense A/B/parents or call Torch GEMM')

    monkeypatch.setattr(module, 'gather_rows', forbidden)
    for name in ('matmul', 'index_select', 'cat', 'stack', 'empty', 'empty_like'):
        monkeypatch.setattr(torch, name, forbidden)
    for name in ('matmul', '__matmul__', 'index_select', 'clone', 'add_'):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    result = buf.compose(0)
    assert torch.all(result[0] == 2)
    assert torch.count_nonzero(result[1:]) == 0
    buf.publish(False)
