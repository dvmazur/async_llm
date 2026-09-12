"""Prepared GDN against the pre-graph affine algorithm, including actual replay."""

import gc

import pytest
import torch

from minisgl.shared_cache.gdn import SharedCacheGDN
from minisgl.shared_cache.gdn_affine import compose_gdn_affines, init_gdn_affine
from minisgl.shared_cache.gdn_decode import GDNDecodeBuffers, prefix_links
from minisgl.shared_cache.shared_block import CacheBlock


def legacy_compose(chains, layer, h, dk, dv, device):
    # Original prefix_memo/full-affine algorithm from c0670e1. Deliberately
    # independent of the prepared links and its state-only computation.
    memo, result = {}, []
    for chain in chains:
        key, acc = (), None
        for block in chain:
            key += (id(block),)
            pair = block.linear_affine.get(layer)
            if pair is None:
                continue
            if key in memo:
                acc = memo[key]
                continue
            a, b = (t.float().to(device) for t in pair)
            if acc is None:
                acc = (a, b)
            else:
                acc = compose_gdn_affines(A_first=acc[0], B_first=acc[1],
                                          A_second=a, B_second=b)
            memo[key] = acc
        if acc is None:
            acc = init_gdn_affine(batch_size=1, num_heads=h, d_k=dk, d_v=dv,
                                  dtype=torch.float32, device=device)
        result.append(acc[1])
    return torch.cat(result).transpose(-1, -2).contiguous()


def test_links_share_prefixes_not_equal_suffixes():
    a, b, c, d = [object() for _ in range(4)]
    levels, parents, terminal = prefix_links([[a, b, c], [a, b, d], [d, b], []], 6, 5)
    assert levels == [[a, d], [b, b], [c, d], [], []]
    assert parents[1][:2] == [0, 1]
    assert parents[2][:2] == [0, 0]
    assert terminal[2][:4] == [0, 1, -1, -1]
    assert terminal[1][:4] == [-1, -1, 1, -1]
    assert all(row[4:] == [-1, -1] for row in terminal)
    with pytest.raises(ValueError, match="capacity"):
        prefix_links([[a, b]], 1, 1)
    with pytest.raises(ValueError, match="capacity"):
        prefix_links([[a], [b]], 1, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA state ownership')
@pytest.mark.parametrize('prefill', [False, True])
@torch.inference_mode()
def test_frozen_worker_does_not_retain_other_workers_output_storage(prefill):
    from minisgl.shared_cache.gdn_prefill import GDNPrefillBuffers
    ar = SharedCacheGDN(num_heads=2, head_k_dim=8, head_v_dim=6,
                         conv_dim=7, conv_kernel=4, device=torch.device('cuda'))
    targets = [CacheBlock(ar.device) for _ in range(4)]
    cls = GDNPrefillBuffers if prefill else GDNDecodeBuffers
    buf = cls(ar, 2, 4, 1, torch.bfloat16, **({'rows': 4} if prefill else {}))

    def publish(group, value):
        args = ([[t] for t in group], group)
        buf.prepare(*args, **({'lengths': [1]*len(group)} if prefill else {}))
        for outputs in buf._current[1:4]:
            for tensor in outputs:
                tensor.fill_(value)
        buf.publish()
        torch.cuda.synchronize()
        buf.retire_completed()
        assert not buf._pending

    publish(targets, 1)
    expected = (2*2*8*8*4, 2*2*6*8*4, 2*7*4*2)
    for kind, size in enumerate(expected):
        storages = []
        for target in targets:
            rows = [(*target.linear_affine[layer], target.linear_conv_state[layer])[kind]
                    for layer in range(2)]
            # Layers share a worker's storage, but workers never share storage.
            assert rows[0].untyped_storage().data_ptr() == rows[1].untyped_storage().data_ptr()
            assert rows[0].untyped_storage().nbytes() == size
            storages.append(rows[0].untyped_storage().data_ptr())
        assert len(set(storages)) == len(targets)
    saved = targets[1].linear_affine[0][0]
    publish(targets[1:], 2)  # worker0 freezes while the others keep decoding
    assert torch.all(saved == 1)
    assert torch.all(targets[0].linear_affine[1][0] == 1)
    assert torch.all(targets[1].linear_affine[0][0] == 2)


def make_case(dtype, dk=8, dv=6):
    torch.manual_seed(718)
    ar = SharedCacheGDN(num_heads=2, head_k_dim=dk, head_v_dim=dv,
                         conv_dim=7, conv_kernel=4, device=torch.device('cuda'))
    blocks = [CacheBlock(ar.device) for _ in range(8)]
    for i, block in enumerate(blocks):
        for layer in range(2):
            # Different presence per layer, plus completely empty write blocks.
            if i < 5 and (i + layer) % 3:
                a = torch.eye(dk, device='cuda').expand(1, 2, dk, dk).clone()
                a += 0.01 * torch.randn_like(a)
                b = torch.randn(1, 2, dv, dk, device='cuda') * 0.1
                block.linear_affine[layer] = (a, b)
            if i < 4 and (i + layer) % 2:
                block.linear_conv_state[layer] = torch.randn(7, 4, device='cuda', dtype=dtype)
    a, b, c, d, e, f, g, z = blocks
    return ar, [[a, b, c], [a, b, d], [a, b, e, f], [g], []], [c, d, f, g, z], blocks


def clone_blocks(chains, targets, device):
    copies = {}
    for block in [b for chain in chains for b in chain] + targets:
        if id(block) not in copies:
            copy = CacheBlock(device)
            copy.linear_affine = {l: (a.clone(), b.clone()) for l, (a, b) in block.linear_affine.items()}
            copy.linear_conv_state = {l: c.clone() for l, c in block.linear_conv_state.items()}
            copies[id(block)] = copy
    return [[copies[id(b)] for b in c] for c in chains], [copies[id(t)] for t in targets]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pointer IO")
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('captured', [False, True])
@torch.inference_mode()
def test_compose_capture_conv_dynamic_addresses(dtype, captured):
    ar, chains, targets, blocks = make_case(dtype)
    buf = GDNDecodeBuffers(ar, layers=2, workers=6, depth=5, dtype=dtype)
    key = torch.randn(6, 1, 2, 8, device='cuda', dtype=dtype)
    value = torch.randn(6, 1, 2, 6, device='cuda', dtype=dtype)
    alpha = torch.rand(6, 1, 2, device='cuda')
    beta = torch.rand(6, 1, 2, device='cuda', dtype=dtype)
    conv = torch.randn(6, 7, 5, device='cuda', dtype=dtype)[..., 1:]
    initials = [torch.empty_like(buf.initial) for _ in range(2)]
    convs = [torch.empty_like(buf.conv_input) for _ in range(2)]

    def body():
        for layer in range(2):
            initials[layer].copy_(buf.compose(layer))
            convs[layer].copy_(buf.conv(layer))
            buf.capture(layer, key, value, alpha, beta, 1e-6)
            buf.store_conv(layer, conv)

    graph = None
    if captured:
        # Warm/capture with unrelated scratch blocks, never production states.
        dummy = [CacheBlock(ar.device) for _ in targets]
        buf.prepare([[d] for d in dummy], dummy)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            body()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        buf.publish(False)

    for step in range(4):
        if step == 1:
            # Replacing allocations must not leave cached pointers; also accept
            # non-contiguous debug tensors after preparing a contiguous copy.
            a = torch.eye(8, device='cuda').expand(1, 2, 8, 8).clone()
            blocks[0].linear_affine[1] = (a.transpose(-1, -2),
                                          torch.ones(1, 2, 8, 6, device='cuda').transpose(-1, -2))
        if step == 2:
            chains = [[blocks[1], targets[0]], [blocks[0], targets[1]],
                      [targets[2]], [], [blocks[0], blocks[1], targets[4]]]
        if step == 3:
            chains, targets = chains[:2], targets[:2]  # capacity stays 6, dummy writes disabled
            for block in blocks:
                block.linear_affine.pop(0, None)
                block.linear_conv_state.pop(0, None)
        ref_chains, ref_targets = clone_blocks(chains, targets, ar.device)
        reference = SharedCacheGDN(num_heads=2, head_k_dim=8, head_v_dim=6,
                                   conv_dim=7, conv_kernel=4, device=ar.device)
        reference.set_context(ref_chains, ref_targets)
        n = len(targets)
        expected_initial = [legacy_compose(chains, l, 2, 8, 6, ar.device) for l in range(2)]
        expected_conv = [reference.prior_conv_states(l) for l in range(2)]
        old_pairs = [(a, a.clone(), b, b.clone()) for t in targets for a, b in t.linear_affine.values()]
        buf.prepare(chains, targets)
        assert buf._current is not None
        if captured:
            graph.replay()
        else:
            body()
        buf.publish()
        for layer in range(2):
            torch.testing.assert_close(initials[layer][:n], expected_initial[layer], atol=2e-6, rtol=2e-6)
            assert torch.count_nonzero(initials[layer][n:]) == 0
            expected = expected_conv[layer]
            if expected is None:
                expected = torch.zeros_like(convs[layer][:n])
            torch.testing.assert_close(convs[layer][:n], expected, atol=0, rtol=0)
            reference.capture_token_affines(layer, key[:n], value[:n], alpha[:n], beta[:n])
            reference.set_conv_states(layer, conv[:n])
            for target, ref in zip(targets, ref_targets):
                for actual, want in zip(target.linear_affine[layer], ref.linear_affine[layer]):
                    # Padded W=6 vs the independent unpadded W=5/2 reference:
                    # cuBLAS GEMV can differ by FP32 rounding (observed 3.7e-9).
                    # This is not a change to learned-model accuracy gates.
                    torch.testing.assert_close(actual, want, atol=2e-7, rtol=2e-6)
                torch.testing.assert_close(target.linear_conv_state[layer], ref.linear_conv_state[layer], atol=0, rtol=0)
        for a, saved_a, b, saved_b in old_pairs:
            torch.testing.assert_close(a, saved_a, atol=0, rtol=0)
            torch.testing.assert_close(b, saved_b, atol=0, rtol=0)
        gc.collect()
    assert buf.prepare_count == (5 if captured else 4)
    assert buf.compose_count == (4 if captured else 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pointer IO")
@torch.inference_mode()
def test_failed_step_does_not_publish_and_duplicate_writers_rejected():
    ar, chains, targets, _ = make_case(torch.bfloat16)
    buf = GDNDecodeBuffers(ar, 2, 5, 4, torch.bfloat16)
    original = dict(targets[0].linear_affine)
    buf.prepare(chains, targets)
    buf.publish(False)
    assert all(targets[0].linear_affine[l][0] is pair[0] for l, pair in original.items())
    with pytest.raises(ValueError, match="distinct write"):
        buf.prepare(chains, [targets[0]] * 5)
