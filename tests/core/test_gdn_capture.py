"""Capture ports: independent old arithmetic, active pointers and real replay."""
import pytest
import torch

from minisgl.shared_cache.gdn import SharedCacheGDN
from minisgl.shared_cache.gdn_decode import GDNDecodeBuffers
from minisgl.shared_cache.shared_block import CacheBlock
from _gdn_reference import SharedCacheGDN as ReferenceGDN
from test_gdn_compose import capture as capture_graph

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA affine capture')


@pytest.mark.parametrize('dk,dv', [(8, 6), (37, 23), (128, 128)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('eps', [1e-6, 1e-3])
@torch.inference_mode()
def test_decode_pointer_capture_replay_against_frozen_reference(dk, dv, dtype, eps):
    torch.manual_seed(7101)
    ar = SharedCacheGDN(num_heads=2, head_k_dim=dk, head_v_dim=dv,
                        conv_dim=7, conv_kernel=4, device=torch.device('cuda'))
    buf = GDNDecodeBuffers(ar, 2, 8, 1, dtype)
    targets = [CacheBlock(ar.device) for _ in range(6)]
    for i, block in enumerate(targets):
        for layer in range(2):
            if (i+layer) % 3:
                a = torch.eye(dk, device='cuda').repeat(1, 2, 1, 1)
                a += torch.randn_like(a)*.01
                block.linear_affine[layer] = (a, torch.randn(1, 2, dv, dk, device='cuda')*.1)
    # Small k makes a non-default epsilon materially affect the output.
    key = torch.randn(8, 1, 2, dk, device='cuda', dtype=dtype)*.01
    value = torch.randn(8, 1, 2, dv, device='cuda', dtype=dtype)
    alpha = torch.rand(8, 1, 2, device='cuda')*.1+.9
    beta = torch.rand(8, 1, 2, device='cuda', dtype=dtype)

    def body():
        for layer in range(2):
            buf.capture(layer, key, value, alpha, beta, eps)

    buf.prepare([[t] for t in targets], targets)
    graph = capture_graph(body)
    buf.publish(False)
    assert not hasattr(buf, 'a'), 'decode capture no longer needs dense A staging'
    for step, n in enumerate((6, 2, 5, 1)):
        if step == 2:
            targets[0].linear_affine.clear()
        refs = [CacheBlock(ar.device) for _ in range(n)]
        for ref, target in zip(refs, targets):
            ref.linear_affine = {l: (a.clone(), b.clone()) for l, (a, b) in target.linear_affine.items()}
        old = [(tensor, tensor.clone()) for t in targets[:n]
               for pair in t.linear_affine.values() for tensor in pair]
        reference = ReferenceGDN(num_heads=2, head_k_dim=dk, head_v_dim=dv,
                                 conv_dim=7, conv_kernel=4, device=ar.device)
        reference.set_context([[t] for t in refs], refs)
        key.normal_(0, .01)
        key[n:] = float('nan')
        buf.prepare([[t] for t in targets[:n]], targets[:n])
        # A padded worker must not even dereference its read pointers.
        buf.read_ptrs[:, n:, :2] = 1
        graph.replay()
        buf.publish()
        for layer in range(2):
            reference.capture_token_affines(layer, key[:n], value[:n], alpha[:n], beta[:n],
                                            l2norm_eps=eps)
            for target, ref in zip(targets, refs):
                for actual, expected in zip(target.linear_affine[layer], ref.linear_affine[layer]):
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        for tensor, saved in old:
            torch.testing.assert_close(tensor, saved, atol=0, rtol=0)


@torch.inference_mode()
def test_decode_capture_cannot_fall_back_to_dense_torch_update(monkeypatch):
    import minisgl.shared_cache.gdn_decode as module
    ar = SharedCacheGDN(num_heads=2, head_k_dim=8, head_v_dim=6,
                        conv_dim=7, conv_kernel=4, device=torch.device('cuda'))
    buf = GDNDecodeBuffers(ar, 1, 4, 1, torch.bfloat16)
    target = CacheBlock(ar.device)
    buf.prepare([[target]], [target])
    key = torch.randn(4, 1, 2, 8, device='cuda')
    value = torch.randn(4, 1, 2, 6, device='cuda')
    gate = torch.ones(4, 1, 2, device='cuda')*.5
    seen = []
    pointer = module.capture_affine_scan

    def observe(*args, **kwargs):
        assert len(args) == 6  # no sequence-offset table in decode
        seen.append(kwargs['l2norm_eps'])
        return pointer(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('No dense A/B gather, GEMV or scatter in decode capture')

    monkeypatch.setattr(module, 'capture_affine_scan', observe)
    monkeypatch.setattr(module, 'gather_rows', forbidden)
    monkeypatch.setattr(module, 'scatter_rows', forbidden)
    monkeypatch.setattr(torch, 'matmul', forbidden)
    buf.capture(0, key, value, gate, gate, 1e-4)
    buf.publish()
    assert seen == [1e-4]
    assert all(torch.isfinite(t).all() for t in target.linear_affine[0])
