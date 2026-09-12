"""The extra FLA pass updates A/B only; compare to the frozen token reference."""
import pytest
import torch

from minisgl.shared_cache.gdn import SharedCacheGDN
from minisgl.shared_cache.gdn_prefill import GDNPrefillBuffers
from minisgl.shared_cache.shared_block import CacheBlock
from _gdn_reference import SharedCacheGDN as ReferenceGDN
from test_gdn_compose import capture as capture_graph

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA FLA capture')


def buffers(dk, dv, workers=4, rows=256):
    ar = SharedCacheGDN(num_heads=2, head_k_dim=dk, head_v_dim=dv,
                        conv_dim=7, conv_kernel=4, device=torch.device('cuda'))
    return GDNPrefillBuffers(ar, 2, workers, 3, torch.bfloat16, rows)


@pytest.mark.parametrize('dk,dv', [(8, 6), (37, 23), (128, 128)])
@torch.inference_mode()
def test_packed_initial_and_owned_final_preserve_layout_missing_states_and_layers(dk, dv, monkeypatch):
    import minisgl.shared_cache.gdn_prefill as module
    torch.manual_seed(118)
    buf = buffers(dk, dv, workers=6)
    targets = [CacheBlock(buf.device) for _ in range(4)]
    a = torch.randn(1, 2, dk, dk, device='cuda').transpose(-1, -2)
    b = torch.randn(1, 2, dk, dv, device='cuda').transpose(-1, -2)
    targets[1].linear_affine[1] = (a, b)
    targets[2].linear_affine[1] = (a, None)
    targets[3].linear_affine[1] = (None, b)
    targets[1].linear_affine[0] = (a+10, b+10)
    buf.prepare([[] for _ in targets], targets, [1]*len(targets))
    buf.read_ptrs[:, len(targets):, :2] = 1  # pack must not dereference inactive pointers
    final = torch.randn_like(buf.affine_initial)
    expected = final.clone()

    def observe(key, value, g, beta, initial, *metadata):
        for w, target in enumerate(targets):
            aa, bb = target.linear_affine.get(1, (None, None))
            if aa is None:
                aa = torch.eye(dk, device='cuda').repeat(1, 2, 1, 1)
            if bb is None:
                bb = torch.zeros(1, 2, dv, dk, device='cuda')
            torch.testing.assert_close(initial[w, ..., :dk], aa[0].transpose(-1,-2), atol=0, rtol=0)
            torch.testing.assert_close(initial[w, ..., dk:], bb[0].transpose(-1,-2), atol=0, rtol=0)
        assert torch.count_nonzero(initial[len(targets):]) == 0
        return final

    monkeypatch.setattr(module, 'chunk_gdn_final_state', observe)
    key = torch.zeros(256, 2, dk, device='cuda')
    value = torch.zeros(256, 2, dv, device='cuda')
    gate = torch.zeros(256, 2, device='cuda')
    buf.capture_prefill(1, key, value, gate.exp(), gate, g=gate)
    # Populate the other layer/window outputs for this isolated IO test.
    for outputs in buf._current[1:4]:
        for out in outputs:
            out[0].zero_()
    buf.publish()
    final.fill_(float('nan'))
    buf.affine_initial.zero_()
    for w, target in enumerate(targets):
        aa, bb = target.linear_affine[1]
        torch.testing.assert_close(aa[0], expected[w, ..., :dk].transpose(-1,-2), atol=0, rtol=0)
        torch.testing.assert_close(bb[0], expected[w, ..., dk:].transpose(-1,-2), atol=0, rtol=0)
        assert aa.untyped_storage().data_ptr() != final.untyped_storage().data_ptr()


@pytest.mark.parametrize('dk,dv', [(16, 12), (128, 128)])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('captured', [False, True])
@torch.inference_mode()
def test_one_auxiliary_fla_pass_updates_existing_blocks_and_replays(dk, dv, dtype, captured, monkeypatch):
    pytest.importorskip('fla.ops.gated_delta_rule')
    import minisgl.shared_cache.gdn_prefill as module
    torch.manual_seed(684)
    buf = buffers(dk, dv)
    targets = [CacheBlock(buf.device) for _ in range(3)]
    for layer in range(2):
        a = torch.eye(dk, device='cuda').repeat(1, 2, 1, 1) + torch.randn(1, 2, dk, dk, device='cuda')*.02
        targets[1].linear_affine[layer] = (a, torch.randn(1, 2, dv, dk, device='cuda')*.1)
    k = torch.randn(2, dk, 256, device='cuda', dtype=dtype).permute(2,0,1)
    v = torch.randn(256, 2, dv, device='cuda', dtype=dtype)
    g = -torch.rand(256, 2, device='cuda')*.04
    beta = torch.rand(256, 2, device='cuda', dtype=dtype)
    calls = []
    original = module.chunk_gdn_final_state

    def observed(key, values, gates, bet, initial, *metadata):
        assert values.shape == (1, 256, 2, dk+dv)
        assert initial.shape == (4, 2, dk, dk+dv)
        assert key.dtype == values.dtype == initial.dtype == torch.float32
        calls.append(1)
        return original(key, values, gates, bet, initial, *metadata)

    def forbidden(*args, **kwargs):
        raise AssertionError('Native prefill capture must use FLA, not the token scan')

    monkeypatch.setattr(module, 'chunk_gdn_final_state', observed)
    monkeypatch.setattr(module, 'capture_affine_scan', forbidden)

    def body():
        for layer in range(2):
            buf.capture_prefill(layer, k, v, g.exp(), beta, g=g)

    dummy = [CacheBlock(buf.device) for _ in targets]
    buf.prepare([[t] for t in dummy], dummy, [1,1,1])
    graph = capture_graph(body) if captured else None
    buf.publish(False)
    addresses = (buf.affine_initial.data_ptr(), buf.affine_values.data_ptr())
    for step, lengths in enumerate(([63,65], [1], [7,5,9], [138,91])):
        n = len(lengths)
        if step == 2:
            targets[0].linear_affine.clear()
        refs = [CacheBlock(buf.device) for _ in lengths]
        for ref, target in zip(refs, targets):
            ref.linear_affine = {l: (a.clone(), b.clone()) for l,(a,b) in target.linear_affine.items()}
        saved = [(t, t.clone()) for target in targets[:n]
                 for pair in target.linear_affine.values() for t in pair]
        reference = ReferenceGDN(num_heads=2, head_k_dim=dk, head_v_dim=dv,
                                 conv_dim=7, conv_kernel=4, device=buf.device)
        reference.set_context([[t] for t in refs], refs)
        k.normal_()
        k[sum(lengths):] = float('nan')
        buf.prepare([[] for _ in lengths], targets[:n], lengths)
        buf.read_ptrs[:, n:, :2] = 1
        if graph is not None:
            graph.replay()
        else:
            body()
        buf.publish()
        for layer in range(2):
            start = 0
            for w, length in enumerate(lengths):
                section = slice(start, start+length)
                reference.capture_token_affines(layer, k[None,section], v[None,section],
                    g[None,section].exp(), beta[None,section], workers=[w])
                for actual, expected in zip(targets[w].linear_affine[layer], refs[w].linear_affine[layer]):
                    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)
                start += length
        for tensor, old in saved:
            torch.testing.assert_close(tensor, old, atol=0, rtol=0)
        assert addresses == (buf.affine_initial.data_ptr(), buf.affine_values.data_ptr())
    assert len(calls) == (4 if captured else 8)  # one auxiliary pass/layer, not one/worker
