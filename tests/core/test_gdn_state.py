"""CPU metadata caching and GPU pointer lifetimes; no learned checkpoint needed."""
import gc
import weakref

import numpy as np
import pytest
import torch

from minisgl.shared_cache.gdn import SharedCacheGDN
from minisgl.shared_cache.gdn_decode import GDNDecodeBuffers
from minisgl.shared_cache.gdn_state import StateDict, prepare_state
from minisgl.shared_cache.session import SharedCacheSession
from minisgl.shared_cache.shared_block import CacheBlock


def signature(device='cpu', layers=3):
    return (torch.device(device), layers, 2, 8, 6, (7, 4), torch.bfloat16)


def populated(device='cpu', layers=3):
    block = CacheBlock(torch.device(device))
    for layer in range(layers):
        block.linear_affine[layer] = (torch.eye(8, device=device).expand(1, 2, 8, 8).clone(),
                                      torch.full((1, 2, 6, 8), float(layer), device=device))
        block.linear_conv_state[layer] = torch.full((7, 4), float(layer), device=device,
                                                     dtype=torch.bfloat16)
    return block


def check_addresses(block, state):
    for layer in range(len(state.pointers)):
        tensors = (*block.linear_affine.get(layer, (None, None)),
                   block.linear_conv_state.get(layer))
        assert state.pointers[layer].tolist() == [0 if t is None else t.data_ptr() for t in tensors]


def test_unchanged_state_is_cached_and_value_edits_remain_visible():
    block, sig = populated(), signature()
    state = prepare_state(block, sig)
    assert prepare_state(block, sig) is state
    block.linear_affine[0][0].add_(3)
    assert prepare_state(block, sig) is state
    assert state.owners[0][0, 0, 0, 0] == 4
    check_addresses(block, state)


@pytest.mark.parametrize('field', ['linear_affine', 'linear_conv_state'])
@pytest.mark.parametrize('operation', ['set', 'delete', 'pop', 'popitem', 'clear', 'update',
                                       'setdefault', 'ior', 'partial_update'])
def test_all_dict_mutations_invalidate(field, operation):
    block, sig = populated(), signature()
    before = prepare_state(block, sig)
    mapping = getattr(block, field)
    value = mapping[1]
    if operation == 'set': mapping[0] = value
    elif operation == 'delete': del mapping[0]
    elif operation == 'pop': mapping.pop(0)
    elif operation == 'popitem': mapping.popitem()
    elif operation == 'clear': mapping.clear()
    elif operation == 'update': mapping.update({0: value})
    elif operation == 'setdefault': mapping.setdefault(3, value)
    elif operation == 'ior': mapping |= {0: value}
    else:
        def broken():
            yield 0, value
            raise RuntimeError('partial update')
        with pytest.raises(RuntimeError, match='partial'):
            mapping.update(broken())
    assert block._gdn_state_cache.prepared is None
    after = prepare_state(block, sig)
    assert after is not before
    check_addresses(block, after)


@pytest.mark.parametrize('field', ['linear_affine', 'linear_conv_state'])
def test_externally_assigned_dict_preserves_alias_and_is_not_cached(field):
    block, sig = populated(), signature()
    prepare_state(block, sig)
    external = dict(getattr(block, field))
    setattr(block, field, external)
    assert getattr(block, field) is external
    assert block._gdn_state_cache.prepared is None
    first = prepare_state(block, sig)
    external.pop(0)
    second = prepare_state(block, sig)
    assert first is not second and block._gdn_state_cache.prepared is None
    check_addresses(block, second)


def test_converted_noncontiguous_debug_states_are_not_cached():
    block, sig = populated(), signature()
    original = block.linear_affine[0][0].transpose(-1, -2)
    block.linear_affine[0] = (original, block.linear_affine[0][1])
    first = prepare_state(block, sig)
    assert block._gdn_state_cache.prepared is None
    original.add_(2)
    second = prepare_state(block, sig)
    torch.testing.assert_close(second.owners[0], original)
    assert first.owners[0][0, 0, 0, 0] == 1
    assert second.owners[0][0, 0, 0, 0] == 3


def test_shape_errors_and_explicit_layout_invalidation():
    block, sig = populated(), signature()
    prepare_state(block, sig)
    block.linear_conv_state[0].resize_(8, 4)
    block.invalidate_gdn_state()
    with pytest.raises(ValueError, match='Unexpected GDN state shape'):
        prepare_state(block, sig)


def test_signature_change_does_not_reuse_incompatible_description():
    block = populated()
    state = prepare_state(block, signature())
    smaller = prepare_state(block, signature(layers=2))
    assert smaller is not state and smaller.pointers.shape == (2, 3)
    sig = (*signature()[:-1], torch.float32)
    converted = prepare_state(block, sig)
    assert converted.owners[2].dtype == torch.float32
    assert prepare_state(block, signature(layers=2)) is smaller


def test_clear_and_block_destruction_do_not_retain_cached_storage():
    block = populated()
    ref = weakref.ref(block.linear_affine[0][0])
    prepare_state(block, signature())
    block.clear()
    assert ref() is None and block._gdn_state_cache.prepared is None
    block = populated()
    ref, block_ref = weakref.ref(block.linear_affine[0][0]), weakref.ref(block)
    prepare_state(block, signature())
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        del block
        assert ref() is None and block_ref() is None, 'metadata must not form an ownership cycle'
    finally:
        if was_enabled: gc.enable()


@pytest.mark.parametrize('self_append', [False, True])
def test_merge_replaces_state_and_preserves_owned_tracking(self_append):
    left, right = populated(), populated()
    previous = prepare_state(left, signature())
    old_b = left.linear_affine[1][1].clone()
    if self_append: right = left
    session = SharedCacheSession.__new__(SharedCacheSession)
    session._finish_block_merge(destination=left, left=left, right=right,
                                left_span=0, right_span=0, keep_left_state=True)
    assert left._gdn_state_cache.prepared is None
    assert isinstance(left.linear_affine, StateDict)
    torch.testing.assert_close(left.linear_affine[1][1], old_b*2)
    current = prepare_state(left, signature())
    assert current is not previous and prepare_state(left, signature()) is current
    check_addresses(left, current)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA pointer tables')
@torch.inference_mode()
def test_published_slabs_bypass_per_layer_tensor_dispatch(monkeypatch):
    ar = SharedCacheGDN(num_heads=2, head_k_dim=8, head_v_dim=6,
                       conv_dim=7, conv_kernel=4, device=torch.device('cuda'))
    buf = GDNDecodeBuffers(ar, 3, 4, 3, torch.bfloat16)
    blocks = [CacheBlock(ar.device) for _ in range(3)]
    buf.prepare([[b] for b in blocks], blocks)
    for kind in buf._current[1:4]:
        for tensor in kind: tensor.fill_(2)
    buf.publish()
    originals = [b._gdn_state_cache.prepared for b in blocks]
    assert all(len(s.owners) == 3 for s in originals)
    for b, s in zip(blocks, originals): check_addresses(b, s)
    chains = [[blocks[0], blocks[1]], [blocks[1], blocks[0]], [blocks[2]]]

    def forbidden(*args, **kwargs):
        raise AssertionError('cached state must not inspect per-layer tensors')
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'to', forbidden)
        patch.setattr(torch.Tensor, 'contiguous', forbidden)
        patch.setattr(torch.Tensor, 'record_stream', forbidden)
        patch.setattr(torch.Tensor, '__getitem__', forbidden)
        patch.setattr(StateDict, 'get', forbidden)
        buf.prepare(chains, blocks)
    reads = buf.read_ptrs.cpu().numpy()
    for w,s in enumerate(originals): np.testing.assert_array_equal(reads[:,w,:2],s.pointers[:,:2])
    assert not reads[:,3].any()
    saved = originals[0].owners[0].clone()
    # Replacing/clearing block state after prepare cannot invalidate in-flight reads.
    blocks[0].clear()
    torch.testing.assert_close(buf.conv(0)[:3], torch.full_like(buf.conv_input[:3],2),atol=0,rtol=0)
    buf.publish(False)
    torch.testing.assert_close(originals[0].owners[0], saved, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA all-layer lifetimes')
@torch.inference_mode()
def test_cached_owners_survive_cross_stream_replacement():
    ar = SharedCacheGDN(num_heads=2, head_k_dim=8, head_v_dim=6,
                       conv_dim=7, conv_kernel=4, device=torch.device('cuda'))
    buf = GDNDecodeBuffers(ar, 3, 1, 1, torch.bfloat16)
    source, target = CacheBlock(ar.device), CacheBlock(ar.device)
    buf.prepare([[source]], [source])
    for kind in buf._current[1:4]: kind[0].fill_(7)
    del kind  # the loop variable must not keep the old conv allocation alive
    buf.publish()
    torch.cuda.synchronize()
    buf.retire_completed()
    state = source._gdn_state_cache.prepared
    refs = [weakref.ref(t) for t in state.owners]
    del state
    consumer = torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(consumer):
        buf.prepare([[source]], [target])
        source.clear()
        assert all(r() is not None for r in refs)
        conv = buf.conv(0).clone()
        buf.publish(False)
    consumer.synchronize()
    torch.testing.assert_close(conv, torch.full_like(conv, 7), atol=0, rtol=0)
    buf.retire_completed()
    assert all(r() is None for r in refs)
