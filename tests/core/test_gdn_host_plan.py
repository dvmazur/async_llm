"""Host plan reuse, exact old topology, and live state/invalidation semantics."""
import gc
import random
import weakref

import pytest
import torch
import minisgl.shared_cache.gdn as module
from minisgl.shared_cache.gdn import SharedCacheGDN, MixedSharedCacheGDN
from minisgl.shared_cache.gdn_compose_cache import GDNComposeStateCache
from minisgl.shared_cache.shared_block import CacheBlock
from reference_gdn_host_plan import _plan_affine_prefixes as reference_plan, write_eligibility
from test_gdn_compose import _reference_compose_initial_recurrent_state
from test_gdn_successor import save, world


class Block:
    def __init__(self, layers=()):
        self.linear_affine = dict.fromkeys(layers, True)
        self.linear_affine_revision = {}


def context(chains, writes):
    gdn = SharedCacheGDN(num_heads=1, head_k_dim=2, head_v_dim=2,
                         conv_dim=1, conv_kernel=1, device=torch.device('cpu'))
    gdn.set_context(chains, writes)
    return gdn


def plan(gdn, layer):
    return gdn._get_host_plan().prefix_plan(layer)


def test_one_build_across_30_layers_and_one_read_per_unique_block(monkeypatch):
    class Reads(dict):
        calls = 0
        def get(self, *args):
            self.calls += 1
            return super().get(*args)
    common = [Block(range(30)) for _ in range(5)]
    tails = [Block(range(30)) for _ in range(64)]
    for b in common+tails: b.linear_affine = Reads(b.linear_affine)
    gdn = context([common+[b] for b in tails], tails)
    builds = []
    original = module._plan_affine_prefixes
    def counted(*args):
        builds.append(args[1])
        return original(*args)
    monkeypatch.setattr(module, '_plan_affine_prefixes', counted)
    first = plan(gdn, 0)
    for b in common+tails: b.linear_affine.calls = 0
    for layer in range(1,30):
        assert plan(gdn,layer) is first
    assert builds == [0]
    assert all(b.linear_affine.calls == 29 for b in common+tails)
    assert first == reference_plan(gdn.cache_structure, 17)


def test_layer_presence_not_just_layer_number_and_same_count_different_blocks():
    p,a,b,t = Block([0,1,2]),Block([0,2]),Block([1]),Block([])
    gdn = context([[p,a,b,t], [p,b,a,t], []], [t,t,Block()])
    for layer in [0,1,2,0,1]:
        assert plan(gdn,layer) == reference_plan(gdn.cache_structure,layer)
    assert [len(f) for f in plan(gdn,0).frontiers] == [1,1]


def test_alternating_layer_masks_build_once_each_and_memo_is_bounded(monkeypatch):
    p,a,b = Block(range(30)),Block(range(0,30,2)),Block(range(1,30,2))
    gdn = context([[p,a,b]], [b])
    builds = []
    original = module._plan_affine_prefixes
    def counted(*args):
        builds.append(args[1])
        return original(*args)
    monkeypatch.setattr(module,'_plan_affine_prefixes',counted)
    expected = [reference_plan(gdn.cache_structure,layer) for layer in (0,1)]
    for layer in range(30):
        assert plan(gdn,layer) == expected[layer%2]
    assert builds == [0,1]
    # A third population evicts the older one, rather than retaining all history.
    b.linear_affine[0] = True
    plan(gdn,0)
    del b.linear_affine[0]
    plan(gdn,0)
    assert builds == [0,1,0,0]


def test_fill_clear_and_equal_value_replacement_do_not_reuse_stale_topology():
    p,a,b = Block([0]),Block(),Block([0])
    gdn = context([[p,a,b]], [b])
    old = plan(gdn,0)
    a.linear_affine[0] = True
    assert plan(gdn,0) != old
    assert plan(gdn,0) == reference_plan(gdn.cache_structure,0)
    a.linear_affine.clear()
    assert plan(gdn,0) == old
    replacement = Block([0])
    gdn.cache_structure[0][0] = replacement
    assert plan(gdn,0).frontiers[0][0].block is replacement


def test_identity_reorders_worker_subsets_and_direct_assignments():
    p,a,b = [Block([0]) for _ in range(3)]
    gdn = context([[p,a,b], [p,b,a]], [b,a])
    plan(gdn,0)
    for mutation in (
        lambda: gdn.cache_structure[0].reverse(),
        lambda: gdn.cache_structure.reverse(),
        lambda: gdn.write_to.reverse(),
        lambda: setattr(gdn,'cache_structure',[[a,p,b],[],[p,p]]),
        lambda: setattr(gdn,'write_to',[b,a,p]),
    ):
        mutation()
        assert plan(gdn,0) == reference_plan(gdn.cache_structure,0)


def test_randomized_topologies_and_write_eligibility_match_literal_reference():
    rng = random.Random(2718)
    blocks = [Block(i for i in range(4) if rng.randrange(2)) for _ in range(12)]
    gdn = context([],[])
    for _ in range(120):
        chains = [rng.choices(blocks,k=rng.randrange(9)) for _ in range(rng.randrange(20))]
        writes = [rng.choice(blocks) for _ in chains]
        gdn.set_context(chains,writes)
        for layer in range(4):
            assert plan(gdn,layer) == reference_plan(chains,layer)
        assert list(gdn._get_host_plan().write_eligible) == write_eligibility(chains,writes)


def test_set_context_releases_old_plan_blocks_and_caller_lists_are_copied():
    block = Block([0])
    ref = weakref.ref(block)
    chains,writes = [[block]],[block]
    gdn = context(chains,writes)
    chains.clear(); writes.clear()
    assert plan(gdn,0).worker_terminals == ((1,0),)
    del block
    gdn.set_context([],[])
    gc.collect()
    assert ref() is None
    assert plan(gdn,0).worker_terminals == ()


def test_mixed_contexts_share_live_blocks_not_stale_effective_plans():
    gdn, common, tails = world(workers=1,dk=2,dv=2,heads=1)
    gdn.configure_successor_cache(0)
    prefill = gdn.context_view(gdn.cache_structure,tails,[2])
    decode = gdn.context_view(gdn.cache_structure,tails)
    mixed = MixedSharedCacheGDN(split=2,prefill=prefill,decode=decode)
    assert plan(mixed.decode,0).worker_terminals == ((1,0),)
    # A prefill fills the previously empty tail after decode's host plan exists.
    tails[0].set_linear_affine(0,tuple(t.clone() for t in common.linear_affine[0]))
    for view in (mixed.prefill,mixed.decode):
        actual = view.compose_initial_recurrent_state(0,torch.float32)
        expected = _reference_compose_initial_recurrent_state(view,0,torch.float32)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert plan(view,0).worker_terminals == ((2,0),)
    assert prefill._host_plan is not decode._host_plan


def test_reused_plan_does_not_cache_revision_or_affine_values():
    gdn,common,tails = world(workers=2,dk=2,dv=2,heads=1)
    save(gdn)
    before = plan(gdn,0)
    # Same presence and same topology, but a new upstream state must miss cache.
    pair = tuple(t.clone() for t in common.linear_affine[0])
    pair[1].add_(1)
    common.set_linear_affine(0,pair)
    actual,_ = gdn.begin_decode_state(0)
    expected = _reference_compose_initial_recurrent_state(gdn,0,torch.float32).transpose(-1,-2)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert plan(gdn,0) is before
    assert gdn.successor_state_cache.stats['hits'] == 0


def test_partial_miss_view_reused_but_respects_new_miss_subset():
    gdn,common,tails = world(workers=3,dk=2,dv=2,heads=1)
    state = save(gdn)
    tails[0].set_linear_affine(0,tails[0].linear_affine[0])
    gdn.begin_decode_state(0)
    view = gdn._compose_miss_view[2]
    first = plan(view,0)
    gdn.begin_decode_state(0)
    assert gdn._compose_miss_view[2] is view
    assert plan(view,0) is first
    tails[2].set_linear_affine(0,tails[2].linear_affine[0])
    actual,_ = gdn.begin_decode_state(0)
    assert gdn._compose_miss_view[1] == (0,2)
    assert gdn._compose_miss_view[2] is not view
    torch.testing.assert_close(actual[1],state[1])
    expected = _reference_compose_initial_recurrent_state(gdn,0,torch.float32).transpose(-1,-2)
    torch.testing.assert_close(actual[[0,2]],expected[[0,2]],rtol=0,atol=0)


def test_topology_changes_invalidate_precomputed_write_eligibility():
    gdn,_,tails = world(workers=2)
    gdn.begin_decode_state(0)
    assert gdn._get_host_plan().write_eligible == (True,True)
    gdn.cache_structure[1].insert(1,tails[0])
    _,tickets = gdn.begin_decode_state(0)
    assert tickets[0] is not None and tickets[1] is None
    gdn.write_to[1] = tails[0]
    _,tickets = gdn.begin_decode_state(0)
    assert tickets == [None,None]


@pytest.mark.parametrize('configure', ['configure_compose_cache','configure_successor_cache'])
def test_reconfiguring_cache_drops_subview_with_old_cache_instance(configure):
    gdn,_,tails = world(workers=2)
    save(gdn)
    tails[0].set_linear_affine(0,tails[0].linear_affine[0])
    gdn.begin_decode_state(0)
    assert gdn._compose_miss_view is not None
    getattr(gdn,configure)(0)
    assert gdn._compose_miss_view is None


def test_prefix_cache_values_stats_and_frontier_work_match_uncached_host_plan(monkeypatch):
    import minisgl.kernel as kernels
    calls = []
    def cpu_pointer(parents,As,Bs):
        calls.append(len(parents))
        return torch.cat([p@a+b for p,a,b in zip(parents,As,Bs)],dim=0)
    monkeypatch.setattr(kernels,'apply_gdn_affine_pointer_nodes',cpu_pointer)
    gdn,common,tails = world(workers=3,dk=2,dv=2,heads=1)
    gdn.configure_successor_cache(0)
    branches = [CacheBlock(gdn.device) for _ in range(2)]
    for i,b in enumerate(branches):
        A,B = (t.clone() for t in common.linear_affine[0])
        A[0,0,0,1] = .1*(i+1)
        B.add_(.2*i)
        b.set_linear_affine(0,(A,B))
    gdn.set_context([[common,branches[i%2],tail] for i,tail in enumerate(tails)],tails)
    # CPU test intentionally supplies the GPU-independent LRU plus a numerical
    # pointer-kernel stand-in. No claim to test CUDA code with this test.
    gdn.compose_state_cache = GDNComposeStateCache(32)
    oracle_cache = GDNComposeStateCache(32)
    for step in range(14):
        if step == 4:
            branches[0].set_linear_affine(0,tuple(t.clone()+.05 for t in branches[0].linear_affine[0]))
        if step == 7:
            tails[1].set_linear_affine(0,tuple(t.clone() for t in common.linear_affine[0]))
        if step == 9:
            branches[1].clear()
        if step == 11:
            gdn.cache_structure.reverse(); gdn.write_to.reverse()
        calls.clear()
        expected = module._evaluate_affine_prefix_plan_cached(
            reference_plan(gdn.cache_structure,0),lin_idx=0,num_heads=1,d_k=2,d_v=2,
            device=gdn.device,write_to=gdn.write_to,cache=oracle_cache)
        oracle_calls = list(calls)
        calls.clear()
        actual = gdn.compose_initial_recurrent_state(0,torch.float32,state_v_first=True)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert calls == oracle_calls
        assert gdn.compose_state_cache.stats == oracle_cache.stats
