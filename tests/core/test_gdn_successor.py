"""Last-decode state reuse: topology, lifetime, and recomposing old oracle."""
import copy
import gc

import pytest
import torch

import minisgl.models.qwen3_5_delta as delta
from minisgl.shared_cache.gdn import SharedCacheGDN
from minisgl.shared_cache.gdn_successor_cache import GDNSuccessorCache, assemble_state_rows
from minisgl.shared_cache.shared_block import CacheBlock
from test_gdn_compose import _reference_compose_initial_recurrent_state


def world(workers=3, device="cpu", dk=7, dv=5, heads=2):
    device = torch.device(device)
    common = CacheBlock(device)
    common.set_linear_affine(0, (
        torch.eye(dk, device=device).expand(1, heads, dk, dk).clone(),
        torch.randn(1, heads, dv, dk, device=device) * .1,
    ))
    tails = [CacheBlock(device) for _ in range(workers)]
    chains = [[common, tail] for tail in tails]
    gdn = SharedCacheGDN(num_heads=heads, head_k_dim=dk, head_v_dim=dv,
        conv_dim=1, conv_kernel=1, device=device, gdn_storage_bytes=8 * 2**20)
    gdn.configure_compose_cache(0)
    gdn.set_context(chains, tails)
    return gdn, common, tails


def save(gdn, value=2.):
    _, tickets = gdn.begin_decode_state(0)
    state = torch.full((gdn.num_workers, gdn.num_heads, gdn.head_v_dim, gdn.head_k_dim),
                       value, device=gdn.device)
    for target in gdn.write_to:
        pair = (torch.eye(gdn.head_k_dim, device=gdn.device).expand(
            1, gdn.num_heads, -1, -1).clone(), torch.zeros_like(state[:1]))
        target.set_linear_affine(0, pair)
    gdn.finish_decode_state(0, state, tickets)
    return state


def test_hits_have_no_compose_or_copy_and_reorder_subset_partial_miss(monkeypatch):
    gdn, common, tails = world()
    state = save(gdn)
    state[1].fill_(3.)
    state[2].fill_(4.)
    original = SharedCacheGDN._compose_initial_recurrent_parts
    composed = []

    def count(self, *args, **kwargs):
        composed.append(self.num_workers)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SharedCacheGDN, "_compose_initial_recurrent_parts", count)
    actual, _ = gdn.begin_decode_state(0)
    assert actual.data_ptr() == state.data_ptr()
    assert not composed
    view = gdn.context_view([gdn.cache_structure[i] for i in [2, 0]], [tails[2], tails[0]])
    actual, _ = view.begin_decode_state(0)
    torch.testing.assert_close(actual, state[[2, 0]])
    assert not composed
    # Replace just one write state: only that worker must be recomposed.
    tails[0].set_linear_affine(0, tails[0].linear_affine[0])
    expected = _reference_compose_initial_recurrent_state(view, 0, torch.float32).transpose(-1, -2)
    actual, _ = view.begin_decode_state(0)
    assert composed == [1]
    torch.testing.assert_close(actual[:1], state[2:3])
    torch.testing.assert_close(actual[1:], expected[1:])


@pytest.mark.parametrize("change", ["upstream", "own", "clear", "replace", "order", "fill_empty"])
def test_context_mutation_cannot_hit(change):
    gdn, common, tails = world(workers=1)
    extra = CacheBlock(gdn.device)
    gdn.set_context([[common, extra, tails[0]]], tails)
    save(gdn)
    assert gdn.successor_state_cache.stats["hits"] == 0
    if change == "upstream":
        common.set_linear_affine(0, common.linear_affine[0])
    elif change == "own":
        tails[0].set_linear_affine(0, tails[0].linear_affine[0])
    elif change == "clear":
        tails[0].clear()
    elif change == "replace":
        common.linear_affine[0] = tuple(t.clone() for t in common.linear_affine[0])
    elif change == "order":
        gdn.set_context([[extra, common, tails[0]]], tails)
    else:
        extra.set_linear_affine(0, tuple(t.clone() for t in common.linear_affine[0]))
    actual, _ = gdn.begin_decode_state(0)
    expected = _reference_compose_initial_recurrent_state(gdn, 0, torch.float32).transpose(-1, -2)
    torch.testing.assert_close(actual, expected)
    assert gdn.successor_state_cache.stats["hits"] == 0


@pytest.mark.parametrize("case", ["middle_write", "peer_writer", "duplicate_write", "repeat_own"])
def test_unsafe_writes_are_not_admitted(case):
    gdn, common, tails = world(workers=2)
    a, b = tails
    if case == "middle_write":
        gdn.set_context([[common, a, b]], [a])
    elif case == "peer_writer":
        gdn.set_context([[common, a], [common, a, b]], [a, b])
    elif case == "duplicate_write":
        gdn.set_context([[common, a], [common, a]], [a, a])
    else:
        gdn.set_context([[common, a, a]], [a])
    save(gdn)
    assert gdn.successor_state_cache.resident_entries == (1 if case == "peer_writer" else 0)


def test_commit_checks_prewrite_upstream_and_exact_own_revision():
    for change in ("upstream", "twice", "not_written"):
        gdn, common, tails = world(workers=1)
        initial, ticket = gdn.begin_decode_state(0)
        if change != "not_written":
            tails[0].set_linear_affine(0, tuple(t.clone() for t in common.linear_affine[0]))
        if change == "upstream":
            common.set_linear_affine(0, common.linear_affine[0])
        if change == "twice":
            tails[0].set_linear_affine(0, tails[0].linear_affine[0])
        gdn.finish_decode_state(0, initial, ticket)
        assert gdn.successor_state_cache.resident_entries == 0


def test_allocation_budget_counts_full_backing_and_releases_on_gc():
    blocks = [CacheBlock(torch.device("cpu")) for _ in range(4)]
    state = torch.zeros(4, 1, 2, 2)
    cache = GDNSuccessorCache(state.untyped_storage().nbytes())
    cache.put(0, state, [(0, blocks[0], ("one",)), (3, blocks[3], ("four",))])
    assert cache.resident_bytes == 64  # two views must not be charged as only 32 bytes
    cache.discard_block(blocks[0])
    assert cache.resident_bytes == 64
    cache.put(1, state.clone(), [(1, blocks[1], ("new",))])
    assert cache.resident_bytes == 64 and cache.resident_entries == 1
    assert cache.stats["evicted_batches"] == 1
    blocks.clear()
    gc.collect()
    assert cache.resident_bytes == 0 and cache.resident_entries == 0


def test_oversize_layer_partition_and_disable():
    gdn, _, tails = world(1)
    tiny = GDNSuccessorCache(1)
    tiny.put(0, torch.zeros(1, 1, 2, 2), [(0, tails[0], ())])
    assert tiny.resident_bytes == 0 and tiny.stats["skipped_oversize"] == 1
    state = save(gdn)
    assert gdn.successor_state_cache.get(tails[0], 1, ()) is None
    gdn.configure_successor_cache(0)
    assert gdn.successor_state_cache is None
    actual, ticket = gdn.begin_decode_state(0)
    assert ticket is None and not torch.equal(actual, state)


def test_session_free_releases_cached_state_and_prefill_never_uses_successor():
    from minisgl.shared_cache.session import SharedCacheSession
    gdn, _, tails = world(1)
    save(gdn)
    view = gdn.context_view(gdn.cache_structure, tails, prefill_segments=[2])
    _, ticket = view.begin_decode_state(0)
    assert ticket is None and gdn.successor_state_cache.stats["hits"] == 0
    session = object.__new__(SharedCacheSession)
    session.sc_gdn = gdn
    session._validate_block = lambda block: None
    session.free_block(tails[0])  # no pages: only the affine/state lifetime is exercised
    assert gdn.successor_state_cache.resident_bytes == 0
    assert not tails[0].linear_affine


@pytest.mark.parametrize("backend", ["cpu", "cuda_fla", "cuda_no_fla"])
@torch.inference_mode()
def test_long_recurrence_against_old_affine_oracle_with_upstream_change(monkeypatch, backend):
    if backend.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if backend == "cuda_fla" and delta._fla_recurrent is None:
        pytest.skip("FLA required")
    if backend != "cuda_fla":
        monkeypatch.setattr(delta, "_fla_recurrent", None)
    torch.manual_seed(731)
    device = "cpu" if backend == "cpu" else "cuda"
    dk, dv = (7, 5) if backend == "cpu" else (128, 64)
    gdn, common, tails = world(3, device, dk, dv, heads=4)
    reference = copy.copy(gdn)
    reference.cache_structure = copy.deepcopy(gdn.cache_structure)
    reference.write_to = [chain[-1] for chain in reference.cache_structure]
    reference.configure_successor_cache(0)
    dtype = torch.bfloat16 if backend == "cuda_fla" else torch.float32
    for step in range(48):
        if step == 12:
            # Two successor hits plus one invalidated terminal: exercise the
            # single-assembly path through the actual FLA AND no-FLA consumer.
            for target in (tails[1], reference.write_to[1]):
                pair = tuple(t.clone() for t in target.linear_affine[0])
                pair[1].add_(.025)
                target.set_linear_affine(0, pair)
        if step == 24:
            pair = tuple(t.clone() for t in common.linear_affine[0])
            pair[1].add_(.05)
            common.set_linear_affine(0, pair)
            reference.cache_structure[0][0].set_linear_affine(0, tuple(t.clone() for t in pair))
        q = torch.randn(3, 1, 4, dk, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn(3, 1, 4, dv, device=device, dtype=dtype)
        g = -torch.rand(3, 1, 4, device=device) * .2
        beta = torch.rand(3, 1, 4, device=device, dtype=dtype)
        start, tickets = gdn.begin_decode_state(0)
        old_start = _reference_compose_initial_recurrent_state(reference, 0, torch.float32).transpose(-1, -2).contiguous()
        out, final = delta._recurrent_delta(q, k, v, g, beta, start, state_v_first=True)
        old_out, old_final = delta._recurrent_delta(q, k, v, g, beta, old_start, state_v_first=True)
        tol = 2e-2 if backend == "cuda_fla" else 2e-5
        torch.testing.assert_close(out, old_out, rtol=tol, atol=tol * .1)
        torch.testing.assert_close(final, old_final, rtol=tol, atol=tol * .1)
        gdn.capture_token_affines(0, k, v, g.exp(), beta)
        reference.capture_token_affines(0, k, v, g.exp(), beta)
        gdn.finish_decode_state(0, final, tickets)
    assert gdn.successor_state_cache.stats["hits"] == (48 - 2) * 3 - 1


@pytest.mark.parametrize("low_precision", [False, True])
@torch.inference_mode()
def test_actual_layer_decode_uses_successor_and_preserves_no_fla(monkeypatch, low_precision):
    from test_gdn_ar_prefill_batch import _layer, _assign, HIDDEN, V_HEADS, HEAD_DIM, CONV_KERNEL
    monkeypatch.setattr(delta, "_fla_recurrent", None)
    layer = _layer()
    if low_precision:
        for name, tensor in layer.state_dict().items():
            _assign(layer, name, tensor.to(torch.bfloat16))
    gdn, _, tails = world(2, dk=HEAD_DIM, dv=HEAD_DIM, heads=V_HEADS)
    gdn.conv_dim = layer.conv_dim
    gdn.conv_kernel = CONV_KERNEL
    reference = copy.copy(gdn)
    reference.cache_structure = copy.deepcopy(gdn.cache_structure)
    reference.write_to = [c[-1] for c in reference.cache_structure]
    reference.configure_successor_cache(0)
    for _ in range(24):
        x = (torch.randn(2, HIDDEN) * .1).to(torch.bfloat16 if low_precision else torch.float32)
        actual = layer._forward_ar_decode(x, gdn)
        expected = layer._forward_ar_decode(x, reference)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        for a, b in zip(tails, reference.write_to):
            torch.testing.assert_close(a.linear_conv_state[0], b.linear_conv_state[0])
    assert gdn.successor_state_cache.stats["hits"] == (0 if low_precision else 46)
    if low_precision:
        assert gdn.successor_state_cache.resident_entries == 0
