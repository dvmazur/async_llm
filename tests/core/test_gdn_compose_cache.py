"""Correctness and policy tests for persistent GDN prefix-state caching."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
from minisgl.shared_cache.gdn import SharedCacheGDN
from minisgl.shared_cache.gdn_compose_cache import GDNComposeStateCache
from minisgl.shared_cache.shared_block import CacheBlock


@dataclass(eq=False)
class _Block:
    name: str
    linear_affine: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


def _pair(seed, *, heads, d_k, d_v, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    A = torch.eye(d_k, device=device).view(1, 1, d_k, d_k).expand(1, heads, -1, -1)
    A = A.clone() + 0.02 * torch.randn(1, heads, d_k, d_k, device=device, generator=generator)
    B = torch.randn(1, heads, d_v, d_k, device=device, generator=generator) * 0.1
    return A, B


def _reference(chains, *, heads, d_k, d_v, device):
    rows = []
    for chain in chains:
        state = torch.zeros(1, heads, d_v, d_k, dtype=torch.float32, device=device)
        for block in chain:
            pair = block.linear_affine.get(0)
            if pair is not None:
                state = torch.matmul(state, pair[0]) + pair[1]
        rows.append(state)
    return torch.cat(rows)


def test_second_touch_policy_admission_eviction_and_write_filter():
    cache = GDNComposeStateCache(max_bytes=2 * 2 * 4 * 4)
    one = torch.ones(2, 4)
    two = 2 * one
    three = 3 * one

    cache.consider("one", one, current_write_prefix=False)
    assert cache.resident_entries == 0
    cache.consider("one", one, current_write_prefix=False)
    assert torch.equal(cache.get("one"), one)
    cache.consider("write", two, current_write_prefix=True)
    cache.consider("write", two, current_write_prefix=True)
    assert cache.get("write") is None

    cache.consider("two", two, current_write_prefix=False)
    cache.consider("two", two, current_write_prefix=False)
    cache.consider("three", three, current_write_prefix=False)
    cache.consider("three", three, current_write_prefix=False)
    assert cache.resident_entries == 2
    assert cache.get("one") is None
    assert cache.stats["evictions"] == 1


def test_cache_block_affine_revision_survives_clear_and_reuse():
    block = CacheBlock(torch.device("cpu"))
    pair = (torch.ones(1), torch.zeros(1))
    block.set_linear_affine(3, pair)
    assert block.linear_affine_revision[3] == 1
    block.set_linear_affine(3, pair)
    assert block.linear_affine_revision[3] == 2
    block.clear()
    assert block.linear_affine_revision[3] == 3
    block.set_linear_affine(3, pair)
    assert block.linear_affine_revision[3] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cached_compose_matches_reference_and_invalidates_replaced_middle_block():
    workers, heads, d_k, d_v = 2, 4, 19, 13
    common = _Block("common", {0: _pair(1, heads=heads, d_k=d_k, d_v=d_v, device="cuda")})
    middle = _Block("middle", {0: _pair(2, heads=heads, d_k=d_k, d_v=d_v, device="cuda")})
    tails = [
        _Block(
            f"tail-{worker}", {0: _pair(3 + worker, heads=heads, d_k=d_k, d_v=d_v, device="cuda")}
        )
        for worker in range(workers)
    ]
    chains = [[common, middle, tail] for tail in tails]
    gdn = SharedCacheGDN(
        num_heads=heads,
        head_k_dim=d_k,
        head_v_dim=d_v,
        conv_dim=1,
        conv_kernel=1,
        device=torch.device("cuda"),
    )
    gdn.compose_state_cache = GDNComposeStateCache(max_bytes=32 * 2**20)
    gdn.set_context(chains, tails)

    # First observation is ghost-only, second admits, third consumes a hit.
    for _ in range(3):
        expected = _reference(chains, heads=heads, d_k=d_k, d_v=d_v, device="cuda")
        actual = gdn.compose_initial_recurrent_state(0, torch.float32, state_v_first=True)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    # Only [common, middle] is worth caching.  A depth-one prefix performs no
    # state GEMM, so admitting [common] would add storage/copy overhead without
    # avoiding any compose work.
    assert gdn.compose_state_cache.stats["admissions"] >= 1
    assert gdn.compose_state_cache.stats["hits"] >= 1
    assert gdn.compose_state_cache.stats["skipped_write_prefixes"] >= 2

    # Direct replacement bypasses CacheBlock revisions, so pointer identity in
    # the key must still prevent a stale middle-prefix hit.
    middle.linear_affine[0] = _pair(99, heads=heads, d_k=d_k, d_v=d_v, device="cuda")
    expected = _reference(chains, heads=heads, d_k=d_k, d_v=d_v, device="cuda")
    actual = gdn.compose_initial_recurrent_state(0, torch.float32, state_v_first=True)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cache_budget_is_a_ratio_of_gdn_storage(monkeypatch):
    monkeypatch.setenv("MINISGL_GDN_COMPOSE_CACHE_RATIO", "0.25")
    gdn = SharedCacheGDN(
        num_heads=1,
        head_k_dim=2,
        head_v_dim=2,
        conv_dim=1,
        conv_kernel=1,
        device=torch.device("cuda"),
        gdn_storage_bytes=4_000,
    )
    assert gdn.compose_state_cache is not None
    assert gdn.compose_cache_ratio == 0.25
    assert gdn.compose_state_cache.max_bytes == 1_000

    gdn.configure_compose_cache(0)
    assert gdn.compose_state_cache is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("layout", ["ordered", "duplicate", "mixed_depth", "empty"])
def test_cached_terminal_assembly_matches_old_reference_and_avoids_redundant_cat(monkeypatch, layout):
    import minisgl.kernel as kernel
    from test_gdn_compose import _reference_compose_initial_recurrent_state

    device = torch.device("cuda")
    common, middle, left, right, sink = [CacheBlock(device) for _ in range(5)]
    for seed, block in enumerate((common, middle, left, right)):
        block.set_linear_affine(0, _pair(seed, heads=2, d_k=19, d_v=13, device=device))
    chains = [[common, middle, left], [common, middle, right]]
    if layout == "duplicate":
        chains.append(chains[0])
    elif layout == "mixed_depth":
        chains.append([common])
    elif layout == "empty":
        chains.append([])
    gdn = SharedCacheGDN(num_heads=2, head_k_dim=19, head_v_dim=13,
                         conv_dim=1, conv_kernel=1, device=device)
    gdn.compose_state_cache = GDNComposeStateCache(32 * 2**20)
    # Exclude terminal states from admission, but allow [common,middle] hits.
    gdn.set_context(chains, [left, right] + ([sink] if len(chains) == 3 else []))
    for _ in range(2):
        gdn.compose_initial_recurrent_state(0, torch.float32, state_v_first=True)
    expected = _reference_compose_initial_recurrent_state(gdn, 0, torch.float32).transpose(-1, -2)
    original_nodes, original_cat = kernel.apply_gdn_affine_pointer_nodes, torch.cat
    outputs, cats = [], []

    def nodes(*a, **kw):
        result = original_nodes(*a, **kw)
        outputs.append(result)
        return result

    def cat(*a, **kw):
        cats.append(1)
        return original_cat(*a, **kw)

    monkeypatch.setattr(kernel, "apply_gdn_affine_pointer_nodes", nodes)
    monkeypatch.setattr(torch, "cat", cat)
    actual = gdn.compose_initial_recurrent_state(0, torch.float32, state_v_first=True)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    if layout == "ordered":
        assert actual is outputs[-1]
        assert not cats
    else:
        assert cats, "nontrivial terminal mapping must not blindly return a frontier"
