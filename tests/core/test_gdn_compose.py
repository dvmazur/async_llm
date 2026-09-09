"""Tests for batched GDN initial-state composition.

The numerical oracle below is a literal snapshot of the original production
``SharedCacheGDN.compose_initial_recurrent_state`` implementation.  The apparent
duplication is deliberate: refactoring it to use the optimized prefix planner or
evaluator would destroy its value as an independent regression oracle.
"""

from __future__ import annotations

import inspect
import random
from dataclasses import dataclass, field

import pytest
import torch
from minisgl.models.qwen3_5_delta import _fla_recurrent, _recurrent_delta
from minisgl.shared_cache.gdn import (
    SharedCacheGDN,
    _evaluate_affine_prefix_plan,
    _plan_affine_prefixes,
)
from minisgl.shared_cache.gdn_affine import compose_gdn_affines, init_gdn_affine


# Literal snapshot of the complete pre-optimization production method body.
# Keep this independent from all new planning/evaluation helpers.
def _reference_compose_initial_recurrent_state(
    self, lin_idx: int, dtype: torch.dtype
) -> torch.Tensor | None:
    """Compose each worker's chain into an initial recurrent state.

    Returns ``[num_workers, H, d_k, d_v]`` in HF convention (or ``None`` if no
    block in any chain has an affine for this layer).
    """
    if not self.has_previous_affine(lin_idx):
        return None

    # Worker chains typically share leading blocks (e.g. [prompt, thinker] is a
    # prefix of [prompt, thinker, writer]).  Memoize each composed prefix by its
    # block-id tuple so a shared prefix is composed once per call, not per worker.
    # Bit-identical to composing each chain independently (compose is deterministic).
    prefix_memo: dict = {}

    def compose_chain(chain):
        acc = None  # (A, B) once we hit the first block with an affine
        key: tuple = ()
        for block in chain:
            key = key + (id(block),)
            pair = block.linear_affine.get(lin_idx)
            if pair is None:
                continue  # block has no affine for this layer -> acc unchanged
            if key in prefix_memo:
                acc = prefix_memo[key]
                continue
            A_b = pair[0].to(dtype=torch.float32, device=self.device)
            B_b = pair[1].to(dtype=torch.float32, device=self.device)
            if acc is None:
                acc = (A_b, B_b)  # first real block: no identity compose needed
            else:
                acc = compose_gdn_affines(
                    A_first=acc[0], B_first=acc[1], A_second=A_b, B_second=B_b
                )
            prefix_memo[key] = acc
        if acc is None:
            acc = init_gdn_affine(
                batch_size=1,
                num_heads=self.num_heads,
                d_k=self.head_k_dim,
                d_v=self.head_v_dim,
                dtype=torch.float32,
                device=self.device,
            )
        return acc

    per_worker = [compose_chain(chain)[1] for chain in self.cache_structure]
    S_block = torch.cat(per_worker, dim=0)  # [W, H, d_v, d_k]
    S_hf = S_block.transpose(-1, -2).contiguous()  # [W, H, d_k, d_v]
    return S_hf.to(dtype=dtype)


@dataclass(eq=False)
class _FakeBlock:
    name: str
    linear_affine: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


def _make_gdn(chains, *, heads=2, d_k=5, d_v=3, device="cpu") -> SharedCacheGDN:
    gdn = SharedCacheGDN(
        num_heads=heads,
        head_k_dim=d_k,
        head_v_dim=d_v,
        conv_dim=1,
        conv_kernel=1,
        device=torch.device(device),
    )
    gdn.set_context(chains, [chain[-1] if chain else _FakeBlock("sink") for chain in chains])
    return gdn


def _pair(*, heads=2, d_k=5, d_v=3, seed=0, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(seed)
    eye = torch.eye(d_k, dtype=torch.float32, device=device).view(1, 1, d_k, d_k)
    # Dense, non-commuting matrices are required to make ordering mistakes visible.
    A = eye.expand(1, heads, d_k, d_k).clone()
    A.add_(0.05 * torch.randn(1, heads, d_k, d_k, generator=generator, device=device))
    B = 0.1 * torch.randn(1, heads, d_v, d_k, generator=generator, device=device)
    return A, B


def _block(name, *, lin_idx=0, seed=0, heads=2, d_k=5, d_v=3, device="cpu"):
    block = _FakeBlock(name)
    block.linear_affine[lin_idx] = _pair(heads=heads, d_k=d_k, d_v=d_v, seed=seed, device=device)
    return block


def _packed_blocks(names, *, seed=0, heads=2, d_k=5, d_v=3, device="cpu"):
    pairs = [
        _pair(heads=heads, d_k=d_k, d_v=d_v, seed=seed + row, device=device)
        for row in range(len(names))
    ]
    A_batch = torch.cat([pair[0] for pair in pairs], dim=0)
    B_batch = torch.cat([pair[1] for pair in pairs], dim=0)
    return [
        _FakeBlock(name, {0: (A_batch[row : row + 1], B_batch[row : row + 1])})
        for row, name in enumerate(names)
    ]


def _assert_reference_parity(gdn: SharedCacheGDN, lin_idx: int, dtype: torch.dtype):
    expected = _reference_compose_initial_recurrent_state(gdn, lin_idx, dtype)
    actual = gdn.compose_initial_recurrent_state(lin_idx, dtype)
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if dtype == torch.bfloat16:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def _node_paths(plan):
    """Return object-id paths for every node, grouped by depth."""
    paths = []
    previous = []
    for depth, frontier in enumerate(plan.frontiers, start=1):
        current = []
        for node in frontier:
            prefix = () if depth == 1 else previous[node.parent_index]
            current.append(prefix + (id(node.block),))
        paths.append(current)
        previous = current
    return paths


def test_no_affine_history_returns_none():
    chains = [[], [_FakeBlock("empty")], [_FakeBlock("also-empty")]]
    gdn = _make_gdn(chains)
    _assert_reference_parity(gdn, lin_idx=7, dtype=torch.float32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_numerical_parity_variable_shared_reordered_and_missing(dtype):
    p = _block("P", seed=1)
    a = _block("A", seed=2)
    b = _block("B", seed=3)
    t0 = _block("T0", seed=4)
    t1 = _block("T1", seed=5)
    missing_begin = _FakeBlock("missing-begin")
    missing_middle = _FakeBlock("missing-middle")
    missing_end = _FakeBlock("missing-end")
    chains = [
        [],
        [p],
        [p, a],
        [p, a, t0],
        [p, a, t1],
        [p, a, t0],  # duplicate complete chain
        [p, a, b],
        [p, b, a],  # reordered, must not alias
        [missing_begin, p, missing_middle, a, missing_end],
    ]
    gdn = _make_gdn(chains)
    _assert_reference_parity(gdn, lin_idx=0, dtype=dtype)

    out = gdn.compose_initial_recurrent_state(0, dtype)
    assert out is not None
    assert torch.count_nonzero(out[0]).item() == 0  # root worker gets zero state
    assert torch.equal(out[3], out[5])  # duplicate chain maps to same value
    assert not torch.allclose(out[6].float(), out[7].float())  # order is observable


def test_source_affines_are_not_mutated():
    blocks = [_block(name, seed=i + 10) for i, name in enumerate(("P", "A", "B"))]
    original_pairs = {id(block): block.linear_affine[0] for block in blocks}
    before = {id(block): tuple(t.clone() for t in block.linear_affine[0]) for block in blocks}
    gdn = _make_gdn([[blocks[0], blocks[1]], [blocks[0], blocks[2]]])
    _assert_reference_parity(gdn, 0, torch.float32)
    for block in blocks:
        assert block.linear_affine[0][0] is original_pairs[id(block)][0]
        assert block.linear_affine[0][1] is original_pairs[id(block)][1]
        for actual, expected in zip(block.linear_affine[0], before[id(block)]):
            assert torch.equal(actual, expected)


def test_affine_scan_storage_packs_contiguous_component_rows_once_per_batch():
    heads, d_k, d_v, workers = 2, 5, 3, 3
    targets = [_FakeBlock(f"target-{worker}") for worker in range(workers)]
    gdn = _make_gdn([[] for _ in targets], heads=heads, d_k=d_k, d_v=d_v)
    gdn.set_context([[] for _ in targets], targets)
    generator = torch.Generator().manual_seed(91)
    state = torch.randn(workers, heads, d_k + d_v, d_k, generator=generator)

    gdn.store_affine_scan_state(0, state, d_k=d_k)

    A_storage = None
    B_storage = None
    for worker, target in enumerate(targets):
        A_hat, B_hat = target.linear_affine[0]
        assert A_hat.is_contiguous()
        assert B_hat.is_contiguous()
        torch.testing.assert_close(A_hat, state[worker : worker + 1, :, :d_k, :])
        torch.testing.assert_close(B_hat, state[worker : worker + 1, :, d_k:, :])
        A_storage = A_hat.untyped_storage() if A_storage is None else A_storage
        B_storage = B_hat.untyped_storage() if B_storage is None else B_storage
        assert A_hat.untyped_storage().data_ptr() == A_storage.data_ptr()
        assert B_hat.untyped_storage().data_ptr() == B_storage.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pointer compose required")
def test_affine_scan_storage_can_be_reused_as_single_shared_pointer_parent():
    device = "cuda"
    heads, d_k, d_v = 2, 16, 8
    common = _FakeBlock("scan-common")
    writer = _make_gdn([[]], heads=heads, d_k=d_k, d_v=d_v, device=device)
    writer.set_context([[]], [common])
    generator = torch.Generator(device=device).manual_seed(92)
    state = torch.randn(
        1, heads, d_k + d_v, d_k, device=device, generator=generator
    )
    writer.store_affine_scan_state(0, state, d_k=d_k)
    left = _block(
        "left", seed=93, heads=heads, d_k=d_k, d_v=d_v, device=device
    )
    right = _block(
        "right", seed=94, heads=heads, d_k=d_k, d_v=d_v, device=device
    )
    gdn = _make_gdn(
        [[common, left], [common, right]],
        heads=heads,
        d_k=d_k,
        d_v=d_v,
        device=device,
    )

    _assert_reference_parity(gdn, 0, torch.float32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_v_first_output_matches_hf_reference_with_rectangular_state(dtype):
    heads, d_k, d_v = 2, 7, 3
    p = _block("P", seed=1, heads=heads, d_k=d_k, d_v=d_v)
    a = _block("A", seed=2, heads=heads, d_k=d_k, d_v=d_v)
    b = _block("B", seed=3, heads=heads, d_k=d_k, d_v=d_v)
    gdn = _make_gdn([[p, a], [p, b]], heads=heads, d_k=d_k, d_v=d_v)

    expected_hf = _reference_compose_initial_recurrent_state(gdn, 0, dtype)
    actual_v_first = gdn.compose_initial_recurrent_state(0, dtype, state_v_first=True)
    assert expected_hf is not None and actual_v_first is not None
    assert actual_v_first.shape == (2, heads, d_v, d_k)
    assert actual_v_first.is_contiguous()
    torch.testing.assert_close(
        actual_v_first.float().transpose(-1, -2),
        expected_hf.float(),
        rtol=2e-2 if dtype == torch.bfloat16 else 2e-5,
        atol=2e-2 if dtype == torch.bfloat16 else 2e-5,
    )


def test_representative_batch64_parity_with_mixed_tails():
    heads, d_k, d_v = 3, 7, 4
    common = _block("common", seed=100, heads=heads, d_k=d_k, d_v=d_v)
    branches = [
        _block(f"branch-{i}", seed=200 + i, heads=heads, d_k=d_k, d_v=d_v) for i in range(8)
    ]
    tails = [
        (
            _block(f"tail-{i}", seed=300 + i, heads=heads, d_k=d_k, d_v=d_v)
            if i % 3
            else _FakeBlock(f"empty-tail-{i}")
        )
        for i in range(64)
    ]
    chains = [[common, branches[i % 8], tails[i]] for i in range(64)]
    gdn = _make_gdn(chains, heads=heads, d_k=d_k, d_v=d_v)
    _assert_reference_parity(gdn, 0, torch.float32)


def test_required_prefix_trie_topology():
    p = _block("P", seed=1)
    a = _block("A", seed=2)
    b = _block("B", seed=3)
    t0 = _block("T0", seed=4)
    t1 = _block("T1", seed=5)
    t2 = _block("T2", seed=6)
    n = _FakeBlock("N")  # no affine for this layer
    chains = [
        [p, a, t0],
        [p, a, t1],
        [p, b],
        [p, a, t0],
        [p, n, a, t2],
    ]
    plan = _plan_affine_prefixes(chains, lin_idx=0)

    assert [len(frontier) for frontier in plan.frontiers] == [1, 2, 3]
    assert [[node.block.name for node in f] for f in plan.frontiers] == [
        ["P"],
        ["A", "B"],
        ["T0", "T1", "T2"],
    ]
    # P/A is shared by w0, w1, w3, and w4. Duplicate chains share a terminal.
    assert plan.worker_terminals[0] == plan.worker_terminals[3]
    assert plan.worker_terminals == ((3, 0), (3, 1), (2, 1), (3, 0), (3, 2))
    assert sum(len(f) for f in plan.frontiers[1:]) == 5
    assert all(node.block is not n for frontier in plan.frontiers for node in frontier)

    paths = _node_paths(plan)
    assert paths[1][0] == (id(p), id(a))
    assert paths[1][1] == (id(p), id(b))


def test_topology_keeps_reorders_and_equal_valued_distinct_blocks_separate():
    p = _block("P", seed=1)
    a = _block("A", seed=2)
    b = _block("B", seed=3)
    a_clone = _FakeBlock("A-clone", {0: tuple(t.clone() for t in a.linear_affine[0])})
    plan = _plan_affine_prefixes([[p, a, b], [p, b, a], [p, a_clone, b], []], lin_idx=0)
    paths = _node_paths(plan)
    all_paths = {path for frontier in paths for path in frontier}
    assert (id(p), id(a), id(b)) in all_paths
    assert (id(p), id(b), id(a)) in all_paths
    assert (id(p), id(a_clone), id(b)) in all_paths
    assert len(plan.frontiers[1]) == 3  # P/A, P/B, and distinct P/A-clone
    assert plan.worker_terminals[-1] == (0, 0)  # root / zero


def test_generated_topologies_match_tuple_prefix_oracle():
    rng = random.Random(1234)
    blocks = [_block(f"b{i}", seed=20 + i) for i in range(7)]
    # This block exists in chains but is a no-op at lin_idx=0.
    missing = _FakeBlock("missing", {1: _pair(seed=999)})
    pool = blocks + [missing]

    for _ in range(40):
        chains = [rng.sample(pool, rng.randint(0, 5)) for _ in range(rng.randint(1, 12))]
        plan = _plan_affine_prefixes(chains, lin_idx=0)
        planned_paths = {path for frontier in _node_paths(plan) for path in frontier}

        effective = [tuple(id(b) for b in chain if 0 in b.linear_affine) for chain in chains]
        expected_prefixes = {
            chain[:depth] for chain in effective for depth in range(1, len(chain) + 1)
        }
        assert planned_paths == expected_prefixes

        terminal_paths = []
        paths_by_depth = _node_paths(plan)
        for depth, index in plan.worker_terminals:
            terminal_paths.append(() if depth == 0 else paths_by_depth[depth - 1][index])
        assert terminal_paths == effective

        gdn = _make_gdn(chains)
        _assert_reference_parity(gdn, 0, torch.float32)


def test_cpu_evaluator_preserves_frontier_shapes_for_packed_inputs():
    p = _block("P", seed=1)
    a, b = _packed_blocks(["A", "B"], seed=2)
    t0, t1, t2 = _packed_blocks(["T0", "T1", "T2"], seed=4)
    plan = _plan_affine_prefixes([[p, a, t0], [p, a, t1], [p, b], [p, a, t2]], lin_idx=0)
    states = _evaluate_affine_prefix_plan(
        plan,
        lin_idx=0,
        num_heads=2,
        d_k=5,
        d_v=3,
        device=torch.device("cpu"),
    )
    assert [state.shape[0] for state in states] == [1, 2, 3]


def test_production_compose_has_no_storage_layout_dispatch():
    source = inspect.getsource(__import__("minisgl.shared_cache.gdn", fromlist=["_"]))
    assert "_contiguous_frontier_view" not in source
    assert "untyped_storage" not in source


def test_production_compose_does_not_construct_unused_composite_A():
    source = inspect.getsource(SharedCacheGDN.compose_initial_recurrent_state)
    assert "compose_gdn_affines" not in source


@pytest.mark.skipif(
    not torch.cuda.is_available() or _fla_recurrent is None,
    reason="CUDA and flash-linear-attention are required",
)
def test_fla_recurrent_matches_with_reference_and_batched_initial_states():
    device = "cuda"
    workers, heads, d_k, d_v = 4, 16, 128, 64
    p = _block("P", seed=1, heads=heads, d_k=d_k, d_v=d_v, device=device)
    a = _block("A", seed=2, heads=heads, d_k=d_k, d_v=d_v, device=device)
    b = _block("B", seed=3, heads=heads, d_k=d_k, d_v=d_v, device=device)
    chains = [[p, a], [p, b], [p, a, b], []]
    gdn = _make_gdn(chains, heads=heads, d_k=d_k, d_v=d_v, device=device)
    old_state = _reference_compose_initial_recurrent_state(gdn, 0, torch.float32)
    new_state_v_first = gdn.compose_initial_recurrent_state(0, torch.float32, state_v_first=True)
    assert old_state is not None and new_state_v_first is not None

    generator = torch.Generator(device=device).manual_seed(77)
    q = torch.randn(
        workers, 1, heads, d_k, dtype=torch.bfloat16, device=device, generator=generator
    )
    k = torch.randn_like(q)
    v = torch.randn(
        workers, 1, heads, d_v, dtype=torch.bfloat16, device=device, generator=generator
    )
    gate = -torch.rand(workers, 1, heads, dtype=torch.float32, device=device, generator=generator)
    beta = torch.rand(workers, 1, heads, dtype=torch.bfloat16, device=device, generator=generator)

    old_out, old_final = _fla_recurrent(
        q,
        k,
        v,
        g=gate,
        beta=beta,
        initial_state=old_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    new_out, new_final_v_first = _recurrent_delta(
        q,
        k,
        v,
        gate,
        beta,
        new_state_v_first,
        state_v_first=True,
    )
    torch.testing.assert_close(new_out.float(), old_out.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(new_final_v_first.transpose(-1, -2), old_final, rtol=2e-4, atol=2e-4)
