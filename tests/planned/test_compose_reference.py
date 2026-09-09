"""Numerical CPU oracle for prepared pool indices, not a new GPU executor."""
import ast
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from minisgl.planned.forward_plan import BlockState, DecodeRequest, PlanCapacity, prepare_forward
from reference_gdn_compose import _reference_compose_initial_recurrent_state


def evaluate(phase, pool, layer):
    """Test-only interpreter of packed tables; outputs cannot alias the pool."""
    width = len(phase.active)
    output = torch.zeros((width, *pool.shape[3:]), dtype=pool.dtype)
    previous = []
    gemms = 0
    for depth, count in enumerate(phase.level_counts):
        current = []
        for row in range(count):
            node = depth * width + row
            slot = phase.node_slots[node]
            A, B = pool[layer, slot]
            value = B.clone() if depth == 0 else previous[phase.parent_rows[node]] @ A + B
            gemms += depth > 0
            current.append(value)
            lo, hi = phase.sink_offsets[node:node+2]
            for worker in phase.sink_workers[lo:hi]:
                output[worker] = value
        previous = current
    return output[:sum(phase.active)], gemms


@pytest.mark.parametrize("seed", range(20))
def test_four_layer_pool_matches_literal_old_reference(seed):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    layers, slots, heads, dim = 4, 32, 2, 5
    pool = torch.randn(layers, slots, 2, heads, dim, dim) * .1
    saved = pool.clone()
    slot_ids = rng.sample(range(slots), 16)
    blocks = {i: BlockState(i, slot_ids[i], i % 4 != 0, i % 4 != 0) for i in range(16)}
    chains = [(1,), (1, 2), (1, 2, 3), (1, 2, 3), (), (1, 0, 2)]
    chains += [tuple(rng.randrange(8) for _ in range(10)) for _ in range(2)]
    cap = PlanCapacity(0, 0, 8, 12, slots)
    plan = prepare_forward(blocks, capacity=cap,
        decode=[DecodeRequest(c, 8+i) for i,c in enumerate(chains)])
    oracle_blocks = {i: SimpleNamespace(linear_affine={
        l: (pool[l, b.slot, 0][None], pool[l, b.slot, 1][None]) for l in range(layers)
    } if b.populated else {}) for i,b in blocks.items()}
    old = SimpleNamespace(device=torch.device("cpu"), num_heads=heads, head_k_dim=dim, head_v_dim=dim,
                          cache_structure=[[oracle_blocks[b] for b in c] for c in chains])
    old.has_previous_affine = lambda l: any(l in b.linear_affine for c in old.cache_structure for b in c)
    for layer in range(layers):
        expected = _reference_compose_initial_recurrent_state(old, layer, torch.float32).transpose(-1, -2)
        actual, gemms = evaluate(plan.decode, pool, layer)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        assert gemms == plan.decode.trie.gemm_count
        assert actual.data_ptr() != pool.data_ptr()
        actual.fill_(float("nan"))
        torch.testing.assert_close(pool, saved, rtol=0, atol=0)


def test_literal_reference_body_is_unchanged_from_existing_snapshot():
    repo = Path(__file__).resolve().parents[2]
    old_tree = ast.parse((repo / "tests/core/test_gdn_compose.py").read_text())
    new_tree = ast.parse((Path(__file__).parent / "reference_gdn_compose.py").read_text())
    def method(tree):
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_reference_compose_initial_recurrent_state")
    assert ast.dump(method(old_tree)) == ast.dump(method(new_tree))
