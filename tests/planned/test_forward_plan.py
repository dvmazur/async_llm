"""CPU-only contract tests; no checkpoint, torch import, or GPU required."""
from collections import Counter
from dataclasses import FrozenInstanceError, replace
import json
import random
import subprocess
import sys

import pytest

from minisgl.planned.forward_plan import (
    BlockState, DecodeRequest, PlanCapacity, PrefillRequest, build_trie, prepare_forward,
)


CAP = PlanCapacity(4, 512, 8, 12, 128)


def state(b, *, populated=True, slot=None):
    return BlockState(b, b if slot is None else slot, populated, populated, generation=2, revision=7)


def unpack(phase, width):
    """Independent reconstruction of actual ancestry using the packed tables."""
    out = []
    for worker, active in enumerate(phase.active):
        if not active:
            continue
        chain, index = [], phase.terminal_nodes[worker]
        while index >= 0:
            depth, row = divmod(index, width)
            assert row < phase.level_counts[depth]
            chain.append(phase.node_slots[index])
            parent = phase.parent_rows[index]
            index = (depth - 1) * width + parent if depth else -1
        out.append(tuple(reversed(chain)))
    return out


def assert_sinks(phase):
    seen = []
    for node, (start, end) in enumerate(zip(phase.sink_offsets, phase.sink_offsets[1:])):
        for w in phase.sink_workers[start:end]:
            assert phase.active[w]
            assert phase.terminal_nodes[w] == node
            seen.append(w)
    expected = [w for w, (a, t) in enumerate(zip(phase.active, phase.terminal_nodes)) if a and t >= 0]
    assert sorted(seen) == expected
    assert all(w == -1 for w in phase.sink_workers[phase.sink_offsets[-1]:])


def test_import_is_independent_of_torch_and_cuda():
    code = "import sys; from minisgl.planned.forward_plan import prepare_forward; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_mixed_uses_old_decode_and_new_prefill_without_mutating_snapshot():
    blocks = {1: state(1), 2: state(2, populated=False), 3: state(3), 4: state(4, populated=False)}
    before = dict(blocks)
    plan = prepare_forward(blocks, capacity=CAP,
        prefill=[PrefillRequest((1, 3), 2, 7)],
        decode=[DecodeRequest((1, 2, 3), 3), DecodeRequest((1, 2, 4), 4)])
    assert unpack(plan.prefill, 4) == [(1, 3)]
    assert unpack(plan.decode, 8) == [(1, 2, 3), (1, 2)]
    assert plan.prefill.prior_conv_slots[0] == 3
    assert plan.decode.prior_conv_slots[:2] == (3, 2)
    assert plan.prefill.write_fresh[0]
    assert plan.decode.write_fresh[:2] == (False, True)
    assert blocks == before
    assert plan.blocks[1].populated  # block 3 was read once, still old logical state
    assert [(w.phase, w.old_revision, w.new_revision, w.added_tokens) for w in plan.writes] == [
        ("prefill", 7, 8, 7), ("decode", 7, 8, 1), ("decode", 7, 8, 1)]
    assert_sinks(plan.prefill); assert_sinks(plan.decode)


def test_prefill_includes_old_write_block_and_preserves_duplicate_read_order():
    blocks = {i: state(i) for i in range(4)}
    plan = prepare_forward(blocks, capacity=CAP,
        prefill=[PrefillRequest((0, 1, 0), 2, 3)])
    assert unpack(plan.prefill, 4) == [(0, 1, 0, 2)]
    assert plan.prefill.prior_conv_slots[0] == 2


def test_decode_implicit_self_is_not_silently_appended_to_gdn_read_chain():
    plan = prepare_forward({0: state(0), 1: state(1)}, capacity=CAP,
        decode=[DecodeRequest((0,), 1)])
    assert unpack(plan.decode, 8) == [(0,)]
    assert plan.decode.write_slots[0] == 1


def test_capacity_rows_have_holes_and_fixed_decode_boundary():
    blocks = {i: state(i, populated=False) for i in range(5)}
    plan = prepare_forward(blocks, capacity=CAP,
        prefill=[PrefillRequest((), i, n) for i, n in enumerate((36, 138, 91))],
        decode=[DecodeRequest((), i) for i in (3, 4)])
    assert plan.rows.prefill_offsets == (0, 36, 174, 265, 265)
    assert sum(plan.rows.active) == 267
    assert not any(plan.rows.active[265:512])
    assert plan.rows.request[512:514] == (4, 5)
    assert plan.rows.output_rows == (35, 173, 264, -1, 512, 513, -1, -1, -1, -1, -1, -1)
    assert plan.rows.chunk_offsets == (0, 1, 4, 6, 6)
    assert plan.rows.chunk_request[:6] == (0, 1, 1, 1, 2, 2)
    assert plan.rows.chunk_local[:6] == (0, 0, 1, 2, 0, 1)
    assert len(plan.rows.chunk_request) == 12


def test_graph_key_and_all_buffer_shapes_ignore_real_lengths_ids_and_population():
    blocks = {i: state(i) for i in range(8)}
    plans = [prepare_forward(blocks, capacity=CAP,
        prefill=[PrefillRequest((7,), i, n) for i, n in enumerate(lengths)],
        decode=[DecodeRequest((7, 6), 6)]) for lengths in ((36, 138, 91), (64, 64, 137), (1,))]
    assert len({p.graph_key for p in plans}) == 1
    assert [p.rows.chunk_offsets[len(ls)] for p, ls in zip(plans, ((36,138,91),(64,64,137),(1,)))] == [6, 5, 1]
    for obj in ("rows", "prefill", "decode"):
        baseline = getattr(plans[0], obj)
        for field in baseline.__dataclass_fields__:
            if field == "trie": continue
            assert len({len(getattr(getattr(p, obj), field)) for p in plans}) == 1, (obj, field)


def test_inactive_and_fresh_are_distinct_and_empty_chains_produce_zero_roots():
    plan = prepare_forward({5: state(5, populated=False)}, capacity=CAP,
        decode=[DecodeRequest((5,), 5)])
    assert plan.decode.active == (True,) + (False,) * 7
    assert plan.decode.write_fresh == (True,) + (False,) * 7
    assert plan.decode.write_slots == (5,) + (-1,) * 7
    assert plan.decode.terminal_nodes == (-1,) * 8
    assert not any(plan.decode.level_counts)
    assert_sinks(plan.decode)


def test_empty_phases_and_short_prefill_recipes_do_not_create_graph_keys():
    blocks={i:state(i) for i in range(4)}
    pf=[PrefillRequest((0,),1,1),PrefillRequest((0,),2,1)]
    dec=[DecodeRequest((0,3),3)]
    cases=[(pf,dec),(pf[:1],dec),([],dec),(pf,[]),([],[]),
           ([PrefillRequest((0,),1,65)],dec)]
    plans=[prepare_forward(blocks,capacity=CAP,prefill=p,decode=d) for p,d in cases]
    assert len({p.graph_key for p in plans})==1
    assert {p.mode for p in plans}=={"mixed","prefill","decode","empty"}
    assert {p.rows.prefill_recipe for p in plans}=={(0,),(1,),(2,)}


def test_empty_plan_and_zero_phase_capacities():
    cap = PlanCapacity(0, 0, 0, 0, 1)
    plan = prepare_forward({}, capacity=cap)
    assert plan.mode == "empty" and not plan.writes
    assert plan.rows.prefill_offsets == (0,)
    assert plan.prefill.sink_offsets == (0,)
    assert plan.rows.output_rows == ()


@pytest.mark.parametrize("kwargs", [
    {"decode": [DecodeRequest((0,), 1), DecodeRequest((0,), 1)]},
    {"prefill": [PrefillRequest((0,), 1, 1)], "decode": [DecodeRequest((0,), 1)]},
    {"prefill": [PrefillRequest((2,), 1, 1), PrefillRequest((), 2, 1)]},
])
def test_reject_same_phase_dependencies_and_duplicate_writers(kwargs):
    blocks = {i: state(i, populated=False) for i in range(3)}
    before = dict(blocks)
    with pytest.raises(ValueError): prepare_forward(blocks, capacity=CAP, **kwargs)
    assert blocks == before


@pytest.mark.parametrize("cap,kwargs", [
    (replace(CAP, prefill_tokens=1), {"prefill": [PrefillRequest((), 1, 2)]}),
    (replace(CAP, prefill_requests=0), {"prefill": [PrefillRequest((), 1, 1)]}),
    (replace(CAP, decode_workers=0), {"decode": [DecodeRequest((0,), 1)]}),
    (replace(CAP, chain_depth=1), {"decode": [DecodeRequest((0, 1), 1)]}),
    (replace(CAP, block_slots=1), {"decode": [DecodeRequest((0,), 1)]}),
])
def test_capacity_overflows_before_any_mutation(cap, kwargs):
    blocks = {i: state(i) for i in range(2)}
    before = dict(blocks)
    with pytest.raises(ValueError): prepare_forward(blocks, capacity=cap, **kwargs)
    assert blocks == before


def test_post_prefill_population_can_exceed_depth_even_if_prefill_fits():
    blocks = {0: state(0), 1: state(1, populated=False), 2: state(2)}
    with pytest.raises(ValueError, match="chain capacity"):
        prepare_forward(blocks, capacity=replace(CAP, chain_depth=2),
            prefill=[PrefillRequest((0,), 1, 2)], decode=[DecodeRequest((0,1,2), 2)])
    assert not blocks[1].populated


@pytest.mark.parametrize("blocks", [
    {0: state(0), 1: BlockState(1, None)},
    {0: state(0), 1: state(1, slot=0)},
    {0: state(0), 1: state(2)},
])
def test_invalid_registry_or_unreserved_write_is_rejected(blocks):
    with pytest.raises(ValueError):
        prepare_forward(blocks, capacity=CAP, decode=[DecodeRequest((0,), 1)])


def test_all_layer_reuse_has_one_lookup_per_block_and_snapshot_is_immutable():
    class Registry(dict):
        calls = Counter()
        def __getitem__(self, key):
            self.calls[key] += 1
            return super().__getitem__(key)
    blocks = Registry({i: state(i) for i in range(5)})
    source = [0, 1]
    request = DecodeRequest(source, 3)
    plan = prepare_forward(blocks, capacity=CAP,
        prefill=[PrefillRequest((0, 1), 2, 1)], decode=[request, DecodeRequest((0,1,2), 4)])
    assert blocks.calls == {i: 1 for i in range(5)}
    for _ in range(30):
        assert plan.decode is plan.decode
        unpack(plan.decode, 8)
    assert blocks.calls == {i: 1 for i in range(5)}
    source.clear(); blocks.clear()
    assert unpack(plan.decode, 8) == [(0,1), (0,1,2)]
    with pytest.raises(FrozenInstanceError): plan.mode = "prefill"


def test_large_slot_indices_are_not_truncated_to_int32():
    slot = 2**31 + 3
    plan = prepare_forward({0: state(0, slot=slot)},
        capacity=replace(CAP, block_slots=slot+1), decode=[DecodeRequest((0,), 0)])
    assert plan.decode.node_slots[0] == slot


@pytest.mark.parametrize("seed", range(40))
def test_randomized_chains_match_independent_prefix_set_and_packed_traversal(seed):
    rng = random.Random(seed)
    blocks = {i: state(i, populated=rng.random() > .25) for i in range(24)}
    raw = [tuple(rng.randrange(16) for _ in range(rng.randrange(12))) for _ in range(8)]
    expected = [tuple(b for b in c if blocks[b].populated) for c in raw]
    plan = prepare_forward(blocks, capacity=CAP,
        decode=[DecodeRequest(c, 16+i) for i,c in enumerate(raw)])
    assert unpack(plan.decode, 8) == expected
    prefixes = {c[:n] for c in expected for n in range(1, len(c)+1)}
    assert plan.decode.trie.gemm_count == sum(len(p) > 1 for p in prefixes)
    assert sum(plan.decode.level_counts) == len(prefixes)
    assert_sinks(plan.decode)


def test_shared_prefix_savings_examples_and_early_terminals():
    assert build_trie([(0, i) for i in range(10, 42)]).gemm_count == 32
    assert build_trie([(0, 1, 2, i) for i in range(10, 42)]).gemm_count == 34
    chains = [(0,), (0,1,2), (0,1), (0,1,2), (), (0,2,1)]
    blocks = {i: state(i) for i in range(12)}
    plan = prepare_forward(blocks, capacity=CAP,
        decode=[DecodeRequest(c, i+6) for i,c in enumerate(chains)])
    assert unpack(plan.decode, 8) == chains
    assert_sinks(plan.decode)


@pytest.mark.parametrize("make", [
    lambda: BlockState(-1, 0), lambda: BlockState(0, -1),
    lambda: BlockState(0, None, True), lambda: BlockState(0, 0, False, True),
    lambda: BlockState(0, 0, 1), lambda: DecodeRequest((True,), 1),
    lambda: PrefillRequest((), 1, 0), lambda: PrefillRequest((1,), 1, 1),
    lambda: replace(CAP, chunk_size=0), lambda: replace(CAP, chain_depth=-1),
])
def test_invalid_input_types_and_bounds(make):
    with pytest.raises(ValueError): make()
