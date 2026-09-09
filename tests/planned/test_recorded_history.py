"""Opt-in CPU parity on normalized historical real operations (no weights)."""
from collections import defaultdict
import json
import os
from pathlib import Path

import pytest

from minisgl.planned.forward_plan import BlockState, DecodeRequest, PlanCapacity, PrefillRequest, prepare_forward
from test_forward_plan import assert_sinks, unpack


@pytest.mark.parametrize("trace_index", [0, 1, 2], ids=["real9_mixed", "real15_mixed", "real20_nonmixed"])
def test_every_recorded_phase_has_same_states_and_transition_count(trace_index):
    path = os.environ.get("MINISGL_TRIE_HISTORY")
    if path is None:
        pytest.skip("set MINISGL_TRIE_HISTORY for historical replay tests")
    trace = json.loads(Path(path).read_text())["traces"][trace_index]
    by_op = defaultdict(dict)
    for event in trace["events"]:
        by_op[event["op_index"]][event["phase"]] = event
    totals = {"prefill": 0, "decode": 0}
    keys = set()
    # The 20-pipeline trace really contains 40 simultaneous prefill jobs.
    # These are test bounds, not a recommended production memory reservation.
    cap = PlanCapacity(64, 4096, 64, 16, 512)
    for phases in by_op.values():
        ids = sorted({b for e in phases.values() for c in [*e["chains"], e["writes"]] for b in c})
        population = dict.fromkeys(ids, True)
        # Snapshot before either phase. Fresh PF targets become populated only
        # inside prepare_forward's post-prefill view, never in the original map.
        for e in phases.values():
            population.update(zip(e["writes"], e["write_populated"]))
        blocks = {b: BlockState(b, slot, population[b], population[b]) for slot, b in enumerate(ids)}
        prefill, decode = [], []
        if "prefill" in phases:
            e = phases["prefill"]
            for c, w, n in zip(e["chains"], e["writes"], e["new_lengths"]):
                context = c[:-1] if c and c[-1] == w else c
                prefill.append(PrefillRequest(context, w, n))
        if "decode" in phases:
            e = phases["decode"]
            decode = [DecodeRequest(c, w) for c, w in zip(e["chains"], e["writes"])]
        plan = prepare_forward(blocks, capacity=cap, prefill=prefill, decode=decode)
        keys.add(plan.graph_key)
        for phase, width in (("prefill", cap.prefill_requests), ("decode", cap.decode_workers)):
            actual = getattr(plan, phase)
            if phase not in phases:
                assert not any(actual.active)
                continue
            e = phases[phase]
            expected = [tuple(blocks[b].slot for b in c) for c in e["chains"]]
            assert unpack(actual, width) == expected
            assert actual.trie.gemm_count == e["trie_gemms"]
            totals[phase] += actual.trie.gemm_count
            assert_sinks(actual)
    assert len(keys) <= 3  # mode only; not 410 keys for 410 observed batches
    for phase in totals:
        assert totals[phase] == trace["summary"][phase]["trie_gemms"]
