from dataclasses import replace

import pytest

from minisgl.planned.forward_plan import DecodeRequest, PlanCapacity, PrefillRequest
from minisgl.planned.slots import SlotRegistry


class Completion:
    def __init__(self, ready=False): self.ready = ready
    def query(self): return self.ready


def capacity(slots=3): return PlanCapacity(3, 32, 3, 8, slots)


def populate(registry, block, n=3):
    tx = registry.begin(capacity=capacity(registry.capacity), prefill=[PrefillRequest((), block, n)])
    registry.mark_submitted(tx)
    assert registry.finish(tx, Completion(True))
    return registry[block]


def test_empty_handles_do_not_allocate_gpu_slots():
    reg = SlotRegistry(2)
    ids = [reg.create() for _ in range(64)]
    assert reg.free_slots == 2
    assert all(reg[b].slot is None for b in ids)
    for b in ids: reg.release_handle(b)
    assert reg.live_handles == 0


def test_commit_is_once_after_whole_forward_and_event():
    reg = SlotRegistry(3)
    a, b = reg.create(), reg.create()
    tx = reg.begin(capacity=capacity(), prefill=[PrefillRequest((), a, 7)], decode=[DecodeRequest((a,), b)])
    assert tx.plan.decode.level_counts[0] == 1  # planned PF population
    assert not reg[a].populated and not reg[b].populated
    event = Completion()
    reg.mark_submitted(tx)
    assert not reg.finish(tx, event)
    assert reg.token_count(a) == 0 and reg[a].revision == 0
    event.ready = True
    assert reg.finish(tx, event)
    assert reg.token_count(a) == 7 and reg.token_count(b) == 1
    assert reg[a].revision == reg[b].revision == 1
    assert reg[a].has_conv and reg[b].has_conv
    with pytest.raises(RuntimeError): reg.finish(tx, event)


def test_invalid_plan_rolls_back_only_new_reservations():
    reg = SlotRegistry(2)
    a, b = reg.create(), reg.create()
    before = reg[a], reg[b]
    with pytest.raises(ValueError):
        reg.begin(capacity=replace(capacity(2), prefill_requests=1),
                  prefill=[PrefillRequest((), a, 1), PrefillRequest((), b, 1)])
    assert (reg[a], reg[b]) == before and reg.free_slots == 2
    with pytest.raises(KeyError):
        reg.begin(capacity=capacity(2), decode=[DecodeRequest((987,), a)])
    assert (reg[a], reg[b]) == before and reg.free_slots == 2


def test_physical_exhaustion_and_profile_overflow_are_distinct():
    reg = SlotRegistry(1)
    a, b = reg.create(), reg.create()
    populate(reg, a)
    with pytest.raises(RuntimeError, match="requested=1, free=0, capacity=1"):
        reg.begin(capacity=capacity(1), decode=[DecodeRequest((a,), b)])
    assert reg[b].slot is None and reg[a].revision == 1
    with pytest.raises(ValueError, match="capacities disagree"):
        reg.begin(capacity=capacity(3), decode=[DecodeRequest((a,), a)])


def test_cancellation_and_single_inflight_guards():
    reg = SlotRegistry(2)
    a = reg.create()
    tx = reg.begin(capacity=capacity(2), decode=[DecodeRequest((), a)])
    with pytest.raises(RuntimeError, match="one model forward"):
        reg.begin(capacity=capacity(2), decode=[DecodeRequest((), a)])
    with pytest.raises(RuntimeError): reg.finish(tx, Completion(True))
    reg.cancel_before_launch(tx)
    assert reg.free_slots == 2 and reg[a].slot is None
    tx = reg.begin(capacity=capacity(2), decode=[DecodeRequest((), a)])
    reg.mark_submitted(tx)
    with pytest.raises(RuntimeError): reg.cancel_before_launch(tx)
    with pytest.raises(RuntimeError): reg.mark_submitted(tx)
    reg.finish(tx, Completion(True))


def test_generation_reuse_clear_and_existing_handle_semantics():
    reg = SlotRegistry(1)
    a = reg.create()
    first = populate(reg, a)
    reg.clear(a)
    assert reg[a].slot is None and reg.token_count(a) == 0
    second = populate(reg, a, 2)
    assert first.slot == second.slot
    assert second.generation > first.generation
    assert second.revision == first.revision + 2  # clear + write
    assert reg.token_count(a) == 2


def test_release_of_inflight_readers_and_writers_is_deferred():
    reg = SlotRegistry(3)
    common, target = reg.create(), reg.create()
    populate(reg, common)
    tx = reg.begin(capacity=capacity(), decode=[DecodeRequest((common,), target)])
    reg.mark_submitted(tx)
    with pytest.raises(RuntimeError, match="in flight"): reg.clear(common)
    reg.release_handle(common); reg.release_handle(target)
    assert reg.free_slots == 1
    assert reg.live_handles == 2
    with pytest.raises(RuntimeError, match="released"): reg[common]
    assert not reg.finish(tx, Completion(False))
    assert reg.finish(tx, Completion(True))
    assert reg.free_slots == 3 and reg.live_handles == 0


def test_failed_write_invalidated_until_completion_then_clear():
    reg = SlotRegistry(2)
    common, target = reg.create(), reg.create()
    populate(reg, common)
    tx = reg.begin(capacity=capacity(2), decode=[DecodeRequest((common,), target)])
    reg.mark_submitted(tx)
    reg.mark_failed(tx)
    assert reg.free_slots == 0
    with pytest.raises(RuntimeError, match="partial failed write"): reg[target]
    with pytest.raises(RuntimeError, match="in flight"): reg.clear(target)
    assert not reg.finish(tx, Completion(False))
    reg.finish(tx, Completion(True))
    assert reg[common].populated
    with pytest.raises(RuntimeError, match="partial failed write"): reg[target]
    reg.clear(target)
    assert reg.free_slots == 1 and not reg[target].populated


def test_fatal_event_failure_poison_runtime_instead_of_retry():
    class BadCompletion:
        def query(self): raise RuntimeError("device failed")
    reg = SlotRegistry(1)
    b = reg.create()
    tx = reg.begin(capacity=capacity(1), decode=[DecodeRequest((), b)])
    reg.mark_submitted(tx)
    with pytest.raises(RuntimeError, match="device failed"): reg.finish(tx, BadCompletion())
    with pytest.raises(RuntimeError, match="rebuild"): reg.create()
    with pytest.raises(RuntimeError, match="rebuild"): reg.clear(b)


def test_gc_protocol_does_not_retain_unbounded_logical_handle_history():
    reg = SlotRegistry(2)
    for _ in range(200):
        b = reg.create()
        populate(reg, b)
        reg.release_handle(b)
        assert reg.live_handles == 0 and reg.free_slots == 2


def test_cross_runtime_or_stale_transaction_is_rejected():
    a, b = SlotRegistry(1), SlotRegistry(1)
    x, y = a.create(), b.create()
    tx = a.begin(capacity=capacity(1), decode=[DecodeRequest((), x)])
    ty = b.begin(capacity=capacity(1), decode=[DecodeRequest((), y)])
    with pytest.raises(RuntimeError): b.mark_submitted(tx)
    b.cancel_before_launch(ty); a.cancel_before_launch(tx)
