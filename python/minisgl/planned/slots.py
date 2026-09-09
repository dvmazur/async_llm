"""CPU slot ownership and all-layer commit protocol (internal runtime API).

The session adapter will own logical handles and call release_handle when a
handle dies. Public clear/free semantics must not be confused with GC of the
handle: clearing leaves a valid empty logical block. No GPU tensors live here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

from .forward_plan import BlockState, ForwardPlan, PlanCapacity, WriteIntent, prepare_forward


@dataclass(frozen=True, slots=True)
class MutationPlan:
    """Non-decoder append/merge transaction: one all-layer ownership update."""
    blocks: tuple[BlockState, ...]
    writes: tuple[WriteIntent, ...]


@dataclass(slots=True)
class Transaction:
    plan: ForwardPlan | MutationPlan
    reserved: tuple[tuple[int, BlockState], ...]
    stage: str = "prepared"


class SlotRegistry:
    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("positive slot capacity required")
        self.capacity = capacity
        self._free = deque(range(capacity))
        self._slot_generation = [0] * capacity
        self._states: dict[int, BlockState] = {}
        self._lengths: dict[int, int] = {}
        self._next_block_id = 0
        self._pending: Transaction | None = None
        self._invalid: set[int] = set()
        self._drop_after: set[int] = set()
        self._poisoned = False

    def _healthy(self):
        if self._poisoned:
            raise RuntimeError("runtime completion failed; rebuild the runtime/context")

    def __getitem__(self, block_id):
        self._healthy()
        if block_id in self._invalid:
            raise RuntimeError("block has a partial failed write; clear/rebuild it first")
        if block_id in self._drop_after:
            raise RuntimeError("logical handle was released")
        return self._states[block_id]

    @property
    def free_slots(self): return len(self._free)

    @property
    def live_handles(self): return len(self._states)

    def token_count(self, block_id):
        self[block_id]
        return self._lengths[block_id]

    def create(self):
        self._healthy()
        block_id = self._next_block_id
        self._next_block_id += 1
        self._states[block_id] = BlockState(block_id, None)
        self._lengths[block_id] = 0
        return block_id

    def _pinned(self, block_id):
        return self._pending is not None and any(b.block_id == block_id for b in self._pending.plan.blocks)

    def clear(self, block_id):
        self._healthy()
        if self._pinned(block_id):
            raise RuntimeError("block in flight: wait for completion before clear")
        old = self._states[block_id]
        if old.slot is not None:
            self._free.append(old.slot)
        self._states[block_id] = BlockState(block_id, None, generation=old.generation, revision=old.revision+1)
        self._lengths[block_id] = 0
        self._invalid.discard(block_id)

    def release_handle(self, block_id):
        self._healthy()
        if block_id not in self._states:
            return
        if self._pinned(block_id):
            self._drop_after.add(block_id)
            return
        self.clear(block_id)
        del self._states[block_id], self._lengths[block_id]
        self._drop_after.discard(block_id)

    def begin(self, *, capacity: PlanCapacity, prefill=(), decode=()):
        self._healthy()
        if self._pending is not None:
            raise RuntimeError("only one model forward may be in flight")
        if capacity.block_slots != self.capacity:
            raise ValueError("profile and physical pool capacities disagree")
        prefill, decode = tuple(prefill), tuple(decode)
        writes = tuple(dict.fromkeys(r.write_to for r in (*prefill, *decode)))
        missing = [b for b in writes if self[b].slot is None]
        if len(missing) > len(self._free):
            raise RuntimeError(f"GDN pool exhausted: requested={len(missing)}, free={len(self._free)}, capacity={self.capacity}")
        reserved = []
        try:
            for b in missing:
                old = self[b]
                slot = self._free.popleft()
                self._slot_generation[slot] += 1
                reserved.append((b, old))
                self._states[b] = replace(old, slot=slot, generation=self._slot_generation[slot])
            plan = prepare_forward(self, capacity=capacity, prefill=prefill, decode=decode)
        except Exception:
            self._rollback_reservations(reserved)
            raise
        self._pending = Transaction(plan, tuple(reserved))
        return self._pending

    def begin_mutation(self, *, readers, destination, num_tokens):
        """Reserve/pin a nonempty result; do not publish its new state yet.

        Snapshot includes the destination before mutation, even for self-append
        or destination==right. Its all-layer revision changes only at finish.
        """
        self._healthy()
        if self._pending is not None:
            raise RuntimeError("only one model forward may be in flight")
        if type(num_tokens) is not int or num_tokens<1:
            raise ValueError("nonempty mutation result required; clear empty results")
        ids=tuple(dict.fromkeys((*readers,destination)))
        snapshot={b:self[b] for b in ids}
        old=snapshot[destination]
        reserved=[]
        if old.slot is None:
            if not self._free:raise RuntimeError("GDN pool exhausted before mutation")
            slot=self._free.popleft()
            self._slot_generation[slot]+=1
            reserved.append((destination,old))
            self._states[destination]=replace(old,slot=slot,generation=self._slot_generation[slot])
            snapshot[destination]=self._states[destination]
        target=snapshot[destination]
        intent=WriteIntent(destination,target.slot,target.generation,target.revision,target.revision+1,
                           num_tokens-self._lengths[destination],"mutation")
        self._pending=Transaction(MutationPlan(tuple(snapshot.values()),(intent,)),tuple(reserved))
        return self._pending

    def _rollback_reservations(self, reserved):
        for b, old in reversed(reserved):
            self._free.appendleft(self._states[b].slot)
            self._states[b] = old

    def _check(self, transaction):
        self._healthy()
        if transaction is not self._pending:
            raise RuntimeError("transaction is stale or belongs to another runtime")

    def mark_submitted(self, transaction):
        """Call BEFORE enqueuing the first operation that can write state."""
        self._check(transaction)
        if transaction.stage != "prepared":
            raise RuntimeError("transaction was already submitted")
        transaction.stage = "submitted"

    def cancel_before_launch(self, transaction):
        self._check(transaction)
        if transaction.stage != "prepared":
            raise RuntimeError("cannot roll back after GPU writes may have started")
        self._rollback_reservations(transaction.reserved)
        self._pending = None
        transaction.stage = "cancelled"
        self._release_dropped()

    def mark_failed(self, transaction):
        self._check(transaction)
        if transaction.stage not in ("submitted", "failed"):
            raise RuntimeError("use cancel_before_launch for unsubmitted work")
        transaction.stage = "failed"
        self._invalid.update(w.block_id for w in transaction.plan.writes)
        # Keep slots pinned until the completion event, even on a Python error.

    def finish(self, transaction, completion):
        """Nonblocking commit after an event recorded AFTER all GPU readers.

        The owner must include retained-output copies in that event dependency.
        This is not a device synchronize or a per-layer commit.
        """
        self._check(transaction)
        if transaction.stage not in ("submitted", "failed"):
            raise RuntimeError("cannot finish unsubmitted work")
        try:
            ready = completion.query()
        except Exception:
            self._poisoned = True
            self._invalid.update(w.block_id for w in transaction.plan.writes)
            raise
        if not ready:
            return False
        if transaction.stage == "submitted":
            for w in transaction.plan.writes:
                old = self._states[w.block_id]
                if (old.slot, old.generation, old.revision) != (w.slot, w.generation, w.old_revision):
                    self._poisoned = True
                    raise RuntimeError("state changed while its transaction was in flight")
            for w in transaction.plan.writes:
                self._states[w.block_id] = replace(self._states[w.block_id],
                    populated=True, has_conv=True, revision=w.new_revision)
                self._lengths[w.block_id] += w.added_tokens
            transaction.stage = "committed"
        else:
            transaction.stage = "failed_complete"
        self._pending = None
        self._release_dropped()
        return True

    def _release_dropped(self):
        for b in tuple(self._drop_after):
            self.release_handle(b)
