"""First CPU slice of ForwardPlan: all-layer GDN topology and physical rows.

No tensors, per-layer cache lookups, pointer tables, memoization, or mutations.
Caller supplies a snapshot with reserved write slots and pins its lifetime.
Attention/KV/mRoPE adapters, allocation/commit and the GPU body are NOT here yet.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import accumulate
from typing import Mapping, Sequence


def _uint(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class BlockState:
    block_id: int
    slot: int | None
    populated: bool = False
    has_conv: bool = False
    generation: int = 0
    revision: int = 0

    def __post_init__(self):
        for name in ("block_id", "generation", "revision"):
            _uint(getattr(self, name), name)
        if self.slot is not None:
            _uint(self.slot, "slot")
        if type(self.populated) is not bool or type(self.has_conv) is not bool:
            raise ValueError("population flags must be booleans")
        if (self.populated or self.has_conv) and self.slot is None:
            raise ValueError("populated state must have a slot")
        if self.has_conv and not self.populated:
            raise ValueError("partial-layer/incomplete block state is unsupported")


@dataclass(frozen=True, slots=True)
class PrefillRequest:
    context: tuple[int, ...]
    write_to: int
    length: int

    def __post_init__(self):
        object.__setattr__(self, "context", tuple(self.context))
        for b in (*self.context, self.write_to):
            _uint(b, "block ID")
        _uint(self.length, "prefill length")
        if not self.length:
            raise ValueError("prefill length must be positive")
        if self.write_to in self.context:
            raise ValueError("prefill context must exclude its write block")


@dataclass(frozen=True, slots=True)
class DecodeRequest:
    read_blocks: tuple[int, ...]
    write_to: int

    def __post_init__(self):
        object.__setattr__(self, "read_blocks", tuple(self.read_blocks))
        for b in (*self.read_blocks, self.write_to):
            _uint(b, "block ID")


@dataclass(frozen=True, slots=True)
class PlanCapacity:
    prefill_requests: int
    prefill_tokens: int
    decode_workers: int
    chain_depth: int
    block_slots: int
    chunk_size: int = 64

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _uint(getattr(self, name), name)
        if not self.chunk_size or not self.block_slots:
            raise ValueError("chunk_size and block_slots must be positive")


@dataclass(frozen=True, slots=True)
class AffineNode:
    slot: int
    parent_row: int  # previous level, -1 for first B


@dataclass(frozen=True, slots=True)
class TriePlan:
    levels: tuple[tuple[AffineNode, ...], ...]
    terminals: tuple[tuple[int, int], ...]  # 1-based depth; (0, 0) = zero state

    @property
    def gemm_count(self) -> int:
        return sum(map(len, self.levels[1:]))


def build_trie(chains: Sequence[Sequence[int]]) -> TriePlan:
    """Deduplicate ordered prefixes by slot identity, never tensor values."""
    levels: list[list[AffineNode]] = []
    children: dict[tuple[int, int, int], tuple[int, int]] = {}
    terminals = []
    for chain in chains:
        depth = row = 0
        for slot in chain:
            _uint(slot, "slot")
            key = (depth, row, slot)
            child = children.get(key)
            if child is None:
                if depth == len(levels):
                    levels.append([])
                child = (depth + 1, len(levels[depth]))
                levels[depth].append(AffineNode(slot, row if depth else -1))
                children[key] = child
            depth, row = child
        terminals.append((depth, row))
    return TriePlan(tuple(tuple(level) for level in levels), tuple(terminals))


@dataclass(frozen=True, slots=True)
class PhasePlan:
    # All arrays have capacity shapes except the diagnostic immutable trie.
    trie: TriePlan
    active: tuple[bool, ...]
    write_slots: tuple[int, ...]
    write_fresh: tuple[bool, ...]
    prior_conv_slots: tuple[int, ...]  # -1 = no prior; separate from active
    level_counts: tuple[int, ...]
    node_slots: tuple[int, ...]  # [chain_depth, worker_capacity], flattened
    parent_rows: tuple[int, ...]
    terminal_nodes: tuple[int, ...]  # flattened node; -1 = zero or inactive
    sink_offsets: tuple[int, ...]  # CSR for terminal producer -> worker outputs
    sink_workers: tuple[int, ...]  # padded to worker capacity with -1


def _phase(chains, writes, states, width, max_depth):
    effective = [tuple(states[b].slot for b in chain if states[b].populated) for chain in chains]
    trie = build_trie(effective)
    if len(trie.levels) > max_depth:
        raise ValueError(f"GDN chain capacity exceeded: need {len(trie.levels)}, have {max_depth}")
    slots, parents = [-1] * (width * max_depth), [-1] * (width * max_depth)
    counts = [0] * max_depth
    for depth, level in enumerate(trie.levels):
        counts[depth] = len(level)
        for row, node in enumerate(level):
            index = depth * width + row
            slots[index], parents[index] = node.slot, node.parent_row
    sinks: list[list[int]] = [[] for _ in slots]
    terminals = [-1] * width
    for worker, (depth, row) in enumerate(trie.terminals):
        if depth:
            index = (depth - 1) * width + row
            sinks[index].append(worker)
            terminals[worker] = index
    sink_workers = [w for node_sinks in sinks for w in node_sinks]
    conv = [next((states[b].slot for b in reversed(chain) if states[b].has_conv), -1)
            for chain in chains]
    n = len(writes)
    return PhasePlan(
        trie=trie, active=(True,) * n + (False,) * (width - n),
        write_slots=tuple(states[b].slot for b in writes) + (-1,) * (width - n),
        write_fresh=tuple(not states[b].populated for b in writes) + (False,) * (width - n),
        prior_conv_slots=tuple(conv) + (-1,) * (width - n),
        level_counts=tuple(counts), node_slots=tuple(slots), parent_rows=tuple(parents),
        terminal_nodes=tuple(terminals),
        sink_offsets=(0, *accumulate(map(len, sinks))),
        sink_workers=tuple(sink_workers) + (-1,) * (width - len(sink_workers)),
    )


@dataclass(frozen=True, slots=True)
class RowPlan:
    active: tuple[bool, ...]
    request: tuple[int, ...]  # D request IDs begin at R_cap, not actual_R
    local_token: tuple[int, ...]
    prefill_offsets: tuple[int, ...]
    output_rows: tuple[int, ...]  # PF slots then D slots; -1 for inactive
    chunk_request: tuple[int, ...]
    chunk_local: tuple[int, ...]  # -1 for inactive chunk
    chunk_offsets: tuple[int, ...]

    @property
    def chunk_indices(self):
        """FLA's interleaved [request, local_chunk] table, prepared on CPU."""
        return tuple(value for pair in zip(self.chunk_request, self.chunk_local) for value in pair)

    @property
    def prefill_recipe(self):
        """Old layer's finite precision recipes; a device value, not a graph key.

        0: ragged combined FLA; 1: lone token chunk output + FP32 capture;
        2: multiple one-token requests, FP32-normalized recurrent + capture.
        """
        lengths=[b-a for a,b in zip(self.prefill_offsets,self.prefill_offsets[1:]) if b>a]
        return (0 if not lengths or any(n!=1 for n in lengths) else (1 if len(lengths)==1 else 2),)


def _rows(prefill, decode_count, cap):
    size = cap.prefill_tokens + cap.decode_workers
    owner, local, active = [-1] * size, [-1] * size, [False] * size
    outputs = [-1] * (cap.prefill_requests + cap.decode_workers)
    offsets, chunk_offsets = [0], [0]
    chunk_owner, chunk_local = [], []
    for r, request in enumerate(prefill):
        start = offsets[-1]
        stop = start + request.length
        owner[start:stop] = [r] * request.length
        local[start:stop] = range(request.length)
        active[start:stop] = [True] * request.length
        offsets.append(stop)
        outputs[r] = stop - 1
        chunks = (request.length + cap.chunk_size - 1) // cap.chunk_size
        chunk_owner.extend([r] * chunks)
        chunk_local.extend(range(chunks))
        chunk_offsets.append(len(chunk_owner))
    offsets.extend([offsets[-1]] * (cap.prefill_requests - len(prefill)))
    chunk_offsets.extend([chunk_offsets[-1]] * (cap.prefill_requests - len(prefill)))
    for d in range(decode_count):
        index = cap.prefill_tokens + d
        owner[index], local[index], active[index] = cap.prefill_requests + d, 0, True
        outputs[cap.prefill_requests + d] = index
    chunk_cap = (cap.prefill_tokens + cap.chunk_size - 1) // cap.chunk_size + cap.prefill_requests
    pad = chunk_cap - len(chunk_owner)
    return RowPlan(tuple(active), tuple(owner), tuple(local), tuple(offsets), tuple(outputs),
                   tuple(chunk_owner) + (-1,) * pad, tuple(chunk_local) + (-1,) * pad,
                   tuple(chunk_offsets))


@dataclass(frozen=True, slots=True)
class WriteIntent:
    block_id: int
    slot: int
    generation: int
    old_revision: int
    new_revision: int
    added_tokens: int
    phase: str


@dataclass(frozen=True, slots=True)
class ForwardPlan:
    capacity: PlanCapacity
    mode: str
    blocks: tuple[BlockState, ...]  # snapshot, no live objects or tensors
    prefill: PhasePlan
    decode: PhasePlan
    rows: RowPlan
    writes: tuple[WriteIntent, ...]
    prefill_requests: tuple[PrefillRequest, ...]
    decode_requests: tuple[DecodeRequest, ...]

    @property
    def graph_key(self):
        # Model/device/precision/recipe are owned by the future runner instance.
        # Actual occupancy is data. A mixed-capacity program must keep its graph
        # when the PF or decode group temporarily becomes empty.
        cap=self.capacity
        mode=("mixed" if cap.prefill_tokens and cap.decode_workers else "prefill"
              if cap.prefill_tokens else "decode" if cap.decode_workers else "empty")
        return mode, cap


def prepare_forward(
    blocks: Mapping[int, BlockState], *, capacity: PlanCapacity,
    prefill: Sequence[PrefillRequest] = (), decode: Sequence[DecodeRequest] = (),
) -> ForwardPlan:
    """Plan both phases once, without reading/mutating any per-layer data.

    This function neither reserves GPU slots nor commits writes. An eventual
    runtime must validate/pin generations and retain the plan until completion.
    """
    prefill, decode = tuple(prefill), tuple(decode)
    if len(prefill) > capacity.prefill_requests or len(decode) > capacity.decode_workers:
        raise ValueError("request capacity exceeded")
    if sum(r.length for r in prefill) > capacity.prefill_tokens:
        raise ValueError("prefill token capacity exceeded")
    pf_writes, dec_writes = [r.write_to for r in prefill], [r.write_to for r in decode]
    writes = pf_writes + dec_writes
    if len(set(writes)) != len(writes):
        raise ValueError("duplicate writers within forward")
    pf_ids = set(pf_writes)
    if any(pf_ids.intersection(r.context) for r in prefill):
        raise ValueError("prefill reads another same-phase write")
    pf_chains = [(*r.context, r.write_to) for r in prefill]
    dec_chains = [r.read_blocks for r in decode]
    referenced = dict.fromkeys(b for chain in (*pf_chains, *dec_chains) for b in chain)
    referenced.update(dict.fromkeys(writes))
    snapshot = {}
    slot_owners = {}
    for b in referenced:
        state = blocks[b]  # one lookup per referenced block, independent of L
        if state.block_id != b:
            raise ValueError("registry key and block identity disagree")
        if state.slot is not None:
            if state.slot >= capacity.block_slots:
                raise ValueError("slot exceeds pool capacity")
            if state.slot in slot_owners:
                raise ValueError("two distinct live blocks share a physical slot")
            slot_owners[state.slot] = b
        snapshot[b] = state
    if any(snapshot[b].slot is None for b in writes):
        raise ValueError("write slot must be reserved before preparing GPU metadata")
    pf_plan = _phase(pf_chains, pf_writes, snapshot, capacity.prefill_requests, capacity.chain_depth)
    post_prefill = dict(snapshot)
    for b in pf_writes:
        post_prefill[b] = replace(snapshot[b], populated=True, has_conv=True)
    dec_plan = _phase(dec_chains, dec_writes, post_prefill, capacity.decode_workers, capacity.chain_depth)
    intents = tuple(WriteIntent(b, snapshot[b].slot, snapshot[b].generation,
                               snapshot[b].revision, snapshot[b].revision + 1, length, phase)
                    for b, length, phase in
                    [*((r.write_to, r.length, "prefill") for r in prefill),
                     *((r.write_to, 1, "decode") for r in decode)])
    mode = "mixed" if prefill and decode else "prefill" if prefill else "decode" if decode else "empty"
    return ForwardPlan(capacity, mode, tuple(snapshot.values()), pf_plan, dec_plan,
                       _rows(prefill, len(decode), capacity), intents, prefill, decode)
