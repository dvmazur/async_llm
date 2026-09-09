"""
Async-reasoning support for Qwen3.5 Gated DeltaNet (GDN) linear-attention layers.

Full-attention layers reuse ``SharedCacheAttention`` (token-addressable KV blocks).
GDN layers instead keep a monolithic recurrent state, so a worker's chain of blocks
is composed via the **affine summary** ``(A_hat, B_hat)`` stored on each ``CacheBlock``
(see ``gdn_affine``): the initial recurrent state for a chain is
``S0 = 0 @ A_chain + B_chain = B_chain`` (block convention), transposed to the HF
kernel convention ``[B, H, d_k, d_v]`` at the model interface.

``SharedCacheGDN`` is created once by the session (it knows the GDN dims) and handed the
current forward's ``(cache_structure, write_to)`` via ``set_context``.  The patched
``Qwen3_5GatedDeltaNet.forward`` reads it off ``get_global_ctx().gdn_ar``.
"""

from __future__ import annotations

import os
from collections import Counter
from copy import copy
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import torch

from .gdn_affine import init_gdn_affine, update_affine_summary
from .gdn_compose_cache import GDNComposeStateCache
from .gdn_successor_cache import (
    GDNSuccessorCache, assemble_state_rows,
)

if TYPE_CHECKING:
    from .shared_block import CacheBlock


_StateParts = torch.Tensor | List[torch.Tensor]


def _materialize_state_parts(parts: _StateParts) -> torch.Tensor:
    """Preserve a ready packed batch, otherwise gather terminal rows once."""
    if isinstance(parts, torch.Tensor):
        return parts
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


@dataclass(frozen=True)
class _AffineTrieNode:
    """One unique effective block prefix at a particular affine depth.

    ``parent_index`` addresses the preceding depth frontier.  It is ``-1`` for
    depth-one nodes, whose parent is the implicit zero-state root.
    """

    parent_index: int
    block: "CacheBlock"


@dataclass(frozen=True)
class _AffinePrefixPlan:
    """Pure host-side topology plan for batched affine-prefix evaluation."""

    frontiers: Tuple[Tuple[_AffineTrieNode, ...], ...]
    # Per worker: ``(depth, row-within-that-depth)``.  ``(0, 0)`` is the root.
    worker_terminals: Tuple[Tuple[int, int], ...]


def _block_affine_signature(block: "CacheBlock", lin_idx: int) -> tuple[int, int, int, int]:
    """Stable identity of one block's current affine contents."""

    pair = block.linear_affine[lin_idx]
    block_id = int(getattr(block, "block_id", id(block)))
    revisions = getattr(block, "linear_affine_revision", {})
    revision = int(revisions.get(lin_idx, 0))
    # Pointers also detect direct test/user assignments which bypass the
    # production CacheBlock.set_linear_affine revision bump.
    return block_id, revision, pair[0].data_ptr(), pair[1].data_ptr()


def _prefix_keys(
    plan: _AffinePrefixPlan, lin_idx: int
) -> Tuple[Tuple[tuple[tuple[int, int, int, int], ...], ...], ...]:
    """Return each trie node's versioned full-prefix key, grouped by depth."""

    keys_by_depth = []
    previous = ()
    for depth, frontier in enumerate(plan.frontiers):
        current = []
        for node in frontier:
            parent = () if depth == 0 else previous[node.parent_index]
            current.append(parent + (_block_affine_signature(node.block, lin_idx),))
        keys_by_depth.append(tuple(current))
        previous = keys_by_depth[-1]
    return tuple(keys_by_depth)


def _plan_affine_prefixes(
    cache_structure: Sequence[Sequence["CacheBlock"]], lin_idx: int
) -> _AffinePrefixPlan:
    """Build a trie of unique affine-bearing block prefixes.

    Blocks without an affine for ``lin_idx`` are mathematical no-ops and do not
    consume a trie depth.  Children are keyed by object identity, so equal-valued
    but distinct cache blocks remain distinct topology nodes.
    """

    frontiers: List[List[_AffineTrieNode]] = []
    # (parent depth, parent row, child block identity) -> (child depth, child row)
    children: dict[Tuple[int, int, int], Tuple[int, int]] = {}
    worker_terminals: List[Tuple[int, int]] = []

    for chain in cache_structure:
        depth, row = 0, 0  # implicit root / zero state
        for block in chain:
            if block.linear_affine.get(lin_idx) is None:
                continue

            child_key = (depth, row, id(block))
            child = children.get(child_key)
            if child is None:
                child_depth = depth + 1
                while len(frontiers) < child_depth:
                    frontiers.append([])
                child_row = len(frontiers[child_depth - 1])
                frontiers[child_depth - 1].append(
                    _AffineTrieNode(parent_index=-1 if depth == 0 else row, block=block)
                )
                child = (child_depth, child_row)
                children[child_key] = child
            depth, row = child
        worker_terminals.append((depth, row))

    return _AffinePrefixPlan(
        frontiers=tuple(tuple(frontier) for frontier in frontiers),
        worker_terminals=tuple(worker_terminals),
    )


class _GDNHostPlan:
    """One context's topology, not a cache of tensors, pointers or revisions.

    A layer may reuse the previous layer's effective trie only if exactly the
    same blocks currently have affines. Re-read that mask on every lookup:
    prefill can fill an empty block before decode in the same mixed forward.
    Keep the two most recent masks/plans so alternating layer populations do
    not rebuild every time. This remains bounded for manually reused contexts.
    """

    def __init__(self, chains, key):
        self.key = key
        self.chains = tuple(tuple(chain) for chain in chains)
        self.blocks = tuple({id(b): b for chain in self.chains for b in chain}.values())
        chain_ids, write_ids = key
        counts = Counter(write_ids)
        writers = set(write_ids)
        self.write_eligible = tuple(
            bool(chain) and chain[-1] == target and counts[target] == 1
            and all(block not in writers for block in chain[:-1])
            for chain, target in zip(chain_ids, write_ids)
        )
        self._presence = None
        self._prefix_plan = None
        self._previous = None

    def prefix_plan(self, lin_idx):
        presence = tuple(b.linear_affine.get(lin_idx) is not None for b in self.blocks)
        if self._prefix_plan is None or presence != self._presence:
            if self._previous is not None and self._previous[0] == presence:
                next_plan = self._previous[1]
            else:
                next_plan = _plan_affine_prefixes(self.chains, lin_idx)
            self._previous = (self._presence, self._prefix_plan)
            self._prefix_plan = next_plan
            self._presence = presence
        return self._prefix_plan


def _apply_affine_frontier(
    parent_states: torch.Tensor,
    frontier: Sequence[_AffineTrieNode],
    lin_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """Evaluate one non-root trie frontier without constructing composite A."""

    parent_indices = [node.parent_index for node in frontier]
    pairs = [node.block.linear_affine[lin_idx] for node in frontier]
    A_rows = [pair[0].to(dtype=torch.float32, device=device) for pair in pairs]
    B_rows = [pair[1].to(dtype=torch.float32, device=device) for pair in pairs]
    if parent_states.is_cuda:
        from minisgl.kernel import apply_gdn_affine_pointer_frontier

        return apply_gdn_affine_pointer_frontier(parent_states, parent_indices, A_rows, B_rows)

    # CPU remains a simple numerical/reference path; production CUDA always
    # uses the same pointer kernel regardless of A/B storage placement.
    child_rows = [
        torch.matmul(parent_states[node.parent_index : node.parent_index + 1], A_row).add_(B_row)
        for node, A_row, B_row in zip(frontier, A_rows, B_rows)
    ]
    return torch.cat(child_rows, dim=0)


def _evaluate_affine_prefix_plan(
    plan: _AffinePrefixPlan,
    *,
    lin_idx: int,
    num_heads: int,
    d_k: int,
    d_v: int,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    """Evaluate trie frontiers and return FP32 block-convention states by depth."""

    if not plan.frontiers:
        return ()

    first_rows = [
        node.block.linear_affine[lin_idx][1].to(dtype=torch.float32, device=device)
        for node in plan.frontiers[0]
    ]
    first_states = first_rows[0] if len(first_rows) == 1 else torch.cat(first_rows, dim=0)
    expected_tail = (num_heads, d_v, d_k)
    if first_states.shape[1:] != expected_tail:
        raise ValueError(
            "invalid GDN affine B shape: "
            f"expected [N, {num_heads}, {d_v}, {d_k}], got {list(first_states.shape)}"
        )

    states_by_depth = [first_states]
    for frontier in plan.frontiers[1:]:
        states_by_depth.append(
            _apply_affine_frontier(states_by_depth[-1], frontier, lin_idx, device)
        )
    return tuple(states_by_depth)


def _terminal_states(
    plan: _AffinePrefixPlan,
    states_by_depth: Sequence[torch.Tensor],
    *,
    num_heads: int,
    d_k: int,
    d_v: int,
    device: torch.device,
    materialize: bool = True,
) -> _StateParts:
    """Gather worker terminal rows from a fully evaluated prefix plan."""

    final_depth = len(states_by_depth)
    terminals_are_final_frontier = len(plan.worker_terminals) == states_by_depth[-1].shape[
        0
    ] and all(
        depth == final_depth and row == worker
        for worker, (depth, row) in enumerate(plan.worker_terminals)
    )
    if terminals_are_final_frontier:
        return states_by_depth[-1]

    zero = None
    per_worker = []
    for depth, row in plan.worker_terminals:
        if depth == 0:
            if zero is None:
                zero = torch.zeros(
                    1,
                    num_heads,
                    d_v,
                    d_k,
                    dtype=torch.float32,
                    device=device,
                )
            per_worker.append(zero)
        else:
            per_worker.append(states_by_depth[depth - 1][row : row + 1])
    return torch.cat(per_worker, dim=0) if materialize else per_worker


def _evaluate_affine_prefix_plan_cached(
    plan: _AffinePrefixPlan,
    *,
    lin_idx: int,
    num_heads: int,
    d_k: int,
    d_v: int,
    device: torch.device,
    write_to: Sequence["CacheBlock"],
    cache: GDNComposeStateCache,
    materialize: bool = True,
) -> _StateParts:
    """Evaluate only suffixes after the deepest resident prefix per worker."""

    keys_by_depth = _prefix_keys(plan, lin_idx)
    terminal_keys = [
        None if depth == 0 else keys_by_depth[depth - 1][row]
        for depth, row in plan.worker_terminals
    ]
    state_by_key: dict[tuple, torch.Tensor] = {}
    required = set()
    any_hit = False
    for terminal in terminal_keys:
        if terminal is None:
            continue
        deepest_hit = 0
        # A depth-one prefix is already returned as its B summary without a
        # compose GEMM. Caching it cannot save math and may force a large final
        # cat, so only prefixes containing at least two affine blocks qualify.
        for depth in range(len(terminal), 1, -1):
            key = terminal[:depth]
            cached = cache.get(key)
            if cached is not None:
                state_by_key[key] = cached
                deepest_hit = depth
                any_hit = True
                break
        required.update(terminal[:depth] for depth in range(deepest_hit + 1, len(terminal) + 1))

    write_ids = {id(block) for block in write_to}
    if not any_hit:
        states_by_depth = _evaluate_affine_prefix_plan(
            plan,
            lin_idx=lin_idx,
            num_heads=num_heads,
            d_k=d_k,
            d_v=d_v,
            device=device,
        )
        for depth, (frontier, keys) in enumerate(zip(plan.frontiers, keys_by_depth)):
            if depth == 0:
                continue
            for row, (node, key) in enumerate(zip(frontier, keys)):
                cache.consider(
                    key,
                    states_by_depth[depth][row : row + 1],
                    current_write_prefix=id(node.block) in write_ids,
                )
        return _terminal_states(
            plan,
            states_by_depth,
            num_heads=num_heads,
            d_k=d_k,
            d_v=d_v,
            device=device,
            materialize=materialize,
        )

    node_by_key = {
        key: node
        for frontier, keys in zip(plan.frontiers, keys_by_depth)
        for node, key in zip(frontier, keys)
    }
    computed = []
    terminal_output = None
    for depth, keys in enumerate(keys_by_depth, start=1):
        missing = [key for key in keys if key in required and key not in state_by_key]
        if not missing:
            continue
        if depth == 1:
            for key in missing:
                state_by_key[key] = (
                    node_by_key[key]
                    .block.linear_affine[lin_idx][1]
                    .to(dtype=torch.float32, device=device)
                )
                computed.append(key)
            continue

        parents = [state_by_key[key[:-1]] for key in missing]
        pairs = [node_by_key[key].block.linear_affine[lin_idx] for key in missing]
        from minisgl.kernel import apply_gdn_affine_pointer_nodes

        output = apply_gdn_affine_pointer_nodes(
            parents,
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
        )
        # This launch already produced exactly the requested worker order.
        # Do not split it into row views and copy the same batch back together.
        # Reordered, duplicate, mixed-depth or partially cached terminals still
        # use the general assembly below.
        if missing == terminal_keys:
            terminal_output = output
        for row, key in enumerate(missing):
            state_by_key[key] = output[row : row + 1]
            computed.append(key)

    for key in computed:
        if len(key) < 2:
            continue
        node = node_by_key[key]
        cache.consider(
            key,
            state_by_key[key],
            current_write_prefix=id(node.block) in write_ids,
        )

    if terminal_output is not None:
        return terminal_output

    zero = None
    per_worker = []
    for terminal in terminal_keys:
        if terminal is None:
            if zero is None:
                zero = torch.zeros(
                    1,
                    num_heads,
                    d_v,
                    d_k,
                    dtype=torch.float32,
                    device=device,
                )
            per_worker.append(zero)
        else:
            per_worker.append(state_by_key[terminal])
    return _materialize_state_parts(per_worker) if materialize else per_worker


@dataclass
class MixedSharedCacheGDN:
    split: int
    prefill: SharedCacheGDN
    decode: SharedCacheGDN


class SharedCacheGDN:
    """Per-forward composer/capturer of GDN affine summaries over worker chains."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_dim: int,
        conv_kernel: int,
        device: torch.device,
        gdn_storage_bytes: Optional[int] = None,
        num_linear_layers: Optional[int] = None,
    ) -> None:
        self.device = device
        self.num_heads = num_heads  # H (post GQA-repeat, = linear_num_value_heads)
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_dim = conv_dim
        self.conv_kernel = conv_kernel
        # Set per forward pass.
        self.cache_structure: List[List[CacheBlock]] = []
        self.write_to: List[CacheBlock] = []
        # Prefill only: per-request token counts, in row order, when the forward
        # batches several prefills.  ``None`` for decode (one token per worker)
        # and for a single-request prefill.
        self.prefill_segments: Optional[List[int]] = None
        self.prefill_cu_seqlens: Optional[torch.Tensor] = None
        self.prefill_cu_seqlens_cpu: Optional[torch.Tensor] = None
        self._host_plan: Optional[_GDNHostPlan] = None
        self._compose_miss_view = None
        self.gdn_storage_bytes = gdn_storage_bytes
        self.num_linear_layers = num_linear_layers
        cache_ratio = float(os.environ.get("MINISGL_GDN_COMPOSE_CACHE_RATIO", "1.6"))
        self.configure_compose_cache(cache_ratio)
        self.configure_successor_cache(
            float(os.environ.get("MINISGL_GDN_SUCCESSOR_CACHE_RATIO", "1.0"))
        )

    def configure_successor_cache(self, ratio: float) -> None:
        """Independent bounded cache; zero disables last-decode state reuse."""
        self._compose_miss_view = None
        self.successor_cache_ratio = float(ratio)
        budget = int((self.gdn_storage_bytes or 0) * ratio)
        self.successor_state_cache = GDNSuccessorCache(budget) if budget > 0 else None

    def _decode_signature(self, chain, lin_idx):
        # Include empty blocks too: filling one later must invalidate the key.
        return tuple(
            _block_affine_signature(block, lin_idx)
            if lin_idx in block.linear_affine else (
                int(getattr(block, "block_id", id(block))),
                block.linear_affine_revision.get(lin_idx, 0), None, None,
            )
            for block in chain
        )

    def begin_decode_state(self, lin_idx, *, materialize=True):
        """Return a v-first initial state and a pre-write validation ticket.

        Only this decode entry point consults successors. Prefill always uses
        the original compose API; its writes naturally invalidate saved keys.
        A pointer-capable consumer may request strong row views instead of a
        concatenation. The default tensor API and validation tickets are unchanged.
        """
        cache = self.successor_state_cache
        if cache is None or self.prefill_segments is not None:
            if not materialize:
                return self._compose_initial_recurrent_parts(lin_idx), None
            return self.compose_initial_recurrent_state(
                lin_idx, torch.float32, state_v_first=True
            ), None

        host_plan = self._get_host_plan()
        tickets, rows = [], []
        for chain, target, structurally_eligible in zip(
            self.cache_structure, self.write_to, host_plan.write_eligible
        ):
            eligible = (
                structurally_eligible
                and all(hasattr(block, "linear_affine_revision") for block in chain)
            )
            signature = self._decode_signature(chain, lin_idx) if eligible else None
            tickets.append(None if signature is None else (id(target), signature))
            rows.append(None if signature is None else cache.get(target, lin_idx, signature))

        missing = [w for w, row in enumerate(rows) if row is None]
        if len(missing) == self.num_workers:
            if not materialize:
                return self._compose_initial_recurrent_parts(lin_idx), tickets
            return self.compose_initial_recurrent_state(
                lin_idx, torch.float32, state_v_first=True
            ), tickets
        if missing:
            indices = tuple(missing)
            cached_view = self._compose_miss_view
            if cached_view is None or cached_view[0] is not host_plan or cached_view[1] != indices:
                view = self.context_view(
                    [self.cache_structure[w] for w in missing], [self.write_to[w] for w in missing]
                )
                self._compose_miss_view = (host_plan, indices, view)
            else:
                view = cached_view[2]
            # Compose must NOT first concatenate its misses: the recurrent
            # consumer needs one batch containing both those rows and hits.
            # A pointer recurrent consumer can use those strong rows directly.
            parts = view._compose_initial_recurrent_parts(lin_idx)
            if parts is None:
                zero = torch.zeros(
                    1, self.num_heads, self.head_v_dim, self.head_k_dim,
                    dtype=torch.float32, device=self.device,
                )
                parts = [zero] * len(missing)
            elif isinstance(parts, torch.Tensor):
                parts = list(parts.split(1))
            resolved = [row.tensor() if row is not None else None for row in rows]
            for worker, part in zip(missing, parts, strict=True):
                resolved[worker] = part
            return (torch.cat(resolved, dim=0) if materialize else resolved), tickets
        return assemble_state_rows(rows, materialize=materialize), tickets

    def finish_decode_state(self, lin_idx, state, tickets):
        """Commit only after capture, under the resulting affine revision.

        Never label an old recurrent state with a newly mutated upstream. The
        own write must advance exactly once; all upstream signatures must match.
        """
        cache = self.successor_state_cache
        if cache is None or tickets is None:
            return
        records = []
        for row, (chain, target, ticket) in enumerate(
            zip(self.cache_structure, self.write_to, tickets)
        ):
            if ticket is None or id(target) != ticket[0] or not chain or chain[-1] is not target:
                continue
            before = ticket[1]
            after = self._decode_signature(chain, lin_idx)
            if (after[:-1] == before[:-1] and after[-1][0] == before[-1][0]
                and after[-1][1] == before[-1][1] + 1
                and lin_idx in target.linear_affine):
                records.append((row, target, after))
        cache.put(lin_idx, state, records)

    def configure_compose_cache(self, ratio: float) -> None:
        """Reset the cache to *ratio* of the engine's allocated GDN state pool."""

        self._compose_miss_view = None
        self.compose_cache_ratio = float(ratio)
        if (
            self.device.type != "cuda"
            or self.gdn_storage_bytes is None
            or self.compose_cache_ratio <= 0
        ):
            self.compose_state_cache = None
            return
        budget = max(1, int(self.gdn_storage_bytes * self.compose_cache_ratio))
        self.compose_state_cache = GDNComposeStateCache(budget)

    def set_context(
        self,
        cache_structure: Sequence[Sequence["CacheBlock"]],
        write_to: Sequence["CacheBlock"],
        prefill_segments: Optional[Sequence[int]] = None,
    ) -> None:
        self.cache_structure = [list(c) for c in cache_structure]
        self.write_to = list(write_to)
        self._host_plan = None
        self._compose_miss_view = None
        self.prefill_segments = None if prefill_segments is None else list(prefill_segments)
        if self.prefill_segments is not None and len(self.prefill_segments) > 1:
            offsets = [0, *accumulate(self.prefill_segments)]
            self.prefill_cu_seqlens_cpu = torch.tensor(offsets, dtype=torch.long)
            self.prefill_cu_seqlens = self.prefill_cu_seqlens_cpu.to(self.device)
        else:
            self.prefill_cu_seqlens = None
            self.prefill_cu_seqlens_cpu = None

    def context_view(self, cache_structure, write_to, prefill_segments=None):
        """Per-forward metadata view, sharing blocks and the bounded compose cache.

        No state tensors/pools are copied. Distinct views prevent the decode
        context from overwriting prefill offsets during a mixed model pass.
        """
        view = copy(self)
        view.set_context(cache_structure, write_to, prefill_segments)
        return view

    @property
    def num_workers(self) -> int:
        return len(self.cache_structure)

    def _get_host_plan(self) -> _GDNHostPlan:
        # Public lists have historically been mutable, including direct test/
        # user assignments. Check identities, not tensor equality or only list
        # identity, so an in-place reorder cannot silently reuse the old trie.
        key = (tuple(tuple(map(id, chain)) for chain in self.cache_structure),
               tuple(map(id, self.write_to)))
        if self._host_plan is None or self._host_plan.key != key:
            self._host_plan = _GDNHostPlan(self.cache_structure, key)
            self._compose_miss_view = None
        return self._host_plan

    # ------------------------------------------------------------------
    # Reads: compose prior state for the current forward
    # ------------------------------------------------------------------

    def has_previous_affine(self, lin_idx: int) -> bool:
        for chain in self.cache_structure:
            for block in chain:
                if lin_idx in block.linear_affine:
                    return True
        return False

    def compose_initial_recurrent_state(
        self, lin_idx: int, dtype: torch.dtype, *, state_v_first: bool = False
    ) -> Optional[torch.Tensor]:
        """Evaluate each worker's affine-prefix chain from the zero state.

        By default returns ``[num_workers, H, d_k, d_v]`` in HF convention.
        With ``state_v_first=True``, returns the native affine-cache convention
        ``[num_workers, H, d_v, d_k]`` without a materializing transpose. Returns
        ``None`` if no block in any chain has an affine for this layer.
        """
        parts = self._compose_initial_recurrent_parts(lin_idx)
        if parts is None:
            return None
        S_block = _materialize_state_parts(parts)
        if state_v_first:
            return S_block.to(dtype=dtype)  # [W, H, d_v, d_k]
        S_hf = S_block.transpose(-1, -2).contiguous()  # [W, H, d_k, d_v]
        return S_hf.to(dtype=dtype)

    def _compose_initial_recurrent_parts(self, lin_idx: int) -> Optional[_StateParts]:
        """Native FP32 terminals, deferring any worker-order materialization.

        A ready ordered frontier is returned as one tensor. Other layouts are
        single-row views (including one shared zero row for empty workers).
        These are temporary strong references, not a new persistent cache.
        """
        plan = self._get_host_plan().prefix_plan(lin_idx)
        if not plan.frontiers:
            return None

        if self.compose_state_cache is not None:
            return _evaluate_affine_prefix_plan_cached(
                plan,
                lin_idx=lin_idx,
                num_heads=self.num_heads,
                d_k=self.head_k_dim,
                d_v=self.head_v_dim,
                device=self.device,
                write_to=self.write_to,
                cache=self.compose_state_cache,
                materialize=False,
            )
        else:
            states_by_depth = _evaluate_affine_prefix_plan(
                plan,
                lin_idx=lin_idx,
                num_heads=self.num_heads,
                d_k=self.head_k_dim,
                d_v=self.head_v_dim,
                device=self.device,
            )
            return _terminal_states(
                plan,
                states_by_depth,
                num_heads=self.num_heads,
                d_k=self.head_k_dim,
                d_v=self.head_v_dim,
                device=self.device,
                materialize=False,
            )

    def prior_conv_states(self, lin_idx: int) -> Optional[torch.Tensor]:
        """Per-worker most-recent conv window along the chain, ``[W, conv_dim, k]``
        (zeros for a worker whose chain has none), or ``None`` if all are empty."""
        per_worker: List[Optional[torch.Tensor]] = []
        present: Optional[torch.Tensor] = None
        for chain in self.cache_structure:
            found: Optional[torch.Tensor] = None
            for block in reversed(chain):
                c = block.linear_conv_state.get(lin_idx)
                if c is not None:
                    found = c
                    break
            if found is not None:
                present = found
            per_worker.append(found)
        if present is None:
            return None
        # Match the stored states' dtype: stacking a float32 filler with bf16
        # states would silently promote the whole window and break the conv.
        zeros = torch.zeros(
            self.conv_dim, self.conv_kernel, device=self.device, dtype=present.dtype
        )
        filled = [(c if c is not None else zeros).to(device=self.device) for c in per_worker]
        return torch.stack(filled, dim=0)  # [W, conv_dim, k]

    # ------------------------------------------------------------------
    # Writes: capture this forward's tokens into the write-target blocks
    # ------------------------------------------------------------------

    def affine_scan_initial_state(
        self,
        lin_idx: int,
        *,
        num_heads: int,
        d_k: int,
        d_v: int,
        workers: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Pack the current block summaries as ``[A_hat; B_hat]`` scan state.

        In block convention the augmented state is ``[W,H,d_k+d_v,d_k]``.
        Running the GDN recurrence with augmented values ``[0; v]`` updates its
        first rows exactly like ``A_hat`` and its remaining rows like ``B_hat``.
        Fresh blocks start from ``[I; 0]``; an append starts from the summary
        already stored on the selected write block.
        """
        targets = self.write_to if workers is None else [self.write_to[w] for w in workers]
        state = torch.zeros(
            len(targets),
            num_heads,
            d_k + d_v,
            d_k,
            dtype=torch.float32,
            device=self.device,
        )
        eye = torch.eye(d_k, dtype=torch.float32, device=self.device)
        state[:, :, :d_k, :] = eye
        for row, target in enumerate(targets):
            pair = target.linear_affine.get(lin_idx)
            if pair is None:
                continue
            A_hat, B_hat = pair
            state[row, :, :d_k, :].copy_(A_hat[0])
            state[row, :, d_k:, :].copy_(B_hat[0])
        return state

    def store_affine_scan_state(
        self,
        lin_idx: int,
        state: torch.Tensor,
        *,
        d_k: int,
        workers: Optional[Sequence[int]] = None,
    ) -> None:
        """Store pointer-compose-compatible ``A_hat``/``B_hat`` row views.

        FLA returns one augmented ``[W,H,d_k+d_v,d_k]`` tensor. Production
        Qwen blocks scatter their row into a block-owned all-layer slab: A/B
        remain contiguous for pointer compose, while block lifetimes no longer
        pin unrelated workers from the same Hogwild batch. Fake/rectangular
        reference blocks retain the historical packed-batch fallback.
        """
        targets = self.write_to if workers is None else [self.write_to[w] for w in workers]
        if state.shape[0] != len(targets):
            raise ValueError(
                f"affine scan returned {state.shape[0]} rows for {len(targets)} targets"
            )
        A_view = state[:, :, :d_k, :]
        B_view = state[:, :, d_k:, :]
        output_pairs = self._block_owned_affine_pairs(
            targets,
            lin_idx=lin_idx,
            num_heads=state.shape[1],
            d_k=d_k,
            d_v=B_view.shape[-2],
        )
        if output_pairs is not None:
            if state.is_cuda:
                from minisgl.kernel import store_gdn_affine_pointer

                store_gdn_affine_pointer(state, output_pairs, d_k=d_k)
            else:
                for row, (A_out, B_out) in enumerate(output_pairs):
                    A_out.copy_(A_view[row : row + 1])
                    B_out.copy_(B_view[row : row + 1])
        else:
            if B_view.shape[-2] == d_k:
                # Reference/fake blocks preserve the historical packed batch.
                packed = torch.stack((A_view, B_view), dim=0)
                A_batch, B_batch = packed[0], packed[1]
            else:
                A_batch = A_view.contiguous()
                B_batch = B_view.contiguous()
            output_pairs = [
                (A_batch[row : row + 1], B_batch[row : row + 1])
                for row in range(len(targets))
            ]
        assert output_pairs is not None
        for target, pair in zip(targets, output_pairs):
            if hasattr(target, "set_linear_affine"):
                target.set_linear_affine(lin_idx, pair)
            else:
                target.linear_affine[lin_idx] = pair

    def _block_owned_affine_pairs(
        self,
        targets: Sequence["CacheBlock"],
        *,
        lin_idx: int,
        num_heads: int,
        d_k: int,
        d_v: int,
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Return per-block slab views when production dimensions support them."""

        if (
            self.num_linear_layers is None
            or d_k != d_v
            or not all(hasattr(target, "affine_storage_pair") for target in targets)
        ):
            return None
        return [
            target.affine_storage_pair(
                lin_idx,
                num_layers=self.num_linear_layers,
                num_heads=num_heads,
                d_k=d_k,
                d_v=d_v,
            )
            for target in targets
        ]

    def capture_token_affines(
        self,
        lin_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        l2norm_eps: float = 1e-6,
        workers: Optional[Sequence[int]] = None,
    ) -> None:
        """Accumulate per-token affine updates into each worker's write block.

        ``key/value`` are ``[W, seq, H, d]``; ``alpha/beta`` are ``[W, seq, H]``.
        The key is L2-normed to match the kernel's ``use_qk_l2norm_in_kernel=True``.

        *workers* selects which write blocks the ``W`` rows correspond to
        (default: all of them, in order).  A batched prefill has a different
        token count per request, so it captures one request at a time.

        The rank-1 update is batched over workers (one call per token, not per
        worker), so the only Python loop is the inherently-sequential token scan
        (length 1 for decode; the block length for a prefill).
        """
        targets = self.write_to if workers is None else [self.write_to[w] for w in workers]
        W, seq, H, dk = key.shape
        dv = value.shape[-1]
        key_f = key.float()
        key_f = key_f * torch.rsqrt((key_f * key_f).sum(dim=-1, keepdim=True) + l2norm_eps)
        value_f, alpha_f, beta_f = value.float(), alpha.float(), beta.float()

        previous = [target.linear_affine.get(lin_idx) for target in targets]
        output_pairs = self._block_owned_affine_pairs(
            targets,
            lin_idx=lin_idx,
            num_heads=H,
            d_k=dk,
            d_v=dv,
        )
        pointer_compatible = key.is_cuda and all(
            pair is None
            or (
                pair[0].shape == (1, H, dk, dk)
                and pair[1].shape == (1, H, dv, dk)
                and pair[0].dtype == torch.float32
                and pair[1].dtype == torch.float32
                and pair[0].device == key.device
                and pair[1].device == key.device
                and pair[0].is_contiguous()
                and pair[1].is_contiguous()
            )
            for pair in previous
        )
        if seq == 1 and pointer_compatible:
            from minisgl.kernel import capture_gdn_affine_pointer_update

            packed_output = capture_gdn_affine_pointer_update(
                previous,
                key_f[:, 0].contiguous(),
                value_f[:, 0].contiguous(),
                alpha_f[:, 0].contiguous(),
                beta_f[:, 0].contiguous(),
                output_pairs=output_pairs,
            )
            if packed_output is not None:
                A, B = packed_output
        else:
            # General CPU/multi-token prefill reference and fallback path.
            A, B = init_gdn_affine(
                batch_size=W,
                num_heads=H,
                d_k=dk,
                d_v=dv,
                dtype=torch.float32,
                device=key.device,
            )
            for w, pair in enumerate(previous):
                if pair is not None:
                    A[w] = pair[0][0].to(dtype=torch.float32, device=key.device)
                    B[w] = pair[1][0].to(dtype=torch.float32, device=key.device)

            for t in range(seq):
                A, B = update_affine_summary(
                    A_hat=A,
                    B_hat=B,
                    k=key_f[:, t],
                    v=value_f[:, t],
                    alpha=alpha_f[:, t],
                    beta=beta_f[:, t],
                )

            if output_pairs is not None:
                for w, (A_out, B_out) in enumerate(output_pairs):
                    A_out.copy_(A[w : w + 1])
                    B_out.copy_(B[w : w + 1])

        if output_pairs is None:
            output_pairs = [
                (A[w : w + 1], B[w : w + 1]) for w in range(len(targets))
            ]
        for target, pair in zip(targets, output_pairs):
            if hasattr(target, "set_linear_affine"):
                target.set_linear_affine(lin_idx, pair)
            else:
                target.linear_affine[lin_idx] = pair

    def set_conv_states(
        self, lin_idx: int, conv: torch.Tensor, workers: Optional[Sequence[int]] = None
    ) -> None:
        """Store per-worker conv windows ``[W, conv_dim, k]`` into write blocks
        (``workers`` selects which, as in :meth:`capture_token_affines`)."""
        targets = self.write_to if workers is None else [self.write_to[w] for w in workers]
        for w, target in enumerate(targets):
            target.linear_conv_state[lin_idx] = conv[w].detach().clone()


__all__ = ["SharedCacheGDN"]
