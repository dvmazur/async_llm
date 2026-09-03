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
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import torch

from .gdn_affine import init_gdn_affine, update_affine_summary
from .gdn_compose_cache import GDNComposeStateCache

if TYPE_CHECKING:
    from .shared_block import CacheBlock


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
) -> torch.Tensor:
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
    return torch.cat(per_worker, dim=0)


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
) -> torch.Tensor:
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
        )

    node_by_key = {
        key: node
        for frontier, keys in zip(plan.frontiers, keys_by_depth)
        for node, key in zip(frontier, keys)
    }
    computed = []
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
    return per_worker[0] if len(per_worker) == 1 else torch.cat(per_worker, dim=0)


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
        self.gdn_storage_bytes = gdn_storage_bytes
        cache_ratio = float(os.environ.get("MINISGL_GDN_COMPOSE_CACHE_RATIO", "1.6"))
        self.configure_compose_cache(cache_ratio)

    def configure_compose_cache(self, ratio: float) -> None:
        """Reset the cache to *ratio* of the engine's allocated GDN state pool."""

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
        self.prefill_segments = None if prefill_segments is None else list(prefill_segments)

    @property
    def num_workers(self) -> int:
        return len(self.cache_structure)

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
        plan = _plan_affine_prefixes(self.cache_structure, lin_idx)
        if not plan.frontiers:
            return None

        if self.compose_state_cache is not None:
            S_block = _evaluate_affine_prefix_plan_cached(
                plan,
                lin_idx=lin_idx,
                num_heads=self.num_heads,
                d_k=self.head_k_dim,
                d_v=self.head_v_dim,
                device=self.device,
                write_to=self.write_to,
                cache=self.compose_state_cache,
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
            S_block = _terminal_states(
                plan,
                states_by_depth,
                num_heads=self.num_heads,
                d_k=self.head_k_dim,
                d_v=self.head_v_dim,
                device=self.device,
            )
        if state_v_first:
            return S_block.to(dtype=dtype)  # [W, H, d_v, d_k]
        S_hf = S_block.transpose(-1, -2).contiguous()  # [W, H, d_k, d_v]
        return S_hf.to(dtype=dtype)

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

            A, B = capture_gdn_affine_pointer_update(
                previous,
                key_f[:, 0].contiguous(),
                value_f[:, 0].contiguous(),
                alpha_f[:, 0].contiguous(),
                beta_f[:, 0].contiguous(),
            )
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

        for w, target in enumerate(targets):
            pair = (A[w : w + 1], B[w : w + 1])
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
