"""Literal pre-optimization topology builder (2026-09-07).

Only the return-record types are shared. Do not replace this oracle's algorithm
with the production planner or its presence-mask cache.
"""
from typing import List, Sequence, Tuple
from minisgl.shared_cache.gdn import _AffinePrefixPlan, _AffineTrieNode


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


def write_eligibility(cache_structure, write_to):
    # Literal eligibility calculation from old begin_decode_state.
    write_ids = [id(block) for block in write_to]
    write_set = set(write_ids)
    return [bool(
        chain and chain[-1] is target
        and write_ids.count(id(target)) == 1
        and all(id(block) not in write_set for block in chain[:-1])
        and all(hasattr(block, "linear_affine_revision") for block in chain)
    ) for chain, target in zip(cache_structure, write_to)]
