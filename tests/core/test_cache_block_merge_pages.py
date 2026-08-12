"""Exact cache-block merge tests for non-trivial paged layouts.

The full-model merge test uses FlashInfer, whose engine page size is one.  This
module exercises the same ``SharedCacheSession.merge_blocks`` implementation
with page size four, where the right block may have to move into the unused tail
of the left block.  Values are deliberately synthetic so every KV slot can be
checked exactly rather than through a logits tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from minisgl.kvcache.page_allocator import PageAllocator
from minisgl.shared_cache import CacheBlock, SharedCacheSession


CPU = torch.device("cpu")
PAGE_SIZE = 4


class TinyKVCache:
    def __init__(self, num_pages: int = 32, num_layers: int = 2) -> None:
        shape = (num_layers, num_pages, PAGE_SIZE, 1, 2)
        self.keys = torch.full(shape, -1.0)
        self.values = torch.full(shape, -1.0)
        self.num_layers = num_layers

    def k_cache(self, layer_idx: int) -> torch.Tensor:
        return self.keys[layer_idx]

    def v_cache(self, layer_idx: int) -> torch.Tensor:
        return self.values[layer_idx]


class AdditiveRope:
    """Exact stand-in that makes the corrected key range directly observable."""

    @staticmethod
    def _rope(keys: torch.Tensor, corrections: torch.Tensor) -> torch.Tensor:
        return keys + corrections.to(keys.dtype)[:, None, None]


@dataclass
class BlockSnapshot:
    token_ids: list[int]
    pages: list[int]
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    affine: dict[int, tuple[torch.Tensor, torch.Tensor]]
    conv: dict[int, torch.Tensor]


def make_session() -> SharedCacheSession:
    # Initializing a full SharedCacheSession requires a model.  merge_blocks
    # itself only needs these four real session components.
    session = object.__new__(SharedCacheSession)
    session.device = CPU
    session.page_size = PAGE_SIZE
    session.page_allocator = PageAllocator(32, PAGE_SIZE, CPU)
    session.kv_cache = TinyKVCache()
    session.sc_attn = AdditiveRope()
    return session


def fill_block(
    session: SharedCacheSession,
    num_tokens: int,
    *,
    value_base: int,
) -> CacheBlock:
    block = session.create_block()
    if num_tokens:
        pages, slots = session._alloc_token_storage(num_tokens, write_to=block)
        block.grow_pages(pages, num_tokens)
        block.token_ids = list(range(value_base, value_base + num_tokens))
        slots = slots.to(torch.int64)
        token_values = torch.arange(num_tokens, dtype=torch.float32) + value_base
        for layer_idx in range(session.kv_cache.num_layers):
            layer_values = token_values + 100 * layer_idx
            keys = torch.stack((layer_values, layer_values + 0.25), dim=-1)[:, None, :]
            values = torch.stack((layer_values + 0.5, layer_values + 0.75), dim=-1)[
                :, None, :
            ]
            session.kv_cache.k_cache(layer_idx).reshape(-1, 1, 2).index_copy_(
                0, slots, keys
            )
            session.kv_cache.v_cache(layer_idx).reshape(-1, 1, 2).index_copy_(
                0, slots, values
            )

        # Exercise mRoPE span bookkeeping independently of token count.
        block.mrope_span_override = max(1, num_tokens - 2)

        # Both-sided summaries exercise composition.  The one-sided summaries
        # exercise clone-versus-transfer behavior under each consume flag.
        scalar = float(value_base)
        block.linear_affine[0] = (
            torch.tensor([[[[scalar + 1]]]]),
            torch.tensor([[[[scalar + 2]]]]),
        )
        block.linear_affine[value_base] = (
            torch.tensor([[[[scalar + 3]]]]),
            torch.tensor([[[[scalar + 4]]]]),
        )
        block.linear_conv_state[0] = torch.tensor([[scalar, scalar + 1]])
    return block


def snapshot(session: SharedCacheSession, block: CacheBlock) -> BlockSnapshot:
    slots = block.token_slots_tensor().to(torch.int64)
    return BlockSnapshot(
        token_ids=block.token_ids.copy(),
        pages=block.page_starts.copy(),
        keys=[
            session.kv_cache.k_cache(layer_idx)
            .reshape(-1, 1, 2)
            .index_select(0, slots)
            .clone()
            for layer_idx in range(session.kv_cache.num_layers)
        ],
        values=[
            session.kv_cache.v_cache(layer_idx)
            .reshape(-1, 1, 2)
            .index_select(0, slots)
            .clone()
            for layer_idx in range(session.kv_cache.num_layers)
        ],
        affine={
            layer_idx: (pair[0].clone(), pair[1].clone())
            for layer_idx, pair in block.linear_affine.items()
        },
        conv={
            layer_idx: state.clone()
            for layer_idx, state in block.linear_conv_state.items()
        },
    )


def assert_block_unchanged(
    session: SharedCacheSession, block: CacheBlock, expected: BlockSnapshot
) -> None:
    actual = snapshot(session, block)
    assert actual.token_ids == expected.token_ids
    assert actual.pages == expected.pages
    for actual_layer, expected_layer in zip(actual.keys, expected.keys):
        torch.testing.assert_close(actual_layer, expected_layer, rtol=0, atol=0)
    for actual_layer, expected_layer in zip(actual.values, expected.values):
        torch.testing.assert_close(actual_layer, expected_layer, rtol=0, atol=0)
    for layer_idx, expected_pair in expected.affine.items():
        torch.testing.assert_close(block.linear_affine[layer_idx][0], expected_pair[0])
        torch.testing.assert_close(block.linear_affine[layer_idx][1], expected_pair[1])
    for layer_idx, expected_state in expected.conv.items():
        torch.testing.assert_close(block.linear_conv_state[layer_idx], expected_state)


@pytest.mark.parametrize("consume_left", [False, True])
@pytest.mark.parametrize("consume_right", [False, True])
@pytest.mark.parametrize(
    ("left_tokens", "right_tokens"),
    [
        (5, 10),  # right crosses the partial left page and several own pages
        (5, 2),  # right fits completely into the partial left page
        (8, 6),  # page-aligned concatenation needs no compaction
        (0, 6),  # empty left
        (5, 0),  # empty right
    ],
)
def test_merge_page_size_four_exact(
    left_tokens: int,
    right_tokens: int,
    consume_left: bool,
    consume_right: bool,
) -> None:
    session = make_session()
    free_pages_before = session.page_allocator.num_free_pages
    left = fill_block(session, left_tokens, value_base=10)
    right = fill_block(session, right_tokens, value_base=50)
    left_before = snapshot(session, left)
    right_before = snapshot(session, right)
    left_span = left.mrope_span
    right_span = right.mrope_span
    left_tail_capacity = left.free_tail

    merged = session.merge_blocks(
        left,
        right,
        consume_left=consume_left,
        consume_right=consume_right,
    )

    assert merged.num_tokens == left_tokens + right_tokens
    assert merged.token_ids == left_before.token_ids + right_before.token_ids
    assert merged.mrope_span == left_span + right_span

    left_page_count = len(left_before.pages)
    right_tokens_after_left_tail = max(0, right_tokens - left_tail_capacity)
    right_output_pages = (right_tokens_after_left_tail + PAGE_SIZE - 1) // PAGE_SIZE
    merged_left_pages = merged.page_starts[:left_page_count]
    merged_right_pages = merged.page_starts[left_page_count:]
    if consume_left:
        assert merged_left_pages == left_before.pages
    else:
        assert set(merged_left_pages).isdisjoint(left_before.pages)
    if consume_right:
        assert merged_right_pages == right_before.pages[:right_output_pages]
        returned_pages = set(session.page_allocator.free_page_starts.tolist())
        assert set(right_before.pages[right_output_pages:]) <= returned_pages
    else:
        assert set(merged_right_pages).isdisjoint(right_before.pages)

    merged_after = snapshot(session, merged)
    for layer_idx in range(session.kv_cache.num_layers):
        expected_keys = torch.cat(
            (
                left_before.keys[layer_idx],
                right_before.keys[layer_idx] + left_span,
            )
        )
        expected_values = torch.cat(
            (left_before.values[layer_idx], right_before.values[layer_idx])
        )
        torch.testing.assert_close(merged_after.keys[layer_idx], expected_keys, rtol=0, atol=0)
        torch.testing.assert_close(
            merged_after.values[layer_idx], expected_values, rtol=0, atol=0
        )

    if left_tokens and right_tokens:
        left_pair = left_before.affine[0]
        right_pair = right_before.affine[0]
        torch.testing.assert_close(
            merged.linear_affine[0][0], left_pair[0] @ right_pair[0]
        )
        torch.testing.assert_close(
            merged.linear_affine[0][1], left_pair[1] @ right_pair[0] + right_pair[1]
        )
        torch.testing.assert_close(merged.linear_conv_state[0], right_before.conv[0])

    if consume_left:
        assert left.is_consumed
        with pytest.raises(RuntimeError, match="consumed"):
            session.free_block(left)
        with pytest.raises(RuntimeError, match="consumed"):
            session.merge_blocks(left, merged)
    else:
        assert_block_unchanged(session, left, left_before)

    if consume_right:
        assert right.is_consumed
        with pytest.raises(RuntimeError, match="consumed"):
            session.free_block(right)
        with pytest.raises(RuntimeError, match="consumed"):
            session.merge_blocks(merged, right)
    else:
        assert_block_unchanged(session, right, right_before)

    session.free_block(merged)
    if not consume_left:
        session.free_block(left)
    if not consume_right:
        session.free_block(right)
    assert session.page_allocator.num_free_pages == free_pages_before


def test_merge_rejects_consuming_same_block() -> None:
    session = make_session()
    block = fill_block(session, 5, value_base=10)

    with pytest.raises(ValueError, match="itself"):
        session.merge_blocks(block, block, consume_left=True)

    assert not block.is_consumed
    assert block.num_tokens == 5
    session.free_block(block)
