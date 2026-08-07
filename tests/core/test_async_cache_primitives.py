"""
Unit tests for the async-cache primitives:
``CacheBlock`` / ``CacheView`` / ``AsyncContext`` and the context-based
``WorkerGroup``.  Pure Python/torch — no model or GPU needed::

    pytest tests/core/test_async_cache_primitives.py -v
"""

from __future__ import annotations

import pytest
import torch
from minisgl.async_cache import AsyncContext, CacheBlock, WorkerGroup
from minisgl.shared_cache import SharedBlock

CPU = torch.device("cpu")


def _filled_block(num_tokens: int, first_page: int = 0, page_size: int = 1) -> CacheBlock:
    block = CacheBlock(CPU, page_size=page_size)
    num_pages = -(-num_tokens // page_size)
    starts = torch.arange(first_page, first_page + num_pages, dtype=torch.int32) * page_size
    block.grow_pages(starts, num_tokens)
    return block


class TestCacheBlock:
    def test_shared_block_alias(self):
        assert SharedBlock is CacheBlock

    def test_token_ids_cleared(self):
        block = _filled_block(3)
        block.token_ids.extend([7, 8, 9])
        block.clear()
        assert block.token_ids == []
        assert block.num_tokens == 0


class TestAsyncContext:
    def test_output_block_defaults_to_last_view_block(self):
        a, b = _filled_block(2), _filled_block(3, first_page=2)
        ctx = AsyncContext(cache_view=[a, b])
        assert ctx.output_block is b
        assert ctx.num_cached_tokens == 5

    def test_output_block_outside_view(self):
        a, out = _filled_block(2), CacheBlock(CPU)
        ctx = AsyncContext(cache_view=[a], output_block=out)
        assert ctx.output_block is out
        assert ctx.num_cached_tokens == 2

    def test_empty_context_rejected(self):
        with pytest.raises(AssertionError):
            AsyncContext(cache_view=[])


class TestWorkerGroup:
    def _two_worker_group(self):
        prompt = _filled_block(4)
        w1 = _filled_block(2, first_page=4)
        w2 = _filled_block(3, first_page=6)
        group = WorkerGroup(
            [
                AsyncContext(cache_view=[prompt, w2, w1]),
                AsyncContext(cache_view=[prompt, w1, w2]),
            ]
        )
        return group, prompt, w1, w2

    def test_workers_form(self):
        group, prompt, w1, w2 = self._two_worker_group()
        assert group.num_workers == 2
        assert group.cache_structure == [[prompt, w2, w1], [prompt, w1, w2]]
        assert group.write_to == [w1, w2]
        assert group.worker_cache_length(0) == 9
        assert group.max_cache_length() == 9

    def test_legacy_form_matches_workers_form(self):
        prompt = _filled_block(4)
        w1 = _filled_block(2, first_page=4)
        w2 = _filled_block(3, first_page=6)
        legacy = WorkerGroup(
            cache_structure=[[prompt, w2, w1], [prompt, w1, w2]],
            write_to=[w1, w2],
        )
        assert legacy.cache_structure == [[prompt, w2, w1], [prompt, w1, w2]]
        assert legacy.write_to == [w1, w2]

    def test_legacy_default_write_to(self):
        a, b, c = _filled_block(1), _filled_block(1, 1), _filled_block(1, 2)
        group = WorkerGroup(cache_structure=[[a, b], [a, c]])
        assert group.write_to == [b, c]

    def test_indexing_and_slicing(self):
        group, _, w1, w2 = self._two_worker_group()
        assert isinstance(group[0], AsyncContext)
        assert group[0].output_block is w1
        assert group[1].output_block is w2

        tail = group[1:]
        assert isinstance(tail, WorkerGroup)
        assert tail.num_workers == 1
        assert tail.write_to == [w2]

        assert len(group) == 2
        assert [ctx.output_block for ctx in group] == [w1, w2]

    def test_duplicate_output_block_rejected(self):
        prompt, w = _filled_block(2), _filled_block(1, first_page=2)
        with pytest.raises(ValueError, match="writing the same block"):
            WorkerGroup(
                [
                    AsyncContext(cache_view=[prompt, w]),
                    AsyncContext(cache_view=[w]),
                ]
            )

    def test_both_forms_rejected(self):
        block = _filled_block(1)
        ctx = AsyncContext(cache_view=[block])
        with pytest.raises(AssertionError):
            WorkerGroup([ctx], cache_structure=[[block]])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
