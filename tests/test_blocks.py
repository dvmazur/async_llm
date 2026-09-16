import asyncio

import pytest

from experiment_runner.blocks import BlockHandle
from test_policy import FakeEngine


def test_each_owner_closes_independently_and_last_owner_frees():
    async def run():
        engine = FakeEngine()
        original = await BlockHandle.create(engine, 'block')
        raw = original.raw
        reader = original.share()
        assert reader is not original and reader.raw is raw
        producer = object()
        original.producer = producer
        assert reader.producer is producer
        await original.aclose()
        await original.aclose()  # idempotent; cannot drop reader's reference
        assert reader.raw is raw and id(raw) in engine.live_blocks
        with pytest.raises(RuntimeError, match='closed block handle'):
            original.share()
        with pytest.raises(RuntimeError, match='closed block handle'):
            _ = original.raw
        async with reader:
            assert reader.raw is raw
        assert not engine.live_blocks
        await reader.aclose()
    asyncio.run(run())


def test_context_releases_on_error_and_snapshot_is_independent():
    async def run():
        engine = FakeEngine()
        with pytest.raises(ValueError, match='consumer failed'):
            async with await BlockHandle.create(engine, 'source') as source:
                source.raw.num_tokens = 7
                snapshot = await source.snapshot('snapshot')
                assert snapshot.raw is not source.raw
                assert snapshot.raw.num_tokens == 7
                raise ValueError('consumer failed')
        assert list(engine.live_blocks.values()) == [snapshot.raw]
        await snapshot.aclose()
        assert not engine.live_blocks
    asyncio.run(run())


def test_snapshot_retains_source_during_async_copy():
    async def run():
        entered, resume = asyncio.Event(), asyncio.Event()
        class Engine(FakeEngine):
            async def merge_blocks(self, left, right):
                entered.set()
                await resume.wait()
                assert id(left) in self.live_blocks
                return await super().merge_blocks(left, right)
        engine = Engine()
        source = await BlockHandle.create(engine, 'source')
        copy = asyncio.create_task(source.snapshot('copy'))
        await entered.wait()
        await source.aclose()
        resume.set()
        snapshot = await copy
        assert list(engine.live_blocks.values()) == [snapshot.raw]
        await snapshot.aclose()
        assert not engine.live_blocks
    asyncio.run(asyncio.wait_for(run(), 3))


def test_failed_snapshot_drops_temporary_and_source_reference():
    class Engine(FakeEngine):
        async def merge_blocks(self, left, right):
            raise RuntimeError('copy failed')
    async def run():
        engine = Engine()
        source = await BlockHandle.create(engine, 'source')
        with pytest.raises(RuntimeError, match='copy failed'):
            await source.snapshot('copy')
        assert list(engine.live_blocks.values()) == [source.raw]
        await source.aclose()
        assert not engine.live_blocks
    asyncio.run(run())


def test_rejected_free_keeps_last_handle_retryable():
    class Engine(FakeEngine):
        reject = True
        async def free_block(self, block):
            if self.reject:
                raise RuntimeError('block still queued')
            await super().free_block(block)
    async def run():
        engine = Engine()
        handle = await BlockHandle.create(engine, 'block')
        with pytest.raises(RuntimeError, match='still queued'):
            await handle.aclose()
        assert not handle.closed and id(handle.raw) in engine.live_blocks
        engine.reject = False
        await handle.aclose()
        assert not engine.live_blocks
    asyncio.run(run())
