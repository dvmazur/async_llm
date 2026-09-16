"""Shared ownership of engine blocks, without an allocator or block registry.

Each owner holds a separate handle: share() acquires it, aclose() releases it.
Use async with for scoped ownership. No GPU calls from __del__: freeing a block
requires the live engine/event loop, so cleanup is explicit and deterministic.
"""
from dataclasses import dataclass
from itertools import count


@dataclass
class _Ownership:
    backend: object
    raw: object
    name: str
    users: int = 1
    producer: object = None


class BlockHandle:
    _serial = count(1)

    def __init__(self, ownership):
        self._ownership = ownership
        self.closed = False

    @classmethod
    def _adopt(cls, backend, raw, name):
        return cls(_Ownership(backend, raw, f'{name}:{next(cls._serial)}'))

    @classmethod
    async def create(cls, backend, name):
        return cls._adopt(backend, await backend.create_block(), name)

    def _check_open(self):
        if self.closed:
            raise RuntimeError(f'closed block handle: {self.name}')

    @property
    def raw(self):
        self._check_open()
        return self._ownership.raw

    @property
    def name(self):
        return self._ownership.name

    @property
    def producer(self):
        return self._ownership.producer

    @producer.setter
    def producer(self, value):
        self._check_open()
        self._ownership.producer = value

    def share(self):
        self._check_open()
        self._ownership.users += 1
        return BlockHandle(self._ownership)

    async def aclose(self):
        if self.closed:
            return
        state = self._ownership
        self.closed = True
        state.users -= 1
        if state.users == 0:
            try:
                await state.backend.free_block(state.raw)
            except BaseException:
                state.users += 1
                self.closed = False
                raise
            state.raw = state.producer = None

    async def __aenter__(self):
        self._check_open()
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def snapshot(self, name):
        # Keep source alive even if its original owner closes during the copy.
        async with self.share() as source:
            backend = self._ownership.backend
            async with await BlockHandle.create(backend, name + '/empty') as empty:
                raw = await backend.merge_blocks(source.raw, empty.raw)
            return self._adopt(backend, raw, name)
