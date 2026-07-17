# Async cache scheduler plan

This document describes basic primitives for the implementation of async cache
scheduling. Note that this doc uses the term "async cache" instead of
"async reasoning".

## Main primitives

```python
@dataclass
class CacheBlock:
    ...
```

```python
CacheView = list[CacheBlock]
```

```python
@dataclass
class AsyncContext
    cache_view: CacheView
    output_block: CacheBlock
```

```python
@dataclass
class WorkerGroup:
    workers: list[AsyncContext]

    def __getitem__(self, key):
        """Index a specific worker or slice a couple of them"""
```

## Scheduler API

> NOTE: Make the mapping to existing API primitives more obvious

```python
class AsyncCacheEngine:
    def prefill_block(self, token_ids: list[int]) -> CacheBlock:
        ...

    def decode_step(self, context: AsyncContext) -> ?:
        ...
```

## Scheduler internals

The scheduler's internals should be switched entirely to processing async cache
requests, not that vanilla decoding/prefill requests can be expressed through
async cache terms if a `CacheView` of a single `CacheBlock` is passed with the
same `CacheBlock` as the output.

### Decoding request queue

The scheduler keeps a queue of `AsyncContext`s which is drained at every engine
tick. Once a new batch is formed, separate `AsyncContext` requests are used to
create a new `WorkerGroup` that is then executed.

