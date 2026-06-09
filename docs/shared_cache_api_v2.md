# Shared Cache API — V2 Plan

This document captures the known improvements deferred from V1.

## 1. Inter-group decode batching

**V1 behaviour**: the scheduler runs one `WorkerGroup`'s decode step per loop
iteration. If two API clients both have active shared-cache sessions, their
steps are serialized across consecutive loop iterations.

**V2 goal**: accumulate all pending `WorkerGroup`s at the start of each
iteration and fuse them into a single `session.decode_step()` call (i.e. a
single engine forward pass). This requires:

- `_pending_sc_decodes` to store `(group, input_ids, uid, remaining_steps)`
  tuples so they can all be consumed together.
- `SharedCacheSession.decode_step()` extended (or a new batch variant added)
  to accept a list of `(WorkerGroup, input_ids)` pairs and merge them into one
  `Batch`. The tricky part is that each group may have a different
  `cache_structure` depth; padding / per-group table indices need care.
- Reply routing: after the fused forward pass, split logits back by group and
  send per-group `SharedCacheReply` events.

## 2. Shared-cache + normal decode co-batching

**V1 behaviour**: a shared-cache decode step occupies the engine for that
iteration; no normal (scheduler-owned) requests are batched alongside it.

**V2 goal**: merge shared-cache decode requests into the same `Batch` as
normal decode requests so the engine runs one forward pass that serves both.
This is deeper than inter-group batching and requires:

- `Req` objects created for shared-cache workers (they already have a `Req`
  shape inside `SharedCacheSession._build_batch`; expose them to the
  scheduler's decode manager).
- Page table entries and `out_loc` for shared-cache workers filled by
  `SharedCacheSession._fill_page_tables`, then stitched into the scheduler's
  combined `Batch`.
- Post-forward split: shared-cache logits go to `SharedCacheSession._record_writes`
  and generate `SharedCacheReply`; normal-request logits go to the existing
  detokenizer path.

This also unblocks CUDA-graph capture across the combined batch, which is
currently only done for normal requests.

## 3. Dynamic page pool coordination

**V1 behaviour**: pages are statically partitioned at startup via
`shared_cache_page_budget` in `ServerArgs`. Neither side can use the other's
headroom even when idle.

**V2 goal**: let the scheduler lend idle pages to `SharedCacheSession` and
reclaim them when a normal request needs them. Requires:

- A shared page allocator (or a thin borrow/return protocol) sitting between
  the scheduler's cache and `SharedCacheSession._free_pages`.
- Eviction policy: if the shared-cache session holds borrowed pages that a
  normal request needs, either block the normal request briefly or evict the
  least-recently-used shared block (invalidating its `block_id`; callers get
  an error and must re-prefill).
