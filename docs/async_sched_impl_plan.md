# Async cache scheduler — implementation plan

Implementation plan for `ASYNC_SCHED_DESIGN.md`: the scheduler/API side of
async cache. Decisions locked (2026-07-15):

- **Full switch** — async-cache contexts become the scheduler's only internal
  request representation; vanilla traffic rides on single-block contexts.
  Radix prefix cache, chunked prefill, CUDA-graph decode and overlap
  scheduling are dropped for now (research fork; can be re-added behind the
  unified path later).
- **Asyncio-native tick** — single process, single thread; the engine tick is
  an async task in the user's event loop. No background thread, no ZMQ for
  the async API.
- **Engine-side sampling** — `async_generate` yields token ids; `forbid_ids`
  becomes a logit-bias feature of the decode request. `prefill_block` keeps
  returning last-token logits (covers the probe).
- **Scope: demo port only** — no mixed prefill+decode merge (v1 alternates
  prefill/decode ticks, probe briefly stalls decode exactly as today's demo
  does), no HTTP/server async API. The `dmazur/mixed-prefill-decode-async-context`
  branch plugs in later as a pure tick-policy upgrade.

## Where we start

- `shared_cache/session.py` — `SharedCacheSession.prefill_block` (`:198`) /
  `decode_step` (`:294`) already execute exactly the forwards the scheduler
  needs, eagerly (`_forward`, `:448`), sharing the engine's `PageAllocator`.
  This stays the **execution backend**; the new scheduler owns *when* it runs.
- `shared_cache/worker_group.py` — lock-step `WorkerGroup(cache_structure,
  write_to)`; becomes derived from `AsyncContext`s.
- `scheduler/` — vanilla managers (`PrefillManager`, `DecodeManager`,
  `CacheManager`, `TableManager`) get replaced by the async-cache queue; the
  IO mixin / message layer stays so server mode keeps working (slower, no
  radix cache).
- Prior art: `async-reasoning-scheduler` branch (scheduler-owned service) and
  `docs/async_thoughts_architecture.md` — same policy/mechanism split, but a
  lock-step blocking API. We reuse its ideas (borrowed-page integrity checks,
  builder), not its code, since the redesign changes the request model.

## Phase 0 — Primitives (mechanical, no behavior change)

New module `python/minisgl/async_cache/` (name per design doc; re-exported
from `minisgl`):

```python
CacheBlock = SharedBlock          # rename; SharedBlock kept as alias for tests
CacheView  = list[CacheBlock]

@dataclass
class AsyncContext:
    cache_view: CacheView
    output_block: CacheBlock
```

- `CacheBlock` gains host-side `token_ids: list[int]`, appended on
  prefill/decode. The design doc's probe reads `thinker_block.token_ids`, and
  `ends_with_double_newline(block)` needs it; today the demo tracks these
  lists by hand.
- `WorkerGroup` is rebuilt as `workers: list[AsyncContext]` with
  `__getitem__` (index/slice). `cache_structure` / `write_to` become derived
  properties so `SharedCacheAttention.prepare` and `SharedCacheGDN` are
  untouched.
- Validation moves here: distinct `output_block`s per group; `output_block`
  must appear in its own `cache_view` **or** trigger the aux path exactly as
  today (write-block-not-in-view is supported by the attention).

Gate: existing `tests/core/test_shared_cache*.py` green after the rename
(they may construct groups via a compat shim).

## Phase 1 — `AsyncCacheEngine`: queues + synchronous tick

`python/minisgl/scheduler/async_engine.py`. Deliberately **sync** — the
asyncio layer (Phase 2) is a thin wrapper — so all GPU logic is testable
without an event loop.

Request records (internal):

```python
@dataclass
class PrefillRequest:
    token_ids: torch.Tensor        # 1-D int32 cpu
    context: CacheView             # may be empty
    into: CacheBlock               # fresh block
    capture_affine: bool
    future: ...                    # resolved with last-token logits

@dataclass
class DecodeRequest:
    context: AsyncContext
    input_id: int                  # last token of output_block
    forbid_ids: list[int]
    sampling_params: SamplingParams
    future: ...                    # resolved with next token id
```

Tick (one call = one forward at most):

1. Drain `prefill_queue` first (matches today's prefill-first policy,
   `scheduler.py:221`). v1 runs **one prefill per tick** via
   `session.prefill_block` — plural batched prefill and mixed batches are
   out of scope.
2. Else, take **all** pending `DecodeRequest`s, assemble a `WorkerGroup`
   from their contexts, run `session.decode_step`.
   - If two pending requests share an `output_block` (user error — one agent,
     two writers), fail the second future rather than the whole tick.
3. Sampling: apply forbid mask (`logits[:, forbid_ids] -= inf`-style bias),
   then greedy/temperature via the existing `Sampler` machinery
   (`engine/sample.py`); append the chosen token to
   `output_block.token_ids`. KV/page append already happens inside
   `decode_step` (post-forward commit, `session.py:356-358`).
4. Resolve futures.

Block lifecycle: engine-owned `create_block()` / `free_block()` delegate to
the session; `free_block` asserts the block is not referenced by any queued
request (simple liveness check; refcounting deferred).

Gate: sync unit tests — tick batching semantics (N queued decodes → one
group), prefill-first ordering, duplicate-output error, forbid-mask
equivalence with the demo's `logits[ids] -= 100` masking.

## Phase 2 — `AsyncLLM`: asyncio frontend

`python/minisgl/llm/async_llm.py`, exported as `minisgl.AsyncLLM`.

- Owns the engine + `AsyncCacheEngine` + tokenizer. Public API is the design
  doc's snippet:
  - `await create_block()` / `await free_block(block)`
  - `await prefill_block(token_ids, context=None, into=None,
    capture_affine=True)` → returns the filled `CacheBlock`; last-token
    logits are exposed as `result.logits` (small `PrefillResult` — the doc's
    snippet uses both return shapes, this covers both; doc gets a one-line
    fix).
  - `async_generate(context, forbid_ids=(), sampling_params=...,
    max_steps=None)` → async generator of token ids. One `DecodeRequest` per
    step: enqueue → await future → yield → repeat. Breaking out of the
    generator stops the stream cleanly (nothing further is queued). EOS
    policy stays demo-side per `IDEAS.md`.
- Engine task `_engine_loop()`: started lazily on first request.

  ```python
  while self._live_requests or not self._closed:
      await self._work_available.wait()
      await asyncio.sleep(0)     # let all just-woken coroutines enqueue
      self.async_engine.tick()   # blocking GPU forward — fine, only work
  ```

  **Pacing note (the one real subtlety):** after a decode tick resolves the
  thinker's and writer's futures, both coroutines must get a chance to
  enqueue their next step *before* the next tick, or the group composition
  flip-flops (still correct, just batch-of-1 ticks). The `sleep(0)` after
  the wakeup handles the common case; the equivalence test in Phase 3 pins
  it. If scheduling order proves flaky, upgrade to counting live streams and
  waiting until each has re-enqueued or finished.

Gate: single-agent smoke test (prefill → N decode steps == lock-step
`SharedCacheSession` run, greedy, token-identical).

## Phase 3 — Demo port (validates the API end-to-end)

Rewrite `scripts/async_thoughts` on the new API, following the
`DEMO.md` / design-doc snippet: `thinker_coro`, `writer_coro`, `probe_coro`,
`asyncio.Event` handoff, probe via throwaway `prefill_block(...,
capture_affine=False)`.

- All prompts, forbidden sets, `\n\n` end-of-step detection, probe text and
  yes/no comparison stay demo-side (`IDEAS.md` #2).
- Oracle test: run the old lock-step `_run_loop` and the new asyncio demo on
  the same model/problem with greedy sampling and a deterministic probe
  cadence; assert identical thinker/writer token streams while both streams
  are active in the same ticks. (When the writer parks, the thinker decodes
  in batch-of-1 ticks — same as the lock-step `thinker_only` state, so
  streams stay comparable.)

Gate: demo runs on Qwen3-32B and a Qwen3.5 hybrid (GDN path exercised);
oracle test green on Qwen2.5-0.5B in CI.

## Phase 4 — Vanilla traffic on the unified path + deletion

1. Re-express `LLM.generate` (offline) per design doc: for each prompt,
   `create_block` → `prefill_block` (empty context) → decode loop with
   `AsyncContext(cache_view=[block], output_block=block)`; EOS/max_tokens in
   the frontend. Single-block decode through the query-rotation path has
   rotation offset 0 — numerically standard attention, eager, no graphs.
2. Server mode: `Scheduler.run_forever` keeps the IO mixin/message layer but
   translates `UserMsg` → single-block context requests into the same
   queues. `DetokenizeMsg` replies unchanged, so the tokenizer/API processes
   don't notice.
3. Delete: `PrefillManager`/`DecodeManager`/`CacheManager`/`TableManager`,
   radix/naive prefix cache, overlap-scheduling machinery
   (`ForwardInput`/`ForwardData`), `GraphRunner` usage on the decode path
   (keep the module until a follow-up decides), `test_scheduler.py` rewritten
   against the queue engine. `CLEANUP_PLAN.md` loc count updated.

Gate: `LLM.generate` greedy outputs match current `main` on a small model;
server smoke test (launch + one completion) passes.

## Risks

| risk | mitigation |
|---|---|
| Tick pacing puts thinker/writer in different batches → composition (and numerics) wobble | `sleep(0)` gather + Phase 3 oracle pins the 2-coroutine case; stream-counting fallback described in Phase 2 |
| Vanilla throughput regression (no graphs/radix/overlap) | accepted by decision; single-block fast-path dispatch is a known follow-up |
| Probe stalls decode one tick per probe | accepted; mixed-batch branch is the designed fix, plugs into the tick as a policy change |
| `free_block` while another view references it | liveness assert in Phase 1; refcount if the demo ever frees non-probe blocks |
| Hybrid (GDN) state under the new grouping | no change — `WorkerGroup.cache_structure/write_to` derived identically; covered by running the demo on Qwen3.5 |

## Order & estimate

Phases land as separate PRs in order; each is green on the previous gates.

| phase | deliverable | est. |
|---|---|---|
| 0 | primitives + renames | 0.5 d |
| 1 | `AsyncCacheEngine` sync tick + tests | 1 d |
| 2 | `AsyncLLM` asyncio frontend | 0.5–1 d |
| 3 | demo port + equivalence oracle | 1 d |
| 4 | vanilla-on-unified + deletion | 1 d |

Doc nits to fold back into `ASYNC_SCHED_DESIGN.md` once implemented:
`prefill_block` return shape (`PrefillResult` vs block-or-logits),
`decode_step(context) -> ?` becomes the internal `DecodeRequest`/tick pair,
and `WorkerGroup` moves from user-facing to scheduler-internal.
