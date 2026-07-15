"""
Queue-based scheduling core for async cache requests (see ASYNC_SCHED_DESIGN.md
and docs/async_sched_impl_plan.md, Phase 1).

``AsyncCacheEngine`` owns two request queues and drains them one forward per
``tick()``:

* ``PrefillRequest`` — fill a fresh ``CacheBlock`` with tokens (optionally
  attending to a context view); resolves with the last-token logits.
* ``DecodeRequest`` — one decode step for one ``AsyncContext``; all pending
  decode requests are batched into a single ``WorkerGroup`` forward and each
  resolves with its sampled token id.

The engine is deliberately synchronous: ``tick()`` blocks on the GPU forward
and resolves plain future objects.  The asyncio frontend (``AsyncLLM``,
Phase 2) drives ``tick()`` from an event-loop task and swaps
``future_factory`` for ``asyncio`` futures — same single thread, so plain
``set_result`` is safe.

Token selection is engine-side: per-request ``forbid_ids`` are masked out of
the logits (a logit bias), then the engine's ``Sampler`` picks the token.
Logits are returned only on request (``return_logits=True``): a prefill then
resolves with the last-token logits (e.g. for the mode-switching probe), a
decode with ``(token_id, raw_logits)``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Deque, List, Optional, Sequence

import torch
from minisgl.core import SamplingParams
from minisgl.shared_cache import (
    AsyncContext,
    CacheBlock,
    CacheView,
    SharedCacheSession,
    WorkerGroup,
)

if TYPE_CHECKING:
    from minisgl.engine import Engine
    from minisgl.engine.sample import Sampler

_GREEDY = SamplingParams(temperature=0.0, max_tokens=1)


class SimpleFuture:
    """Minimal future with the same surface the engine needs from
    ``asyncio.Future`` (``set_result`` / ``set_exception`` / ``done``)."""

    __slots__ = ("_result", "_exception", "_done")

    def __init__(self) -> None:
        self._result: Any = None
        self._exception: Optional[BaseException] = None
        self._done = False

    def set_result(self, result: Any) -> None:
        assert not self._done, "future already resolved"
        self._result = result
        self._done = True

    def set_exception(self, exception: BaseException) -> None:
        assert not self._done, "future already resolved"
        self._exception = exception
        self._done = True

    def done(self) -> bool:
        return self._done

    def exception(self) -> Optional[BaseException]:
        assert self._done, "future not resolved yet"
        return self._exception

    def result(self) -> Any:
        assert self._done, "future not resolved yet"
        if self._exception is not None:
            raise self._exception
        return self._result


@dataclass
class PrefillRequest:
    token_ids: torch.Tensor  # 1-D int32 cpu
    context: CacheView  # may be empty
    into: CacheBlock  # fresh block to fill
    capture_affine: bool
    return_logits: bool  # resolve with last-token logits instead of None
    future: Any  # resolved with logits [vocab] if return_logits else None


@dataclass
class DecodeRequest:
    context: AsyncContext
    input_id: int  # token to feed (KV not yet stored; typically last sampled)
    forbid_ids: Sequence[int]
    sampling_params: SamplingParams
    return_logits: bool  # also return the raw (pre-mask) logits row
    future: Any  # resolved with token id, or (token id, logits) if return_logits


class AsyncCacheEngine:
    """
    Drains async-cache request queues, one forward per ``tick()``.

    Scheduling policy (v1): prefills first, one per tick; otherwise every
    pending decode request is batched into a single ``WorkerGroup`` step.
    Mixed prefill+decode forwards are a later tick-policy upgrade (see
    docs/mixed_batch_plan.md).
    """

    def __init__(
        self,
        engine: "Engine | None" = None,
        *,
        session: "SharedCacheSession | None" = None,
        sampler: "Sampler | None" = None,
    ):
        if session is None:
            assert engine is not None, "AsyncCacheEngine needs an engine or a session"
            session = SharedCacheSession(engine)
        self.session = session
        if sampler is None:
            assert engine is not None, "AsyncCacheEngine needs an engine or a sampler"
            sampler = engine.sampler
        self.sampler = sampler
        # Swapped for asyncio's loop.create_future by the asyncio frontend.
        self.future_factory: Callable[[], Any] = SimpleFuture
        self._prefill_queue: Deque[PrefillRequest] = deque()
        self._decode_queue: Deque[DecodeRequest] = deque()

    # ------------------------------------------------------------------
    # Block lifecycle
    # ------------------------------------------------------------------

    def create_block(self) -> CacheBlock:
        return self.session.create_block()

    def free_block(self, block: CacheBlock) -> None:
        if self._block_in_use(block):
            raise RuntimeError(f"cannot free {block!r}: referenced by a queued request")
        self.session.free_block(block)

    def _block_in_use(self, block: CacheBlock) -> bool:
        for pf in self._prefill_queue:
            if block is pf.into or any(block is b for b in pf.context):
                return True
        for dec in self._decode_queue:
            if block is dec.context.output_block or any(block is b for b in dec.context.cache_view):
                return True
        return False

    # ------------------------------------------------------------------
    # Request submission
    # ------------------------------------------------------------------

    def submit_prefill(
        self,
        token_ids: torch.Tensor,
        *,
        into: CacheBlock,
        context: Optional[CacheView] = None,
        capture_affine: bool = True,
        return_logits: bool = False,
    ) -> Any:
        """Queue a prefill of *into* (a fresh block); returns a future resolved
        with the last-token logits ``[vocab]`` if *return_logits* else None."""
        assert into.num_tokens == 0, "prefill target must be a fresh block"
        future = self.future_factory()
        self._prefill_queue.append(
            PrefillRequest(
                token_ids=token_ids,
                context=list(context or []),
                into=into,
                capture_affine=capture_affine,
                return_logits=return_logits,
                future=future,
            )
        )
        return future

    def submit_decode(
        self,
        context: AsyncContext,
        input_id: int,
        *,
        forbid_ids: Sequence[int] = (),
        sampling_params: Optional[SamplingParams] = None,
        return_logits: bool = False,
    ) -> Any:
        """Queue one decode step for *context*; returns a future resolved with
        the sampled token id — or ``(token_id, logits)`` if *return_logits*,
        where *logits* is the raw ``[vocab]`` row before forbid masking."""
        future = self.future_factory()
        self._decode_queue.append(
            DecodeRequest(
                context=context,
                input_id=int(input_id),
                forbid_ids=list(forbid_ids),
                sampling_params=sampling_params or _GREEDY,
                return_logits=return_logits,
                future=future,
            )
        )
        return future

    @property
    def has_work(self) -> bool:
        return bool(self._prefill_queue or self._decode_queue)

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self) -> Optional[str]:
        """Run at most one forward; returns ``"prefill"``/``"decode"`` or
        ``None`` when both queues are empty."""
        if self._prefill_queue:
            self._run_prefill(self._prefill_queue.popleft())
            return "prefill"
        if self._decode_queue:
            self._run_decode_batch()
            return "decode"
        return None

    def _run_prefill(self, req: PrefillRequest) -> None:
        try:
            logits = self.session.prefill_block(
                req.into,
                req.token_ids,
                context=req.context or None,
                capture_affine=req.capture_affine,
            )
        except Exception as exc:
            req.future.set_exception(exc)
            raise
        req.future.set_result(logits[0] if req.return_logits else None)

    def _run_decode_batch(self) -> None:
        # Take everything queued; reject late duplicates of an output block
        # (one agent, two concurrent steps — a user error) without failing the
        # whole tick.
        reqs: List[DecodeRequest] = []
        seen_outputs = set()
        while self._decode_queue:
            req = self._decode_queue.popleft()
            out_id = id(req.context.output_block)
            if out_id in seen_outputs:
                req.future.set_exception(
                    ValueError("two decode requests write the same block in one step")
                )
                continue
            seen_outputs.add(out_id)
            reqs.append(req)
        if not reqs:
            return

        group = WorkerGroup([req.context for req in reqs])
        input_ids = torch.tensor([req.input_id for req in reqs], dtype=torch.int32)
        try:
            with torch.inference_mode():
                logits = self.session.decode_step(group, input_ids)
                # Snapshot raw rows before the forbid mask mutates them.
                raw_logits = {
                    i: logits[i].clone() for i, req in enumerate(reqs) if req.return_logits
                }
                next_tokens = self._select_tokens(logits, reqs)
        except Exception as exc:
            for req in reqs:
                req.future.set_exception(exc)
            raise
        for i, (req, token) in enumerate(zip(reqs, next_tokens.tolist())):
            token = int(token)
            req.future.set_result((token, raw_logits[i]) if req.return_logits else token)

    def _select_tokens(self, logits: torch.Tensor, reqs: List[DecodeRequest]) -> torch.Tensor:
        for i, req in enumerate(reqs):
            if req.forbid_ids:
                idx = torch.tensor(req.forbid_ids, dtype=torch.int64, device=logits.device)
                logits[i, idx] = float("-inf")
        args = self.sampler.prepare_params([req.sampling_params for req in reqs])
        return self.sampler.sample(logits, args).cpu()
