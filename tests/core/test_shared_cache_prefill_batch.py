"""
Batched shared-cache prefill: ``SharedCacheSession.prefill_batch``.

Every test here pins the same contract — **one forward for N prefills must
produce exactly what N separate forwards produce** — for the shapes the async
engine actually forms a batch out of: plain prefills, in-context prefills over
a shared prompt, ragged lengths, extensions of non-empty blocks, and a mix of
all of them.  A batched block must also be usable afterwards: the final tests
decode from it and compare against the serialized run.

Run (needs GPU + weights)::

    MINISGL_E2E_MODEL=Qwen/Qwen3-0.6B pytest tests/core/test_shared_cache_prefill_batch.py -v
"""

from __future__ import annotations

import os
from typing import List, Sequence

import pytest
import torch
from minisgl.shared_cache import AsyncContext, PrefillJob, SharedCacheSession, WorkerGroup

E2E_MODEL_PATH = os.environ.get("MINISGL_E2E_MODEL", "")

requires_e2e = pytest.mark.skipif(
    not E2E_MODEL_PATH or not torch.cuda.is_available(),
    reason="Set MINISGL_E2E_MODEL to an HF model path and ensure CUDA is available",
)


def _build_engine(model_path: str):
    """Lazy import — importing Engine eagerly pulls in CUDA-only deps."""
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig

    config = EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        max_running_req=8,
        cuda_graph_bs=[2, 4],
        cuda_graph_max_bs=4,
        page_size=int(os.environ.get("MINISGL_TEST_PAGE_SIZE", "1")),
        memory_ratio=0.7,
        max_seq_len_override=2048,
    )
    return Engine(config)


@pytest.fixture(scope="module")
def engine_and_session():
    if not E2E_MODEL_PATH or not torch.cuda.is_available():
        pytest.skip("e2e tests disabled")
    engine = _build_engine(E2E_MODEL_PATH)
    yield engine, SharedCacheSession(engine)
    engine.shutdown()


def _ids(*values: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


def _seq(start: int, length: int) -> torch.Tensor:
    return torch.arange(start, start + length, dtype=torch.int32)


def _assert_exact(batched: torch.Tensor, single: torch.Tensor, what: str) -> None:
    """Batching must be *transparent*: one forward for N requests has to give
    each request bit-for-bit what its own forward on the same path gives."""
    b, s = batched.float().cpu(), single.float().cpu()
    assert b.shape == s.shape, what
    torch.testing.assert_close(b, s, rtol=0, atol=0, msg=lambda m: f"{what}: {m}")


def _assert_close(batched: torch.Tensor, serial: torch.Tensor, what: str) -> None:
    """Batched vs the established single-request ``prefill_block``.

    Exact for an in-context prefill (same shared-cache op).  A *plain* prefill
    is the one case where the two differ by construction: alone it runs on the
    page-table path (stock attention backend, fused rotary), while a batch must
    put it on the shared-cache op — same math, different kernels, so bf16-scale
    drift is expected and only the argmax is pinned.
    """
    b, s = batched.float().cpu(), serial.float().cpu()
    assert b.shape == s.shape, what
    assert int(b.argmax()) == int(s.argmax()), f"{what}: argmax differs"
    torch.testing.assert_close(b, s, rtol=5e-2, atol=2e-1, msg=lambda m: f"{what}: {m}")


class _Run:
    """Runs the same set of prefills three ways on one session — as one batch,
    as one fused forward each, and via ``prefill_block`` — onto fresh blocks."""

    def __init__(self, session: SharedCacheSession):
        self.session = session
        self.blocks: List = []

    def compare(self, jobs: Sequence[dict]) -> None:
        """*jobs*: dicts of ``ids`` plus optional ``context`` (indices into the
        blocks created here) and ``write_to`` (index of a block to extend)."""
        batched = self._materialize(jobs, mode="batch")
        fused = self._materialize(jobs, mode="fused_single")
        serial = self._materialize(jobs, mode="serial")
        for i, (b, f, s) in enumerate(zip(batched, fused, serial)):
            _assert_exact(b, f, f"request {i}")
            _assert_close(b, s, f"request {i}")

    def _materialize(self, jobs: Sequence[dict], mode: str) -> List[torch.Tensor]:
        blocks: dict = {}

        def block_for(job: dict):
            key = job.get("write_to")
            if key is None:
                return self.session.create_block()
            if key not in blocks:
                blocks[key] = self.session.create_block()
                # seed the block so the batched write is an *extension*
                self.session.prefill_block(blocks[key], _seq(1000 + 7 * key, 3))
            return blocks[key]

        ctx_blocks: dict = {}
        for job in jobs:
            for c in job.get("context", ()):
                if c not in ctx_blocks:
                    ctx_blocks[c] = self.session.create_block()
                    self.session.prefill_block(ctx_blocks[c], _seq(500 + 11 * c, 5))

        prepared = [
            PrefillJob(
                block=block_for(job),
                input_ids=job["ids"],
                context=[ctx_blocks[c] for c in job.get("context", ())],
            )
            for job in jobs
        ]
        if mode == "batch":
            out = self.session.prefill_batch(prepared)
        elif mode == "fused_single":
            out = [self.session._prefill_batch_fused([p])[0] for p in prepared]
        else:
            out = [
                self.session.prefill_block(p.block, p.input_ids, context=p.context or None)
                for p in prepared
            ]
        # token bookkeeping must match the serialized run too
        for job, p in zip(jobs, prepared):
            assert p.block.token_ids[-len(job["ids"]) :] == job["ids"].tolist()
        self.blocks.append(prepared)
        return [row for row in out]


@requires_e2e
class TestPrefillBatchEquivalence:
    def test_plain_prefills(self, engine_and_session):
        _, session = engine_and_session
        _Run(session).compare([{"ids": _seq(10, 6)}, {"ids": _seq(40, 4)}, {"ids": _seq(70, 9)}])

    def test_single_job_delegates_to_prefill_block(self, engine_and_session):
        """A group of one keeps the established single-request path untouched."""
        _, session = engine_and_session
        ids = _seq(21, 7)
        batched = session.prefill_batch([PrefillJob(block=session.create_block(), input_ids=ids)])[0]
        serial = session.prefill_block(session.create_block(), ids)
        _assert_exact(batched, serial, "single job")

    def test_shared_context(self, engine_and_session):
        _, session = engine_and_session
        _Run(session).compare(
            [{"ids": _seq(10, 5), "context": [0]}, {"ids": _seq(60, 3), "context": [0]}]
        )

    def test_different_context_depths(self, engine_and_session):
        _, session = engine_and_session
        _Run(session).compare(
            [
                {"ids": _seq(10, 4), "context": [0]},
                {"ids": _seq(30, 6), "context": [0, 1]},
                {"ids": _seq(50, 2), "context": [1, 0, 2]},
            ]
        )

    def test_context_and_plain_mixed(self, engine_and_session):
        """A no-context request in a batch whose other requests have context:
        its rows must merge only their own (single) self segment."""
        _, session = engine_and_session
        _Run(session).compare(
            [
                {"ids": _seq(10, 5)},
                {"ids": _seq(30, 4), "context": [0, 1]},
                {"ids": _seq(80, 3)},
            ]
        )

    def test_extension_of_non_empty_blocks(self, engine_and_session):
        _, session = engine_and_session
        _Run(session).compare(
            [
                {"ids": _seq(10, 4), "write_to": 0},
                {"ids": _seq(30, 5), "write_to": 1, "context": [0]},
                {"ids": _seq(60, 2)},
            ]
        )

    def test_single_token_requests(self, engine_and_session):
        _, session = engine_and_session
        _Run(session).compare(
            [{"ids": _ids(11)}, {"ids": _ids(12), "context": [0]}, {"ids": _ids(13)}]
        )


@requires_e2e
class TestPrefillBatchRejects:
    def test_two_jobs_writing_one_block(self, engine_and_session):
        _, session = engine_and_session
        block = session.create_block()
        jobs = [PrefillJob(block=block, input_ids=_ids(1)), PrefillJob(block=block, input_ids=_ids(2))]
        with pytest.raises(AssertionError, match="write the same block"):
            session.prefill_batch(jobs)

    def test_context_written_by_another_job(self, engine_and_session):
        _, session = engine_and_session
        head, tail = session.create_block(), session.create_block()
        jobs = [
            PrefillJob(block=head, input_ids=_seq(10, 3)),
            PrefillJob(block=tail, input_ids=_seq(20, 3), context=[head]),
        ]
        with pytest.raises(ValueError, match="another request's write block"):
            session.prefill_batch(jobs)

    def test_write_block_in_own_context(self, engine_and_session):
        _, session = engine_and_session
        prompt, block = session.create_block(), session.create_block()
        session.prefill_block(prompt, _seq(1, 4))
        session.prefill_block(block, _seq(50, 3))
        jobs = [
            PrefillJob(block=block, input_ids=_seq(60, 2), context=[prompt, block]),
            PrefillJob(block=session.create_block(), input_ids=_seq(70, 2)),
        ]
        with pytest.raises(ValueError, match="must not appear in context"):
            session.prefill_batch(jobs)


@requires_e2e
class TestPrefillBatchThenDecode:
    def test_decode_after_batched_prefill(self, engine_and_session):
        """Blocks filled by one batched forward must decode exactly like blocks
        filled one at a time — the KV, and for hybrid models the GDN state, are
        what a serialized run would have left behind."""
        _, session = engine_and_session
        prompt_ids, a_ids, b_ids = _seq(100, 6), _seq(200, 4), _seq(300, 5)

        def run(batched: bool):
            prompt = session.create_block()
            session.prefill_block(prompt, prompt_ids)
            a, b = session.create_block(), session.create_block()
            jobs = [
                PrefillJob(block=a, input_ids=a_ids, context=[prompt]),
                PrefillJob(block=b, input_ids=b_ids, context=[prompt]),
            ]
            if batched:
                session.prefill_batch(jobs)
            else:
                for job in jobs:
                    session.prefill_block(job.block, job.input_ids, context=job.context)
            group = WorkerGroup(
                [
                    AsyncContext(cache_view=[prompt, b, a]),
                    AsyncContext(cache_view=[prompt, a, b]),
                ]
            )
            return session.decode_step(group, torch.tensor([7, 9], dtype=torch.int32))

        batched, serial = run(True), run(False)
        for w in range(2):
            _assert_close(batched[w], serial[w], f"worker {w} decode")
