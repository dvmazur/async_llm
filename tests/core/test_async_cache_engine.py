"""
Tests for ``minisgl.scheduler.async_engine.AsyncCacheEngine``.

* **Unit tests** (always run) drive the queue/tick mechanics against a stub
  session: prefill-first ordering, decode batching into one ``WorkerGroup``,
  duplicate-output rejection, forbid-id masking, free-block liveness, and
  failure propagation.

* **End-to-end tests** (require ``MINISGL_E2E_MODEL`` + CUDA) check the engine
  against lock-step ``SharedCacheSession`` calls on a real model: identical
  greedy token streams for a single agent, and identical batched logits for a
  two-agent tick.

Run unit tests only::

    pytest tests/core/test_async_cache_engine.py -v

Run everything (needs GPU + model weights; build one engine per process)::

    MINISGL_E2E_MODEL=Qwen/Qwen3-0.6B pytest tests/core/test_async_cache_engine.py -v
"""

from __future__ import annotations

import os
from typing import List, Optional

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.engine.sample import Sampler
from minisgl.scheduler import AsyncCacheEngine
from minisgl.shared_cache import AsyncContext, CacheBlock, CacheView, WorkerGroup

CPU = torch.device("cpu")
VOCAB = 32

# =============================================================================
# Unit tests — no model/GPU needed
# =============================================================================


class StubSession:
    """Mimics the ``SharedCacheSession`` surface the engine touches.

    Deterministic logits: after feeding token ``t``, the "wanted" next token
    ``(t + 1) % VOCAB`` scores 10 and a "decoy" ``(t + 2) % VOCAB`` scores 20,
    so unmasked argmax picks the decoy and masking it flips the choice —
    making forbid-id behavior observable.
    """

    def __init__(self):
        self.device = CPU
        self.page_size = 1
        self.prefill_calls: List[dict] = []
        self.decode_calls: List[dict] = []
        self._next_page = 0
        self.fail_next: Optional[Exception] = None

    def _pages(self, n: int) -> torch.Tensor:
        start = self._next_page
        self._next_page += n
        return torch.arange(start, start + n, dtype=torch.int32)

    def create_block(self) -> CacheBlock:
        return CacheBlock(self.device)

    def free_block(self, block: CacheBlock) -> None:
        block.clear()

    def prefill_block(
        self,
        block: CacheBlock,
        input_ids: torch.Tensor,
        context: Optional[CacheView] = None,
        capture_affine: bool = True,
    ) -> torch.Tensor:
        if self.fail_next is not None:
            exc, self.fail_next = self.fail_next, None
            raise exc
        ids = input_ids.tolist()
        self.prefill_calls.append(
            {"block": block, "ids": ids, "context": context, "capture_affine": capture_affine}
        )
        block.grow_pages(self._pages(len(ids)), len(ids))
        block.token_ids.extend(ids)
        logits = torch.zeros(1, VOCAB)
        logits[0, (ids[-1] + 1) % VOCAB] = 10.0
        logits[0, (ids[-1] + 2) % VOCAB] = 20.0
        return logits

    def decode_step(self, group: WorkerGroup, input_ids: torch.Tensor) -> torch.Tensor:
        if self.fail_next is not None:
            exc, self.fail_next = self.fail_next, None
            raise exc
        ids = input_ids.reshape(group.num_workers).tolist()
        self.decode_calls.append(
            {
                "structure": group.cache_structure,
                "write_to": group.write_to,
                "input_ids": ids,
            }
        )
        logits = torch.zeros(group.num_workers, VOCAB)
        for i, (tok, wt) in enumerate(zip(ids, group.write_to)):
            wt.append_token(None if wt.has_capacity else int(self._pages(1)[0]))
            wt.token_ids.append(int(tok))
            logits[i, (tok + 1) % VOCAB] = 10.0
            logits[i, (tok + 2) % VOCAB] = 20.0
        return logits


@pytest.fixture()
def stub_engine():
    session = StubSession()
    engine = AsyncCacheEngine(session=session, sampler=Sampler(CPU, VOCAB))
    return engine, session


def _prefilled_block(engine: AsyncCacheEngine, ids: List[int]) -> CacheBlock:
    block = engine.create_block()
    engine.submit_prefill(torch.tensor(ids, dtype=torch.int32), into=block)
    assert engine.tick() == "prefill"
    return block


class TestQueueMechanics:
    def test_empty_tick(self, stub_engine):
        engine, _ = stub_engine
        assert not engine.has_work
        assert engine.tick() is None

    def test_prefill_resolves_with_last_token_logits(self, stub_engine):
        engine, session = stub_engine
        block = engine.create_block()
        fut = engine.submit_prefill(
            torch.tensor([1, 2, 3], dtype=torch.int32), into=block, return_logits=True
        )
        assert engine.has_work and not fut.done()
        assert engine.tick() == "prefill"
        assert fut.done()
        logits = fut.result()
        assert logits.shape == (VOCAB,)
        assert int(logits.argmax()) == 5  # decoy of last token 3
        assert block.token_ids == [1, 2, 3]
        assert session.prefill_calls[0]["capture_affine"] is True

    def test_prefill_default_resolves_none(self, stub_engine):
        engine, _ = stub_engine
        block = engine.create_block()
        fut = engine.submit_prefill(torch.tensor([1], dtype=torch.int32), into=block)
        engine.tick()
        assert fut.result() is None
        assert block.token_ids == [1]

    def test_prefill_runs_before_decode(self, stub_engine):
        engine, session = stub_engine
        block = _prefilled_block(engine, [1, 2])
        ctx = AsyncContext(cache_view=[block])
        engine.submit_decode(ctx, input_id=7)
        pf_block = engine.create_block()
        engine.submit_prefill(torch.tensor([4], dtype=torch.int32), into=pf_block)
        assert engine.tick() == "prefill"  # prefill jumps the earlier decode
        assert engine.tick() == "decode"
        assert not engine.has_work

    def test_decode_batches_all_pending(self, stub_engine):
        engine, session = stub_engine
        prompt = _prefilled_block(engine, [1, 2])
        w1 = _prefilled_block(engine, [3])
        w2 = _prefilled_block(engine, [4])
        ctx1 = AsyncContext(cache_view=[prompt, w2, w1])
        ctx2 = AsyncContext(cache_view=[prompt, w1, w2])
        fut1 = engine.submit_decode(ctx1, input_id=5)
        fut2 = engine.submit_decode(ctx2, input_id=9)

        assert engine.tick() == "decode"
        assert len(session.decode_calls) == 1
        call = session.decode_calls[0]
        assert call["structure"] == [[prompt, w2, w1], [prompt, w1, w2]]
        assert call["write_to"] == [w1, w2]
        assert call["input_ids"] == [5, 9]
        # unmasked argmax picks the decoy (t + 2)
        assert fut1.result() == 7
        assert fut2.result() == 11
        assert w1.token_ids == [3, 5]
        assert w2.token_ids == [4, 9]

    def test_forbid_ids_mask(self, stub_engine):
        engine, _ = stub_engine
        block = _prefilled_block(engine, [1])
        ctx = AsyncContext(cache_view=[block])
        fut = engine.submit_decode(ctx, input_id=5, forbid_ids=[7])  # mask the decoy
        engine.tick()
        assert fut.result() == 6  # falls back to the wanted token (t + 1)

    def test_decode_return_logits_is_raw_row(self, stub_engine):
        engine, _ = stub_engine
        block = _prefilled_block(engine, [1])
        ctx = AsyncContext(cache_view=[block])
        fut = engine.submit_decode(ctx, input_id=5, forbid_ids=[7], return_logits=True)
        engine.tick()
        token, logits = fut.result()
        assert token == 6  # masked selection
        assert logits.shape == (VOCAB,)
        assert int(logits.argmax()) == 7  # raw row: decoy unmasked

    def test_duplicate_output_block_fails_late_request(self, stub_engine):
        engine, session = stub_engine
        block = _prefilled_block(engine, [1])
        other = _prefilled_block(engine, [2])
        ctx_a = AsyncContext(cache_view=[block])
        ctx_b = AsyncContext(cache_view=[other, block], output_block=block)
        fut_a = engine.submit_decode(ctx_a, input_id=3)
        fut_b = engine.submit_decode(ctx_b, input_id=4)

        assert engine.tick() == "decode"
        assert fut_a.result() == 5
        assert isinstance(fut_b.exception(), ValueError)
        # only the first request ran
        assert session.decode_calls[0]["write_to"] == [block]

    def test_free_block_liveness(self, stub_engine):
        engine, _ = stub_engine
        prompt = _prefilled_block(engine, [1])
        block = _prefilled_block(engine, [2])
        engine.submit_decode(AsyncContext(cache_view=[prompt, block]), input_id=3)
        with pytest.raises(RuntimeError, match="referenced by a queued request"):
            engine.free_block(prompt)  # in a queued view
        with pytest.raises(RuntimeError, match="referenced by a queued request"):
            engine.free_block(block)  # queued output block
        engine.tick()
        engine.free_block(block)
        assert block.num_tokens == 0

    def test_prefill_appends_to_non_empty_block(self, stub_engine):
        engine, session = stub_engine
        block = _prefilled_block(engine, [1, 2])
        engine.submit_prefill(torch.tensor([3, 4], dtype=torch.int32), into=block)
        assert engine.tick() == "prefill"
        assert block.num_tokens == 4
        assert block.token_ids == [1, 2, 3, 4]
        # queued prefills for one block chain in submission order
        assert [c["ids"] for c in session.prefill_calls] == [[1, 2], [3, 4]]

    def test_failure_propagates_to_all_futures(self, stub_engine):
        engine, session = stub_engine
        b1 = _prefilled_block(engine, [1])
        b2 = _prefilled_block(engine, [2])
        fut1 = engine.submit_decode(AsyncContext(cache_view=[b1]), input_id=3)
        fut2 = engine.submit_decode(AsyncContext(cache_view=[b2]), input_id=4)
        session.fail_next = RuntimeError("forward exploded")
        with pytest.raises(RuntimeError, match="forward exploded"):
            engine.tick()
        assert isinstance(fut1.exception(), RuntimeError)
        assert isinstance(fut2.exception(), RuntimeError)


# =============================================================================
# End-to-end tests — need MINISGL_E2E_MODEL + CUDA
# =============================================================================

E2E_MODEL_PATH = os.environ.get("MINISGL_E2E_MODEL", "")

requires_e2e = pytest.mark.skipif(
    not E2E_MODEL_PATH or not torch.cuda.is_available(),
    reason="Set MINISGL_E2E_MODEL to an HF model path and ensure CUDA is available",
)


@pytest.fixture(scope="module")
def real_engine():
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig

    config = EngineConfig(
        model_path=E2E_MODEL_PATH,
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


def _encode(text: str) -> torch.Tensor:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(E2E_MODEL_PATH)
    return tok.encode(text, return_tensors="pt").view(-1).to(torch.int32)


@requires_e2e
class TestAsyncCacheEngineE2E:
    def test_single_agent_matches_lockstep_session(self, real_engine):
        """Greedy chain through the engine == lock-step session calls."""
        from minisgl.shared_cache import SharedCacheSession

        n_steps = 8
        prompt_ids = _encode("The capital of France is")

        # Reference: direct session, lock-step.
        session = SharedCacheSession(real_engine)
        ref_block = session.create_block()
        logits = session.prefill_block(ref_block, prompt_ids)
        ref_tokens = [int(logits[0].argmax())]
        ref_group = WorkerGroup(cache_structure=[[ref_block]])
        for _ in range(n_steps):
            logits = session.decode_step(
                ref_group, torch.tensor([ref_tokens[-1]], dtype=torch.int32)
            )
            ref_tokens.append(int(logits[0].argmax()))

        # Engine: same chain through submit + tick.
        engine = AsyncCacheEngine(real_engine)
        block = engine.create_block()
        fut = engine.submit_prefill(prompt_ids, into=block, return_logits=True)
        assert engine.tick() == "prefill"
        tokens = [int(fut.result().argmax())]
        ctx = AsyncContext(cache_view=[block])
        for _ in range(n_steps):
            fut = engine.submit_decode(ctx, input_id=tokens[-1])
            assert engine.tick() == "decode"
            tokens.append(fut.result())

        assert tokens == ref_tokens
        # KV-backed bookkeeping: prompt + the n_steps fed tokens
        assert block.token_ids == prompt_ids.tolist() + tokens[:-1]

        session.free_block(ref_block)
        engine.free_block(block)

    def test_two_agent_tick_matches_direct_group(self, real_engine):
        """Two decode submissions in one tick == one direct 2-worker group step."""
        from minisgl.shared_cache import SharedCacheSession

        session = SharedCacheSession(real_engine)
        prompt_ids = _encode("Q: What is 2 + 2?\n")
        a_ids = _encode("A:")
        b_ids = _encode("Hint:")

        def build(sess_or_engine, prefill):
            prompt, a, b = (prefill(ids) for ids in (prompt_ids, a_ids, b_ids))
            return prompt, a, b

        # Reference blocks via direct session.
        def ref_prefill(ids):
            blk = session.create_block()
            session.prefill_block(blk, ids)
            return blk

        r_prompt, r_a, r_b = build(session, ref_prefill)
        ref_group = WorkerGroup(
            cache_structure=[[r_prompt, r_b, r_a], [r_prompt, r_a, r_b]],
            write_to=[r_a, r_b],
        )
        ref_logits = session.decode_step(ref_group, torch.tensor([11, 13], dtype=torch.int32))
        ref_next = [int(row.argmax()) for row in ref_logits]

        # Engine blocks via queue.
        engine = AsyncCacheEngine(real_engine)

        def eng_prefill(ids):
            blk = engine.create_block()
            engine.submit_prefill(ids, into=blk)
            engine.tick()
            return blk

        e_prompt, e_a, e_b = build(engine, eng_prefill)
        fut_a = engine.submit_decode(AsyncContext(cache_view=[e_prompt, e_b, e_a]), input_id=11)
        fut_b = engine.submit_decode(AsyncContext(cache_view=[e_prompt, e_a, e_b]), input_id=13)
        assert engine.tick() == "decode"

        assert [fut_a.result(), fut_b.result()] == ref_next
        for blk in (r_prompt, r_a, r_b):
            session.free_block(blk)
        for blk in (e_prompt, e_a, e_b):
            engine.free_block(blk)

    def test_greedy_sampling_params_with_forbid(self, real_engine):
        """Masking the greedy argmax forces the runner-up token."""
        engine = AsyncCacheEngine(real_engine)
        block = engine.create_block()
        prompt_ids = _encode("The capital of France is")
        fut = engine.submit_prefill(prompt_ids, into=block, return_logits=True)
        engine.tick()
        logits = fut.result().float()
        top2 = logits.topk(2).indices.tolist()

        ctx = AsyncContext(cache_view=[block])
        fut = engine.submit_decode(
            ctx,
            input_id=int(top2[0]),
            forbid_ids=[],
            sampling_params=SamplingParams(temperature=0.0),
        )
        engine.tick()
        unmasked = fut.result()

        # New chain on a fresh block: same input, now with the winner masked.
        block2 = engine.create_block()
        engine.submit_prefill(prompt_ids, into=block2)
        engine.tick()
        ctx2 = AsyncContext(cache_view=[block2])
        fut = engine.submit_decode(ctx2, input_id=int(top2[0]), forbid_ids=[unmasked])
        engine.tick()
        assert fut.result() != unmasked

        engine.free_block(block)
        engine.free_block(block2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
