"""
Chunked shared-cache prefill: ``SharedCacheSession.max_prefill_rows``.

A long prefill is split into waves of at most ``max_prefill_rows`` query rows, where
a row is one (token, segment) pair -- the unit the shared-cache planner actually
materializes.  Image tokens split like any others, including mid-image, because the
vision tower runs once up front and each chunk carries the matching slice of its output.

Three things are pinned, in descending strictness:

* **The committed block is exact.**  Token ids, page count and mRoPE span must not
  depend on where the boundaries fell.  The span is the sharp one: an image's span is
  not its token count, so a slicing error moves it.
* **Chunkings agree with each other, and with unchunked, on the choice.**  Different
  budgets put boundaries in different places -- including inside an image run -- so a
  mis-sliced embedding range or a stale mRoPE shift cannot survive it.
* **Logits are within a measured bound, not equal.**  On a hybrid model each boundary
  hands the recurrent state over through the affine summary rather than the kernel's
  in-forward carry.  Measured 0.13-0.19 on Qwen3.5-0.8B and flat from 2 waves to 1024,
  so it is an offset rather than drift -- and the unchunked extension path already pays
  it.  Greedy identity is therefore asserted only on non-hybrid models.

``TestChunkSlicing`` pins the slicing arithmetic and needs no GPU; the rest is e2e.

Run (needs GPU + weights)::

    pytest tests/core/test_chunked_prefill.py -v
    MINISGL_E2E_MODEL=Qwen/Qwen3-0.6B pytest tests/core/test_chunked_prefill.py -v
"""

from __future__ import annotations

import os
from typing import List

import pytest
import torch
from minisgl.shared_cache import AsyncContext, CacheBlock, PrefillJob, PrefillPlan, WorkerGroup
from minisgl.shared_cache.session import _chunk_job, _job_rows

VL_MODEL_ID = "Qwen/Qwen3.5-0.8B"
# Qwen3.5-0.8B is multimodal *and* hybrid, so it covers the GDN state hand-off too.
E2E_MODEL_PATH = os.environ.get("MINISGL_E2E_MODEL", VL_MODEL_ID)

requires_e2e = pytest.mark.skipif(
    not E2E_MODEL_PATH or not torch.cuda.is_available(),
    reason="Set MINISGL_E2E_MODEL to an HF model path and ensure CUDA is available",
)


# =============================================================================
# Chunk slicing (CPU only -- pure index arithmetic)
# =============================================================================


MERGE = 2


def _plan(types: List[int], hidden: int = 4) -> PrefillPlan:
    """A plan whose embedding rows carry their own image-token index, so a wrong
    slice shows up as wrong *values* rather than only a wrong shape.

    Each contiguous image run becomes one single-frame square grid, so image runs
    must be perfect squares.  Single-frame matters: with ``t > 1`` the running mRoPE
    position advances by ``max(h, w) // merge`` while the largest position sits on
    the *temporal* axis, and the two stop agreeing.  Video is unsupported.
    """
    from minisgl.models.qwen3_5_mrope import get_rope_index

    types_t = torch.tensor(types, dtype=torch.int64)
    n_img = int((types_t == 1).sum())
    grids, i = [], 0
    while i < len(types):
        if types[i] != 1:
            i += 1
            continue
        j = i
        while j < len(types) and types[j] == 1:
            j += 1
        side = int(round((j - i) ** 0.5))
        assert side * side == j - i, f"image run {j - i} is not a perfect square"
        grids.append([1, side * MERGE, side * MERGE])
        i = j
    ids = torch.arange(len(types), dtype=torch.int32)
    rel_pos = get_rope_index(
        ids, types_t, MERGE, torch.tensor(grids, dtype=torch.int64) if grids else None
    )
    embeds = torch.arange(n_img, dtype=torch.float32)[:, None].repeat(1, hidden)
    return PrefillPlan(ids=ids, types=types_t, rel_pos=rel_pos, embeds=embeds)


def _job(plan: PrefillPlan) -> PrefillJob:
    return PrefillJob(
        block=CacheBlock(torch.device("cpu")),
        input_ids=plan.ids,
        pixel_values=torch.zeros(1, 1),  # marks the job multimodal
        image_grid_thw=torch.zeros(1, 3, dtype=torch.int64),
        mm_token_type_ids=plan.types,
    )


class TestChunkSlicing:
    """``_chunk_job`` must tile the request exactly once and hand every chunk the
    embedding rows its own image tokens need -- no more, no fewer."""

    @pytest.mark.parametrize(
        "types",
        [
            [0, 0, 0, 0, 0, 0],  # text only
            [1, 1, 1, 1, 0, 0],  # leading image
            [0, 0, 1, 1, 1, 1],  # trailing image
            [0, 1, 1, 1, 1, 0],  # image in the middle
            [1] * 9,  # image only, nothing to anchor against
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],  # two images
            [0, 1, 0, 1, 0],  # single-token images
        ],
    )
    @pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7, 100])
    def test_tiles_exactly_once_with_aligned_embeds(self, types, size):
        plan = _plan(types)
        job = _job(plan)
        total = len(types)
        chunks = [_chunk_job(job, plan, s, min(s + size, total)) for s in range(0, total, size)]

        # every token appears exactly once, in order
        assert torch.equal(torch.cat([c.input_ids for c in chunks]), plan.ids)
        assert torch.equal(torch.cat([c.mm_token_type_ids for c in chunks]), plan.types)

        # what `x[image_mask] = image_embeds` in the model forward requires
        seen = 0
        for c in chunks:
            n_img = int((c.mm_token_type_ids == 1).sum())
            got = c.image_embeds.shape[0]
            assert got == n_img, f"chunk mask wants {n_img} embeds, got {got}"
            # rows must be *this* chunk's image tokens, not merely the right count
            expected = torch.arange(seen, seen + n_img, dtype=torch.float32)
            assert torch.equal(c.image_embeds[:, 0], expected)
            seen += n_img
        assert seen == int((plan.types == 1).sum())

        # pixels are consumed by the plan, never forwarded to a chunk
        assert all(c.pixel_values is None and c.image_grid_thw is None for c in chunks)

    @pytest.mark.parametrize(
        "types", [[0, 1, 1, 1, 1, 0, 0], [1] * 9, [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]]
    )
    def test_positions_stay_absolute_across_chunks(self, types):
        """A chunk's ``mrope_rel`` is relative to the block's span at that point, so
        re-adding the span each chunk must reproduce the whole-request layout —
        including when a boundary lands inside an image run."""
        for base in (0, 7):
            plan = _plan(types)
            plan.base_span = base
            job = _job(plan)
            for size in (1, 2, 3, 5):
                span, rebuilt = base, []
                for s in range(0, len(plan.ids), size):
                    chunk = _chunk_job(job, plan, s, min(s + size, len(plan.ids)))
                    # mirrors _prefill_batch_fused: rotate at the block's span, then
                    # take the chunk's own span as the block's new one
                    rebuilt.append(chunk.mrope_rel + span)
                    span = chunk.mrope_span
                got = torch.cat(rebuilt, dim=1)
                assert torch.equal(got, plan.rel_pos + base), f"base={base} size={size}"
                assert span == base + int(plan.rel_pos.max()) + 1

    def test_text_only_plan_leaves_job_untouched(self):
        plan = PrefillPlan(ids=torch.arange(6, dtype=torch.int32))
        job = PrefillJob(block=CacheBlock(torch.device("cpu")), input_ids=plan.ids)
        chunk = _chunk_job(job, plan, 2, 4)
        assert torch.equal(chunk.input_ids, plan.ids[2:4])
        assert chunk.mm_token_type_ids is None
        assert chunk.image_embeds is None and chunk.mrope_rel is None


class TestJobRows:
    def test_rows_count_context_segments(self):
        dev = torch.device("cpu")
        empty, filled = CacheBlock(dev), CacheBlock(dev)
        filled.num_tokens = 3
        ids = torch.arange(10, dtype=torch.int32)
        assert _job_rows(PrefillJob(block=empty, input_ids=ids)) == 10
        # an empty context block is dropped by the planner, so it costs no rows
        assert _job_rows(PrefillJob(block=empty, input_ids=ids, context=[empty])) == 10
        assert _job_rows(PrefillJob(block=empty, input_ids=ids, context=[filled])) == 20


# =============================================================================
# End-to-end equivalence
# =============================================================================


def _build_engine(model_path: str, **overrides):
    """Lazy import — importing Engine eagerly pulls in CUDA-only deps."""
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig

    kwargs = dict(
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
    kwargs.update(overrides)
    return Engine(EngineConfig(**kwargs))


@pytest.fixture(scope="module")
def session():
    """One engine for the whole module: ``Engine.__init__`` asserts CUDA is not yet
    initialized, so a second one cannot be built in the same process."""
    if not E2E_MODEL_PATH or not torch.cuda.is_available():
        pytest.skip("e2e tests disabled")
    from minisgl.shared_cache import SharedCacheSession

    engine = _build_engine(E2E_MODEL_PATH, num_page_override=4096, max_seq_len_override=4096)
    yield SharedCacheSession(engine)
    engine.shutdown()


def _seq(start: int, length: int) -> torch.Tensor:
    return torch.arange(start, start + length, dtype=torch.int32)


# Logit gap a chunked prefill may show against another chunking or against an
# unchunked run.  On a hybrid model each chunk boundary hands the recurrent state over
# through the fp32 affine summary instead of the kernel's in-forward carry, which costs
# a fixed offset -- measured at 0.13-0.19 on Qwen3.5-0.8B and, importantly, flat from 2
# waves to 1024.  The unchunked block-extension path already pays the same gap (~0.2,
# see ``test_prefill_extends_non_empty_block``), so chunking inherits it rather than
# introducing it.  A text-only model has no GDN and stays far tighter.
_HYBRID_ATOL = 1.0
_TEXT_ATOL = 2e-2


def _atol(session) -> float:
    return _HYBRID_ATOL if session._is_hybrid else _TEXT_ATOL


def _assert_same_choice(chunked: torch.Tensor, whole: torch.Tensor, atol: float, what: str) -> None:
    """Chunked vs unchunked: pin the decision and bound the gap."""
    assert int(chunked.argmax()) == int(whole.argmax()), f"{what}: argmax differs"
    gap = (chunked.float() - whole.float()).abs().max().item()
    assert gap <= atol, f"{what}: logit gap {gap:.4f} exceeds {atol}"


def _assert_chunkings_agree(rows: List[torch.Tensor], atol: float, what: str) -> None:
    """Every chunking of one prefill must agree with every other.

    The budgets place boundaries in different places -- including inside an image run --
    so a mis-sliced embedding range, a stale mRoPE shift or a dropped token cannot
    survive this.  What it does not pin is the state hand-off, which is why the
    tolerance is the measured one rather than an exact match.
    """
    for i, row in enumerate(rows[1:], start=1):
        gap = (row.float() - rows[0].float()).abs().max().item()
        assert int(row.argmax()) == int(rows[0].argmax()), f"{what}: chunking {i} argmax differs"
        assert gap <= atol, f"{what}: chunking {i} vs 0 gap {gap:.4f} exceeds {atol}"


def _decode(session, blocks, first: int, steps: int = 8) -> List[int]:
    group = WorkerGroup([AsyncContext(cache_view=blocks)])
    out, tok = [], first
    for _ in range(steps):
        logits = session.decode_step(group, torch.tensor([tok], dtype=torch.int32))
        tok = int(logits[0].argmax())
        out.append(tok)
    return out


@requires_e2e
class TestChunkedEquivalence:
    @pytest.mark.parametrize("rows", [1, 3, 8, 64, 10_000])
    def test_plain_prefill(self, session, rows):
        ids = _seq(10, 33)
        whole = session.create_block()
        ref = session.prefill_block(whole, ids)

        chunked = session.create_block()
        session.max_prefill_rows = rows
        try:
            got = session.prefill_block(chunked, ids)
        finally:
            session.max_prefill_rows = None

        assert chunked.token_ids == ids.tolist()
        assert chunked.num_tokens == whole.num_tokens
        _assert_same_choice(got[0], ref[0], _atol(session), f"rows={rows}")
        for blk in (whole, chunked):
            session.free_block(blk)

    @pytest.mark.parametrize("rows", [2, 7, 64])
    def test_in_context_prefill(self, session, rows):
        """With a context view a row costs (len(context)+1) per token, so the same
        budget yields more waves — the point of counting rows rather than tokens."""
        prompt_ids, tail_ids = _seq(100, 12), _seq(300, 21)
        prompt = session.create_block()
        session.prefill_block(prompt, prompt_ids)

        whole = session.create_block()
        ref = session.prefill_block(whole, tail_ids, context=[prompt])

        chunked = session.create_block()
        session.max_prefill_rows = rows
        try:
            got = session.prefill_block(chunked, tail_ids, context=[prompt])
        finally:
            session.max_prefill_rows = None

        assert chunked.token_ids == tail_ids.tolist()
        _assert_same_choice(got[0], ref[0], _atol(session), f"rows={rows}")
        for blk in (prompt, whole, chunked):
            session.free_block(blk)

    def test_decode_after_chunked_prefill(self, session):
        """A chunked prefill must leave a block that decodes like an unchunked one.

        Greedy identity is only asserted without GDN: on a hybrid model the state
        hand-off can flip a near-tie, so there the bound is on the logits (checked by
        the tests above) and on the committed block state (checked here).
        """
        ids = _seq(50, 40)
        whole = session.create_block()
        ref_logits = session.prefill_block(whole, ids)
        # Snapshot before decoding: a decode step writes into the block it reads.
        want = (list(whole.token_ids), whole.num_pages, whole.mrope_span)
        ref = _decode(session, [whole], int(ref_logits[0].argmax()))

        for budget in (9, 4, 1):
            blk = session.create_block()
            session.max_prefill_rows = budget
            try:
                logits = session.prefill_block(blk, ids)
            finally:
                session.max_prefill_rows = None
            assert (list(blk.token_ids), blk.num_pages, blk.mrope_span) == want, f"budget={budget}"
            got = _decode(session, [blk], int(logits[0].argmax()))
            if not session._is_hybrid:
                assert got == ref, f"budget={budget}: decode diverged: {got} vs {ref}"
            session.free_block(blk)
        session.free_block(whole)

    def test_batch_of_ragged_jobs(self, session):
        """A batch over budget splits into waves; each request must land what it
        would have alone, and short requests must not be padded or truncated."""
        specs = [_seq(10, 5), _seq(40, 26), _seq(80, 13)]
        refs = []
        for ids in specs:
            blk = session.create_block()
            refs.append((blk, session.prefill_batch([PrefillJob(block=blk, input_ids=ids)])[0]))

        blocks = [session.create_block() for _ in specs]
        jobs = [PrefillJob(block=b, input_ids=i) for b, i in zip(blocks, specs)]
        session.max_prefill_rows = 12
        try:
            got = session.prefill_batch(jobs)
        finally:
            session.max_prefill_rows = None

        for i, (ids, blk, row) in enumerate(zip(specs, blocks, got)):
            assert blk.token_ids == ids.tolist()
            _assert_same_choice(row[0], refs[i][1][0], _atol(session), f"request {i}")
        for blk, _ in refs:
            session.free_block(blk)
        for blk in blocks:
            session.free_block(blk)

    def test_extension_is_chunked_too(self, session):
        """Chunking a prefill that extends a non-empty block: the block's own prefix
        grows under it wave by wave."""
        head, tail = _seq(10, 7), _seq(60, 22)
        whole = session.create_block()
        session.prefill_block(whole, head)
        ref = session.prefill_block(whole, tail)

        chunked = session.create_block()
        session.prefill_block(chunked, head)
        session.max_prefill_rows = 5
        try:
            got = session.prefill_block(chunked, tail)
        finally:
            session.max_prefill_rows = None

        assert chunked.token_ids == head.tolist() + tail.tolist()
        _assert_same_choice(got[0], ref[0], _atol(session), "chunked extension")
        for blk in (whole, chunked):
            session.free_block(blk)

    def test_budget_off_by_default(self, session):
        """The knob must be opt-in: an unset budget changes nothing."""
        assert session.max_prefill_rows is None
        ids = _seq(10, 20)
        a, b = session.create_block(), session.create_block()
        ref, got = session.prefill_block(a, ids), session.prefill_block(b, ids)
        torch.testing.assert_close(got.float().cpu(), ref.float().cpu(), rtol=0, atol=0)
        for blk in (a, b):
            session.free_block(blk)


# =============================================================================
# Images
# =============================================================================


requires_vl = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="VL chunking tests need CUDA"
)


@pytest.fixture(scope="module")
def vl_session(session):
    """The same session — Qwen3.5-0.8B is multimodal *and* hybrid, so these tests
    cover the GDN state hand-off across chunk boundaries as well as image slicing."""
    if not session.engine.config.model_config.is_multimodal:
        pytest.skip(f"{E2E_MODEL_PATH} is not multimodal")
    return session


@pytest.fixture(scope="module")
def image_inputs():
    """The same deterministic image prompt the existing vision tests use.  Processor
    only — loading the HF reference model would initialize CUDA before the engine."""
    from transformers import AutoProcessor

    from test_qwen3_5_vision_encoder import _make_hf_inputs

    inputs = _make_hf_inputs(AutoProcessor.from_pretrained(VL_MODEL_ID), 1)
    return dict(
        input_ids=inputs["input_ids"][0].to(torch.int32),
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        mm_token_type_ids=inputs["mm_token_type_ids"][0],
    )


def _mm_kwargs(image_inputs):
    return {k: v for k, v in image_inputs.items() if k != "input_ids"}


@requires_vl
class TestChunkedImagePrefill:
    def test_boundaries_inside_an_image(self, vl_session, image_inputs):
        """Budgets down to one row put boundaries all over the image run — a case with
        no unchunked analogue — and every one of them must land the same answer."""
        ids = image_inputs["input_ids"]
        n_img = int((image_inputs["mm_token_type_ids"] == 1).sum())
        assert n_img > 8, "image run too short for this sweep to be meaningful"

        whole = vl_session.create_block()
        ref = vl_session.prefill_block(whole, ids, **_mm_kwargs(image_inputs))

        rows, blocks = [], []
        for budget in (len(ids) // 2, len(ids) // 3, 16, 8, 4, 2, 1):
            blk = vl_session.create_block()
            vl_session.max_prefill_rows = budget
            try:
                rows.append(vl_session.prefill_block(blk, ids, **_mm_kwargs(image_inputs))[0])
            finally:
                vl_session.max_prefill_rows = None
            assert blk.token_ids == ids.tolist(), f"budget={budget}"
            assert blk.num_tokens == whole.num_tokens, f"budget={budget}"
            # An image's mRoPE span is not the token count; splitting must not shift it.
            assert blk.mrope_span == whole.mrope_span, f"budget={budget}: span diverged"
            blocks.append(blk)

        chunking_atol = _atol(vl_session)
        if vl_session._is_hybrid:
            # User-approved 2026-09-16: 1.0 -> 1.1 (+10%) for this comparison
            # only. Original FP8 revision 46f8eca also measured 1.015869.
            # This is a tolerance adjustment, not a numerical fix; keep argmax,
            # token/mRoPE bookkeeping and the unchunked comparison unchanged.
            chunking_atol *= 1.1
        _assert_chunkings_agree(rows, chunking_atol, "image prefill")
        _assert_same_choice(rows[0], ref[0], _atol(vl_session), "image prefill vs unchunked")
        for blk in (whole, *blocks):
            vl_session.free_block(blk)

    def test_tower_runs_once_per_request(self, vl_session, image_inputs):
        """The point of precomputing: chunking must not re-embed the image."""
        ids = image_inputs["input_ids"]
        visual = vl_session.engine.model.model.visual
        original, calls = visual.forward, []

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        visual.forward = counting
        block = vl_session.create_block()
        vl_session.max_prefill_rows = max(1, len(ids) // 4)
        try:
            vl_session.prefill_block(block, ids, **_mm_kwargs(image_inputs))
        finally:
            vl_session.max_prefill_rows = None
            visual.forward = original
        assert len(calls) == 1, f"tower ran {len(calls)} times, expected once"
        vl_session.free_block(block)

    def test_decode_after_chunked_image_prefill(self, vl_session, image_inputs):
        """A chunk-filled image block must be usable: decoding from it has to start
        from the same token an unchunked prefill picks and then stay on a plausible
        continuation rather than falling apart.

        The streams are not required to match step for step — this model is hybrid, so
        the state hand-off can flip a near-tie a few tokens in.  What is required is
        that the block itself is committed identically, which the assertions below and
        ``test_boundaries_inside_an_image`` cover.
        """
        ids = image_inputs["input_ids"]
        whole = vl_session.create_block()
        ref_logits = vl_session.prefill_block(whole, ids, **_mm_kwargs(image_inputs))
        # Snapshot before decoding: a decode step writes into the block it reads.
        want = (list(whole.token_ids), whole.num_pages, whole.mrope_span)
        ref = _decode(vl_session, [whole], int(ref_logits[0].argmax()))

        for budget in (len(ids) // 3, 8, 1):
            blk = vl_session.create_block()
            vl_session.max_prefill_rows = budget
            try:
                logits = vl_session.prefill_block(blk, ids, **_mm_kwargs(image_inputs))
            finally:
                vl_session.max_prefill_rows = None
            assert (list(blk.token_ids), blk.num_pages, blk.mrope_span) == want, f"budget={budget}"
            got = _decode(vl_session, [blk], int(logits[0].argmax()))
            assert got[0] == ref[0], f"budget={budget}: first decoded token differs"
            vl_session.free_block(blk)
        vl_session.free_block(whole)

    def test_image_prefill_in_context(self, vl_session, image_inputs):
        """An image appended to a block the new tokens also attend to."""
        ids = image_inputs["input_ids"]
        prompt_ids = _seq(200, 9)

        refs = []
        for rows in (None, max(1, len(ids) // 4)):
            prompt = vl_session.create_block()
            vl_session.prefill_block(prompt, prompt_ids)
            blk = vl_session.create_block()
            vl_session.max_prefill_rows = rows
            try:
                refs.append(
                    vl_session.prefill_block(
                        blk, ids, context=[prompt], **_mm_kwargs(image_inputs)
                    )
                )
            finally:
                vl_session.max_prefill_rows = None
            assert blk.token_ids == ids.tolist()
            vl_session.free_block(prompt)
            vl_session.free_block(blk)
        _assert_same_choice(refs[1][0], refs[0][0], _atol(vl_session), "in-context image chunking")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([sys.argv[0], "-v", "-k", "Slicing or JobRows"]))
