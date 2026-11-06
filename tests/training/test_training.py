import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest
import torch
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM, Trace, replay_trace
from minisgl.training.attention import attention
from minisgl.training.trace import BlockUpdate, ForwardBatch, ForwardRequest
from minisgl_examples.training import sft

from tests.training.fixtures import fp64_attention as fp64_attention
from tests.training.helpers import Session, generate_stress_example, make_llm
from tests.training.models import tiny_model
from tests.training.references import Float64AttentionMath, sequential_attention, visible_sources
from tests.training.workloads import (
    causal_trace,
    example_trace,
    mutation_trace,
    repeated_view_trace,
    single_block_trace,
)


@pytest.mark.parametrize("query_chunk_size", [1, 7, None])
@pytest.mark.parametrize(
    "trace_kind,kv_heads,rotary_width",
    [("single", 4, 8), ("mutation", 2, 8), ("stress", 1, 4), ("repeated", 2, 4)],
)
def test_attention_fp64_algebra_and_fp32_rounding(
    trace_kind, kv_heads, rotary_width, query_chunk_size, record_property, fp64_attention
):
    if trace_kind == "stress":
        data = generate_stress_example()
        trace = Trace.from_dict(data["trace"])
    elif trace_kind == "repeated":
        trace = repeated_view_trace()
    else:
        trace = single_block_trace(True) if trace_kind == "single" else mutation_trace()
    n = len(trace.tokens)
    replay = trace.to(query_chunk_size=query_chunk_size or n)
    torch.manual_seed(7)
    inputs = tuple(
        (torch.randn(n, heads, 8).double().requires_grad_() for heads in (4, kv_heads, kv_heads))
    )
    inv_freq = (10000 ** (-torch.arange(0, rotary_width, 2).float() / rotary_width)).double()
    expected = sequential_attention(*inputs, trace, replay.order, inv_freq, 8 ** (-0.5))
    weights = torch.randn(expected.shape).double()
    expected_grads = torch.autograd.grad((expected * weights).sum(), inputs)
    with Float64AttentionMath():
        actual = attention(*inputs, replay, inv_freq, 8 ** (-0.5))
        actual_grads = torch.autograd.grad((actual * weights).sum(), inputs)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-12, rtol=2e-12)
    record_property("fp64_output_max_abs", (actual - expected).abs().max().item())
    record_property(
        "fp64_gradient_max_abs",
        max(((a - b).abs().max().item() for a, b in zip(actual_grads, expected_grads))),
    )
    inputs32 = tuple((x.detach().float().requires_grad_() for x in inputs))
    actual32 = attention(*inputs32, replay, inv_freq.float(), 8 ** (-0.5))
    grads32 = torch.autograd.grad((actual32 * weights.float()).sum(), inputs32)
    torch.testing.assert_close(actual32.double(), expected, atol=2e-06, rtol=2e-05)
    for actual_grad, expected_grad in zip(grads32, expected_grads):
        torch.testing.assert_close(actual_grad.double(), expected_grad, atol=1e-05, rtol=0.0002)
    record_property("fp32_output_max_abs", (actual32.double() - expected).abs().max().item())
    record_property(
        "fp32_gradient_max_abs",
        max(((a.double() - b).abs().max().item() for a, b in zip(grads32, expected_grads))),
    )


def test_attention_copied_cleared_history_gradcheck(fp64_attention):
    trace = Trace(
        batches=[
            ForwardBatch(0, "prefill", (ForwardRequest((1, 2), 0, 0, ((0, 0),)),)),
            ForwardBatch(
                1,
                "decode",
                (
                    ForwardRequest((3,), 0, 0, ((1, 4), (0, 0))),
                    ForwardRequest((4,), 1, 4, ((0, 0), (1, 4))),
                ),
            ),
        ],
        updates=[
            BlockUpdate(1, 1, refs=((0, 2),)),
            BlockUpdate(1, 1, refs=((1, 2), (1, 2))),
            BlockUpdate(1, 0, refs=()),
        ],
    )
    replay = trace.to(query_chunk_size=2)
    torch.manual_seed(7)
    inputs = tuple(
        torch.randn(4, heads, 2, dtype=torch.float64, requires_grad=True) for heads in (2, 1, 1)
    )
    inv_freq = torch.ones(1, dtype=torch.float64)
    with Float64AttentionMath():
        assert torch.autograd.gradcheck(
            lambda *values: attention(*values, replay, inv_freq, 2**-0.5),
            inputs,
            eps=1e-6,
            atol=1e-7,
            rtol=1e-5,
        )


def test_attention_sparse_self_and_partial_copied_reads():
    copied = ((0, 32), (1, 32), (0, 32))
    trace = Trace(
        batches=[
            ForwardBatch(
                0,
                "prefill",
                tuple(
                    ForwardRequest(tuple(range(32)), block, 0, ((block, 0),)) for block in (0, 1)
                ),
            ),
            ForwardBatch(
                1,
                "decode",
                (
                    ForwardRequest((1,), 2, 96, ((0, 11),)),
                    ForwardRequest((2,), 4, 0, ((2, 74),)),
                ),
            ),
        ],
        updates=[BlockUpdate(1, 2, refs=copied), BlockUpdate(1, 3, refs=copied), BlockUpdate(1, 3)],
    )
    replay = trace.to(query_chunk_size=128)
    torch.manual_seed(7)
    q = torch.randn(len(trace.tokens), 4, 8, requires_grad=True)
    k = torch.randn(len(trace.tokens), 2, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    inv_freq = tiny_model().model.rotary_emb.inv_freq
    expected = sequential_attention(q, k, v, trace, replay.order, inv_freq, 8**-0.5)
    actual = attention(q, k, v, replay, inv_freq, 8**-0.5)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    target, excluded = replay.order.argsort()[[65, 64]]
    expected_grads = torch.autograd.grad(expected[target].square().sum(), (q, k, v))
    actual_grads = torch.autograd.grad(actual[target].square().sum(), (q, k, v))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=2e-4)
    assert not actual_grads[1][excluded].any() and not actual_grads[2][excluded].any()


@pytest.mark.parametrize(
    "rows",
    [
        True,
        1,
        [True],
        [1.5],
        torch.tensor([True]),
        torch.tensor([1.0]),
        torch.tensor(1),
        torch.tensor([[1]]),
        [-1],
        [19],
        torch.tensor([-1]),
        torch.tensor([19]),
    ],
)
def test_replay_trace_rejects_invalid_rows(rows):
    with pytest.raises((TypeError, ValueError, IndexError), match="rows"):
        replay_trace(tiny_model(), mutation_trace(), rows=rows)


def test_serialized_self_append_history_stays_compact():

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                block = await session.create_block()
                await session.prefill_block([1], write_to=block, cache_view=[])
                for _ in range(20):
                    await session.append_block(block, block)
                first_size = len(json.dumps(asdict(session.get_trace())))
                for _ in range(20):
                    await session.append_block(block, block)
                assert block.num_tokens == 2**40
                assert len(json.dumps(asdict(session.get_trace()))) < 3 * first_size
                assert sum((len(update.refs) for update in session.get_trace().updates)) == 80
                assert (
                    session.get_trace().updates[-1].refs
                    == ((block.block_id, block.num_tokens // 2),) * 2
                )
                await session.forward(cache_view=AsyncContext([], block, 2))
                trace = Trace.from_dict(json.loads(json.dumps(asdict(session.get_trace()))))
                replay = trace.to()
                assert sum(map(len, replay.spans)) == 2
                assert visible_sources(replay) == [[0], [1]]
        finally:
            await llm.close()

    return asyncio.run(run())


def test_deep_copy_chain_resolves_without_flattening_or_recursion():

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                block, empty = [await session.create_block() for _ in range(2)]
                await session.prefill_block([1, 2], write_to=block, cache_view=[])
                for _ in range(1100):
                    block = await session.merge_blocks(block, empty)
                await session.forward(cache_view=AsyncContext([block], block, 3))
                replay = session.get_trace().to()
                assert sum(map(len, replay.spans)) == 2
                assert visible_sources(replay) == [[0], [0, 1], [0, 1, 2]]
        finally:
            await llm.close()

    return asyncio.run(run())


def test_nested_copy_ranges_keep_destination_offsets():
    trace = Trace(
        batches=[
            ForwardBatch(
                0,
                "prefill",
                (
                    ForwardRequest((1, 2, 3), 0, 0, ((0, 0),)),
                    ForwardRequest((4, 5), 1, 0, ((1, 0),)),
                ),
            ),
            ForwardBatch(1, "decode", (ForwardRequest((6,), 3, 10, ((3, 10),)),)),
        ],
        updates=[
            BlockUpdate(1, 2, refs=((0, 3), (1, 2), (0, 3))),
            BlockUpdate(1, 3, refs=((2, 8), (1, 2))),
        ],
    )
    replay = trace.to()
    version = replay.reads[-1][0][0]
    assert list(replay.compute_source_spans(version, 2, 9)) == [
        (2, 3, 2),
        (3, 5, 3),
        (0, 3, 5),
        (3, 4, 8),
    ]


@pytest.mark.parametrize(
    "update",
    [
        BlockUpdate(1, 1, refs=((0, -1),)),
        BlockUpdate(1, 1, refs=((0, 1),)),
        BlockUpdate(1, 1, refs=((0, 3),)),
        BlockUpdate(1, 1, refs=((999, 1),)),
        BlockUpdate(1, 1, refs=((-1, 0),)),
        BlockUpdate(0, 1, refs=((0, 2),)),
        BlockUpdate(2, 1, refs=((999, 1),)),
    ],
)
def test_reject_invalid_copy_references(update):
    trace = Trace(
        batches=[
            ForwardBatch(0, "prefill", (ForwardRequest((1, 2), 0, 0, ((0, 0),)),)),
            ForwardBatch(1, "prefill", (ForwardRequest((3,), 2, 0, ((2, 0),)),)),
        ],
        updates=[update],
    )
    with pytest.raises(ValueError):
        trace.to()


def test_recording_and_masks():
    trace = example_trace()
    assert trace.tokens[7].step == trace.tokens[8].step
    assert trace.tokens[9].position == 3
    replay = trace.to()
    visible = visible_sources(replay)
    assert 8 in visible[7] and 7 in visible[8]
    assert 9 in visible[9] and 10 not in visible[9]
    assert 9 not in visible[7]
    assert len(trace.tokens) - 1 in visible[-1] and 7 not in visible[-1]


def test_causal_metadata_is_compact():
    length = 1024
    trace = causal_trace(length)
    replay = trace.to()
    assert replay.spans == [[(0, length, 0)]]
    assert len(replay.reads) == length
    assert sum(map(len, replay.reads)) == length
    for value in vars(replay).values():
        if isinstance(value, torch.Tensor):
            assert value.numel() <= length


@pytest.mark.parametrize(
    "options",
    [
        {"gdn_segment_size": 0},
        {"gdn_segment_size": -1},
        {"gdn_read_mode": "invalid"},
        {"gdn_read_backend": "invalid"},
    ],
)
def test_gdn_options_are_rejected_before_training_data_is_loaded(options):
    with pytest.raises(ValueError, match="gdn_"):
        example_trace().to(**options)
    with pytest.raises(ValueError, match="gdn_"):
        sft.train(NS(**options))


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_replay_preserves_gdn_read_backend_on_transfer(backend):
    replay = example_trace().to(gdn_read_backend=backend)
    replay._gdn["cached"] = object()
    transferred = replay.to("cpu")
    assert replay.gdn_read_backend == transferred.gdn_read_backend == backend
    assert transferred is replay
    assert transferred._gdn["cached"] is replay._gdn["cached"]


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], (8, "expanded", True, "torch")),
        (
            ["--gdn-segment-size", "2", "--gdn-read-mode", "factorized", "--no-gdn-cache-reused"],
            (2, "factorized", False, "torch"),
        ),
        (["--gdn-cache-reused"], (8, "expanded", True, "torch")),
        (["--gdn-read-backend", "triton"], (8, "expanded", True, "triton")),
    ],
)
def test_sft_cli_passes_gdn_options(monkeypatch, tmp_path, flags, expected):
    received = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["sft", "train", *flags])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(sft, "train", received.append)
    sft.main()
    assert len(received) == 1
    args = received[0]
    assert (
        args.gdn_segment_size,
        args.gdn_read_mode,
        args.gdn_cache_reused,
        args.gdn_read_backend,
    ) == expected


@pytest.mark.parametrize(
    "flags",
    [
        ["--gdn-segment-size", "0"],
        ["--gdn-read-mode", "invalid"],
        ["--gdn-read-backend", "invalid"],
    ],
)
def test_sft_cli_rejects_invalid_gdn_options(monkeypatch, tmp_path, flags):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["sft", "train", *flags])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(sft, "train", lambda args: pytest.fail("Invalid options reached training"))
    with pytest.raises(SystemExit) as error:
        sft.main()
    assert error.value.code == 2


def test_copied_spans_keep_duplicate_occurrences():
    replay = mutation_trace().to()
    visible = visible_sources(replay)
    assert len(visible[14]) > len(set(visible[14]))
    assert visible[14].count(2) == 4


def test_reject_unrecorded_blocks():

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                block = NS(block_id=0, num_tokens=1)
                with pytest.raises(ValueError, match="Unrecorded"):
                    await session.forward(cache_view=AsyncContext([block], block, 1))
                with pytest.raises(ValueError, match="Unrecorded"):
                    await session.merge_blocks(block, block)
        finally:
            await llm.close()

    return asyncio.run(run())


def test_recording_mutations():

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                block, empty = [await session.create_block() for _ in range(2)]
                job = NS(block=block, input_ids=torch.tensor([1, 2]), context=[], pixel_values=None)
                await asyncio.gather(
                    *(
                        session.prefill_block(
                            job.input_ids,
                            write_to=job.block,
                            cache_view=job.context,
                            pixel_values=getattr(job, "pixel_values", None),
                            image_grid_thw=getattr(job, "image_grid_thw", None),
                            mm_token_type_ids=getattr(job, "mm_token_type_ids", None),
                        )
                        for job in [job]
                    )
                )
                copied = await session.merge_blocks(empty, block)
                assert await session.append_block(block, block) is block
                assert [u.refs for u in session.get_trace().updates] == [
                    ((empty.block_id, 0), (block.block_id, 2)),
                    ((block.block_id, 2), (block.block_id, 2)),
                ]
                assert all((u.step == 1 for u in session.get_trace().updates))
                await session.free_block(block)
                assert session.get_trace().updates[-1].refs == ()
                await asyncio.gather(
                    *(
                        session.prefill_block(
                            job.input_ids,
                            write_to=job.block,
                            cache_view=job.context,
                            pixel_values=getattr(job, "pixel_values", None),
                            image_grid_thw=getattr(job, "image_grid_thw", None),
                            mm_token_type_ids=getattr(job, "mm_token_type_ids", None),
                        )
                        for job in [job]
                    )
                )
                assert session.get_trace().tokens[2].block == session.get_trace().tokens[0].block
                assert session.get_trace().tokens[2].position == 0
                copied_again = await session.merge_blocks(copied, empty)
                assert session.get_trace().updates[-1].refs == (
                    (copied.block_id, 2),
                    (empty.block_id, 0),
                )
                await session.free_block(copied)
                context = NS(cache_view=[copied_again, block], output_block=block)
                await session.forward(
                    cache_view=AsyncContext(context.cache_view, context.output_block, 3)
                )
                assert len(session.get_trace().tokens) == 5
                replay = session.get_trace().to()
                assert replay.input_ids.numel() == 5
                visible = visible_sources(replay)
                assert visible[2] == [2] and visible[3] == [2, 3]
                assert visible[4] == [0, 1, 2, 3, 4]
        finally:
            await llm.close()

    return asyncio.run(run())


@pytest.mark.parametrize("operation", ["merge_blocks", "append_block", "free_block"])
def test_failed_mutation_does_not_commit_trace(operation):

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                block = await session.create_block()
                await session.prefill_block([1, 2], write_to=block)

                def fail(*args):
                    raise RuntimeError("failed")

                setattr(raw, operation, fail)
                args = (block,) if operation == "free_block" else (block, block)
                with pytest.raises(RuntimeError, match="failed"):
                    await getattr(session, operation)(*args)
                assert session.get_trace().updates == []
                with pytest.raises(RuntimeError, match="Discard"):
                    await session.forward(cache_view=AsyncContext([block], block, 1))
        finally:
            await llm.close()

    return asyncio.run(run())


def test_failed_forward_does_not_commit_trace():

    async def run():
        llm = None
        session = None
        try:

            class FailingSession(Session):

                def decode_step(self, group, input_ids):
                    raise RuntimeError("failed")

            llm, raw = make_llm(FailingSession())
            session = RecordedAsyncLLM(llm)
            session.__enter__()
            block = NS(block_id=0, num_tokens=0)
            group = [NS(cache_view=[block], output_block=block)]
            with pytest.raises(RuntimeError, match="failed"):
                await asyncio.gather(
                    *(
                        session.forward(
                            cache_view=AsyncContext(ctx.cache_view, ctx.output_block, int(token))
                        )
                        for ctx, token in zip(group, torch.tensor([1]), strict=True)
                    )
                )
            assert session.get_trace().tokens == [] and session.get_trace().steps == 0
            with pytest.raises(RuntimeError, match="Discard"):
                await asyncio.gather(
                    *(
                        session.forward(
                            cache_view=AsyncContext(ctx.cache_view, ctx.output_block, int(token))
                        )
                        for ctx, token in zip(group, torch.tensor([1]), strict=True)
                    )
                )
        finally:
            if session is not None:
                session.__exit__(None, None, None)
            if llm is not None:
                await llm.close()

    return asyncio.run(run())


def test_reject_future_visibility():
    trace = Trace(
        batches=[
            ForwardBatch(0, "decode", (ForwardRequest((1,), 0, 0, ((1, 1), (0, 0))),)),
            ForwardBatch(1, "decode", (ForwardRequest((2,), 1, 0, ((1, 0),)),)),
        ]
    )
    with pytest.raises(ValueError, match="future"):
        trace.to()
