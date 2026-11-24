import asyncio
import importlib
import json
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest
import torch
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM, Trace
from minisgl.training.gdn import GDNReplay
from minisgl.training.trace import BlockUpdate, ForwardBatch, ForwardRequest, TensorData, Token

from tests.training.helpers import Session, make_llm
from tests.training.workloads import mutation_trace, randomized_trace, repeated_view_trace


def job(block, ids, context=(), **kwargs):
    return NS(
        **{
            "block": block,
            "input_ids": torch.tensor(ids),
            "context": context,
            "pixel_values": None,
            **kwargs,
        }
    )


def equal_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            equal_tree(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("tokens", [(), (Token(1, 0, 0, 0, ((0, 1),), "prefill"),)])
def test_trace_rejects_token_constructor(tokens, positional):
    with pytest.raises(TypeError):
        Trace(tokens) if positional else Trace(tokens=tokens)


@pytest.mark.parametrize("version", [None, 1])
def test_trace_requires_explicit_current_format_version(version):
    data = asdict(Trace())
    if version is None:
        del data["format_version"]
    else:
        data["format_version"] = version
    with pytest.raises(ValueError, match="Unsupported trace format_version"):
        Trace.from_dict(data)


@pytest.mark.parametrize("sources", [(), (0, 1)])
def test_block_updates_reject_source_rows_including_clear(sources):
    with pytest.raises(TypeError, match="sources"):
        BlockUpdate(1, 2, sources=sources)
    data = asdict(Trace(updates=[BlockUpdate(1, 2)]))
    data["updates"][0]["sources"] = sources
    with pytest.raises(ValueError, match="BlockUpdate.sources is unsupported"):
        Trace.from_dict(data)


def test_empty_current_trace_roundtrip():
    trace = Trace()
    assert Trace.from_dict(json.loads(json.dumps(asdict(trace)))) == trace


def test_current_recording_and_roundtrip():

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                block = await session.create_block()
                await session.prefill_block([1, 2], write_to=block)
                copied = await session.merge_blocks(block, block)
                await session.free_block(block)
                await session.prefill_block([3], write_to=block, cache_view=[copied])
                await session.free_block(copied)
                restored = Trace.from_dict(json.loads(json.dumps(asdict(session.get_trace()))))
                assert restored == session.get_trace()
                assert all(
                    (
                        set(asdict(update)) == {"step", "block", "refs"}
                        for update in restored.updates
                    )
                )
                assert restored.to().num_tokens == 3
        finally:
            await llm.close()

    return asyncio.run(run())


@pytest.mark.parametrize("name", ["sft", "stress"])
def test_training_script_help_available(monkeypatch, capsys, name):
    cli = importlib.import_module(f"minisgl_examples.training.{name}")
    monkeypatch.setattr("sys.argv", [name, "--help"])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize(
    "factory", [mutation_trace, repeated_view_trace, lambda: randomized_trace(7)]
)
def test_request_trace_roundtrip_preserves_compiled_metadata(factory):
    trace = factory()
    encoded = json.loads(json.dumps(asdict(trace)))
    assert encoded["format_version"] == 2 and "tokens" not in encoded
    restored = Trace.from_dict(encoded)
    assert restored == trace
    assert restored.tokens == trace.tokens
    assert restored.num_tokens == len(trace.tokens)
    a, b = restored.to(), trace.to()
    equal_tree(
        (a.input_ids, a.order, a.spans, a.reads, a.copies),
        (b.input_ids, b.order, b.spans, b.reads, b.copies),
    )
    order = a.order
    for mode in ("expanded", "factorized"):
        a = GDNReplay.from_trace(restored, order, read_mode=mode)
        b = GDNReplay.from_trace(trace, order, read_mode=mode)
        equal_tree(
            (a.conv_sources, a.levels, a.reads, a.partitions),
            (b.conv_sources, b.levels, b.reads, b.partitions),
        )


def test_requests_preserve_batch_visibility_order_and_causal_prefixes():

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                prompt, a, b = [await session.create_block() for _ in range(3)]
                await session.prefill_block([1, 2, 3], write_to=prompt, cache_view=[])
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
                        for job in [job(a, [4, 5], [prompt]), job(b, [6, 7], [prompt])]
                    )
                )
                group = [
                    NS(cache_view=[prompt, b, a, b], output_block=a),
                    NS(cache_view=[a, prompt], output_block=b),
                ]
                await asyncio.gather(
                    *(
                        session.forward(
                            cache_view=AsyncContext(ctx.cache_view, ctx.output_block, int(token))
                        )
                        for ctx, token in zip(group, torch.tensor([8, 9]), strict=True)
                    )
                )
                trace = session.get_trace()
                p, x, y = (prompt.block_id, a.block_id, b.block_id)
                expected = [Token(i + 1, 0, p, i, ((p, i + 1),), "prefill") for i in range(3)]
                for block, ids in ((x, [4, 5]), (y, [6, 7])):
                    expected.extend(
                        (
                            Token(token, 1, block, i, ((p, 3), (block, i + 1)), "prefill")
                            for i, token in enumerate(ids)
                        )
                    )
                expected.extend(
                    [
                        Token(8, 2, x, 2, ((p, 3), (y, 3), (x, 3), (y, 3)), "decode"),
                        Token(9, 2, y, 2, ((x, 3), (p, 3)), "decode"),
                    ]
                )
                assert trace.tokens == expected
                assert [
                    (batch.step, batch.kind, len(batch.requests)) for batch in trace.batches
                ] == [(0, "prefill", 1), (1, "prefill", 2), (2, "decode", 2)]
                request = trace.batches[0].requests[0]
                assert (request.input_ids, request.write_to, request.write_start) == (
                    (1, 2, 3),
                    p,
                    0,
                )
                assert trace.batches[-1].requests[0].cache_view == ((p, 3), (y, 2), (x, 2), (y, 2))
                assert trace.batches[-1].requests[1].cache_view == ((x, 2), (p, 3))
        finally:
            await llm.close()

    return asyncio.run(run())


def test_request_snapshot_survives_input_mutation_and_block_reuse():

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                a, b = [await session.create_block() for _ in range(2)]
                first = job(a, [1, 2])
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
                        for job in [first]
                    )
                )
                prefix = asdict(session.get_trace().batches[0])
                first.input_ids.fill_(99)
                await session.append_block(a, a)
                await session.prefill_block([3, 4], write_to=b, cache_view=[a, a])
                await session.free_block(a)
                await session.prefill_block([5], write_to=a, cache_view=[b])
                assert asdict(session.get_trace().batches[0]) == prefix
                assert [
                    r.input_ids for batch in session.get_trace().batches for r in batch.requests
                ] == [(1, 2), (3, 4), (5,)]
                assert [t.position for t in session.get_trace().tokens] == [0, 1, 0, 1, 0]
                assert session.get_trace().batches[1].requests[0].cache_view == (
                    (a.block_id, 4),
                    (a.block_id, 4),
                    (b.block_id, 0),
                )
                assert session.get_trace().updates[-1].refs == ()
                Trace.from_dict(json.loads(json.dumps(asdict(session.get_trace())))).to()
        finally:
            await llm.close()

    return asyncio.run(run())


def test_long_prefill_stores_one_request_without_repeated_metadata(record_property):

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                block = await session.create_block()
                await session.prefill_block(torch.arange(2048), write_to=block)
                trace = session.get_trace()
                data = asdict(trace)
                assert len(data["batches"]) == len(data["batches"][0]["requests"]) == 1
                assert data["batches"][0]["requests"][0]["input_ids"] == tuple(range(2048))
                new_bytes = len(json.dumps(data))
                token_bytes = len(json.dumps([asdict(token) for token in trace.tokens]))
                assert new_bytes < token_bytes / 8
                record_property("request_trace_bytes", new_bytes)
                record_property("token_metadata_bytes", token_bytes)
                assert [token.token_id for token in trace.tokens] == list(range(2048))
                assert [token.position for token in trace.tokens] == list(range(2048))
        finally:
            await llm.close()

    return asyncio.run(run())


def test_image_inputs_and_positions_roundtrip_without_aliasing():

    async def run():
        llm = None
        session = None
        try:
            positions = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]])

            class ImageSession(Session):

                def _mrope_rel(self, input_ids, types, grid):
                    return positions

                def prefill_batch(self, jobs):
                    for item in jobs:
                        start = getattr(item.block, "mrope_span", item.block.num_tokens)
                        super().prefill_batch([item])
                        length = 2 if item.pixel_values is not None else item.input_ids.numel()
                        item.block.mrope_span = start + length
                    return [torch.zeros(1, 32) for _ in jobs]

            llm, raw = make_llm(ImageSession())
            session = RecordedAsyncLLM(llm)
            session.__enter__()
            prompt, block = [await session.create_block() for _ in range(2)]
            await session.prefill_block([1, 2], write_to=prompt)
            pixels = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
            grid, types = (torch.tensor([[1, 2, 2]]), torch.ones(4, dtype=torch.int64))
            await session.prefill_block(
                [3, 4, 5, 6],
                write_to=block,
                cache_view=[prompt],
                pixel_values=pixels,
                image_grid_thw=grid,
                mm_token_type_ids=types,
            )
            request = session.get_trace().batches[-1].requests[0]
            for name, expected in (
                ("pixel_values", pixels),
                ("image_grid_thw", grid),
                ("mm_token_type_ids", types),
                ("mrope_positions", positions),
            ):
                tensor = getattr(request, name)
                assert tensor.shape == tuple(expected.shape)
                assert tensor.dtype == str(expected.dtype).removeprefix("torch.")
                torch.testing.assert_close(tensor.to_tensor(), expected, rtol=0, atol=0)
            assert request.write_span == 0 and request.cache_spans == (2, 0)
            snapshot = asdict(request)
            assert "image_embeds" not in snapshot
            pixels.zero_()
            grid.fill_(99)
            types.zero_()
            positions.fill_(99)
            assert asdict(request) == snapshot
            await session.prefill_block([7, 8], write_to=block)
            extension = session.get_trace().batches[-1].requests[0]
            assert extension.write_start == 4 and extension.write_span == 2
            torch.testing.assert_close(
                extension.mrope_positions.to_tensor(),
                torch.tensor([[2, 3], [2, 3], [2, 3]]),
                rtol=0,
                atol=0,
            )
            restored = Trace.from_dict(json.loads(json.dumps(asdict(session.get_trace()))))
            assert restored == session.get_trace()
            assert restored.batches[1].requests[0].pixel_values.to_tensor().sum().item() == 66
            replay = restored.to()
            assert replay.input_ids.numel() == restored.num_tokens
            gdn = GDNReplay.from_trace(restored, replay.order)
            assert gdn.conv_sources.shape[0] == restored.num_tokens
        finally:
            if session is not None:
                session.__exit__(None, None, None)
            if llm is not None:
                await llm.close()

    return asyncio.run(run())


@pytest.mark.parametrize("has_pixels", [False, True])
@pytest.mark.parametrize("rows", [None, 0, 1])
def test_trace_rejects_image_embeddings_including_null(rows, has_pixels):
    request = ForwardRequest(
        (1,),
        0,
        0,
        ((0, 0),),
        **(
            {
                "pixel_values": TensorData.capture(torch.zeros(4, 12)),
                "image_grid_thw": TensorData.capture(torch.tensor([[1, 2, 2]])),
                "mm_token_type_ids": TensorData.capture(torch.ones(1, dtype=torch.long)),
                "mrope_positions": TensorData.capture(torch.zeros(3, 1, dtype=torch.long)),
            }
            if has_pixels
            else {}
        ),
    )
    trace = Trace(batches=[ForwardBatch(0, "prefill", (request,))], steps=1)
    data = json.loads(json.dumps(asdict(trace)))
    saved = data["batches"][0]["requests"][0]
    assert "image_embeds" not in saved
    saved["image_embeds"] = (
        None if rows is None else asdict(TensorData.capture(torch.zeros(rows, 32)))
    )
    with pytest.raises(ValueError, match="record pixel_values"):
        Trace.from_dict(data)


@pytest.mark.parametrize("rows", [0, 1])
@pytest.mark.parametrize("has_pixels", [False, True])
def test_recorder_rejects_embedding_jobs_before_session_mutation(rows, has_pixels):
    async def run():
        llm, _ = make_llm()
        try:
            with RecordedAsyncLLM(llm) as recorded:
                block = await recorded.create_block()
                before = asdict(recorded.get_trace())
                with pytest.raises(TypeError, match="image_embeds"):
                    await recorded.prefill_block(
                        [1],
                        write_to=block,
                        pixel_values=torch.zeros(4, 12) if has_pixels else None,
                        image_embeds=torch.zeros(rows, 32),
                    )
                assert asdict(recorded.get_trace()) == before
                assert block.num_tokens == 0
                await recorded.prefill_block([2], write_to=block)
                assert recorded.get_trace().num_tokens == block.num_tokens == 1
        finally:
            await llm.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "path,value",
    [
        (("format_version",), 99),
        (("tokens",), []),
        (("batches", 1, "step"), 0),
        (("batches", 0, "step"), 2),
        (("batches", 0, "kind"), "other"),
        (("batches", 0, "requests"), []),
        (("batches", 0, "requests", 0, "input_ids"), []),
        (("batches", 1, "requests", 0, "input_ids"), [3, 4]),
        (("batches", 0, "requests", 0, "write_start"), -1),
        (("batches", 0, "requests", 0, "cache_view"), [[0, -1]]),
        (
            ("batches", 1, "requests"),
            [
                {"input_ids": [3], "write_to": 0, "write_start": 2, "cache_view": [[0, 2]]},
                {"input_ids": [4], "write_to": 0, "write_start": 2, "cache_view": [[0, 2]]},
            ],
        ),
    ],
)
def test_reject_invalid_request_storage(path, value):

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                block = await session.create_block()
                await session.prefill_block([1, 2], write_to=block, cache_view=[])
                await session.forward(cache_view=AsyncContext([block], block, 3))
                data = json.loads(json.dumps(asdict(session.get_trace())))
                target = data
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with pytest.raises(ValueError):
                    Trace.from_dict(data)
        finally:
            await llm.close()

    return asyncio.run(run())


@pytest.mark.parametrize("method", ["prefill", "forward", "decode"])
def test_failed_forward_keeps_request_storage_atomic(method):
    async def run():
        llm, raw = make_llm()
        try:
            with RecordedAsyncLLM(llm) as recorded:
                block = await recorded.create_block()
                await recorded.prefill_block([1, 2], write_to=block)
                before = asdict(recorded.get_trace())

                def fail(*args):
                    block.num_tokens += 1
                    raise RuntimeError("inference failed")

                setattr(raw, "decode_step" if method == "decode" else "prefill_batch", fail)
                with pytest.raises(RuntimeError, match="inference failed"):
                    if method == "prefill":
                        await recorded.prefill_block([3, 4], write_to=block)
                    elif method == "forward":
                        await recorded.forward([3, 4], [block])
                    else:
                        await recorded.forward([3], [block])
                assert asdict(recorded.get_trace()) == before
                with pytest.raises(RuntimeError, match="Discard"):
                    await recorded.prefill_block([5], write_to=block)
        finally:
            await llm.close()

    asyncio.run(run())


def test_stress_cli_writes_request_trace(monkeypatch, tmp_path, capsys):
    from minisgl_examples.training import stress

    instance, _ = make_llm()
    instance.engine = NS(config=NS(model_config=NS(vocab_size=32)))
    monkeypatch.setattr(stress, "AsyncLLM", lambda *args, **kwargs: instance)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["stress", "--output", "trace.json", "--large-size", "4"])
    stress.main()
    example = json.loads((tmp_path / "trace.json").read_text())["examples"][0]
    assert example["trace"]["format_version"] == 2 and "tokens" not in example["trace"]
    count = Trace.from_dict(example["trace"]).num_tokens
    assert count > 0 and count == len(example["targets"]) == len(example["logprobs"])
    assert instance._closed
    assert f"Saved {count} input rows" in capsys.readouterr().out


def test_mining_aligns_targets_with_batched_request_rows(monkeypatch, tmp_path):
    from contextlib import nullcontext

    from minisgl import llm
    from minisgl_examples.training import sft

    from tests.core.test_async_llm import _make_llm

    instance, session = _make_llm()
    engine = instance.async_engine
    instance.tokenizer = NS(
        encode=lambda *a, **kw: [4, 5, 6],
        apply_chat_template=lambda *a, **kw: [1, 2, 3],
        decode=lambda ids, **kw: str(ids),
        eos_token_id=-1,
    )

    def append(left, right):
        ids = list(right.token_ids)
        if ids:
            left.grow_pages(session._pages(len(ids)), len(ids))
        left.token_ids.extend(ids)
        return left

    def merge(left, right):
        block = session.create_block()
        append(block, left)
        return append(block, right)

    decode_step = session.decode_step

    def decode(group, input_ids):
        logits = decode_step(group, input_ids)
        logits.fill_(-20.0)
        for row, block in enumerate(group.write_to):
            logits[row, 1 + block.block_id % 23] = 20.0
        return logits

    monkeypatch.setattr(session, "append_block", append, raising=False)
    monkeypatch.setattr(session, "merge_blocks", merge, raising=False)
    monkeypatch.setattr(session, "decode_step", decode)
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *a, **kw: nullcontext())
    generator = torch.Generator
    monkeypatch.setattr(llm, "AsyncLLM", lambda *a, **kw: instance)
    monkeypatch.setattr(torch, "Generator", lambda **kw: generator(device="cpu"))
    monkeypatch.setattr(torch, "multinomial", lambda values, *a, **kw: values.argmax().reshape(1))
    output = tmp_path / "mined.json"
    args = NS(model="fake", stress=False, max_tokens=2, seed=7, temperature=1.0, output=output)
    asyncio.run(asyncio.wait_for(sft.mine(args), timeout=10))
    examples = json.loads(output.read_text())["examples"]
    assert instance._closed and len(examples) == 3
    assert instance.async_engine is engine and engine.session is session
    for example in examples:
        trace = Trace.from_dict(example["trace"])
        assert any(batch.kind == "decode" and len(batch.requests) == 2 for batch in trace.batches)
        assert sum(target != -100 for target in example["targets"]) == 6
        assert len(example["targets"]) == len(example["logprobs"]) == trace.num_tokens
        for token, target, probability in zip(
            trace.tokens, example["targets"], example["logprobs"], strict=True
        ):
            labelled = token.kind == "decode" and token.position in (2, 3)
            assert target == (1 + token.block % 23 if labelled else -100)
            assert probability == (0.0 if labelled else None)
        assert sum(token.kind == "decode" and token.position == 4 for token in trace.tokens) == 3
