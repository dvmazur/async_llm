import asyncio
import importlib
import json
from dataclasses import asdict, replace
from types import SimpleNamespace as NS

import pytest
import torch
from minisgl.training import Trace, replay_trace
from minisgl.training.gdn import GDNReplay
from minisgl.training.recording import _TraceRecorder
from minisgl.training.trace import (
    BlockUpdate,
    ForwardBatch,
    ForwardRequest,
    TensorData,
    Token,
)
from test_training import (
    Session,
    mutation_trace,
    randomized_trace,
    repeated_view_trace,
    sequential,
    tiny_model,
)


class RequestSession(Session):
    def prefill_batch(self, jobs):
        super().prefill_batch(jobs)
        return [None] * len(jobs)


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
    session = _TraceRecorder(RequestSession())
    block = session.create_block()
    session.prefill_block(block, torch.tensor([1, 2]))
    copied = session.merge_blocks(block, block)
    session.free_block(block)
    session.prefill_block(block, torch.tensor([3]), context=[copied])
    session.free_block(copied)
    restored = Trace.from_dict(json.loads(json.dumps(asdict(session.trace))))
    assert restored == session.trace
    assert all(set(asdict(update)) == {"step", "block", "refs"} for update in restored.updates)
    assert restored.to().num_tokens == 3


@pytest.mark.parametrize("name", ["sft", "stress"])
def test_cli_future_deprecation_warning_keeps_help_available(monkeypatch, capsys, name):
    cli = importlib.import_module(f"minisgl.training.{name}")
    monkeypatch.setattr("sys.argv", [name, "--help"])
    with pytest.warns(FutureWarning, match=f"minisgl.training.{name} CLI"):
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


def test_preparation_keeps_recording_serializable_and_reuses_unchanged_cache():
    trace, model = mutation_trace(), tiny_model()
    recorded = asdict(trace)
    expected = replay_trace(model, trace)
    chunks = next(iter(trace._attention.values()))
    actual = replay_trace(model, trace)
    assert next(iter(trace._attention.values())) is chunks
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert asdict(trace) == recorded
    assert Trace.from_dict(json.loads(json.dumps(asdict(trace)))) == trace


def test_recording_append_recompiles_and_configured_trace_keeps_snapshot():
    session = _TraceRecorder(RequestSession())
    block = session.create_block()
    session.prefill_block(block, torch.tensor([1, 2]))
    trace, model = session.trace, tiny_model()
    before = replay_trace(model, trace, hidden_only=False)
    chunks = next(iter(trace._attention.values()))
    configured = trace.to(query_chunk_size=1, attention_checkpointing=False)
    assert configured is not trace
    assert configured.query_chunk_size == 1 and trace.query_chunk_size == 128
    session.prefill_block(block, torch.tensor([3, 4]))
    actual = replay_trace(model, trace, hidden_only=False)
    torch.testing.assert_close(actual, sequential(model, trace), atol=2e-6, rtol=2e-5)
    assert actual.shape[0] == trace.num_tokens == 4
    assert next(iter(trace._attention.values())) is not chunks
    assert configured.num_tokens == 2 and len(configured.batches) == 1
    torch.testing.assert_close(replay_trace(model, configured, hidden_only=False), before)


def test_recording_can_continue_between_replay_forward_and_backward():
    session = _TraceRecorder(RequestSession())
    block = session.create_block()
    session.prefill_block(block, torch.tensor([1, 2]))
    reference, model = tiny_model(), tiny_model()
    trace = session.trace
    trace.query_chunk_size = 1
    expected = replay_trace(reference, Trace.from_dict(asdict(trace)), hidden_only=False)
    actual = replay_trace(model, trace, hidden_only=False)
    session.prefill_block(block, torch.tensor([3, 4]))
    replay_trace(model, trace, hidden_only=False)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for expected_parameter, actual_parameter in zip(
        reference.parameters(), model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            actual_parameter.grad, expected_parameter.grad, atol=1e-5, rtol=2e-4
        )


def test_replacing_recorded_batch_invalidates_cached_inputs_and_visibility():
    trace, model = mutation_trace(), tiny_model()
    before = replay_trace(model, trace, hidden_only=False)
    chunks = next(iter(trace._attention.values()))
    batch = trace.batches[0]
    first, *rest = batch.requests
    trace.batches[0] = replace(batch, requests=(replace(first, input_ids=(21, 22)), *rest))
    actual = replay_trace(model, trace, hidden_only=False)
    torch.testing.assert_close(actual, sequential(model, trace), atol=2e-6, rtol=2e-5)
    assert not torch.allclose(actual, before)
    assert next(iter(trace._attention.values())) is not chunks


def test_row_selection_uses_token_count_after_recorded_batch_changes():
    trace = Trace(batches=[ForwardBatch(0, "prefill", (ForwardRequest((1, 2), 0, 0, ((0, 0),)),))])
    model = tiny_model()
    replay_trace(model, trace)
    batch = trace.batches[0]
    request = batch.requests[0]
    trace.batches[0] = replace(batch, requests=(replace(request, input_ids=(1, 2, 3)),))
    actual = replay_trace(model, trace, rows=[2], hidden_only=False)
    torch.testing.assert_close(actual, sequential(model, trace)[[2]], atol=2e-6, rtol=2e-5)
    assert trace.num_tokens == 3


def test_replacing_block_update_invalidates_cached_provenance():
    trace = Trace(
        batches=[
            ForwardBatch(
                0,
                "prefill",
                (
                    ForwardRequest((1,), 0, 0, ((0, 0),)),
                    ForwardRequest((2,), 1, 0, ((1, 0),)),
                ),
            ),
            ForwardBatch(1, "decode", (ForwardRequest((3,), 2, 1, ((2, 1),)),)),
        ],
        updates=[BlockUpdate(1, 2, refs=((0, 1),))],
    )
    model = tiny_model()
    before = replay_trace(model, trace, hidden_only=False)
    chunks = next(iter(trace._attention.values()))
    trace.updates[0] = BlockUpdate(1, 2, refs=((1, 1),))
    actual = replay_trace(model, trace, hidden_only=False)
    torch.testing.assert_close(actual, sequential(model, trace), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual[:2], before[:2], atol=0, rtol=0)
    assert not torch.allclose(actual[2], before[2])
    assert next(iter(trace._attention.values())) is not chunks


def test_requests_preserve_batch_visibility_order_and_causal_prefixes():
    session = _TraceRecorder(Session())
    prompt, a, b = [session.create_block() for _ in range(3)]
    session.prefill_batch([job(prompt, [1, 2, 3])])
    session.prefill_batch([job(a, [4, 5], [prompt]), job(b, [6, 7], [prompt])])
    group = [
        NS(cache_view=[prompt, b, a, b], output_block=a),
        NS(cache_view=[a, prompt], output_block=b),
    ]
    session.decode_step(group, torch.tensor([8, 9]))
    trace = session.trace
    p, x, y = prompt.block_id, a.block_id, b.block_id
    expected = [Token(i + 1, 0, p, i, ((p, i + 1),), "prefill") for i in range(3)]
    for block, ids in ((x, [4, 5]), (y, [6, 7])):
        expected.extend(
            Token(token, 1, block, i, ((p, 3), (block, i + 1)), "prefill")
            for i, token in enumerate(ids)
        )
    expected.extend(
        [
            Token(8, 2, x, 2, ((p, 3), (y, 3), (x, 3), (y, 3)), "decode"),
            Token(9, 2, y, 2, ((x, 3), (p, 3)), "decode"),
        ]
    )
    assert trace.tokens == expected
    assert [(batch.step, batch.kind, len(batch.requests)) for batch in trace.batches] == [
        (0, "prefill", 1),
        (1, "prefill", 2),
        (2, "decode", 2),
    ]
    request = trace.batches[0].requests[0]
    assert (request.input_ids, request.write_to, request.write_start) == ((1, 2, 3), p, 0)
    assert trace.batches[-1].requests[0].cache_view == ((p, 3), (y, 2), (x, 2), (y, 2))
    assert trace.batches[-1].requests[1].cache_view == ((x, 2), (p, 3))


def test_request_snapshot_survives_input_mutation_and_block_reuse():
    session = _TraceRecorder(RequestSession())
    a, b = [session.create_block() for _ in range(2)]
    first = job(a, [1, 2])
    session.prefill_batch([first])
    prefix = asdict(session.trace.batches[0])
    first.input_ids.fill_(99)
    session.append_block(a, a)
    session.prefill_block(b, torch.tensor([3, 4]), context=[a, a])
    session.free_block(a)
    session.prefill_block(a, torch.tensor([5]), context=[b])
    assert asdict(session.trace.batches[0]) == prefix
    assert [r.input_ids for batch in session.trace.batches for r in batch.requests] == [
        (1, 2),
        (3, 4),
        (5,),
    ]
    assert [t.position for t in session.trace.tokens] == [0, 1, 0, 1, 0]
    assert session.trace.batches[1].requests[0].cache_view == (
        (a.block_id, 4),
        (a.block_id, 4),
        (b.block_id, 0),
    )
    assert session.trace.updates[-1].refs == ()
    Trace.from_dict(json.loads(json.dumps(asdict(session.trace)))).to()


def test_long_prefill_stores_one_request_without_repeated_metadata(record_property):
    session = _TraceRecorder(RequestSession())
    block = session.create_block()
    session.prefill_block(block, torch.arange(2048))
    trace = session.trace
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


def test_image_inputs_and_positions_roundtrip_without_aliasing():
    positions = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]])

    class ImageSession(RequestSession):
        def _mrope_rel(self, input_ids, types, grid):
            return positions

        def prefill_batch(self, jobs):
            for item in jobs:
                start = getattr(item.block, "mrope_span", item.block.num_tokens)
                super().prefill_batch([item])
                length = 2 if item.pixel_values is not None else item.input_ids.numel()
                item.block.mrope_span = start + length
            return [None] * len(jobs)

    session = _TraceRecorder(ImageSession())
    prompt, block = [session.create_block() for _ in range(2)]
    session.prefill_block(prompt, torch.tensor([1, 2]))
    pixels = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    grid, types = torch.tensor([[1, 2, 2]]), torch.ones(4, dtype=torch.int64)
    session.prefill_block(
        block,
        torch.tensor([3, 4, 5, 6]),
        context=[prompt],
        pixel_values=pixels,
        image_grid_thw=grid,
        mm_token_type_ids=types,
    )
    request = session.trace.batches[-1].requests[0]
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
    session.prefill_block(block, torch.tensor([7, 8]))
    extension = session.trace.batches[-1].requests[0]
    assert extension.write_start == 4 and extension.write_span == 2
    torch.testing.assert_close(
        extension.mrope_positions.to_tensor(),
        torch.tensor([[2, 3], [2, 3], [2, 3]]),
        rtol=0,
        atol=0,
    )
    restored = Trace.from_dict(json.loads(json.dumps(asdict(session.trace))))
    assert restored == session.trace
    assert restored.batches[1].requests[0].pixel_values.to_tensor().sum().item() == 66
    replay = restored.to()
    assert replay.input_ids.numel() == restored.num_tokens
    gdn = GDNReplay.from_trace(restored, replay.order)
    assert gdn.conv_sources.shape[0] == restored.num_tokens


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
    session = _TraceRecorder(RequestSession())
    block = session.create_block()
    before = asdict(session.trace)
    request = job(
        block,
        [1],
        pixel_values=torch.zeros(4, 12) if has_pixels else None,
        image_embeds=torch.zeros(rows, 32),
    )
    with pytest.raises(ValueError, match="before internal prefill chunking"):
        session.prefill_batch([request])
    assert asdict(session.trace) == before
    assert block.num_tokens == 0
    session.prefill_block(block, torch.tensor([2]))
    assert session.trace.num_tokens == block.num_tokens == 1


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
    session = _TraceRecorder(Session())
    block = session.create_block()
    session.prefill_batch([job(block, [1, 2])])
    session.decode_step([NS(cache_view=[block], output_block=block)], torch.tensor([3]))
    data = json.loads(json.dumps(asdict(session.trace)))
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        Trace.from_dict(data)


@pytest.mark.parametrize("method", ["prefill_batch", "prefill_block", "decode_step"])
def test_failed_forward_keeps_request_storage_atomic(method):
    session = _TraceRecorder(Session())
    block = session.create_block()
    session.prefill_batch([job(block, [1, 2])])
    before = asdict(session.trace)

    def fail(*args):
        block.num_tokens += 1
        raise RuntimeError("inference failed")

    operation = "decode_step" if method == "decode_step" else "prefill_batch"
    setattr(session.session, operation, fail)
    args = {
        "prefill_batch": ([job(block, [3, 4])],),
        "prefill_block": (block, torch.tensor([3, 4])),
        "decode_step": ([NS(cache_view=[block], output_block=block)], torch.tensor([3])),
    }
    with pytest.raises(RuntimeError, match="inference failed"):
        getattr(session, method)(*args[method])
    assert asdict(session.trace) == before
    with pytest.raises(RuntimeError, match="Discard"):
        session.prefill_batch([job(block, [5])])


def test_stress_cli_writes_request_trace(monkeypatch, tmp_path, capsys):
    from minisgl import engine, shared_cache
    from minisgl.training import stress

    closed = []
    instance = NS(config=NS(model_config=NS(vocab_size=32)), shutdown=lambda: closed.append(True))
    monkeypatch.setattr(engine, "EngineConfig", lambda **kwargs: NS(**kwargs))
    monkeypatch.setattr(engine, "Engine", lambda config: instance)
    monkeypatch.setattr(shared_cache, "SharedCacheSession", lambda engine: Session())
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["stress", "--output", "trace.json", "--large-size", "4"])
    stress.main()
    example = json.loads((tmp_path / "trace.json").read_text())["examples"][0]
    assert example["trace"]["format_version"] == 2 and "tokens" not in example["trace"]
    count = Trace.from_dict(example["trace"]).num_tokens
    assert count > 0 and count == len(example["targets"]) == len(example["logprobs"])
    assert closed == [True]
    assert f"Saved {count} input rows" in capsys.readouterr().out


def test_mining_aligns_targets_with_batched_request_rows(monkeypatch, tmp_path):
    from contextlib import nullcontext

    from minisgl import llm
    from minisgl.training import sft
    from test_async_llm import _make_llm

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
