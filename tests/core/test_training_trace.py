import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest
import torch
from minisgl.training.gdn import GDNReplay
from minisgl.training.replay import Replay
from minisgl.training.sft import restore_trace
from minisgl.training.trace import RecordingSession, Token, Trace
from test_training import Session, mutation_trace, randomized_trace, repeated_view_trace


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


def legacy_dict(trace):
    return {
        "tokens": [asdict(token) for token in trace.tokens],
        "steps": trace.steps,
        "updates": [asdict(update) for update in trace.updates],
    }


def equal_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            equal_tree(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "factory", [mutation_trace, repeated_view_trace, lambda: randomized_trace(7)]
)
def test_request_and_legacy_trace_compile_identically(factory):
    trace = factory()
    encoded = json.loads(json.dumps(asdict(trace)))
    assert encoded["format_version"] == 2 and "tokens" not in encoded
    restored = restore_trace(encoded)
    legacy = restore_trace(json.loads(json.dumps(legacy_dict(trace))))
    assert restored == trace
    assert restored.tokens == legacy.tokens == trace.tokens
    assert restored.num_tokens == len(trace.tokens)
    a, b = Replay.from_trace(restored), Replay.from_trace(legacy)
    equal_tree(
        (a.input_ids, a.order, a.spans, a.reads, a.copies),
        (b.input_ids, b.order, b.spans, b.reads, b.copies),
    )
    order = a.order
    for mode in ("expanded", "factorized"):
        a = GDNReplay.from_trace(restored, order, read_mode=mode)
        b = GDNReplay.from_trace(legacy, order, read_mode=mode)
        equal_tree(
            (a.conv_sources, a.levels, a.reads, a.partitions),
            (b.conv_sources, b.levels, b.reads, b.partitions),
        )


def test_requests_preserve_batch_visibility_order_and_causal_prefixes():
    session = RecordingSession(Session())
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
    session = RecordingSession(RequestSession())
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
    Replay.from_trace(restore_trace(json.loads(json.dumps(asdict(session.trace)))))


def test_long_prefill_stores_one_request_without_repeated_metadata(record_property):
    session = RecordingSession(RequestSession())
    block = session.create_block()
    session.prefill_block(block, torch.arange(2048))
    trace = session.trace
    data = asdict(trace)
    assert len(data["batches"]) == len(data["batches"][0]["requests"]) == 1
    assert data["batches"][0]["requests"][0]["input_ids"] == tuple(range(2048))
    new_bytes, legacy_bytes = len(json.dumps(data)), len(json.dumps(legacy_dict(trace)))
    assert new_bytes < legacy_bytes / 8
    record_property("request_trace_bytes", new_bytes)
    record_property("legacy_trace_bytes", legacy_bytes)
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

    session = RecordingSession(ImageSession())
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
        torch.testing.assert_close(tensor.tensor(), expected, rtol=0, atol=0)
    assert request.write_span == 0 and request.cache_spans == (2, 0)
    snapshot = asdict(request)
    pixels.zero_()
    grid.fill_(99)
    types.zero_()
    positions.fill_(99)
    assert asdict(request) == snapshot
    session.prefill_block(block, torch.tensor([7, 8]))
    extension = session.trace.batches[-1].requests[0]
    assert extension.write_start == 4 and extension.write_span == 2
    torch.testing.assert_close(
        extension.mrope_positions.tensor(), torch.tensor([[2, 3], [2, 3], [2, 3]]), rtol=0, atol=0
    )
    restored = restore_trace(json.loads(json.dumps(asdict(session.trace))))
    assert restored == session.trace
    assert restored.batches[1].requests[0].pixel_values.tensor().sum().item() == 66
    replay = Replay.from_trace(restored)
    assert replay.input_ids.numel() == restored.num_tokens
    gdn = GDNReplay.from_trace(restored, replay.order)
    assert gdn.conv_sources.shape[0] == restored.num_tokens


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
    session = RecordingSession(Session())
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
    session = RecordingSession(Session())
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
    from minisgl import llm
    from minisgl.training import sft

    class FakeLLM:
        def __init__(self):
            self.async_engine = NS(session=Session())
            self.tokenizer = NS(
                encode=lambda *a, **kw: [4, 5, 6],
                apply_chat_template=lambda *a, **kw: [1, 2, 3],
                decode=lambda ids, **kw: str(ids),
                eos_token_id=-1,
            )
            self.pending = []
            self.closed = False

        async def create_block(self):
            return self.async_engine.session.create_block()

        async def merge_blocks(self, left, right):
            return self.async_engine.session.merge_blocks(left, right)

        async def append_block(self, left, right):
            return self.async_engine.session.append_block(left, right)

        async def free_block(self, block):
            return self.async_engine.session.free_block(block)

        async def close(self):
            self.closed = True

        async def forward(self, ids=None, *, cache_view=None, write_to=None, **kwargs):
            if ids is not None:
                self.async_engine.session.prefill_batch(
                    [job(write_to, ids, (cache_view or [])[:-1])]
                )
                return
            future = asyncio.get_running_loop().create_future()
            self.pending.append((cache_view, future))
            if len(self.pending) == 1:
                asyncio.get_running_loop().call_soon(self.flush)
            return await future

        def flush(self):
            pending, self.pending = self.pending, []
            contexts = [context for context, _ in pending]
            self.async_engine.session.decode_step(
                contexts, torch.tensor([context.next_input_id for context in contexts])
            )
            for context, future in pending:
                logits = torch.full((32,), -20.0)
                logits[1 + context.output_block.block_id % 23] = 20.0
                future.set_result(NS(logits=logits))

    instance = FakeLLM()
    generator = torch.Generator
    monkeypatch.setattr(llm, "AsyncLLM", lambda *a, **kw: instance)
    monkeypatch.setattr(torch, "Generator", lambda **kw: generator(device="cpu"))
    monkeypatch.setattr(torch, "multinomial", lambda values, *a, **kw: values.argmax().reshape(1))
    output = tmp_path / "mined.json"
    args = NS(model="fake", stress=False, max_tokens=2, seed=7, temperature=1.0, output=output)
    asyncio.run(asyncio.wait_for(sft.mine(args), timeout=10))
    examples = json.loads(output.read_text())["examples"]
    assert instance.closed and len(examples) == 3
    for example in examples:
        trace = Trace.from_dict(example["trace"])
        assert any(batch.kind == "decode" and len(batch.requests) == 2 for batch in trace.batches)
        assert sum(target != -100 for target in example["targets"]) == 6
        assert len(example["targets"]) == len(example["logprobs"]) == trace.num_tokens
        for token, target, probability in zip(
            trace.tokens, example["targets"], example["logprobs"], strict=True
        ):
            assert target == (1 + token.block % 23 if token.kind == "decode" else -100)
            assert probability == (0.0 if token.kind == "decode" else None)
