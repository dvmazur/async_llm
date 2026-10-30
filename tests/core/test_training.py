import json
import random
from dataclasses import asdict
from functools import partial
from itertools import groupby
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
from minisgl.training import sft
from minisgl.training.replay import Replay, attention, forward
from minisgl.training.sft import restore_trace, supervised_loss
from minisgl.training.stress import workload
from minisgl.training.trace import BlockUpdate, RecordingSession, Token, Trace
from torch.overrides import TorchFunctionMode
from transformers import Qwen3Config, Qwen3ForCausalLM


class Session:
    def __init__(self):
        self.next_block_id = 100

    def create_block(self):
        block = NS(block_id=self.next_block_id, num_tokens=0)
        self.next_block_id += 1
        return block

    def merge_blocks(self, left, right):
        block = self.create_block()
        block.num_tokens = left.num_tokens + right.num_tokens
        return block

    def append_block(self, left, right):
        left.num_tokens += right.num_tokens
        return left

    def free_block(self, block):
        block.num_tokens = 0

    def prefill_batch(self, jobs):
        for job in jobs:
            job.block.num_tokens += job.input_ids.numel()

    def decode_step(self, group, input_ids):
        for ctx in group:
            ctx.output_block.num_tokens += 1


def example_trace():
    session = RecordingSession(Session())
    prompt, a, b = [NS(block_id=i, num_tokens=0) for i in range(3)]

    def prefill(block, ids, context):
        session.prefill_batch(
            [NS(block=block, input_ids=torch.tensor(ids), context=context, pixel_values=None)]
        )

    prefill(prompt, [1, 2, 3], [])
    prefill(a, [4, 5], [prompt])
    prefill(b, [6, 7], [prompt])
    ca = NS(cache_view=[prompt, b, a], output_block=a)
    cb = NS(cache_view=[prompt, a, b], output_block=b)
    session.decode_step([ca, cb], torch.tensor([8, 9]))
    prefill(a, [10, 11], [prompt, b])
    ca.cache_view = [b, prompt, a]
    session.decode_step([ca, cb], torch.tensor([12, 13]))
    ca.cache_view = [prompt, b]
    session.decode_step([ca], torch.tensor([14]))
    return session.trace


def single_block_trace(mixed_steps):
    session = RecordingSession(Session())
    block = session.create_block()
    groups = [[1, 2, 3], [4], [5, 6], [7], [8, 9]] if mixed_steps else [list(range(1, 10))]
    for ids in groups:
        if len(ids) == 1:
            session.decode_step([NS(cache_view=[block], output_block=block)], torch.tensor(ids))
        else:
            session.prefill_batch(
                [NS(block=block, input_ids=torch.tensor(ids), context=[], pixel_values=None)]
            )
    return session.trace


def mutation_trace():
    session = RecordingSession(Session())
    prompt, a, b, empty = [NS(block_id=i, num_tokens=0) for i in range(4)]

    def prefill(block, ids, context):
        session.prefill_batch(
            [NS(block=block, input_ids=torch.tensor(ids), context=context, pixel_values=None)]
        )

    prefill(prompt, [1, 2], [])
    prefill(a, [3, 4], [prompt])
    prefill(b, [5, 6], [prompt, a])
    merged = session.merge_blocks(a, b)
    copied = session.merge_blocks(empty, merged)
    prefill(a, [7], [prompt])
    session.free_block(b)
    prefill(b, [8, 9], [])
    prefill(merged, [10, 11], [prompt])
    ca = NS(cache_view=[copied, b, merged], output_block=merged)
    cb = NS(cache_view=[prompt, merged, b], output_block=b)
    session.decode_step([ca, cb], torch.tensor([12, 13]))
    session.append_block(merged, merged)
    session.append_block(copied, a)
    session.free_block(a)
    prefill(a, [14], [])
    prefill(merged, [15], [copied])
    ca.cache_view = [copied]
    session.decode_step([ca], torch.tensor([16]))
    session.free_block(copied)
    prefill(copied, [17], [b, merged])
    right_copy = session.merge_blocks(merged, empty)
    empty_copy = session.merge_blocks(empty, empty)
    prefill(empty_copy, [18], [right_copy])
    session.free_block(merged)
    prefill(right_copy, [19], [prompt])
    return session.trace


def repeated_view_trace():
    session = RecordingSession(Session())
    a, b, c, empty = [session.create_block() for _ in range(4)]

    def prefill(block, ids, context):
        session.prefill_batch(
            [NS(block=block, input_ids=torch.tensor(ids), context=context, pixel_values=None)]
        )

    prefill(a, [1, 2, 3, 4, 5], [])
    prefill(b, [6, 7], [a, a])
    prefill(c, [8], [a, b, a])
    session.decode_step(
        [
            NS(cache_view=[a, b, a, empty, b], output_block=a),
            NS(cache_view=[b, a, b, a], output_block=b),
            NS(cache_view=[a, a], output_block=c),
        ],
        torch.tensor([9, 10, 11]),
    )
    copied = session.merge_blocks(a, b)
    session.append_block(copied, copied)
    session.free_block(a)
    prefill(a, [12, 13], [copied, copied, b, copied])
    session.decode_step(
        [
            NS(cache_view=[a, copied, a, copied], output_block=a),
            NS(cache_view=[copied, a, b, copied, b], output_block=b),
        ],
        torch.tensor([14, 15]),
    )
    session.free_block(copied)
    prefill(copied, [16, 17], [a, b, a])
    return session.trace


def randomized_trace(seed):
    rng = random.Random(seed)
    session = RecordingSession(Session())
    blocks = [session.create_block() for _ in range(5)]

    def prefill(block, count, context=()):
        session.prefill_batch(
            [
                NS(
                    block=block,
                    input_ids=torch.tensor([rng.randrange(1, 32) for _ in range(count)]),
                    context=context,
                    pixel_values=None,
                )
            ]
        )

    for block in blocks[:3]:
        prefill(block, 2)
    for round_id in range(12):
        left = rng.choice(blocks)
        right = left if round_id % 3 == 0 else rng.choice(blocks)
        if round_id % 4 == 0:
            session.free_block(left)
        elif round_id % 4 == 1:
            session.append_block(left, right)
        else:
            copied = session.merge_blocks(left, right)
            if round_id % 4 == 2:
                session.free_block(left)
            blocks.append(copied)
        if round_id % 3 == 0:
            output = rng.choice(blocks)
            contexts = [block for block in blocks if block is not output]
            prefill(output, 2, rng.sample(contexts, min(3, len(contexts))))
        else:
            outputs = rng.sample(blocks, 3)
            contexts = [
                NS(cache_view=rng.sample(blocks, rng.randrange(5)), output_block=block)
                for block in outputs
            ]
            session.decode_step(contexts, torch.tensor([rng.randrange(1, 32) for _ in outputs]))
    return session.trace


def tiny_model():
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def visible_sources(replay):
    rows = [[] for _ in replay.reads]
    for original, reads in zip(replay.order.tolist(), replay.reads):
        for version, start, end, _ in reads:
            for source_start, source_end, _ in replay.source_spans(version, start, end):
                rows[original].extend(replay.order[source_start:source_end].tolist())
    return rows


def apply_updates(trace, step, blocks, origins=None):
    for update in trace.updates:
        if update.step != step:
            continue
        if update.refs is None:
            contents = (
                list(update.sources)
                if origins is None
                else [origins[source] for source in update.sources]
            )
        else:
            contents = []
            for block, length in update.refs:
                members = blocks.get(block, [])
                assert len(members) == length
                contents.extend(members)
        blocks[update.block] = contents


def sequential_attention(q, k, v, trace, order, inv_freq, scaling):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    width = 2 * len(inv_freq)

    def rotated(x, positions):
        positions = torch.tensor(positions, dtype=x.dtype, device=x.device)
        angles = torch.outer(positions, inv_freq).repeat(1, 2)[None]
        part = x[..., :width].transpose(0, 1)[None]
        part, _ = apply_rotary_pos_emb(part, part, angles.cos(), angles.sin())
        return torch.cat((part[0].transpose(0, 1), x[..., width:]), -1)

    inverse = order.argsort().tolist()
    blocks, outputs = {}, [None] * len(q)
    for step, entries in groupby(enumerate(trace.tokens), key=lambda entry: entry[1].step):
        entries = list(entries)
        apply_updates(trace, step, blocks)
        for row, token in entries:
            blocks.setdefault(token.block, []).append(row)
        for row, token in entries:
            sources = [
                source for block, length in token.view for source in blocks.get(block, [])[:length]
            ]
            query_position = len(sources) - 1
            positions = list(range(len(sources)))
            if token.block not in dict(token.view):
                sources.append(row)
                positions.append(query_position)
            indices = [inverse[source] for source in sources]
            groups = q.shape[1] // k.shape[1]
            query = rotated(q[inverse[row] : inverse[row] + 1], [query_position])[0]
            key = rotated(k[indices], positions).repeat_interleave(groups, 1)
            value = v[indices].repeat_interleave(groups, 1)
            weights = (torch.einsum("hd,lhd->hl", query, key) * scaling).softmax(-1)
            outputs[inverse[row]] = torch.einsum("hl,lhd->hd", weights, value).flatten()
    return torch.stack(outputs)


class Float64AttentionMath(TorchFunctionMode):
    """Test production algebra in FP64, overriding its explicit FP32 casts only."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.Tensor.float and args[0].dtype == torch.float64:
            return args[0]
        return func(*args, **(kwargs or {}))


@pytest.fixture
def fp64_attention(monkeypatch):
    from minisgl.training import replay as replay_module

    monkeypatch.setattr(
        replay_module,
        "checkpoint",
        partial(
            replay_module.checkpoint,
            context_fn=lambda: (Float64AttentionMath(), Float64AttentionMath()),
        ),
    )


def sequential(model, trace):
    """Independent step-by-step oracle, rotating raw cached keys into each view."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

    cache = [{} for _ in model.model.layers]
    origins = [{} for _ in model.model.layers]
    outputs = {}
    for step in sorted({t.step for t in trace.tokens}):
        rows = [i for i, t in enumerate(trace.tokens) if t.step == step]
        x = model.model.embed_tokens(torch.tensor([trace.tokens[i].token_id for i in rows]))
        for layer_id, layer in enumerate(model.model.layers):
            apply_updates(trace, step, cache[layer_id], origins[layer_id])
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            q = attn.q_norm(attn.q_proj(h).view(len(rows), -1, attn.head_dim))
            k = attn.k_norm(attn.k_proj(h).view(len(rows), -1, attn.head_dim))
            v = attn.v_proj(h).view(len(rows), -1, attn.head_dim)
            for j, row in enumerate(rows):
                t = trace.tokens[row]
                block = cache[layer_id].setdefault(t.block, [])
                assert len(block) == t.position
                origins[layer_id][row] = (k[j], v[j])
                block.append(origins[layer_id][row])
            out = []
            for j, row in enumerate(rows):
                t = trace.tokens[row]
                entries = [
                    cache[layer_id][block][p] for block, length in t.view for p in range(length)
                ]
                query_position = len(entries) - 1
                if t.block not in dict(t.view):
                    entries.append(cache[layer_id][t.block][t.position])
                    positions = list(range(len(entries) - 1)) + [query_position]
                else:
                    positions = list(range(len(entries)))
                keys = torch.stack([kv[0] for kv in entries]).transpose(0, 1)[None]
                values = torch.stack([kv[1] for kv in entries]).transpose(0, 1)[None]
                query = q[j][None, :, None]
                cos, sin = model.model.rotary_emb(x, torch.tensor([positions]))
                _, keys = apply_rotary_pos_emb(keys, keys, cos, sin)
                cos, sin = model.model.rotary_emb(x, torch.tensor([[query_position]]))
                query, _ = apply_rotary_pos_emb(query, query, cos, sin)
                keys = repeat_kv(keys, attn.num_key_value_groups)
                values = repeat_kv(values, attn.num_key_value_groups)
                weights = (query @ keys.transpose(-1, -2) * attn.scaling).softmax(-1)
                out.append((weights @ values).flatten())
            x = x + attn.o_proj(torch.stack(out))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        logits = model.lm_head(model.model.norm(x))
        outputs.update(zip(rows, logits))
    return torch.stack([outputs[i] for i in range(len(trace.tokens))])


@pytest.mark.parametrize("trace_factory", [example_trace, mutation_trace, repeated_view_trace])
@pytest.mark.parametrize("query_chunk_size", [1, 3, 128])
def test_parallel_logits_and_gradients(trace_factory, query_chunk_size):
    model = tiny_model()
    trace = trace_factory()
    replay = Replay.from_trace(trace, query_chunk_size=query_chunk_size)
    expected = sequential(model, trace)
    actual = forward(model, replay)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    parameters = tuple(model.parameters())
    expected_grads = torch.autograd.grad(expected.square().sum(), parameters)
    actual_grads = torch.autograd.grad(actual.square().sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=2e-4)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_randomized_mutations_match_sequential_logits_and_gradients(seed):
    model = tiny_model()
    trace = restore_trace(json.loads(json.dumps(asdict(randomized_trace(seed)))))
    replay = Replay.from_trace(trace, query_chunk_size=5)
    expected = sequential(model, trace)
    actual = forward(model, replay)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    weights = torch.randn_like(actual)
    parameters = tuple(model.parameters())
    expected_grads = torch.autograd.grad((expected * weights).sum(), parameters)
    actual_grads = torch.autograd.grad((actual * weights).sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=3e-4)


def test_synthetic_stress_workload_logits_and_gradients():
    session = RecordingSession(Session())
    data = workload(session, vocab_size=32, large_size=4)
    trace = restore_trace(json.loads(json.dumps(data["trace"])))
    assert trace == session.trace
    assert data["coverage"]["blocks"] >= 24
    assert data["coverage"]["max_view_depth"] == 12
    assert set(data["coverage"]["decode_batch_sizes"]) == {4, 5, 6, 7, 8}
    assert all(
        data["coverage"]["operations"][operation] > 0
        for operation in (
            "empty_left_copy",
            "empty_right_copy",
            "self_append",
            "clear_reuse",
            "extend_copied_source",
            "prefill_extension",
        )
    )
    assert all(update.refs is not None and not update.sources for update in trace.updates)
    assert len(data["targets"]) == len(data["logprobs"]) == len(trace.tokens)
    model = tiny_model()
    replay = Replay.from_trace(trace, query_chunk_size=7)
    expected, actual = sequential(model, trace), forward(model, replay)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    weights = torch.randn_like(expected)
    parameters = tuple(model.parameters())
    expected_grads = torch.autograd.grad((expected * weights).sum(), parameters)
    actual_grads = torch.autograd.grad((actual * weights).sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=3e-4)


@pytest.mark.parametrize("kv_heads", [4, 2, 1])
@pytest.mark.parametrize("rotary_width", [8, 4])
def test_grouped_attention_and_partial_rotary_gradients(kv_heads, rotary_width):
    torch.manual_seed(7)
    trace = mutation_trace()
    replay = Replay.from_trace(trace, query_chunk_size=3)
    q = torch.randn(len(trace.tokens), 4, 8, requires_grad=True)
    k = torch.randn(len(trace.tokens), kv_heads, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    inv_freq = 10000 ** (-torch.arange(0, rotary_width, 2).float() / rotary_width)
    expected = sequential_attention(q, k, v, trace, replay.order, inv_freq, 8**-0.5)
    actual = attention(q, k, v, replay, inv_freq, 8**-0.5)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    expected_grads = torch.autograd.grad(expected.square().sum(), (q, k, v))
    actual_grads = torch.autograd.grad(actual.square().sum(), (q, k, v))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=2e-4)


@pytest.mark.parametrize("query_chunk_size", [1, 7, None])
@pytest.mark.parametrize(
    "trace_kind,kv_heads,rotary_width",
    [("single", 4, 8), ("mutation", 2, 8), ("stress", 1, 4), ("repeated", 2, 4)],
)
def test_attention_fp64_algebra_and_fp32_rounding(
    trace_kind, kv_heads, rotary_width, query_chunk_size, record_property, fp64_attention
):
    if trace_kind == "stress":
        session = RecordingSession(Session())
        workload(session, vocab_size=32, large_size=4)
        trace = session.trace
    elif trace_kind == "repeated":
        trace = repeated_view_trace()
    else:
        trace = single_block_trace(True) if trace_kind == "single" else mutation_trace()
    n = len(trace.tokens)
    replay = Replay.from_trace(trace, query_chunk_size=query_chunk_size or n)
    torch.manual_seed(7)
    inputs = tuple(
        torch.randn(n, heads, 8).double().requires_grad_() for heads in (4, kv_heads, kv_heads)
    )
    inv_freq = (10000 ** (-torch.arange(0, rotary_width, 2).float() / rotary_width)).double()
    expected = sequential_attention(*inputs, trace, replay.order, inv_freq, 8**-0.5)
    weights = torch.randn(expected.shape).double()
    expected_grads = torch.autograd.grad((expected * weights).sum(), inputs)
    with Float64AttentionMath():
        actual = attention(*inputs, replay, inv_freq, 8**-0.5)
        actual_grads = torch.autograd.grad((actual * weights).sum(), inputs)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-12, rtol=2e-12)
    record_property("fp64_output_max_abs", (actual - expected).abs().max().item())
    record_property(
        "fp64_gradient_max_abs",
        max((a - b).abs().max().item() for a, b in zip(actual_grads, expected_grads)),
    )

    inputs32 = tuple(x.detach().float().requires_grad_() for x in inputs)
    actual32 = attention(*inputs32, replay, inv_freq.float(), 8**-0.5)
    grads32 = torch.autograd.grad((actual32 * weights.float()).sum(), inputs32)
    torch.testing.assert_close(actual32.double(), expected, atol=2e-6, rtol=2e-5)
    for actual_grad, expected_grad in zip(grads32, expected_grads):
        torch.testing.assert_close(actual_grad.double(), expected_grad, atol=1e-5, rtol=2e-4)
    record_property("fp32_output_max_abs", (actual32.double() - expected).abs().max().item())
    record_property(
        "fp32_gradient_max_abs",
        max((a.double() - b).abs().max().item() for a, b in zip(grads32, expected_grads)),
    )


def test_attention_copied_cleared_history_gradcheck(fp64_attention):
    trace = Trace(
        [
            Token(1, 0, 0, 0, ((0, 1),), "prefill"),
            Token(2, 0, 0, 1, ((0, 2),), "prefill"),
            Token(3, 1, 0, 0, ((1, 5), (0, 1)), "decode"),
            Token(4, 1, 1, 4, ((0, 1), (1, 5)), "decode"),
        ],
        updates=[
            BlockUpdate(1, 1, refs=((0, 2),)),
            BlockUpdate(1, 1, refs=((1, 2), (1, 2))),
            BlockUpdate(1, 0, refs=()),
        ],
    )
    replay = Replay.from_trace(trace, query_chunk_size=2)
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
    tokens = [
        Token(i % 32, 0, block, i, ((block, i + 1),), "prefill")
        for block in (0, 1)
        for i in range(32)
    ]
    copied = tuple(range(64)) + tuple(range(32))
    trace = Trace(
        tokens
        + [Token(1, 1, 2, 96, ((0, 11),), "decode"), Token(2, 1, 4, 0, ((2, 75),), "decode")],
        updates=[BlockUpdate(1, 2, copied), BlockUpdate(1, 3, copied), BlockUpdate(1, 3, ())],
    )
    replay = Replay.from_trace(trace, query_chunk_size=128)
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


def test_selected_logits_and_gradients():
    model = tiny_model()
    replay = Replay.from_trace(mutation_trace(), query_chunk_size=3)
    rows = torch.tensor([18, 0, 7, 7, 13])
    expected = forward(model, replay)[rows]
    head_rows = []
    hook = model.lm_head.register_forward_pre_hook(
        lambda module, inputs: head_rows.append(inputs[0].shape[-2])
    )
    actual = forward(model, replay, rows=rows)
    hook.remove()
    assert head_rows == [rows.numel()]
    torch.testing.assert_close(actual, expected)
    parameters = tuple(model.parameters())
    expected_grads = torch.autograd.grad(expected.square().sum(), parameters)
    actual_grads = torch.autograd.grad(actual.square().sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=2e-4)


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_chunked_supervised_loss_and_gradients(chunk_size):
    model = tiny_model()
    replay = Replay.from_trace(mutation_trace(), query_chunk_size=3)
    targets = torch.arange(len(replay.input_ids)) % 32
    targets[::3] = -100
    expected = torch.nn.functional.cross_entropy(forward(model, replay), targets, reduction="sum")
    head_rows = []
    hook = model.lm_head.register_forward_pre_hook(
        lambda module, inputs: head_rows.append(inputs[0].shape[-2])
    )
    actual = supervised_loss(model, replay, targets, chunk_size=chunk_size)
    hook.remove()
    assert max(head_rows) <= chunk_size
    assert sum(head_rows) == targets.ne(-100).sum()
    torch.testing.assert_close(actual, expected)
    parameters = tuple(model.parameters())
    expected_grads = torch.autograd.grad(expected, parameters)
    actual_grads = torch.autograd.grad(actual, parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=2e-4)


def test_attention_checkpoint_does_not_save_dense_scores():
    length = 128
    trace = Trace([Token(i % 32, 0, 0, i, ((0, i + 1),), "prefill") for i in range(length)])
    replay = Replay.from_trace(trace, query_chunk_size=7)
    q = torch.randn(length, 4, 8, requires_grad=True)
    k = torch.randn(length, 2, 8, requires_grad=True)
    v = torch.randn(length, 2, 8, requires_grad=True)
    inv_freq = tiny_model().model.rotary_emb.inv_freq
    saved, storage_bytes = [], {}

    def pack(tensor):
        saved.append(tensor.shape)
        storage = tensor.untyped_storage()
        storage_bytes[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = attention(q, k, v, replay, inv_freq, 8**-0.5)
    assert saved
    assert all(shape.numel() < length * length for shape in saved)
    input_bytes = sum(tensor.numel() * tensor.element_size() for tensor in (q, k, v))
    assert sum(storage_bytes.values()) <= 2 * input_bytes
    output.square().mean().backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None and tensor.grad.isfinite().all()


@pytest.mark.parametrize("mixed_steps", [False, True])
def test_causal_forward_and_backward_match_huggingface(mixed_steps):
    reference, model = tiny_model().train(), tiny_model().train()
    if mixed_steps:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    replay = Replay.from_trace(single_block_trace(mixed_steps), query_chunk_size=3)
    expected = reference(replay.input_ids[None], use_cache=False).logits[0]
    actual = forward(model, replay)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    targets = replay.input_ids.roll(-1)
    torch.nn.functional.cross_entropy(expected, targets).backward()
    torch.nn.functional.cross_entropy(actual, targets).backward()
    for (name, parameter), (_, native) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert parameter.grad is not None and native.grad is not None
        torch.testing.assert_close(
            parameter.grad, native.grad, atol=1e-6, rtol=2e-4, msg=f"Gradient mismatch: {name}"
        )


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_forward_after_replay_stays_causal_and_masked(backend, padded, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = tiny_model().to(device)
    model.set_attn_implementation(backend)
    ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 7]], device=device)
    mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]], device=device) if padded else None
    with torch.no_grad():
        expected = model(ids, attention_mask=mask, use_cache=False).logits
        forward(model, Replay.from_trace(mutation_trace(), device=device))
        assert model.config._attn_implementation == backend
        with patch(
            "torch.nn.functional.scaled_dot_product_attention",
            wraps=torch.nn.functional.scaled_dot_product_attention,
        ) as sdpa:
            actual = model(ids, attention_mask=mask, use_cache=False).logits
        assert sdpa.call_count == (len(model.model.layers) if backend == "sdpa" else 0)
        if backend == "sdpa" and not padded:
            assert all(call.kwargs["attn_mask"] is None for call in sdpa.call_args_list)
        torch.testing.assert_close(actual, expected)
        changed = ids.clone()
        changed[0, -1], changed[1, 0] = 10, 11
        perturbed = model(changed, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(perturbed[0, :-1], actual[0, :-1])
    if padded:
        torch.testing.assert_close(perturbed[1, 1:], actual[1, 1:])


@pytest.mark.parametrize("use_reentrant", [None, False, True])
def test_attention_inference_warmup_then_training(use_reentrant):
    reference, model = tiny_model().train(), tiny_model().eval()
    trace = repeated_view_trace()
    with torch.inference_mode():
        replay = Replay.from_trace(trace, query_chunk_size=3).to("cpu")
        forward(model, replay)
    chunks = replay._attention[(torch.device("cpu"), 3)]
    assert not replay.input_ids.is_inference() and not replay.order.is_inference()
    assert all(not entries.is_inference() for _, entries in chunks)
    model.train()
    if use_reentrant is not None:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        )
    expected = forward(reference, Replay.from_trace(trace, query_chunk_size=3))
    actual = forward(model, replay)
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    assert replay._attention[(torch.device("cpu"), 3)] is chunks
    for parameter, target in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, target.grad, atol=1e-6, rtol=2e-4)


@pytest.mark.parametrize("use_reentrant", [False, True])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_replay_gradient_checkpointing(use_reentrant, backend):
    reference, checkpointed = tiny_model().train(), tiny_model().train()
    checkpointed.set_attn_implementation(backend)
    checkpointed.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
    )
    replay = Replay.from_trace(mutation_trace(), query_chunk_size=3)
    expected = forward(reference, replay)
    actual = forward(checkpointed, replay)
    torch.testing.assert_close(actual, expected)
    checkpointed(torch.tensor([[1, 2, 3]]), use_cache=False)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for parameter, target in zip(checkpointed.parameters(), reference.parameters()):
        assert parameter.grad is not None and target.grad is not None
        torch.testing.assert_close(parameter.grad, target.grad, atol=1e-6, rtol=2e-4)


def test_native_sdpa_cache_continues_after_replay():
    reference, model = tiny_model(), tiny_model()
    for m in (reference, model):
        m.set_attn_implementation("sdpa")
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        original = reference(ids, use_cache=True)
        prefix = model(ids, use_cache=True)
        forward(model, Replay.from_trace(repeated_view_trace()))
        expected = reference(torch.tensor([[4]]), past_key_values=original.past_key_values).logits
        with patch(
            "torch.nn.functional.scaled_dot_product_attention",
            wraps=torch.nn.functional.scaled_dot_product_attention,
        ) as sdpa:
            actual = model(torch.tensor([[4]]), past_key_values=prefix.past_key_values).logits
    assert sdpa.call_count == len(model.model.layers)
    torch.testing.assert_close(actual, expected)


def test_replay_checkpoint_reloads_as_native_huggingface(tmp_path):
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3]])
    original_keys = set(model.state_dict())
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        forward(model, Replay.from_trace(mutation_trace()))
    model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert not any("attn_implementation" in key for key in config)
    restored = Qwen3ForCausalLM.from_pretrained(tmp_path, attn_implementation="eager").eval()
    assert restored.config._attn_implementation == "eager"
    assert set(restored.state_dict()) == original_keys
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], parameter, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(restored(ids, use_cache=False).logits, expected)


def test_mutation_trace_json_roundtrip():
    trace = mutation_trace()
    restored = restore_trace(json.loads(json.dumps(asdict(trace))))
    assert restored == trace
    original, loaded = Replay.from_trace(trace), Replay.from_trace(restored)
    assert loaded.spans == original.spans
    assert loaded.copies == original.copies
    assert loaded.reads == original.reads
    for field in ("input_ids", "order"):
        torch.testing.assert_close(getattr(loaded, field), getattr(original, field), rtol=0, atol=0)
    model = tiny_model()
    with torch.no_grad():
        torch.testing.assert_close(forward(model, loaded), forward(model, original), rtol=0, atol=0)


def test_legacy_and_compact_json_replay_match():
    trace = mutation_trace()
    blocks, updates = {}, []
    for step in sorted({t.step for t in trace.tokens} | {u.step for u in trace.updates}):
        for update in trace.updates:
            if update.step == step:
                apply_updates(Trace(updates=[update]), step, blocks)
                updates.append(BlockUpdate(step, update.block, tuple(blocks[update.block])))
        for row, token in enumerate(trace.tokens):
            if token.step == step:
                blocks.setdefault(token.block, []).append(row)
    legacy = {
        "tokens": [asdict(token) for token in trace.tokens],
        "steps": trace.steps,
        "updates": [asdict(update) for update in updates],
    }
    for update in legacy["updates"]:
        del update["refs"]
    compact = asdict(trace)
    for update in compact["updates"]:
        del update["sources"]
    legacy, compact = [restore_trace(json.loads(json.dumps(data))) for data in (legacy, compact)]
    assert all(update.refs is None for update in legacy.updates)
    assert compact == trace
    old_replay, new_replay = Replay.from_trace(legacy), Replay.from_trace(compact)
    assert visible_sources(old_replay) == visible_sources(new_replay)
    model = tiny_model()
    expected, actual = forward(model, old_replay), forward(model, new_replay)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    parameters = tuple(model.parameters())
    expected_grads = torch.autograd.grad(expected.square().sum(), parameters)
    actual_grads = torch.autograd.grad(actual.square().sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=2e-4)


def test_serialized_self_append_history_stays_compact():
    session = RecordingSession(Session())
    block = session.create_block()
    session.prefill_batch(
        [NS(block=block, input_ids=torch.tensor([1]), context=[], pixel_values=None)]
    )
    for _ in range(20):
        session.append_block(block, block)
    first_size = len(json.dumps(asdict(session.trace)))
    for _ in range(20):
        session.append_block(block, block)
    assert block.num_tokens == 2**40
    assert len(json.dumps(asdict(session.trace))) < 3 * first_size
    assert sum(len(update.refs) for update in session.trace.updates) == 80
    assert not any(update.sources for update in session.trace.updates)
    assert session._contents[block.block_id] == block.num_tokens
    session.decode_step([NS(cache_view=[], output_block=block)], torch.tensor([2]))
    trace = restore_trace(json.loads(json.dumps(asdict(session.trace))))
    replay = Replay.from_trace(trace)
    assert sum(map(len, replay.spans)) == 2
    assert visible_sources(replay) == [[0], [1]]


def test_deep_copy_chain_resolves_without_flattening_or_recursion():
    session = RecordingSession(Session())
    block, empty = [session.create_block() for _ in range(2)]
    session.prefill_batch(
        [NS(block=block, input_ids=torch.tensor([1, 2]), context=[], pixel_values=None)]
    )
    for _ in range(1100):
        block = session.merge_blocks(block, empty)
    session.decode_step([NS(cache_view=[block], output_block=block)], torch.tensor([3]))
    replay = Replay.from_trace(session.trace)
    assert sum(map(len, replay.spans)) == 2
    assert visible_sources(replay) == [[0], [0, 1], [0, 1, 2]]


def test_nested_copy_ranges_keep_destination_offsets():
    trace = Trace(
        [Token(i + 1, 0, 0, i, ((0, i + 1),), "prefill") for i in range(3)]
        + [Token(i + 4, 0, 1, i, ((1, i + 1),), "prefill") for i in range(2)]
        + [Token(6, 1, 3, 10, ((3, 11),), "decode")],
        updates=[
            BlockUpdate(1, 2, refs=((0, 3), (1, 2), (0, 3))),
            BlockUpdate(1, 3, refs=((2, 8), (1, 2))),
        ],
    )
    replay = Replay.from_trace(trace)
    version = replay.reads[-1][0][0]
    assert list(replay.source_spans(version, 2, 9)) == [
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
        BlockUpdate(1, 1, (0,), refs=((0, 2),)),
        BlockUpdate(0, 1, refs=((0, 2),)),
        BlockUpdate(2, 1, refs=((999, 1),)),
    ],
)
def test_reject_invalid_copy_references(update):
    trace = Trace(
        [
            Token(1, 0, 0, 0, ((0, 1),), "prefill"),
            Token(2, 0, 0, 1, ((0, 2),), "prefill"),
            Token(3, 1, 2, 0, ((2, 1),), "prefill"),
        ],
        updates=[update],
    )
    with pytest.raises(ValueError):
        Replay.from_trace(trace)


def test_recording_and_masks():
    trace = example_trace()
    assert trace.tokens[7].step == trace.tokens[8].step
    assert trace.tokens[9].position == 3
    replay = Replay.from_trace(trace)
    visible = visible_sources(replay)
    assert 8 in visible[7] and 7 in visible[8]
    assert 9 in visible[9] and 10 not in visible[9]
    assert 9 not in visible[7]
    assert len(trace.tokens) - 1 in visible[-1] and 7 not in visible[-1]


def test_causal_metadata_is_compact():
    length = 1024
    trace = Trace([Token(i % 32, 0, 0, i, ((0, i + 1),), "prefill") for i in range(length)])
    replay = Replay.from_trace(trace)
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
        Replay.from_trace(example_trace(), **options)
    with pytest.raises(ValueError, match="gdn_"):
        sft.train(NS(**options))


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_replay_preserves_gdn_read_backend_on_transfer(backend):
    replay = Replay.from_trace(example_trace(), gdn_read_backend=backend)
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


def test_attention_reuses_only_compact_metadata():
    length = 128
    trace = Trace([Token(i % 32, 0, 0, i, ((0, i + 1),), "prefill") for i in range(length)])
    replay = Replay.from_trace(trace, query_chunk_size=4)
    q, k, v = torch.randn(length, 4, 8), torch.randn(length, 2, 8), torch.randn(length, 2, 8)
    inv_freq = tiny_model().model.rotary_emb.inv_freq
    expected = attention(q, k, v, replay, inv_freq, 8**-0.5)
    chunks = replay._attention[(q.device, 4)]
    actual = attention(q, k, v, replay, inv_freq, 8**-0.5)
    assert replay._attention[(q.device, 4)] is chunks
    assert sum(entries.numel() for _, entries in chunks) == 4 * length
    assert all(entries.dtype == torch.long for _, entries in chunks)
    assert replay.to("cpu")._attention[(q.device, 4)] is chunks
    torch.testing.assert_close(actual, expected)
    replay.query_chunk_size = 7
    torch.testing.assert_close(attention(q, k, v, replay, inv_freq, 8**-0.5), expected)
    assert replay._attention[(q.device, 7)] is not chunks


def test_copied_spans_keep_duplicate_occurrences():
    replay = Replay.from_trace(mutation_trace())
    visible = visible_sources(replay)
    assert len(visible[14]) > len(set(visible[14]))
    assert visible[14].count(2) == 4


def test_overfit():
    model = tiny_model().train()
    replay = Replay.from_trace(example_trace(), query_chunk_size=3)
    targets = torch.arange(len(replay.input_ids)) % 32
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    losses = []
    for _ in range(25):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(forward(model, replay), targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.1


def test_reject_unrecorded_blocks():
    session = RecordingSession(Session())
    block = NS(block_id=0, num_tokens=1)
    with pytest.raises(ValueError, match="Unrecorded"):
        session.decode_step([NS(cache_view=[block], output_block=block)], torch.tensor([1]))
    with pytest.raises(ValueError, match="Unrecorded"):
        session.merge_blocks(block, block)


def test_recording_mutations():
    session = RecordingSession(Session())
    block, empty = [session.create_block() for _ in range(2)]
    job = NS(block=block, input_ids=torch.tensor([1, 2]), context=[], pixel_values=None)
    session.prefill_batch([job])
    copied = session.merge_blocks(empty, block)
    assert session.append_block(block, block) is block
    assert [u.refs for u in session.trace.updates] == [
        ((empty.block_id, 0), (block.block_id, 2)),
        ((block.block_id, 2), (block.block_id, 2)),
    ]
    assert not any(u.sources for u in session.trace.updates)
    assert all(u.step == 1 for u in session.trace.updates)
    session.free_block(block)
    assert session.trace.updates[-1].refs == ()
    session.prefill_batch([job])
    assert session.trace.tokens[2].block == session.trace.tokens[0].block
    assert session.trace.tokens[2].position == 0
    copied_again = session.merge_blocks(copied, empty)
    assert session.trace.updates[-1].refs == ((copied.block_id, 2), (empty.block_id, 0))
    session.free_block(copied)
    context = NS(cache_view=[copied_again, block], output_block=block)
    session.decode_step([context], torch.tensor([3]))
    assert len(session.trace.tokens) == 5
    replay = Replay.from_trace(session.trace)
    assert replay.input_ids.numel() == 5
    visible = visible_sources(replay)
    assert visible[2] == [2] and visible[3] == [2, 3]
    assert visible[4] == [0, 1, 2, 3, 4]


def test_clear_preserves_copied_gradients_and_isolates_reuse():
    session = RecordingSession(Session())
    source, empty = [session.create_block() for _ in range(2)]

    def prefill(block, ids):
        session.prefill_batch(
            [NS(block=block, input_ids=torch.tensor(ids), context=[], pixel_values=None)]
        )

    prefill(source, [1, 2])
    copied = session.merge_blocks(source, empty)
    session.free_block(source)
    prefill(source, [3])
    prefill(copied, [4])
    replay = Replay.from_trace(session.trace)
    visible = visible_sources(replay)
    assert visible[2] == [2]
    assert visible[3] == [0, 1, 3]

    model = tiny_model()
    logits = forward(model, replay)
    embeddings = model.model.embed_tokens.weight
    (copied_grad,) = torch.autograd.grad(logits[3].square().sum(), embeddings, retain_graph=True)
    (reused_grad,) = torch.autograd.grad(logits[2].square().sum(), embeddings)
    assert (copied_grad[[1, 2]].abs().sum(dim=-1) > 0).all()
    assert not reused_grad[[1, 2]].any()


@pytest.mark.parametrize("operation", ["merge_blocks", "append_block", "free_block"])
def test_failed_mutation_does_not_commit_trace(operation):
    session = RecordingSession(Session())
    block = session.create_block()

    def fail(*args):
        raise RuntimeError("failed")

    setattr(session.session, operation, fail)
    args = (block,) if operation == "free_block" else (block, block)
    with pytest.raises(RuntimeError, match="failed"):
        getattr(session, operation)(*args)
    assert session.trace.updates == []
    with pytest.raises(RuntimeError, match="Discard"):
        session.decode_step([NS(cache_view=[block], output_block=block)], torch.tensor([1]))


def test_failed_forward_does_not_commit_trace():
    class FailingSession(Session):
        def decode_step(self, group, input_ids):
            raise RuntimeError("failed")

    session = RecordingSession(FailingSession())
    block = NS(block_id=0, num_tokens=0)
    group = [NS(cache_view=[block], output_block=block)]
    with pytest.raises(RuntimeError, match="failed"):
        session.decode_step(group, torch.tensor([1]))
    assert session.trace.tokens == [] and session.trace.steps == 0
    with pytest.raises(RuntimeError, match="Discard"):
        session.decode_step(group, torch.tensor([1]))


def test_reject_future_visibility():
    trace = Trace(
        [
            Token(1, 0, 0, 0, ((1, 1), (0, 1)), "decode"),
            Token(2, 1, 1, 0, ((1, 1),), "decode"),
        ]
    )
    with pytest.raises(ValueError, match="future"):
        Replay.from_trace(trace)
