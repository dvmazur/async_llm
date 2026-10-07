import json
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest
import torch
from minisgl.training.replay import Replay, forward
from minisgl.training.sft import restore_trace
from minisgl.training.trace import RecordingSession, Token, Trace
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
            for update in trace.updates:
                if update.step == step:
                    cache[layer_id][update.block] = [
                        origins[layer_id][source] for source in update.sources
                    ]
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


@pytest.mark.parametrize("trace_factory", [example_trace, mutation_trace])
def test_parallel_logits_and_gradients(trace_factory):
    model = tiny_model()
    trace = trace_factory()
    replay = Replay.from_trace(trace)
    expected = sequential(model, trace)
    actual = forward(model, replay)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    parameters = tuple(model.parameters())
    expected_grads = torch.autograd.grad(expected.square().sum(), parameters)
    actual_grads = torch.autograd.grad(actual.square().sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=2e-4)


def test_causal_matches_huggingface():
    model = tiny_model()
    trace = Trace([Token(i + 1, 0, 0, i, ((0, i + 1),), "prefill") for i in range(6)])
    replay = Replay.from_trace(trace)
    expected = model(replay.input_ids[None], use_cache=False).logits[0]
    torch.testing.assert_close(forward(model, replay), expected, atol=2e-6, rtol=2e-5)


def test_native_forward_after_replay_stays_causal_and_masked():
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 7]])
    mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
    with torch.no_grad():
        expected = model(ids, attention_mask=mask, use_cache=False).logits
        forward(model, Replay.from_trace(mutation_trace()))
        actual = model(ids, attention_mask=mask, use_cache=False).logits
        torch.testing.assert_close(actual, expected)
        changed = ids.clone()
        changed[0, -1], changed[1, 0] = 10, 11
        perturbed = model(changed, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(perturbed[0, :-1], actual[0, :-1])
    torch.testing.assert_close(perturbed[1, 1:], actual[1, 1:])


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_replay_gradient_checkpointing(use_reentrant):
    reference, checkpointed = tiny_model().train(), tiny_model().train()
    checkpointed.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
    )
    replay = Replay.from_trace(mutation_trace())
    expected = forward(reference, replay)
    actual = forward(checkpointed, replay)
    torch.testing.assert_close(actual, expected)
    checkpointed(torch.tensor([[1, 2, 3]]), use_cache=False)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for parameter, target in zip(checkpointed.parameters(), reference.parameters()):
        assert parameter.grad is not None and target.grad is not None
        torch.testing.assert_close(parameter.grad, target.grad, atol=1e-6, rtol=2e-4)


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
    for field in ("input_ids", "positions", "query_positions", "mask", "order", "sources"):
        torch.testing.assert_close(getattr(loaded, field), getattr(original, field), rtol=0, atol=0)
    model = tiny_model()
    with torch.no_grad():
        torch.testing.assert_close(forward(model, loaded), forward(model, original), rtol=0, atol=0)


def test_recording_and_masks():
    trace = example_trace()
    assert trace.tokens[7].step == trace.tokens[8].step
    assert trace.tokens[9].position == 3
    replay = Replay.from_trace(trace)
    inverse = replay.order.argsort()
    mask = replay.mask[inverse][:, inverse]
    assert mask[7, 8] and mask[8, 7]
    assert mask[9, 9] and not mask[9, 10]
    assert not mask[7, 9]
    assert mask[-1, -1] and not mask[-1, 7]


def test_overfit():
    model = tiny_model().train()
    replay = Replay.from_trace(example_trace())
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
    assert [u.sources for u in session.trace.updates] == [(0, 1), (0, 1, 0, 1)]
    assert all(u.step == 1 for u in session.trace.updates)
    session.free_block(block)
    assert session.trace.updates[-1].sources == ()
    session.prefill_batch([job])
    assert session.trace.tokens[2].block == session.trace.tokens[0].block
    assert session.trace.tokens[2].position == 0
    copied_again = session.merge_blocks(copied, empty)
    assert session.trace.updates[-1].sources == (0, 1)
    session.free_block(copied)
    context = NS(cache_view=[copied_again, block], output_block=block)
    session.decode_step([context], torch.tensor([3]))
    assert len(session.trace.tokens) == 5
    replay = Replay.from_trace(session.trace)
    assert replay.input_ids.numel() == 5
    assert replay.sources.numel() > replay.input_ids.numel()


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
    original_sources = replay.order[replay.sources]
    mask = replay.mask[replay.order.argsort()]
    assert not mask[2, original_sources < 2].any()
    assert set(original_sources[mask[2]].tolist()) == {2}
    assert set(original_sources[mask[3]].tolist()) == {0, 1, 3}

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
