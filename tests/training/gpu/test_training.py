import asyncio
import json
from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch
from minisgl.training import RecordedAsyncLLM, Trace, replay_trace
from minisgl.training.attention import attention
from minisgl_examples.training.loss import compute_supervised_loss
from transformers import Qwen3ForCausalLM

from tests.training.helpers import Session, generate_stress_example, make_llm
from tests.training.models import tiny_model
from tests.training.references import (
    sequential,
    sequential_attention,
    visible_sources,
)
from tests.training.workloads import (
    causal_trace,
    example_trace,
    mutation_trace,
    randomized_trace,
    repeated_view_trace,
    single_block_trace,
)


@pytest.mark.parametrize("trace_factory", [example_trace, mutation_trace, repeated_view_trace])
@pytest.mark.parametrize("query_chunk_size", [1, 3, 128])
def test_parallel_logits_and_gradients(trace_factory, query_chunk_size):
    model = tiny_model().cuda()
    trace = trace_factory()
    plan = trace.to(query_chunk_size=query_chunk_size)
    expected = sequential(model, trace)
    actual = replay_trace(model, plan, hidden_only=False)
    torch.testing.assert_close(actual, expected, atol=2e-06, rtol=2e-05)
    parameters = tuple(model.parameters())
    expected_grads = torch.autograd.grad(expected.square().sum(), parameters)
    actual_grads = torch.autograd.grad(actual.square().sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-05, rtol=0.0002)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_randomized_mutations_match_sequential_logits_and_gradients(seed):
    model = tiny_model().cuda()
    trace = Trace.from_dict(json.loads(json.dumps(asdict(randomized_trace(seed)))))
    plan = trace.to(query_chunk_size=5)
    expected = sequential(model, trace)
    actual = replay_trace(model, plan, hidden_only=False)
    torch.testing.assert_close(actual, expected, atol=3e-06, rtol=3e-05)
    weights = torch.randn_like(actual)
    parameters = tuple(model.parameters())
    expected_grads = torch.autograd.grad((expected * weights).sum(), parameters)
    actual_grads = torch.autograd.grad((actual * weights).sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-05, rtol=0.0003)


def test_synthetic_stress_workload_logits_and_gradients():
    data = generate_stress_example()
    trace = Trace.from_dict(json.loads(json.dumps(data["trace"])))
    assert trace == Trace.from_dict(data["trace"])
    assert data["coverage"]["blocks"] >= 24
    assert data["coverage"]["max_view_depth"] == 12
    assert set(data["coverage"]["decode_batch_sizes"]) == {4, 5, 6, 7, 8}
    assert all(
        (
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
    )
    assert all((isinstance(update.refs, tuple) for update in trace.updates))
    assert len(data["targets"]) == len(data["logprobs"]) == len(trace.tokens)
    model = tiny_model().cuda()
    plan = trace.to(query_chunk_size=7)
    expected, actual = (sequential(model, trace), replay_trace(model, plan, hidden_only=False))
    torch.testing.assert_close(actual, expected, atol=3e-06, rtol=3e-05)
    weights = torch.randn_like(expected)
    parameters = tuple(model.parameters())
    expected_grads = torch.autograd.grad((expected * weights).sum(), parameters)
    actual_grads = torch.autograd.grad((actual * weights).sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-05, rtol=0.0003)


@pytest.mark.parametrize("kv_heads", [4, 2, 1])
@pytest.mark.parametrize("rotary_width", [8, 4])
def test_grouped_attention_and_partial_rotary_gradients(kv_heads, rotary_width):
    torch.manual_seed(7)
    trace = mutation_trace()
    replay = trace.to(query_chunk_size=3)
    q = torch.randn(len(trace.tokens), 4, 8, requires_grad=True, device="cuda")
    k = torch.randn(len(trace.tokens), kv_heads, 8, requires_grad=True, device="cuda")
    v = torch.randn_like(k, requires_grad=True)
    inv_freq = 10000 ** (-torch.arange(0, rotary_width, 2, device="cuda").float() / rotary_width)
    expected = sequential_attention(q, k, v, trace, replay.order, inv_freq, 8 ** (-0.5))
    actual = attention(q, k, v, replay, inv_freq, 8 ** (-0.5))
    torch.testing.assert_close(actual, expected, atol=2e-06, rtol=2e-05)
    expected_grads = torch.autograd.grad(expected.square().sum(), (q, k, v))
    actual_grads = torch.autograd.grad(actual.square().sum(), (q, k, v))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-05, rtol=0.0002)


def test_selected_logits_and_gradients():
    model = tiny_model().cuda()
    plan = mutation_trace().to(query_chunk_size=3)
    rows = torch.tensor([18, 0, 7, 7, 13], device="cuda")
    expected = replay_trace(model, plan, hidden_only=False)[rows]
    head_rows = []
    hook = model.lm_head.register_forward_pre_hook(
        lambda module, inputs: head_rows.append(inputs[0].shape[-2])
    )
    actual = replay_trace(model, plan, rows=rows, hidden_only=False)
    hook.remove()
    assert head_rows == [rows.numel()]
    torch.testing.assert_close(actual, expected)
    parameters = tuple(model.parameters())
    expected_grads = torch.autograd.grad(expected.square().sum(), parameters)
    actual_grads = torch.autograd.grad(actual.square().sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-05, rtol=0.0002)


@pytest.mark.parametrize("rows", [None, [18, 0, 7, 7, 13], torch.tensor([18, 0, 7, 7, 13])])
def test_replay_trace_hidden_states_and_gradients(rows):
    model = tiny_model().cuda()
    plan = mutation_trace().to(query_chunk_size=3)
    head_inputs = []
    hook = model.lm_head.register_forward_pre_hook(
        lambda module, inputs: head_inputs.append(inputs[0])
    )
    expected = replay_trace(model, plan, rows=rows, hidden_only=False)
    hook.remove()
    with patch.object(model.lm_head, "forward", side_effect=AssertionError("lm_head was called")):
        states = replay_trace(model, plan, rows=rows)
    torch.testing.assert_close(states, head_inputs[0][0], atol=0, rtol=0)
    actual = model.lm_head(states)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    weights = torch.cos(torch.arange(expected.numel(), device="cuda")).reshape_as(expected)
    parameters = tuple(model.parameters())
    expected_grads = torch.autograd.grad((expected * weights).sum(), parameters)
    actual_grads = torch.autograd.grad((actual * weights).sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-05, rtol=0.0002)


@pytest.mark.parametrize(
    "rows",
    [
        None,
        [18, 0, 7, 7, 13],
        (18, 0, 7, 7, 13),
        torch.tensor([18, 0, 7, 7, 13], dtype=torch.int32),
        [],
        torch.tensor([], dtype=torch.int64),
    ],
)
@pytest.mark.parametrize("hidden_only", [True, False])
def test_replay_trace_accepts_unprepared_trace_and_row_sequences(rows, hidden_only):
    model = tiny_model().cuda()
    trace = mutation_trace()
    expected = replay_trace(model, trace, hidden_only=hidden_only)
    if rows is not None:
        expected = expected[
            (
                torch.tensor(rows, dtype=torch.long, device="cuda")
                if not isinstance(rows, torch.Tensor)
                else rows
            )
        ]
    actual = replay_trace(model, mutation_trace(), rows=rows, hidden_only=hidden_only)
    torch.testing.assert_close(actual, expected, atol=2e-06, rtol=2e-05)
    assert actual.shape == (trace.num_tokens if rows is None else len(rows), 32)


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_chunked_supervised_loss_and_gradients(chunk_size):
    model = tiny_model().cuda()
    plan = mutation_trace().to(query_chunk_size=3)
    targets = torch.arange(len(plan.input_ids), device="cuda") % 32
    targets[::3] = -100
    expected = torch.nn.functional.cross_entropy(
        replay_trace(model, plan, hidden_only=False), targets, reduction="sum"
    )
    head_rows = []
    hook = model.lm_head.register_forward_pre_hook(
        lambda module, inputs: head_rows.append(inputs[0].shape[-2])
    )
    actual = compute_supervised_loss(model, plan, targets, chunk_size=chunk_size)
    hook.remove()
    assert max(head_rows) <= chunk_size
    assert sum(head_rows) == targets.ne(-100).sum()
    torch.testing.assert_close(actual, expected)
    parameters = tuple(model.parameters())
    expected_grads = torch.autograd.grad(expected, parameters)
    actual_grads = torch.autograd.grad(actual, parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-05, rtol=0.0002)


def test_attention_checkpoint_does_not_save_dense_scores():
    length = 128
    trace = causal_trace(length)
    replay = trace.to(query_chunk_size=7)
    q = torch.randn(length, 4, 8, requires_grad=True, device="cuda")
    k = torch.randn(length, 2, 8, requires_grad=True, device="cuda")
    v = torch.randn(length, 2, 8, requires_grad=True, device="cuda")
    inv_freq = tiny_model().cuda().model.rotary_emb.inv_freq
    saved, storage_bytes = ([], {})

    def pack(tensor):
        saved.append(tensor.shape)
        storage = tensor.untyped_storage()
        storage_bytes[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = attention(q, k, v, replay, inv_freq, 8 ** (-0.5))
    assert saved
    assert all((shape.numel() < length * length for shape in saved))
    input_bytes = sum((tensor.numel() * tensor.element_size() for tensor in (q, k, v)))
    assert sum(storage_bytes.values()) <= 2 * input_bytes
    output.square().mean().backward()
    for tensor in (q, k, v):
        assert tensor.grad is not None and tensor.grad.isfinite().all()


@pytest.mark.parametrize("mixed_steps", [False, True])
def test_causal_forward_and_backward_match_huggingface(mixed_steps):
    reference, model = (tiny_model().cuda().train(), tiny_model().cuda().train())
    if mixed_steps:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    plan = single_block_trace(mixed_steps).to(query_chunk_size=3)
    expected = reference(plan.input_ids[None].cuda(), use_cache=False).logits[0]
    actual = replay_trace(model, plan, hidden_only=False)
    torch.testing.assert_close(actual, expected, atol=2e-06, rtol=2e-05)
    targets = plan.input_ids.roll(-1).cuda()
    torch.nn.functional.cross_entropy(expected, targets).backward()
    torch.nn.functional.cross_entropy(actual, targets).backward()
    for (name, parameter), (_, native) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert parameter.grad is not None and native.grad is not None
        torch.testing.assert_close(
            parameter.grad, native.grad, atol=1e-06, rtol=0.0002, msg=f"Gradient mismatch: {name}"
        )


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("device", ["cuda"])
def test_native_forward_after_replay_stays_causal_and_masked(backend, padded, device):
    if device == "cuda" and (not torch.cuda.is_available()):
        pytest.skip("CUDA is unavailable")
    model = tiny_model().cuda().to(device)
    model.set_attn_implementation(backend)
    ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 7]], device=device)
    mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]], device=device) if padded else None
    with torch.no_grad():
        expected = model(ids, attention_mask=mask, use_cache=False).logits
        replay_trace(model, mutation_trace().to(device=device), hidden_only=False)
        assert model.config._attn_implementation == backend
        with patch(
            "torch.nn.functional.scaled_dot_product_attention",
            wraps=torch.nn.functional.scaled_dot_product_attention,
        ) as sdpa:
            actual = model(ids, attention_mask=mask, use_cache=False).logits
        assert sdpa.call_count == (len(model.model.layers) if backend == "sdpa" else 0)
        if backend == "sdpa" and (not padded):
            assert all((call.kwargs["attn_mask"] is None for call in sdpa.call_args_list))
        torch.testing.assert_close(actual, expected)
        changed = ids.clone()
        changed[0, -1], changed[1, 0] = (10, 11)
        perturbed = model(changed, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(perturbed[0, :-1], actual[0, :-1])
    if padded:
        torch.testing.assert_close(perturbed[1, 1:], actual[1, 1:])


@pytest.mark.parametrize("use_reentrant", [None, False, True])
def test_attention_inference_warmup_then_training(use_reentrant):
    reference, model = (tiny_model().cuda().train(), tiny_model().cuda().eval())
    trace = repeated_view_trace()
    with torch.inference_mode():
        plan = trace.to(query_chunk_size=3).to("cpu")
        replay_trace(model, plan, hidden_only=False)
    chunks = plan._attention[next(model.parameters()).device, 3]
    assert not plan.input_ids.is_inference() and (not plan.order.is_inference())
    assert all((not entries.is_inference() for _, entries in chunks))
    model.train()
    if use_reentrant is not None:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        )
    expected = replay_trace(reference, trace.to(query_chunk_size=3), hidden_only=False)
    actual = replay_trace(model, plan, hidden_only=False)
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    assert plan._attention[next(model.parameters()).device, 3] is chunks
    for parameter, target in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, target.grad, atol=1e-06, rtol=0.0002)


@pytest.mark.parametrize("use_reentrant", [False, True])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_replay_gradient_checkpointing(use_reentrant, backend):
    reference, checkpointed = (tiny_model().cuda().train(), tiny_model().cuda().train())
    checkpointed.set_attn_implementation(backend)
    checkpointed.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
    )
    plan = mutation_trace().to(query_chunk_size=3)
    expected = replay_trace(reference, plan, hidden_only=False)
    actual = replay_trace(checkpointed, plan, hidden_only=False)
    torch.testing.assert_close(actual, expected)
    checkpointed(torch.tensor([[1, 2, 3]], device="cuda"), use_cache=False)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for parameter, target in zip(checkpointed.parameters(), reference.parameters()):
        assert parameter.grad is not None and target.grad is not None
        torch.testing.assert_close(parameter.grad, target.grad, atol=1e-06, rtol=0.0002)


def test_native_sdpa_cache_continues_after_replay():
    reference, model = (tiny_model().cuda(), tiny_model().cuda())
    for m in (reference, model):
        m.set_attn_implementation("sdpa")
    ids = torch.tensor([[1, 2, 3]], device="cuda")
    with torch.no_grad():
        original = reference(ids, use_cache=True)
        prefix = model(ids, use_cache=True)
        replay_trace(model, repeated_view_trace().to(), hidden_only=False)
        expected = reference(
            torch.tensor([[4]], device="cuda"), past_key_values=original.past_key_values
        ).logits
        with patch(
            "torch.nn.functional.scaled_dot_product_attention",
            wraps=torch.nn.functional.scaled_dot_product_attention,
        ) as sdpa:
            actual = model(
                torch.tensor([[4]], device="cuda"), past_key_values=prefix.past_key_values
            ).logits
    assert sdpa.call_count == len(model.model.layers)
    torch.testing.assert_close(actual, expected)


def test_replay_checkpoint_reloads_as_native_huggingface(tmp_path):
    model = tiny_model().cuda()
    ids = torch.tensor([[1, 2, 3]], device="cuda")
    original_keys = set(model.state_dict())
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        replay_trace(model, mutation_trace().to(), hidden_only=False)
    model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert not any(("attn_implementation" in key for key in config))
    restored = Qwen3ForCausalLM.from_pretrained(tmp_path, attn_implementation="eager").cuda().eval()
    assert restored.config._attn_implementation == "eager"
    assert set(restored.state_dict()) == original_keys
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], parameter, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(restored(ids, use_cache=False).logits, expected)


def test_mutation_trace_json_roundtrip():
    trace = mutation_trace()
    restored = Trace.from_dict(json.loads(json.dumps(asdict(trace))))
    assert restored == trace
    original, loaded = (trace.to(), restored.to())
    assert loaded.spans == original.spans
    assert loaded.copies == original.copies
    assert loaded.reads == original.reads
    for field in ("input_ids", "order"):
        torch.testing.assert_close(getattr(loaded, field), getattr(original, field), rtol=0, atol=0)
    model = tiny_model().cuda()
    with torch.no_grad():
        torch.testing.assert_close(
            replay_trace(model, loaded, hidden_only=False),
            replay_trace(model, original, hidden_only=False),
            rtol=0,
            atol=0,
        )


def test_attention_reuses_only_compact_metadata():
    length = 128
    trace = causal_trace(length)
    replay = trace.to(query_chunk_size=4)
    q, k, v = (
        torch.randn(length, 4, 8, device="cuda"),
        torch.randn(length, 2, 8, device="cuda"),
        torch.randn(length, 2, 8, device="cuda"),
    )
    inv_freq = tiny_model().cuda().model.rotary_emb.inv_freq
    expected = attention(q, k, v, replay, inv_freq, 8 ** (-0.5))
    chunks = replay._attention[q.device, 4]
    actual = attention(q, k, v, replay, inv_freq, 8 ** (-0.5))
    assert replay._attention[q.device, 4] is chunks
    assert sum((entries.numel() for _, entries in chunks)) == 4 * length
    assert all((entries.dtype == torch.long for _, entries in chunks))
    assert replay.to("cpu")._attention[q.device, 4] is chunks
    torch.testing.assert_close(actual, expected)
    replay.query_chunk_size = 7
    torch.testing.assert_close(attention(q, k, v, replay, inv_freq, 8 ** (-0.5)), expected)
    assert replay._attention[q.device, 7] is not chunks


def test_overfit():
    model = tiny_model().cuda().train()
    plan = example_trace().to(query_chunk_size=3)
    targets = torch.arange(len(plan.input_ids), device="cuda") % 32
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    losses = []
    for _ in range(25):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            replay_trace(model, plan, hidden_only=False), targets
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.1


def test_clear_preserves_copied_gradients_and_isolates_reuse():

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                source, empty = [await session.create_block() for _ in range(2)]

                async def prefill(block, ids):
                    await session.prefill_block(ids, write_to=block, cache_view=[])

                await prefill(source, [1, 2])
                copied = await session.merge_blocks(source, empty)
                await session.free_block(source)
                await prefill(source, [3])
                await prefill(copied, [4])
                plan = session.get_trace().to()
                visible = visible_sources(plan)
                assert visible[2] == [2]
                assert visible[3] == [0, 1, 3]
                model = tiny_model().cuda()
                logits = replay_trace(model, plan, hidden_only=False)
                embeddings = model.model.embed_tokens.weight
                (copied_grad,) = torch.autograd.grad(
                    logits[3].square().sum(), embeddings, retain_graph=True
                )
                (reused_grad,) = torch.autograd.grad(logits[2].square().sum(), embeddings)
                assert (copied_grad[[1, 2]].abs().sum(dim=-1) > 0).all()
                assert not reused_grad[[1, 2]].any()
        finally:
            await llm.close()

    return asyncio.run(run())
