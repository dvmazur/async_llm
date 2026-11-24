import asyncio
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from minisgl.models.qwen3_5_mrope import get_rope_index
from minisgl.training import RecordedAsyncLLM, replay_trace
from minisgl.training.attention import attention
from minisgl_examples.training.loss import compute_supervised_loss

from tests.training.fixtures import torch_reference_kernels as torch_reference_kernels
from tests.training.helpers import make_llm
from tests.training.models import tiny_image_model
from tests.training.references import assert_gradients, sequential_gdn, sequential_image_attention
from tests.training.workloads import (
    ImageSession,
    cache_image_trace,
    image_inputs,
    image_prefill,
    single_image_trace,
)


@pytest.mark.parametrize("loss_path", ["logits", "checkpointed_sft"])
@pytest.mark.parametrize("image_count", [1, 2])
def test_native_image_logits_and_all_parameter_gradients(loss_path, image_count):
    reference = tiny_image_model().cuda()
    model = deepcopy(reference)
    inputs = image_inputs(device="cuda")
    if image_count == 2:
        second = {
            **inputs,
            "pixel_values": inputs["pixel_values"].cos(),
            "image_grid_thw": torch.tensor([[1, 6, 4]], device="cuda"),
        }
        inputs = {
            name: torch.cat((value, second[name]), dim=1 if name.endswith("ids") else 0)
            for name, value in inputs.items()
        }
    trace = single_image_trace(inputs)
    plan = trace.to(query_chunk_size=3)
    expected = reference(**inputs, use_cache=False).logits[0]
    actual = replay_trace(model, plan, hidden_only=False)
    torch.testing.assert_close(actual, expected, atol=3e-06, rtol=3e-05)
    with torch.no_grad():
        states = replay_trace(
            model, plan, rows=torch.tensor([8, 2, 8], device="cuda"), hidden_only=True
        )
        torch.testing.assert_close(
            model.lm_head(states), expected[[8, 2, 8]], atol=3e-06, rtol=3e-05
        )
    targets = torch.tensor([1, -100, 3, -100, 5, -100, 7, 8, 9], device="cuda").repeat(image_count)
    expected_loss = F.cross_entropy(expected, targets, reduction="sum")
    if loss_path == "checkpointed_sft":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        actual_loss = compute_supervised_loss(
            model, plan, targets, chunk_size=2, checkpointing=True
        )
    else:
        actual_loss = F.cross_entropy(actual, targets, reduction="sum")
    torch.testing.assert_close(actual_loss, expected_loss, atol=3e-06, rtol=3e-05)
    actual_loss.backward()
    expected_loss.backward()
    assert_gradients(model, reference)
    assert model.model.visual.patch_embed.proj.weight.grad.abs().sum() > 0


def test_multiple_images_follow_packed_token_order():

    async def run():
        llm = None
        session = None
        try:
            reference = tiny_image_model().cuda().eval()
            model = deepcopy(reference)
            llm, raw = make_llm(ImageSession())
            session = RecordedAsyncLLM(llm)
            session.__enter__()
            a, b, c, text = [await session.create_block() for _ in range(4)]
            first = image_inputs(device="cuda")
            second = {
                **first,
                "pixel_values": first["pixel_values"].cos(),
                "image_grid_thw": torch.tensor([[1, 6, 4]], device="cuda"),
            }
            await image_prefill(session, c, inputs=first)
            await image_prefill(session, b, inputs=first)
            await image_prefill(session, a, inputs=second)
            await session.prefill_block([2, 3, 4], write_to=text)
            expected = torch.cat(
                [
                    reference(**first, use_cache=False).logits[0],
                    reference(**first, use_cache=False).logits[0],
                    reference(**second, use_cache=False).logits[0],
                    reference(
                        input_ids=torch.tensor([[2, 3, 4]], device="cuda"), use_cache=False
                    ).logits[0],
                ]
            )
            plan = session.get_trace().to(query_chunk_size=4)
            assert not torch.equal(
                plan.order, torch.arange(session.get_trace().num_tokens, device=plan.order.device)
            )
            actual = replay_trace(model, plan, hidden_only=False)
            torch.testing.assert_close(actual, expected, atol=3e-06, rtol=3e-05)
            weights = torch.cos(torch.arange(actual.numel(), device="cuda")).reshape_as(actual)
            (actual * weights).sum().backward()
            (expected * weights).sum().backward()
            assert_gradients(model, reference)
        finally:
            if session is not None:
                session.__exit__(None, None, None)
            if llm is not None:
                await llm.close()

    return asyncio.run(run())


@torch.no_grad()
def test_image_prefill_preserves_nonmonotonic_grid_positions():

    async def run():
        llm = None
        session = None
        try:
            reference = tiny_image_model().cuda().eval()
            model = deepcopy(reference)
            inputs = image_inputs(device="cuda")
            llm, raw = make_llm(ImageSession())
            session = RecordedAsyncLLM(llm)
            session.__enter__()
            block = await session.create_block()
            await session.prefill_block([2, 3], write_to=block)
            ids, types = (inputs["input_ids"][0], inputs["mm_token_type_ids"][0])
            positions = get_rope_index(ids, types, 2, inputs["image_grid_thw"]) + block.mrope_span
            await image_prefill(session, block, inputs=inputs)
            assert (positions[:, 1:] < positions[:, :-1]).any()
            assert len(session.get_trace().batches) == 2
            request = session.get_trace().batches[1].requests[0]
            torch.testing.assert_close(
                request.pixel_values.to_tensor(), inputs["pixel_values"].cpu(), atol=0, rtol=0
            )
            torch.testing.assert_close(
                request.mrope_positions.to_tensor(), positions.cpu(), atol=0, rtol=0
            )
            full = {
                **inputs,
                "input_ids": torch.cat(
                    (torch.tensor([[2, 3]], device="cuda"), inputs["input_ids"]), 1
                ),
                "mm_token_type_ids": torch.cat(
                    (
                        torch.zeros(1, 2, dtype=torch.long, device="cuda"),
                        inputs["mm_token_type_ids"],
                    ),
                    1,
                ),
            }
            expected = reference(**full, use_cache=False).logits[0]
            plan = session.get_trace().to(query_chunk_size=2)
            torch.testing.assert_close(
                replay_trace(model, plan, hidden_only=False), expected, atol=3e-06, rtol=3e-05
            )
        finally:
            if session is not None:
                session.__exit__(None, None, None)
            if llm is not None:
                await llm.close()

    return asyncio.run(run())


@pytest.mark.parametrize("query_chunk_size", [1, 5, 128])
def test_image_cache_attention_matches_sequential_logits_and_gradients(query_chunk_size):
    torch.manual_seed(37)
    trace = cache_image_trace()
    replay = trace.to("cuda", query_chunk_size=query_chunk_size)
    n = trace.num_tokens
    inputs = [torch.randn(n, heads, 12, requires_grad=True, device="cuda") for heads in (4, 2, 2)]
    native = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    inv_freq, section = (torch.tensor([1.0, 0.3, 0.1, 0.03], device="cuda"), [2, 1, 1])
    actual = attention(*inputs, replay, inv_freq, 12 ** (-0.5), mrope_section=section)
    expected = sequential_image_attention(*native, trace, replay.order, inv_freq, section)
    torch.testing.assert_close(actual, expected, atol=3e-06, rtol=3e-05)
    weights = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * weights).sum(), inputs)
    expected_grad = torch.autograd.grad((expected * weights).sum(), native)
    torch.testing.assert_close(actual_grad, expected_grad, atol=3e-06, rtol=4e-05)


def test_async_image_model_and_every_parameter_gradient_match_direct_recurrence():
    reference = tiny_image_model().cuda()
    model = deepcopy(reference)
    trace = cache_image_trace()
    plan = trace.to(query_chunk_size=4)
    backbone = reference.model.language_model
    ids = torch.tensor([token.token_id for token in trace.tokens], device="cuda")
    x = backbone.embed_tokens(ids).clone()
    features = []
    for batch in trace.batches:
        for request in batch.requests:
            if request.pixel_values is not None:
                features.extend(
                    reference.get_image_features(
                        request.pixel_values.to_tensor().to(x.device),
                        request.image_grid_thw.to_tensor().to(x.device),
                        return_dict=True,
                    ).pooler_output
                )
    x[ids == reference.config.image_token_id] = torch.cat(features)
    for layer in backbone.layers:
        h = layer.input_layernorm(x)
        if hasattr(layer, "linear_attn"):
            output = sequential_gdn(layer.linear_attn, h, trace)
        else:
            attn = layer.self_attn
            query, gate = attn.q_proj(h).view(len(h), -1, 2 * attn.head_dim).chunk(2, -1)
            query = attn.q_norm(query)
            key = attn.k_norm(attn.k_proj(h).view(len(h), -1, attn.head_dim))
            value = attn.v_proj(h).view(len(h), -1, attn.head_dim)
            output = sequential_image_attention(
                query,
                key,
                value,
                trace,
                torch.arange(len(h), device="cuda"),
                backbone.rotary_emb.inv_freq,
                backbone.config.rope_parameters["mrope_section"],
            )
            output = attn.o_proj(output * gate.reshape_as(output).sigmoid())
        x = x + output
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    expected = reference.lm_head(backbone.norm(x))
    actual = replay_trace(model, plan, hidden_only=False)
    torch.testing.assert_close(actual, expected, atol=3e-06, rtol=3e-05)
    targets = torch.arange(len(ids), device="cuda") % reference.config.text_config.vocab_size
    F.cross_entropy(actual, targets).backward()
    F.cross_entropy(expected, targets).backward()
    assert_gradients(model, reference)
