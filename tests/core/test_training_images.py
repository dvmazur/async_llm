from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F
from minisgl.models.qwen3_5_mrope import get_rope_index
from minisgl.training.replay import Replay, attention, forward, hidden
from minisgl.training.sft import supervised_loss
from minisgl.training.trace import RecordingSession, TensorData
from test_training import Session
from test_training_gdn import sequential_gdn
from test_training_logits import torch_reference_kernels as torch_reference_kernels
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration


class ImageSession(Session):
    def create_block(self):
        block = super().create_block()
        block.mrope_span = 0
        return block

    def _mrope_rel(self, ids, types, grid):
        return get_rope_index(ids, types, 2, grid)

    def prefill_batch(self, jobs):
        for job in jobs:
            positions = getattr(job, "mrope_rel", None)
            if positions is None:
                positions = (
                    self._mrope_rel(job.input_ids, job.mm_token_type_ids, job.image_grid_thw)
                    if job.pixel_values is not None
                    else torch.arange(job.input_ids.numel()).expand(3, -1)
                )
            job.block.mrope_span += max(0, int(positions.max()) + 1)
        super().prefill_batch(jobs)
        return [None] * len(jobs)

    def decode_step(self, group, ids):
        for ctx in group:
            ctx.output_block.mrope_span += 1
        super().decode_step(group, ids)

    def merge_blocks(self, left, right):
        block = super().merge_blocks(left, right)
        block.mrope_span = left.mrope_span + right.mrope_span
        return block

    def append_block(self, left, right):
        span = left.mrope_span + right.mrope_span
        super().append_block(left, right)
        left.mrope_span = span
        return left

    def free_block(self, block):
        super().free_block(block)
        block.mrope_span = 0


def image_inputs():
    ids = torch.tensor([30, *([29] * 6), 31, 8])
    return {
        "input_ids": ids[None],
        "pixel_values": torch.sin(torch.arange(24 * 12).float()).reshape(24, 12) / 2,
        "image_grid_thw": torch.tensor([[1, 4, 6]]),
        "mm_token_type_ids": (ids == 29).long()[None],
    }


def image_prefill(session, block, context=(), inputs=None):
    inputs = image_inputs() if inputs is None else inputs
    return session.prefill_batch(
        [
            NS(
                block=block,
                input_ids=inputs["input_ids"][0],
                context=list(context),
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                mm_token_type_ids=inputs["mm_token_type_ids"][0],
            )
        ]
    )


def single_image_trace(inputs=None):
    session = RecordingSession(ImageSession())
    image_prefill(session, session.create_block(), inputs=inputs)
    return session.trace


def tiny_image_model():
    torch.manual_seed(71)
    config = Qwen3_5Config(
        image_token_id=29,
        vision_start_token_id=30,
        vision_end_token_id=31,
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 6,
            "linear_conv_kernel_dim": 4,
            "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
        },
    )
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    model = Qwen3_5ForConditionalGeneration(config).train()
    with torch.no_grad():
        model.model.language_model.layers[0].linear_attn.A_log.fill_(-1)
    return model


def assert_gradients(model, reference, frozen_vision=False):
    for (name, parameter), (native_name, native) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == native_name
        if frozen_vision and name.startswith("model.visual."):
            assert parameter.grad is None and native.grad is None, name
        else:
            assert parameter.grad is not None and native.grad is not None, name
            torch.testing.assert_close(parameter.grad, native.grad, atol=3e-6, rtol=4e-4, msg=name)


@pytest.mark.parametrize("loss_path", ["logits", "checkpointed_sft"])
@pytest.mark.parametrize("image_count", [1, 2])
def test_native_image_logits_and_all_parameter_gradients(loss_path, image_count):
    reference = tiny_image_model()
    model = deepcopy(reference)
    inputs = image_inputs()
    if image_count == 2:
        second = {
            **inputs,
            "pixel_values": inputs["pixel_values"].cos(),
            "image_grid_thw": torch.tensor([[1, 6, 4]]),
        }
        inputs = {
            name: torch.cat((value, second[name]), dim=1 if name.endswith("ids") else 0)
            for name, value in inputs.items()
        }
    trace = single_image_trace(inputs)
    plan = Replay.from_trace(trace, query_chunk_size=3)
    expected = reference(**inputs, use_cache=False).logits[0]
    actual = forward(model, plan)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    with torch.no_grad():
        states = hidden(model, plan, rows=torch.tensor([8, 2, 8]))
        torch.testing.assert_close(model.lm_head(states), expected[[8, 2, 8]], atol=3e-6, rtol=3e-5)
    targets = torch.tensor([1, -100, 3, -100, 5, -100, 7, 8, 9]).repeat(image_count)
    expected_loss = F.cross_entropy(expected, targets, reduction="sum")
    if loss_path == "checkpointed_sft":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        actual_loss = supervised_loss(model, plan, targets, chunk_size=2, checkpointing=True)
    else:
        actual_loss = F.cross_entropy(actual, targets, reduction="sum")
    torch.testing.assert_close(actual_loss, expected_loss, atol=3e-6, rtol=3e-5)
    actual_loss.backward()
    expected_loss.backward()
    assert_gradients(model, reference)
    assert model.model.visual.patch_embed.proj.weight.grad.abs().sum() > 0


def test_precomputed_image_embeddings_train_only_decoder():
    reference = tiny_image_model()
    model = deepcopy(reference)
    inputs = image_inputs()
    with torch.no_grad():
        features = torch.cat(
            reference.model.get_image_features(
                inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True
            ).pooler_output
        )
    trace = single_image_trace()
    batch = trace.batches[0]
    request = replace(
        batch.requests[0],
        pixel_values=None,
        image_grid_thw=None,
        image_embeds=TensorData.capture(features),
    )
    trace.batches[0] = replace(batch, requests=(request,))
    plan = Replay.from_trace(trace, query_chunk_size=2)
    embedded = reference.model.language_model.embed_tokens(inputs["input_ids"]).clone()
    embedded[inputs["mm_token_type_ids"].bool()] = features
    positions = request.mrope_positions.tensor()[:, None]
    expected = reference(inputs_embeds=embedded, position_ids=positions, use_cache=False).logits[0]
    actual = forward(model, plan)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    weights = torch.cos(torch.arange(actual.numel())).reshape_as(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    assert_gradients(model, reference, frozen_vision=True)


def test_multiple_images_and_frozen_features_follow_packed_token_order():
    reference = tiny_image_model().eval()
    model = deepcopy(reference)
    session = RecordingSession(ImageSession())
    a, b, c, text = [session.create_block() for _ in range(4)]
    first = image_inputs()
    second = {
        **first,
        "pixel_values": first["pixel_values"].cos(),
        "image_grid_thw": torch.tensor([[1, 6, 4]]),
    }
    features = torch.cat(
        reference.get_image_features(
            first["pixel_values"], first["image_grid_thw"], return_dict=True
        ).pooler_output
    ).detach()
    image_prefill(session, c, inputs=first)
    image_prefill(session, b, inputs=first)
    image_prefill(session, a, inputs=second)
    session.prefill_block(text, torch.tensor([2, 3, 4]))
    frozen_batch = session.trace.batches[1]
    frozen = replace(
        frozen_batch.requests[0],
        pixel_values=None,
        image_grid_thw=None,
        image_embeds=TensorData.capture(features),
    )
    session.trace.batches[1] = replace(frozen_batch, requests=(frozen,))
    last_batch = session.trace.batches[2]
    session.trace.batches[2] = replace(
        last_batch,
        requests=(replace(last_batch.requests[0], image_embeds=TensorData.capture(features * 0)),),
    )
    embedded = reference.get_input_embeddings()(first["input_ids"]).clone()
    embedded[first["mm_token_type_ids"].bool()] = features
    expected = torch.cat(
        [
            reference(**first, use_cache=False).logits[0],
            reference(
                inputs_embeds=embedded,
                position_ids=frozen.mrope_positions.tensor()[:, None],
                use_cache=False,
            ).logits[0],
            reference(**second, use_cache=False).logits[0],
            reference(input_ids=torch.tensor([[2, 3, 4]]), use_cache=False).logits[0],
        ]
    )
    replay = Replay.from_trace(session.trace, query_chunk_size=4)
    assert not torch.equal(replay.order, torch.arange(session.trace.num_tokens))
    actual = forward(model, replay)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    weights = torch.cos(torch.arange(actual.numel())).reshape_as(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    assert_gradients(model, reference)


@torch.no_grad()
def test_image_chunks_keep_positions_when_grid_coordinates_move_backwards():
    reference = tiny_image_model().eval()
    model = deepcopy(reference)
    inputs = image_inputs()
    features = torch.cat(
        reference.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True
        ).pooler_output
    )
    session = RecordingSession(ImageSession())
    block = session.create_block()
    session.prefill_block(block, torch.tensor([2, 3]))
    ids, types = inputs["input_ids"][0], inputs["mm_token_type_ids"][0]
    positions = get_rope_index(ids, types, 2, inputs["image_grid_thw"]) + block.mrope_span
    boundaries = [0, 1, 4, 5, 7, len(ids)]
    for start, end in zip(boundaries, boundaries[1:]):
        before = int(types[:start].sum())
        count = int(types[start:end].sum())
        session.prefill_batch(
            [
                NS(
                    block=block,
                    input_ids=ids[start:end],
                    context=[],
                    pixel_values=None,
                    image_embeds=features[before : before + count],
                    mm_token_type_ids=types[start:end],
                    mrope_rel=positions[:, start:end] - block.mrope_span,
                )
            ]
        )
    backwards = session.trace.batches[3].requests[0]
    assert backwards.mrope_positions.tensor().max() < backwards.write_span
    assert session.trace.batches[1].requests[0].image_embeds.shape == (0, 32)
    full = {
        **inputs,
        "input_ids": torch.cat((torch.tensor([[2, 3]]), inputs["input_ids"]), 1),
        "mm_token_type_ids": torch.cat(
            (torch.zeros(1, 2, dtype=torch.long), inputs["mm_token_type_ids"]), 1
        ),
    }
    expected = reference(**full, use_cache=False).logits[0]
    replay = Replay.from_trace(session.trace, query_chunk_size=2)
    torch.testing.assert_close(forward(model, replay), expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("mrope_positions", torch.zeros(2, 9, dtype=torch.long), "positions"),
        ("mrope_positions", None, "positions"),
        ("write_span", 1, "write span"),
        ("cache_spans", (1,), "cache spans"),
        ("image_grid_thw", None, "image_grid_thw"),
        ("image_grid_thw", torch.tensor([[1, 4, 5]]), "merge-aligned"),
        ("pixel_values", torch.zeros(23, 12), "patch shape"),
        ("mm_token_type_ids", torch.full((9,), 2), "0/1"),
        ("mm_token_type_ids", torch.zeros(9, dtype=torch.long), "disagree"),
    ],
)
def test_malformed_image_metadata_fails_explicitly(field, value, message):
    trace = single_image_trace()
    batch = trace.batches[0]
    value = TensorData.capture(value) if isinstance(value, torch.Tensor) else value
    trace.batches[0] = replace(batch, requests=(replace(batch.requests[0], **{field: value}),))
    with pytest.raises(ValueError, match=message):
        forward(tiny_image_model(), Replay.from_trace(trace))


def test_image_token_ids_require_payload_without_token_types():
    trace = single_image_trace()
    batch = trace.batches[0]
    request = replace(
        batch.requests[0], pixel_values=None, image_grid_thw=None, mm_token_type_ids=None
    )
    trace.batches[0] = replace(batch, requests=(request,))
    with pytest.raises(ValueError, match="require recorded pixels or image embeddings"):
        forward(tiny_image_model(), Replay.from_trace(trace))


def cache_image_trace():
    session = RecordingSession(ImageSession())
    image, a, b = [session.create_block() for _ in range(3)]
    image_prefill(session, image)
    session.prefill_block(a, torch.tensor([2, 3]), context=[image, image])
    image_prefill(session, a, context=[image])
    session.prefill_block(b, torch.tensor([4, 5]), context=[a, image])
    session.decode_step(
        [
            NS(cache_view=[b, image, a, image], output_block=a),
            NS(cache_view=[image, a, b], output_block=b),
        ],
        torch.tensor([6, 7]),
    )
    merged = session.merge_blocks(a, image)
    session.append_block(merged, merged)
    session.free_block(image)
    session.prefill_block(image, torch.tensor([10, 11]), context=[merged, b, merged])
    session.decode_step(
        [
            NS(cache_view=[merged, image, b, merged], output_block=a),
            NS(cache_view=[b, image, b], output_block=b),
        ],
        torch.tensor([12, 13]),
    )
    session.prefill_block(merged, torch.tensor([14, 15]), context=[b, image])
    return session.trace


def sequential_image_attention(q, k, v, trace, order, inv_freq, section):
    """Materialize live KV copies, their coordinates, and each dense attention row."""
    inverse = order.argsort().tolist()
    blocks, spans, outputs = {}, {}, []
    cursor, update_index = 0, 0
    axes = [0] * len(inv_freq)
    for axis in (1, 2):
        for frequency in range(axis, min(3 * section[axis], len(axes)), 3):
            axes[frequency] = axis

    def rotate(x, position):
        angle = torch.tensor(position).float()[axes] * inv_freq
        left, right = x[..., : 2 * len(inv_freq)].chunk(2, -1)
        return torch.cat(
            (
                left * angle.cos() - right * angle.sin(),
                right * angle.cos() + left * angle.sin(),
                x[..., 2 * len(inv_freq) :],
            ),
            -1,
        )

    def evaluate(row, segments):
        scores, values = [], []
        for entries, query_position in segments:
            query = rotate(q[inverse[row]], query_position)
            for source, position in entries:
                key = rotate(k[inverse[source]], position).repeat_interleave(
                    q.shape[1] // k.shape[1], 0
                )
                scores.append((query * key).sum(-1) * q.shape[-1] ** -0.5)
                values.append(v[inverse[source]].repeat_interleave(q.shape[1] // k.shape[1], 0))
        weights = torch.stack(scores, -1).softmax(-1)
        return (weights[..., None] * torch.stack(values, 1)).sum(1).flatten()

    for batch in trace.batches:
        while update_index < len(trace.updates) and trace.updates[update_index].step <= batch.step:
            update = trace.updates[update_index]
            entries, offset = [], 0
            for block, length in update.refs:
                assert length == len(blocks.get(block, []))
                entries.extend(
                    (row, [p + offset for p in pos]) for row, pos in blocks.get(block, [])
                )
                offset += spans.get(block, 0)
            blocks[update.block], spans[update.block] = entries, offset
            update_index += 1
        before_spans = spans.copy()
        written = {request.write_to for request in batch.requests}
        pending = []
        for request in batch.requests:
            positions = (
                request.mrope_positions.tensor().T.tolist()
                if request.mrope_positions is not None
                else [
                    [before_spans.get(request.write_to, 0) + i] * 3
                    for i in range(len(request.input_ids))
                ]
            )
            entries = [(cursor + i, pos) for i, pos in enumerate(positions)]
            blocks.setdefault(request.write_to, []).extend(entries)
            spans[request.write_to] = max(
                before_spans.get(request.write_to, 0), max(max(pos) for pos in positions) + 1
            )
            pending.append((request, entries))
            cursor += len(entries)
        for request, entries in pending:
            view_spans = [before_spans.get(block, 0) for block, _ in request.cache_view]
            if batch.kind == "decode":
                view_spans = [
                    span + int(block in written)
                    for span, (block, _) in zip(view_spans, request.cache_view, strict=True)
                ]
            for offset, (row, position) in enumerate(entries):
                query_position = (
                    [sum(view_spans) - 1] * 3
                    if batch.kind == "decode"
                    else [p + sum(view_spans[:-1]) for p in position]
                )
                segments, prefix = [], 0
                for i, (block, length) in enumerate(request.cache_view):
                    length += (
                        int(block in written)
                        if batch.kind == "decode"
                        else (offset + 1 if i == len(request.cache_view) - 1 else 0)
                    )
                    if length:
                        segments.append(
                            (blocks[block][:length], [p - prefix for p in query_position])
                        )
                    prefix += view_spans[i]
                if all(block != request.write_to for block, _ in request.cache_view):
                    segments.append(([(row, position)], position))
                outputs.append(evaluate(row, segments))
    return torch.stack(outputs)[order]


@pytest.mark.parametrize("query_chunk_size", [1, 5, 128])
def test_image_cache_attention_matches_sequential_logits_and_gradients(query_chunk_size):
    torch.manual_seed(37)
    trace = cache_image_trace()
    replay = Replay.from_trace(trace, query_chunk_size=query_chunk_size)
    n = trace.num_tokens
    inputs = [torch.randn(n, heads, 12, requires_grad=True) for heads in (4, 2, 2)]
    native = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    inv_freq, section = torch.tensor([1.0, 0.3, 0.1, 0.03]), [2, 1, 1]
    actual = attention(*inputs, replay, inv_freq, 12**-0.5, mrope_section=section)
    expected = sequential_image_attention(*native, trace, replay.order, inv_freq, section)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    weights = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * weights).sum(), inputs)
    expected_grad = torch.autograd.grad((expected * weights).sum(), native)
    torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=4e-5)


def test_async_image_model_and_every_parameter_gradient_match_direct_recurrence():
    reference = tiny_image_model()
    model = deepcopy(reference)
    trace = cache_image_trace()
    replay = Replay.from_trace(trace, query_chunk_size=4)
    backbone = reference.model.language_model
    ids = torch.tensor([token.token_id for token in trace.tokens])
    x = backbone.embed_tokens(ids).clone()
    features = []
    for batch in trace.batches:
        for request in batch.requests:
            if request.pixel_values is not None:
                features.extend(
                    reference.get_image_features(
                        request.pixel_values.tensor(),
                        request.image_grid_thw.tensor(),
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
                torch.arange(len(h)),
                backbone.rotary_emb.inv_freq,
                backbone.config.rope_parameters["mrope_section"],
            )
            output = attn.o_proj(output * gate.reshape_as(output).sigmoid())
        x = x + output
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    expected = reference.lm_head(backbone.norm(x))
    actual = forward(model, replay)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    targets = torch.arange(len(ids)) % reference.config.text_config.vocab_size
    F.cross_entropy(actual, targets).backward()
    F.cross_entropy(expected, targets).backward()
    assert_gradients(model, reference)
