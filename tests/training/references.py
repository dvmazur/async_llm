from itertools import groupby

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode


def visible_sources(replay):
    rows = [[] for _ in replay.reads]
    for original, reads in zip(replay.order.tolist(), replay.reads):
        for version, start, end, _ in reads:
            for source_start, source_end, _ in replay.compute_source_spans(version, start, end):
                rows[original].extend(replay.order[source_start:source_end].tolist())
    return rows


def apply_updates(trace, step, blocks):
    for update in trace.updates:
        if update.step != step:
            continue
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


def sequential(model, trace):
    """Independent step-by-step oracle, rotating raw cached keys into each view."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

    cache = [{} for _ in model.model.layers]
    outputs = {}
    for step in sorted({t.step for t in trace.tokens}):
        rows = [i for i, t in enumerate(trace.tokens) if t.step == step]
        x = model.model.embed_tokens(
            torch.tensor(
                [trace.tokens[i].token_id for i in rows],
                device=model.model.embed_tokens.weight.device,
            )
        )
        for layer_id, layer in enumerate(model.model.layers):
            apply_updates(trace, step, cache[layer_id])
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            q = attn.q_norm(attn.q_proj(h).view(len(rows), -1, attn.head_dim))
            k = attn.k_norm(attn.k_proj(h).view(len(rows), -1, attn.head_dim))
            v = attn.v_proj(h).view(len(rows), -1, attn.head_dim)
            for j, row in enumerate(rows):
                t = trace.tokens[row]
                block = cache[layer_id].setdefault(t.block, [])
                assert len(block) == t.position
                block.append((k[j], v[j]))
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
                cos, sin = model.model.rotary_emb(x, torch.tensor([positions], device=x.device))
                _, keys = apply_rotary_pos_emb(keys, keys, cos, sin)
                cos, sin = model.model.rotary_emb(
                    x, torch.tensor([[query_position]], device=x.device)
                )
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


def sequential_gdn(module, hidden, trace):
    """Replay block contents and raw convolution windows, without affine summaries."""
    blocks, origins, outputs = {}, {}, {}
    h, dk, dv = module.num_v_heads, module.head_k_dim, module.head_v_dim
    width = module.conv_kernel_size
    zero_window = hidden.new_zeros(width, module.conv_dim)

    def advance(state, transform):
        key, value, alpha, beta = transform
        decayed = state * alpha[:, None, None]
        residual = value - torch.einsum("hkv,hk->hv", decayed, key)
        return decayed + torch.einsum("hk,hv->hkv", key, beta[:, None] * residual)

    for step, entries in groupby(enumerate(trace.tokens), key=lambda entry: entry[1].step):
        entries = list(entries)
        apply_updates(trace, step, blocks)
        snapshot = {block: list(rows) for block, rows in blocks.items()}
        writers = {token.block for _, token in entries}
        pending = []
        for row, token in entries:
            visible, tail = [], zero_window
            for block, length in token.view:
                if token.kind == "decode":
                    length -= int(block in writers)
                elif block == token.block:
                    length -= 1
                reads = blocks if token.kind == "prefill" and block == token.block else snapshot
                members = reads.get(block, [])[:length]
                assert len(members) == length
                visible.extend(members)
                if members:
                    tail = origins[members[-1]][1]
            x = hidden[row]
            raw = module.in_proj_qkv(x)
            window = torch.cat([tail[1:], raw[None]], dim=0)
            convolved = (window * module.conv1d.weight[:, 0].T).sum(0)
            if module.conv1d.bias is not None:
                convolved = convolved + module.conv1d.bias
            convolved = F.silu(convolved)
            query, key, value = convolved.split([module.key_dim, module.key_dim, module.value_dim])
            query, key = (t.reshape(module.num_k_heads, dk) for t in (query, key))
            query, key = (
                t * torch.rsqrt(t.square().sum(-1, keepdim=True) + 1e-6) for t in (query, key)
            )
            query, key = (t.repeat_interleave(h // module.num_k_heads, dim=0) for t in (query, key))
            value = value.reshape(h, dv)
            beta = module.in_proj_b(x).sigmoid()
            alpha = (
                -module.A_log.float().exp()
                * F.softplus(module.in_proj_a(x).float() + module.dt_bias.float())
            ).exp()
            transform = (key, value, alpha, beta)
            state = hidden.new_zeros(h, dk, dv)
            for source in visible:
                state = advance(state, origins[source][0])
            state = advance(state, transform)
            core = torch.einsum("hkv,hk->hv", state, query / dk**0.5)
            gate = module.in_proj_z(x).reshape(h, dv)
            outputs[row] = module.out_proj(module.norm(core, gate).flatten())
            origins[row] = (transform, window)
            if token.kind == "decode":
                pending.append((token.block, row))
            else:
                blocks.setdefault(token.block, []).append(row)
        for block, row in pending:
            blocks.setdefault(block, []).append(row)
    return torch.stack([outputs[row] for row in range(len(trace.tokens))])


def sequential_delta(query, key, value, alpha, beta, trace, order):
    query, key = (x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (query, key))
    inverse = order.argsort().tolist()
    blocks, outputs = {}, [None] * len(query)
    for step, entries in groupby(enumerate(trace.tokens), key=lambda entry: entry[1].step):
        entries = list(entries)
        apply_updates(trace, step, blocks)
        snapshot = {block: list(rows) for block, rows in blocks.items()}
        writers = {token.block for _, token in entries}
        pending = []
        for row, token in entries:
            visible = []
            for block, length in token.view:
                length -= (
                    int(block in writers) if token.kind == "decode" else int(block == token.block)
                )
                reads = blocks if token.kind == "prefill" and block == token.block else snapshot
                members = reads.get(block, [])[:length]
                assert len(members) == length
                visible.extend(members)
            state = value.new_zeros(key.shape[1], key.shape[2], value.shape[2])
            for source in [*visible, row]:
                source = inverse[source]
                state = state * alpha[source, :, None, None]
                residual = value[source] - torch.einsum("hkv,hk->hv", state, key[source])
                state = state + torch.einsum(
                    "hk,hv->hkv", key[source], beta[source, :, None] * residual
                )
            outputs[inverse[row]] = torch.einsum(
                "hkv,hk->hv", state, query[inverse[row]] / key.shape[2] ** 0.5
            )
            if token.kind == "decode":
                pending.append((token.block, row))
            else:
                blocks.setdefault(token.block, []).append(row)
        for block, row in pending:
            blocks.setdefault(block, []).append(row)
    return torch.stack(outputs)


class Float64DeltaMath(TorchFunctionMode):
    """Preserve double algebra without changing production's FP32 contract."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.Tensor.float and args[0].dtype == torch.float64:
            return args[0]
        if func is torch.eye:
            kwargs = {**(kwargs or {}), "dtype": torch.float64}
        return func(*args, **(kwargs or {}))


def assert_gradients(model, reference):
    for (name, parameter), (native_name, native) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == native_name
        assert parameter.grad is not None and native.grad is not None, name
        torch.testing.assert_close(parameter.grad, native.grad, atol=3e-6, rtol=4e-4, msg=name)


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
        angle = torch.tensor(position, device=x.device).float()[axes] * inv_freq
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
                request.mrope_positions.to_tensor().T.tolist()
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


def sequential_hybrid_logits(model, trace):
    """Evaluate each layer's recorded steps using raw KV and direct GDN recurrence."""
    backbone = model.model
    device = backbone.embed_tokens.weight.device
    ids = torch.tensor([token.token_id for token in trace.tokens], device=device)
    order = torch.arange(len(trace.tokens), device=device)
    x = backbone.embed_tokens(ids)
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
            output = sequential_attention(
                query, key, value, trace, order, backbone.rotary_emb.inv_freq, attn.scaling
            )
            output = attn.o_proj(output * gate.reshape_as(output).sigmoid())
        x = x + output
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return model.lm_head(backbone.norm(x))
