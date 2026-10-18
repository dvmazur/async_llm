from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from itertools import groupby
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint
from transformers import AttentionInterface

from .trace import Trace


@dataclass
class Replay:
    """Spans: (source start/end, block offset); reads: (version, start/end, query position)."""

    input_ids: torch.Tensor
    order: torch.Tensor
    spans: list[list[tuple[int, int, int]]]
    reads: list[list[tuple[int, int, int, int]]]
    copies: list[list[tuple[int, int, int]]]
    query_chunk_size: int = 128
    trace: Trace | None = None
    gdn_segment_size: int = 8
    gdn_read_mode: str = "expanded"
    gdn_cache_reused: bool = True
    gdn_read_backend: str = "torch"
    attention_checkpointing: bool = True
    cache_plans: bool = True
    _gdn: dict = field(default_factory=dict, repr=False)
    _attention: dict = field(default_factory=dict, repr=False)

    def source_spans(self, version, start, end):
        stack = [(version, start, end, 0)]
        while stack:
            version, start, end, offset = stack.pop()
            if version < 0:
                yield start, end, offset
                continue
            for left, right, position in reversed(self.spans[version]):
                lo, hi = max(start, position), min(end, position + right - left)
                if lo < hi:
                    stack.append((-1, left + lo - position, left + hi - position, offset + lo))
            for parent, length, position in reversed(self.copies[version]):
                lo, hi = max(start, position), min(end, position + length)
                if lo < hi:
                    stack.append((parent, lo - position, hi - position, offset + position))

    @torch.inference_mode(False)
    def to(self, device):
        input_ids, order = self.input_ids.to(device), self.order.to(device)
        if input_ids is self.input_ids and order is self.order:
            return self
        return replace(self, input_ids=input_ids, order=order)

    @classmethod
    @torch.inference_mode(False)
    def from_trace(
        cls,
        trace: Trace,
        device="cpu",
        query_chunk_size=128,
        gdn_segment_size=8,
        gdn_read_mode="expanded",
        gdn_cache_reused=True,
        gdn_read_backend="torch",
        attention_checkpointing=True,
        cache_plans=True,
    ):
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if gdn_segment_size < 1:
            raise ValueError("gdn_segment_size must be positive")
        if gdn_read_mode not in {"expanded", "factorized"}:
            raise ValueError("gdn_read_mode must be expanded or factorized")
        if gdn_read_backend not in {"torch", "triton"}:
            raise ValueError("gdn_read_backend must be torch or triton")
        if trace.multimodal:
            raise NotImplementedError("Multimodal training replay is not implemented")
        tokens = trace.tokens
        if not tokens:
            raise ValueError("Cannot replay an empty trace")
        if any(a.step > b.step for a, b in zip(tokens, tokens[1:])) or any(
            a.step > b.step for a, b in zip(trace.updates, trace.updates[1:])
        ):
            raise ValueError("Trace must be in execution order")
        order = sorted(
            range(len(tokens)), key=lambda i: (tokens[i].block, tokens[i].step, tokens[i].position)
        )
        inverse = {row: i for i, row in enumerate(order)}
        spans, copies, lengths, current, rows = [], [], [], {}, []

        def append(version, row):
            source = inverse[row]
            parts = spans[version]
            if parts and parts[-1][1] == source:
                start, _, position = parts[-1]
                parts[-1] = (start, source + 1, position)
            else:
                parts.append((source, source + 1, lengths[version]))
            lengths[version] += 1

        def replace(block, sources=(), refs=None):
            if refs is not None and sources:
                raise ValueError("Mutation cannot combine refs and legacy source rows")
            parents, position = [], 0
            for source, length in refs or ():
                parent = current.get(source)
                if source < 0 or length < 0 or length != (0 if parent is None else lengths[parent]):
                    raise ValueError("Mutation must reference a live block with its exact length")
                if length:
                    parents.append((parent, length, position))
                    position += length
            current[block] = len(spans)
            spans.append([])
            copies.append(parents)
            lengths.append(position)
            for row in sources:
                append(current[block], row)

        def apply_update(update, before):
            if any(i < 0 or i >= before or tokens[i].step >= update.step for i in update.sources):
                raise ValueError("Block update refers to future or unrecorded tokens")
            replace(update.block, update.sources, update.refs)

        update_index = 0
        for step, group in groupby(enumerate(tokens), key=lambda item: item[1].step):
            group = list(group)
            while update_index < len(trace.updates) and trace.updates[update_index].step <= step:
                update = trace.updates[update_index]
                apply_update(update, group[0][0])
                update_index += 1
            for row, token in group:
                if token.block not in current:
                    replace(token.block)
                version = current[token.block]
                if token.position != lengths[version]:
                    raise ValueError("Writes must append to a contiguous block span")
                append(version, row)
            for row, token in group:
                if token.kind not in {"prefill", "decode"}:
                    raise ValueError("Unknown forward kind")
                own_lengths = [length for block, length in token.view if block == token.block]
                if any(length <= token.position for length in own_lengths):
                    raise ValueError("A token must attend to itself")
                if token.kind == "prefill" and (
                    not own_lengths or any(length != token.position + 1 for length in own_lengths)
                ):
                    raise ValueError("Prefill must include its own causal prefix")
                entries, prefix = [], 0
                total = sum(length for _, length in token.view)
                for block, length in token.view:
                    if length < 0:
                        raise ValueError("View length must be nonnegative")
                    if length == 0:
                        continue
                    if block not in current or length > lengths[current[block]]:
                        raise ValueError("View exposes future or unrecorded tokens")
                    entries.append((current[block], 0, length, total - 1 - prefix))
                    prefix += length
                if not own_lengths:
                    entries.append(
                        (current[token.block], token.position, token.position + 1, token.position)
                    )
                rows.append(entries)

        for update in trace.updates[update_index:]:
            apply_update(update, len(tokens))
        return cls(
            input_ids=torch.tensor([tokens[i].token_id for i in order], device=device),
            order=torch.tensor(order, device=device),
            spans=spans,
            reads=[rows[i] for i in order],
            copies=copies,
            query_chunk_size=query_chunk_size,
            attention_checkpointing=attention_checkpointing,
            cache_plans=cache_plans,
            trace=deepcopy(trace),
            gdn_segment_size=gdn_segment_size,
            gdn_read_mode=gdn_read_mode,
            gdn_cache_reused=gdn_cache_reused,
            gdn_read_backend=gdn_read_backend,
        )


def rotate(x, positions, inv_freq):
    angles = positions.float()[:, None] * inv_freq.float()[None, :]
    cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
    width = 2 * inv_freq.numel()
    left, right = x[..., :width].float().chunk(2, dim=-1)
    rotated = torch.cat((left * cos - right * sin, right * cos + left * sin), -1)
    return torch.cat((rotated.to(x.dtype), x[..., width:]), -1)


def attention_chunk(q, k, v, inv_freq, *, replay, blocks, entries, scaling):
    groups = q.shape[1] // k.shape[1]
    sources, positions = [], []
    for version, lo, hi, _ in blocks:
        positions.append(torch.arange(lo, hi, device=q.device))
        for start, end, _ in replay.source_spans(version, lo, hi):
            sources.append(torch.arange(start, end, device=q.device))
    sources, positions = torch.cat(sources), torch.cat(positions)
    key_sizes = [hi - lo for _, lo, hi, _ in blocks]
    row_sizes = [count for _, _, _, count in blocks]
    queries = rotate(q[entries[:, 0]], entries[:, 3], inv_freq).split(row_sizes)
    keys = rotate(k[sources], positions, inv_freq).split(key_sizes)
    values = v[sources].transpose(0, 1)
    scores = []
    for query, key, position, entry in zip(
        queries, keys, positions.split(key_sizes), entries.split(row_sizes), strict=True
    ):
        rows, starts, ends, _ = entry.unbind(1)
        query = query.reshape(-1, k.shape[1], groups, q.shape[-1]).permute(1, 2, 0, 3)
        score = (query.flatten(1, 2) @ key.permute(1, 2, 0)).view(k.shape[1], groups, len(rows), -1)
        score = score.float() * scaling
        visible = (position >= starts[:, None]) & (position < ends[:, None])
        score = score.masked_fill(~visible, float("-inf"))
        scores.append(
            score.new_full((*score.shape[:2], len(q), len(key)), float("-inf")).index_copy(
                2, rows, score
            )
        )
    weights = torch.cat(scores, -1).softmax(-1).to(v.dtype)
    output = (weights.flatten(1, 2) @ values).view(k.shape[1], groups, len(q), -1)
    return output.permute(2, 0, 1, 3).flatten(1)


def attention(q, k, v, replay, inv_freq, scaling):
    cache_key = (q.device, replay.query_chunk_size)
    if cache_key not in replay._attention:
        chunks = []
        for start in range(0, len(q), replay.query_chunk_size):
            grouped = {}
            for row, reads in enumerate(replay.reads[start : start + replay.query_chunk_size]):
                occurrences = {}
                for version, left, right, position in reads:
                    occurrence = occurrences.get(version, 0)
                    occurrences[version] = occurrence + 1
                    grouped.setdefault((version, occurrence), []).append(
                        (row, left, right, position)
                    )
            blocks, entries = [], []
            for (version, _), reads in grouped.items():
                blocks.append(
                    (version, min(r[1] for r in reads), max(r[2] for r in reads), len(reads))
                )
                entries.extend(reads)
            with torch.inference_mode(False):
                chunks.append((blocks, torch.tensor(entries, device=q.device)))
        replay._attention[cache_key] = chunks
    outputs = []
    for query, (blocks, entries) in zip(
        q.split(replay.query_chunk_size), replay._attention[cache_key], strict=True
    ):
        kwargs = {"replay": replay, "blocks": blocks, "entries": entries, "scaling": scaling}
        if (
            replay.attention_checkpointing
            and torch.is_grad_enabled()
            and any(x.requires_grad for x in (q, k, v))
        ):
            output = checkpoint(
                attention_chunk, query, k, v, inv_freq, use_reentrant=False, **kwargs
            )
        else:
            output = attention_chunk(query, k, v, inv_freq, **kwargs)
        outputs.append(output)
    return torch.cat(outputs)


def replay_attention(
    module, query, key, value, attention_mask, scaling, replay=None, inv_freq=None, **kwargs
):
    if query.shape[0] != 1:
        raise ValueError("Replay packs all tokens into one sequence")
    output = attention(
        query[0].transpose(0, 1),
        key[0].transpose(0, 1),
        value[0].transpose(0, 1),
        replay,
        inv_freq,
        scaling,
    )
    return output.view(1, query.shape[2], query.shape[1], query.shape[3]), None


AttentionInterface.register("minisgl_replay", replay_attention)


def replay_self_attention(module, *args, replay=None, **kwargs):
    if replay is None:
        return module._replay_native_forward(*args, **kwargs)
    local = copy(module)
    local.config = copy(module.config)
    local.config._attn_implementation = "minisgl_replay"
    return module._replay_native_forward.__func__(local, *args, replay=replay, **kwargs)


def replay_gdn(
    module, hidden_states, cache_params=None, attention_mask=None, *, gdn_replay=None, **kwargs
):
    if gdn_replay is None:
        return module._replay_native_forward(
            hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs
        )
    from .gdn import gdn_forward

    return gdn_forward(module, hidden_states, gdn_replay)


def _forward(model, replay: Replay, rows, hidden_only):
    if model.config.model_type not in {"qwen3", "qwen3_5_text", "qwen3_5"}:
        raise ValueError("Only dense text Qwen3 and Qwen3.5 are supported")
    backbone = model.model
    if model.config.model_type == "qwen3_5":
        backbone = backbone.language_model
    config = backbone.config
    rotary = backbone.rotary_emb
    if getattr(rotary, "rope_type", "default") != "default":
        raise NotImplementedError("Only default RoPE is supported")
    if getattr(backbone, "has_sliding_layers", False) or config.attention_dropout:
        raise NotImplementedError("Replay requires full attention and zero attention dropout")
    if not replay.cache_plans:
        replay._gdn.clear()
        replay._attention.clear()
        replay = replace(replay, _gdn={}, _attention={})
    kwargs = {}
    if "linear_attention" in config.layer_types:
        from .gdn import GDNReplay

        if replay.trace is None:
            raise ValueError("Qwen3.5 GDN replay requires its original trace")
        if config.hidden_act not in {"silu", "swish"}:
            raise NotImplementedError("GDN replay requires SiLU convolution")
        key = (
            config.linear_conv_kernel_dim,
            replay.input_ids.device,
            replay.gdn_segment_size,
            replay.gdn_read_mode,
            replay.gdn_cache_reused,
            replay.gdn_read_backend,
        )
        if key not in replay._gdn:
            replay._gdn[key] = GDNReplay.from_trace(
                replay.trace,
                replay.order,
                device=key[1],
                conv_kernel_size=key[0],
                segment_size=key[2],
                read_mode=key[3],
                cache_reused=key[4],
                read_backend=key[5],
            )
        kwargs["gdn_replay"] = replay._gdn[key]
    for layer in backbone.layers:
        for name, wrapper in (("self_attn", replay_self_attention), ("linear_attn", replay_gdn)):
            module = getattr(layer, name, None)
            if module is not None and not hasattr(module, "_replay_native_forward"):
                module._replay_native_forward = module.forward
                module.forward = MethodType(wrapper, module)
    indices = replay.order.argsort()
    if rows is not None:
        indices = indices[rows]
    if not hidden_only:
        kwargs["logits_to_keep"] = indices
    output = (backbone if hidden_only else model)(
        input_ids=replay.input_ids[None],
        # Identity native RoPE; the attention backend applies block-relative rotations.
        position_ids=torch.zeros_like(replay.input_ids)[None],
        attention_mask={"full_attention": None, "linear_attention": None},
        use_cache=False,
        replay=replay,
        inv_freq=rotary.inv_freq,
        return_dict=True,
        **kwargs,
    )
    if hidden_only:
        return output.last_hidden_state[0, indices]
    return output.logits[0]


def hidden(model, replay: Replay, rows=None):
    """Native decoder hidden states, optionally selecting rows in original trace order."""
    return _forward(model, replay, rows, hidden_only=True)


def forward(model, replay: Replay, rows=None):
    """Native HF Qwen3/Qwen3.5 logits, optionally selecting original trace rows."""
    return _forward(model, replay, rows, hidden_only=False)
