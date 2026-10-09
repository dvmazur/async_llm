from dataclasses import dataclass, field, replace
from itertools import groupby

import torch
from torch.utils.checkpoint import checkpoint
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import eager_mask
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

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

    def to(self, device):
        return replace(
            self,
            input_ids=self.input_ids.to(device),
            order=self.order.to(device),
            _attention={},
        )

    @classmethod
    def from_trace(
        cls,
        trace: Trace,
        device="cpu",
        query_chunk_size=128,
    ):
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if not trace.tokens:
            raise ValueError("Cannot replay an empty trace")
        tokens = trace.tokens
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
                view = dict(token.view)
                if len(view) != len(token.view):
                    raise ValueError("Repeated blocks in a view are not supported")
                if token.block in view and view[token.block] <= token.position:
                    raise ValueError("A token must attend to itself")
                if token.kind == "prefill" and view.get(token.block) != token.position + 1:
                    raise ValueError("Prefill must include its own causal prefix")
                entries, prefix = [], 0
                total = sum(view.values())
                for block, length in token.view:
                    if length < 0:
                        raise ValueError("View length must be nonnegative")
                    if length == 0:
                        continue
                    if block not in current or length > lengths[current[block]]:
                        raise ValueError("View exposes future or unrecorded tokens")
                    entries.append((current[block], 0, length, total - 1 - prefix))
                    prefix += length
                if token.block not in view:
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
                for version, left, right, position in reads:
                    grouped.setdefault(version, []).append((row, left, right, position))
            blocks, entries = [], []
            for version, reads in grouped.items():
                blocks.append(
                    (version, min(r[1] for r in reads), max(r[2] for r in reads), len(reads))
                )
                entries.extend(reads)
            chunks.append((blocks, torch.tensor(entries, device=q.device)))
        replay._attention[cache_key] = chunks
    outputs = []
    for query, (blocks, entries) in zip(
        q.split(replay.query_chunk_size), replay._attention[cache_key], strict=True
    ):
        kwargs = {"replay": replay, "blocks": blocks, "entries": entries, "scaling": scaling}
        if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
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
    if replay is None:
        return eager_attention_forward(
            module, query, key, value, attention_mask, scaling=scaling, **kwargs
        )
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
AttentionMaskInterface.register("minisgl_replay", eager_mask)


def _forward(model, replay: Replay, rows, hidden_only):
    if model.config.model_type != "qwen3":
        raise ValueError("Only dense text Qwen3 is supported")
    backbone = model.model
    config = backbone.config
    rotary = backbone.rotary_emb
    if getattr(rotary, "rope_type", "default") != "default":
        raise NotImplementedError("Only default RoPE is supported")
    if getattr(backbone, "has_sliding_layers", False) or config.attention_dropout:
        raise NotImplementedError("Replay requires full attention and zero attention dropout")
    kwargs = {}
    backbone.set_attn_implementation("minisgl_replay")
    indices = replay.order.argsort()
    if rows is not None:
        indices = indices[rows]
    if not hidden_only:
        kwargs["logits_to_keep"] = indices
    output = (backbone if hidden_only else model)(
        input_ids=replay.input_ids[None],
        # Identity native RoPE; the attention backend applies block-relative rotations.
        position_ids=torch.zeros_like(replay.input_ids)[None],
        attention_mask={"full_attention": None},
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
    """Native HF Qwen3 logits, optionally selecting original trace rows."""
    return _forward(model, replay, rows, hidden_only=False)
