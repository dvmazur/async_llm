from dataclasses import dataclass
from itertools import groupby

import torch
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import eager_mask
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

from .trace import Trace


@dataclass
class Replay:
    input_ids: torch.Tensor
    positions: torch.Tensor
    query_positions: torch.Tensor
    mask: torch.Tensor
    spans: list[tuple[int, int]]
    order: torch.Tensor
    sources: torch.Tensor

    @classmethod
    def from_trace(cls, trace: Trace, device="cpu"):
        if not trace.tokens:
            raise ValueError("Cannot replay an empty trace")
        tokens = trace.tokens
        if any(a.step > b.step for a, b in zip(tokens, tokens[1:])) or any(
            a.step > b.step for a, b in zip(trace.updates, trace.updates[1:])
        ):
            raise ValueError("Trace must be in execution order")
        versions, current, rows = [], {}, []

        def replace(block, sources=()):
            current[block] = len(versions)
            versions.append((block, list(sources)))

        update_index = 0
        for step, group in groupby(enumerate(tokens), key=lambda item: item[1].step):
            group = list(group)
            while update_index < len(trace.updates) and trace.updates[update_index].step <= step:
                update = trace.updates[update_index]
                if any(
                    i < 0 or i >= group[0][0] or tokens[i].step >= update.step
                    for i in update.sources
                ):
                    raise ValueError("Block update refers to future or unrecorded tokens")
                replace(update.block, update.sources)
                update_index += 1
            for row, token in group:
                if token.block not in current:
                    replace(token.block)
                sources = versions[current[token.block]][1]
                if token.position != len(sources):
                    raise ValueError("Writes must append to a contiguous block span")
                sources.append(row)
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
                    if length == 0:
                        continue
                    if block not in current or not 0 <= length <= len(versions[current[block]][1]):
                        raise ValueError("View exposes future or unrecorded tokens")
                    entries.append((current[block], 0, length, total - 1 - prefix))
                    prefix += length
                if token.block not in view:
                    entries.append(
                        (current[token.block], token.position, token.position + 1, token.position)
                    )
                rows.append(entries)

        order = sorted(
            range(len(tokens)), key=lambda i: (tokens[i].block, tokens[i].step, tokens[i].position)
        )
        inverse = {row: i for i, row in enumerate(order)}
        version_order = sorted(range(len(versions)), key=lambda i: (versions[i][0], i))
        spans, sources, positions = [], [], []
        for version in version_order:
            refs = versions[version][1]
            spans.append((len(sources), len(sources) + len(refs)))
            sources.extend(inverse[i] for i in refs)
            positions.extend(range(len(refs)))
        columns = {version: i for i, version in enumerate(version_order)}
        mask = torch.zeros(len(tokens), len(sources), dtype=torch.bool)
        query_positions = torch.zeros(len(tokens), len(versions), dtype=torch.long)
        for row, original in enumerate(order):
            for version, start, end, position in rows[original]:
                column = columns[version]
                offset = spans[column][0]
                mask[row, offset + start : offset + end] = True
                query_positions[row, column] = position
        return cls(
            torch.tensor([tokens[i].token_id for i in order], device=device),
            torch.tensor(positions, device=device),
            query_positions.to(device),
            mask.to(device),
            spans,
            torch.tensor(order, device=device),
            torch.tensor(sources, device=device),
        )


def rotate(x, positions, inv_freq):
    angles = positions.float()[:, None] * inv_freq.float()[None, :]
    cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
    left, right = x.float().chunk(2, dim=-1)
    return torch.cat((left * cos - right * sin, right * cos + left * sin), -1).to(x.dtype)


def attention(q, k, v, replay, inv_freq, scaling):
    k = rotate(k[replay.sources], replay.positions, inv_freq)
    v = v[replay.sources]
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1).transpose(0, 1)
    v = v.repeat_interleave(groups, dim=1).transpose(0, 1)
    scores = []
    for block, (start, end) in enumerate(replay.spans):
        rotated = rotate(q, replay.query_positions[:, block], inv_freq).transpose(0, 1)
        scores.append(rotated @ k[:, start:end].transpose(-1, -2))
    scores = torch.cat(scores, dim=-1).float() * scaling
    scores = scores.masked_fill(~replay.mask, float("-inf"))
    weights = scores.softmax(-1).to(v.dtype)
    return (weights @ v).transpose(0, 1).flatten(1)


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


def forward(model, replay: Replay):
    """Native HF Qwen3 forward with replay attention, returning logits in trace order."""
    if model.config.model_type != "qwen3":
        raise ValueError("Only dense text Qwen3 is supported")
    backbone = model.model
    rotary = backbone.rotary_emb
    if getattr(rotary, "rope_type", "default") != "default":
        raise NotImplementedError("Only default Qwen3 RoPE is supported")
    if getattr(backbone, "has_sliding_layers", False) or model.config.attention_dropout:
        raise NotImplementedError("Replay requires full attention and zero attention dropout")
    model.set_attn_implementation("minisgl_replay")
    logits = model(
        input_ids=replay.input_ids[None],
        # Identity native RoPE; the attention backend applies block-relative rotations.
        position_ids=torch.zeros_like(replay.input_ids)[None],
        attention_mask={"full_attention": None},
        use_cache=False,
        replay=replay,
        inv_freq=rotary.inv_freq,
        return_dict=True,
    ).logits[0]
    return logits[replay.order.argsort()]
