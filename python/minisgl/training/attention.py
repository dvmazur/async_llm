from copy import copy

import torch
from torch.utils.checkpoint import checkpoint
from transformers import AttentionInterface

from .mrope import rotate


def attention_chunk(q, k, v, inv_freq, *, replay, blocks, entries, scaling, mrope_section=None):
    groups = q.shape[1] // k.shape[1]
    sources, positions, rope_positions = [], [], []
    for version, lo, hi, _ in blocks:
        positions.append(torch.arange(lo, hi, device=q.device))
        if replay.rope_positions is None:
            for start, end, _ in replay.source_spans(version, lo, hi):
                sources.append(torch.arange(start, end, device=q.device))
        else:
            for start, end, _, shift in replay.source_spans(version, lo, hi, with_rope=True):
                sources.append(torch.arange(start, end, device=q.device))
                rope_positions.append(replay.rope_positions[start:end] + shift)
    sources, positions = torch.cat(sources), torch.cat(positions)
    key_sizes = [hi - lo for _, lo, hi, _ in blocks]
    row_sizes = [count for _, _, _, count in blocks]
    query_positions = entries[:, 3:] if rope_positions else entries[:, 3]
    key_positions = torch.cat(rope_positions) if rope_positions else positions
    queries = rotate(q[entries[:, 0]], query_positions, inv_freq, mrope_section).split(row_sizes)
    keys = rotate(k[sources], key_positions, inv_freq, mrope_section).split(key_sizes)
    values = v[sources].transpose(0, 1)
    scores = []
    for query, key, position, entry in zip(
        queries, keys, positions.split(key_sizes), entries.split(row_sizes), strict=True
    ):
        rows, starts, ends = entry[:, :3].unbind(1)
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


def attention(q, k, v, replay, inv_freq, scaling, mrope_section=None):
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
                    position = (position,) if isinstance(position, int) else position
                    grouped.setdefault((version, occurrence), []).append(
                        (row, left, right, *position)
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
        kwargs = {
            "replay": replay,
            "blocks": blocks,
            "entries": entries,
            "scaling": scaling,
            "mrope_section": mrope_section,
        }
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
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    replay=None,
    inv_freq=None,
    mrope_section=None,
    **kwargs,
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
        mrope_section,
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
