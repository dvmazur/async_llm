"""Chunked Torch attention with shared, once-per-forward prepared K/V."""

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from minisgl.training.mrope import rotate


@dataclass(frozen=True)
class _ReadGroup:
    start: int
    end: int
    entries: torch.Tensor


@dataclass(frozen=True)
class _QueryChunk:
    entries: torch.Tensor
    groups: tuple[_ReadGroup, ...]


@dataclass(frozen=True)
class _AttentionPlan:
    k_indices: torch.Tensor
    k_positions: torch.Tensor
    k_offsets: torch.Tensor
    chunks: tuple[_QueryChunk, ...]


@torch.inference_mode(False)
def _build_plan(replay, device):
    """Cache indices/positions only; never retain prepared model activations."""
    bounds = {}
    for reads in replay.reads:
        for version, left, right, _ in reads:
            if left < 0 or right <= left:
                raise ValueError("Attention reads must have a nonempty key range")
            lo, hi = bounds.get(version, (left, right))
            bounds[version] = min(lo, left), max(hi, right)

    indices, positions, offsets, spans = [], [], [], {}
    rope = None if replay.rope_positions is None else replay.rope_positions.detach().cpu()
    for version, (left, right) in bounds.items():
        begin = len(indices)
        if rope is None:
            for start, end, _ in replay.compute_source_spans(version, left, right):
                indices.extend(range(start, end))
            positions.extend(range(left, right))
        else:
            for start, end, _, shift in replay.compute_source_spans(
                version, left, right, with_rope=True
            ):
                indices.extend(range(start, end))
                positions.extend((rope[start:end] + shift).tolist())
        if len(indices) - begin != right - left:
            raise ValueError("Attention read does not resolve to its recorded key count")
        offsets.extend(range(left, right))
        spans[version] = begin, left

    chunks = []
    for start in range(0, len(replay.reads), replay.query_chunk_size):
        grouped = {}
        for row, reads in enumerate(replay.reads[start : start + replay.query_chunk_size]):
            occurrences = {}
            for version, left, right, position in reads:
                occurrence = occurrences.get(version, 0)
                occurrences[version] = occurrence + 1
                position = (position,) if isinstance(position, int) else position
                grouped.setdefault((version, occurrence), []).append((row, left, right, *position))
        groups, combined = [], []
        for (version, _), entries in grouped.items():
            lo, hi = min(row[1] for row in entries), max(row[2] for row in entries)
            begin, left = spans[version]
            groups.append((begin + lo - left, begin + hi - left, len(combined), len(entries)))
            combined.extend(entries)
        entries = torch.tensor(combined, dtype=torch.long, device=device)
        chunks.append(
            _QueryChunk(
                entries,
                tuple(
                    _ReadGroup(begin, end, entries[offset : offset + count])
                    for begin, end, offset, count in groups
                ),
            )
        )
    return _AttentionPlan(
        torch.tensor(indices, dtype=torch.long, device=device),
        torch.tensor(positions, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
        tuple(chunks),
    )


def _bmm(left, right):
    """Use the reference precision and ordinary Torch autograd under autocast."""
    with torch.autocast(device_type=left.device.type, enabled=False):
        return torch.bmm(left, right)


def _attention_chunk(q, keys, values, offsets, inv_freq, chunk, scaling, mrope_section):
    heads, width = q.shape[1:]
    kv_heads = values.shape[0]
    head_groups = heads // kv_heads
    entries = chunk.entries
    query_positions = entries[:, 3] if entries.shape[1] == 4 else entries[:, 3:]
    queries = rotate(q[entries[:, 0]], query_positions, inv_freq, mrope_section)
    queries = queries.split([len(group.entries) for group in chunk.groups])
    scores, chunk_values = [], []
    for query, group in zip(queries, chunk.groups, strict=True):
        entries = group.entries
        rows, starts, ends = entries[:, :3].unbind(1)
        query = query.reshape(-1, kv_heads, head_groups, width).permute(1, 2, 0, 3)
        key = keys[:, :, group.start : group.end]
        score = _bmm(query.flatten(1, 2), key).view(
            kv_heads, head_groups, len(rows), group.end - group.start
        )
        score = score.to(torch.promote_types(score.dtype, torch.float32)) * scaling
        position = offsets[group.start : group.end]
        visible = (position >= starts[:, None]) & (position < ends[:, None])
        score = score.masked_fill(~visible, float("-inf"))
        scores.append(
            score.new_full((*score.shape[:2], len(q), score.shape[-1]), float("-inf")).index_copy(
                2, rows, score
            )
        )
        chunk_values.append(values[:, group.start : group.end, :])

    # Every block contributes to the same denominator, including repeated reads.
    weights = torch.cat(scores, -1).softmax(-1).to(values.dtype)
    output = _bmm(weights.flatten(1, 2), torch.cat(chunk_values, dim=1))
    return output.view(kv_heads, head_groups, len(q), width).permute(2, 0, 1, 3).flatten(1)


def attention(q, k, v, replay, inv_freq, scaling, mrope_section=None):
    """Prepare shared K/V once, then evaluate chunked globally normalized attention.

    Precision and autograd match the Torch reference: FP16/BF16 Q/K/V are
    promoted before gathering and RoPE, all attention math uses FP32, and the
    output is cast to V's original dtype. FP64 inputs retain FP64 matrix math.
    """
    if any(x.ndim != 3 for x in (q, k, v)):
        raise ValueError("Attention Q/K/V must have layout [tokens, heads, head_dim]")
    if any(x.device != q.device or x.dtype != q.dtype for x in (k, v)):
        raise ValueError("Attention Q/K/V must have matching devices and dtypes")
    if (
        len(q) != len(k)
        or k.shape != v.shape
        or q.shape[-1] != k.shape[-1]
        or not k.shape[1]
        or q.shape[1] % k.shape[1]
    ):
        raise ValueError("Unsupported grouped-query attention dimensions")
    replay._prepare()
    if len(q) != len(replay.reads):
        raise ValueError("Q/K/V rows must match the prepared trace")
    output_dtype = v.dtype
    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v = (value.to(compute_dtype) for value in (q, k, v))
    cache_key = ("torch_optimized", q.device, replay.query_chunk_size)
    if cache_key not in replay._attention:
        replay._attention[cache_key] = _build_plan(replay, q.device)
    plan = replay._attention[cache_key]
    # Chunk intervals below are basic views into these shared tensors. Historical
    # boundaries do not require either repeated gathering or repeated K RoPE.
    keys = rotate(k[plan.k_indices], plan.k_positions, inv_freq, mrope_section).permute(1, 2, 0)
    values = v[plan.k_indices].transpose(0, 1)
    outputs = []
    for query, chunk in zip(q.split(replay.query_chunk_size), plan.chunks, strict=True):
        args = (query, keys, values, plan.k_offsets, inv_freq, chunk, scaling, mrope_section)
        if (
            replay.attention_checkpointing
            and torch.is_grad_enabled()
            and any(x.requires_grad for x in (q, k, v))
        ):
            output = checkpoint(_attention_chunk, *args, use_reentrant=False)
        else:
            output = _attention_chunk(*args)
        outputs.append(output)
    return torch.cat(outputs).to(output_dtype)
