"""Compile one masked varlen attention sequence per recorded KV-block version.

The plan contains indices and positions only.  Its tensors can be reused across
layers and forwards without retaining activations or model weights.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlashAttentionPlan:
    q_indices: torch.Tensor
    q_positions: torch.Tensor
    q_starts: torch.Tensor
    q_ends: torch.Tensor
    k_indices: torch.Tensor
    k_positions: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    max_q: int
    max_k: int
    merge_indptr: torch.Tensor
    merge_indices: torch.Tensor
    num_queries: int


@torch.inference_mode(False)
def build_flash_attention_plan(replay, device) -> FlashAttentionPlan:
    """Return packed FA tasks and CSR indices for merging their query outputs.

    Query/key indices address the same packed token order as ``replay.reads``.
    Each version's referenced key span is packed once. ``q_starts``/``q_ends``
    describe each read in that sequence's local key coordinates. Repeated reads
    remain separate query rows, including their individual RoPE positions.
    ``merge_indices`` address partial outputs along the packed query axis.
    """
    replay._prepare()
    device = torch.device(device)
    num_queries = len(replay.reads)
    merge_rows = [[] for _ in range(num_queries)]
    rope_positions = None if replay.rope_positions is None else replay.rope_positions.detach().cpu()
    grouped = {}
    for row, reads in enumerate(replay.reads):
        for version, left, right, position in reads:
            if left < 0 or right <= left:
                raise ValueError("Flash attention reads must have a nonempty key range")
            grouped.setdefault(version, []).append((row, left, right, position))

    q_indices, q_positions, q_starts, q_ends = [], [], [], []
    k_indices, k_positions = [], []
    cu_q, cu_k = [0], [0]
    max_q = max_k = 0
    for version, reads in grouped.items():
        left, right = min(read[1] for read in reads), max(read[2] for read in reads)
        for row, start, end, position in reads:
            merge_rows[row].append(len(q_indices))
            q_indices.append(row)
            q_positions.append(position)
            q_starts.append(start - left)
            q_ends.append(end - left)
        key_start = len(k_indices)
        if rope_positions is None:
            for start, end, _ in replay.compute_source_spans(version, left, right):
                k_indices.extend(range(start, end))
            k_positions.extend(range(left, right))
        else:
            for start, end, _, shift in replay.compute_source_spans(
                version, left, right, with_rope=True
            ):
                k_indices.extend(range(start, end))
                k_positions.extend((rope_positions[start:end] + shift).tolist())
        key_count = len(k_indices) - key_start
        if key_count != right - left:
            raise ValueError("A replay read does not resolve to its recorded key count")
        cu_q.append(len(q_indices))
        cu_k.append(len(k_indices))
        max_q = max(max_q, len(reads))
        max_k = max(max_k, key_count)
    if max(cu_q[-1], cu_k[-1]) > torch.iinfo(torch.int32).max:
        raise ValueError("Packed attention lengths exceed the int32 varlen limit")
    merge_indptr, merge_indices = [0], []
    for partials in merge_rows:
        if not partials:
            raise ValueError("Every replay query must have at least one attention read")
        merge_indices.extend(partials)
        merge_indptr.append(len(merge_indices))
    return FlashAttentionPlan(
        q_indices=torch.tensor(q_indices, dtype=torch.long, device=device),
        q_positions=torch.tensor(q_positions, dtype=torch.long, device=device),
        q_starts=torch.tensor(q_starts, dtype=torch.int32, device=device),
        q_ends=torch.tensor(q_ends, dtype=torch.int32, device=device),
        k_indices=torch.tensor(k_indices, dtype=torch.long, device=device),
        k_positions=torch.tensor(k_positions, dtype=torch.long, device=device),
        cu_q=torch.tensor(cu_q, dtype=torch.int32, device=device),
        cu_k=torch.tensor(cu_k, dtype=torch.int32, device=device),
        max_q=max_q,
        max_k=max_k,
        merge_indptr=torch.tensor(merge_indptr, dtype=torch.int32, device=device),
        merge_indices=torch.tensor(merge_indices, dtype=torch.int32, device=device),
        num_queries=num_queries,
    )
