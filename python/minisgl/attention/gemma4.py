"""Eager paged Gemma: compact page tables and reusable FlashInfer plans.

KV uses block-local RoPE. The ordinary path never gathers cached K/V.
Decode has main-page/edge batches; prefill uses causal self, image self and
historical-context batches. Only window-boundary context needs a packed mask,
bounded by the window, not by the complete context length.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import groupby
from typing import NamedTuple

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import init_logger
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextRotaryEmbedding,
    apply_rotary_pos_emb,
    get_block_sequence_ids_for_mask,
)

from .base import BaseAttnBackend, BaseAttnMetadata

logger = init_logger(__name__)


class Segment(NamedTuple):
    indices: torch.Tensor  # physical page numbers; page_size=1 permits arbitrary debug slots
    origin: int
    local_start: int = 0
    page_size: int = 1
    length: int | None = None
    first_offset: int = 0

    @property
    def num_tokens(self):
        return self.indices.numel() if self.length is None else self.length

    @property
    def slots(self):
        # Inspection/reference only. Production planning works on page numbers.
        p = self.page_size
        slots = (self.indices[:, None] * p + torch.arange(p, device=self.indices.device)).flatten()
        return slots[self.first_offset : self.first_offset + self.num_tokens]


class GemmaView(NamedTuple):
    start: int
    count: int
    query_start: int
    segments: list[Segment]

    @property
    def num_tokens(self):
        return sum(s.num_tokens for s in self.segments)


@dataclass
class GemmaMetadata(BaseAttnMetadata):
    views: list[GemmaView]
    decode: bool = False
    plans: dict = field(default_factory=dict)
    image_spans: list | None = None
    image_indices: torch.Tensor | None = None
    placeholder_indices: torch.Tensor | None = None
    last_indices: torch.Tensor | None = None

    def get_last_indices(self, bs):
        if self.last_indices is None:
            self.last_indices = torch.tensor(
                [v.start + v.count - 1 for v in self.views],
                device=get_global_ctx().kv_cache.device,
                dtype=torch.int64,
            )
        return self.last_indices[:bs]


def attention_ranges(q_start, q_stop, k_start, k_stop, *, causal, window):
    """Exact causal/window intersections; retained as the geometry primitive."""
    if causal:
        k_stop = min(k_stop, q_stop)
        q_start = max(q_start, k_start)
    if window is not None:
        k_start = max(k_start, q_start - window + 1)
        q_stop = min(q_stop, k_stop + window - 1)
    if q_start >= q_stop or k_start >= k_stop:
        return
    cuts = [q_start, q_stop]
    if causal and q_start < k_stop < q_stop:
        cuts.insert(1, k_stop)
    for a, b in zip(cuts, cuts[1:]):
        left = -1 if window is None else window - 1 + k_stop - b
        yield a, b, k_start, k_stop, causal and b <= k_stop, left


@dataclass
class AttentionPart:
    row: int
    count: int
    position: int
    indices: torch.Tensor
    kv_len: int
    page_size: int
    destination: int
    query_global: int
    key_global: int


@dataclass(eq=False)
class AttentionPlan:
    parts: list[AttentionPart]
    decode: bool
    causal: bool
    window_left: int
    custom_window: int | None
    rows: torch.Tensor
    positions: torch.Tensor
    destinations: torch.Tensor
    indices: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    last_page_len: torch.Tensor
    seq_lens: torch.Tensor
    custom_mask: torch.Tensor | None
    slot: object = None

    @property
    def page_size(self):
        return self.parts[0].page_size


class WrapperSlot:
    """A reusable wrapper; rebind old metadata safely after another batch."""

    def __init__(self, wrapper, workspace):
        self.wrapper = wrapper
        self.owner = None
        self.stream_id = None
        self.int_workspace = torch.empty(8 * 1024**2, dtype=torch.uint8, device=workspace.device)
        wrapper.reset_workspace_buffer(workspace, self.int_workspace)
        self.copy_done = torch.cuda.Event()

    def bind(self, plan, heads, kv_heads, dim, dtype):
        stream = torch.cuda.current_stream(plan.indices.device)
        if self.owner is plan and self.stream_id == stream.cuda_stream:
            return
        self.int_workspace.record_stream(stream)
        if self.owner is not None:
            # FI's pinned staging buffer must not be rewritten during its H2D.
            # Wait only for the prior PLAN copy, not the prior model forward.
            self.copy_done.synchronize()
        common = {
            "num_qo_heads": heads,
            "num_kv_heads": kv_heads,
            "page_size": plan.page_size,
            "q_data_type": dtype,
            "kv_data_type": dtype,
            "sm_scale": 1.0,
            "seq_lens": plan.seq_lens,
            "non_blocking": True,
            "disable_split_kv": True,
        }
        if plan.decode:
            self.wrapper.plan(
                indptr=plan.kv_indptr,
                indices=plan.indices,
                last_page_len=plan.last_page_len,
                head_dim=dim,
                **common,
            )
        else:
            # FI 0.6.17's public custom-mask packer requires GPU indptr. Do not
            # use packed_custom_mask: that path leaves bit offsets unconverted
            # to bytes for batched masks. The plan (including this readback)
            # is reused by every layer of the same type.
            device = plan.indices.device if plan.custom_mask is not None else "cpu"
            self.wrapper.plan(
                qo_indptr=plan.qo_indptr.to(device),
                paged_kv_indptr=plan.kv_indptr.to(device),
                paged_kv_indices=plan.indices,
                paged_kv_last_page_len=plan.last_page_len.to(device),
                head_dim_qk=dim,
                causal=plan.causal,
                window_left=plan.window_left,
                custom_mask=plan.custom_mask,
                **common,
            )
        self.copy_done.record()
        self.owner = plan
        self.stream_id = stream.cuda_stream


class Gemma4AttentionBackend(BaseAttnBackend):
    def __init__(self, config):
        self.kv_cache = get_global_ctx().kv_cache
        self.device, self.page_size = self.kv_cache.device, get_global_ctx().page_size
        self.hf_config = config.hf_config
        self.text = getattr(config.hf_config, "text_config", config.hf_config)
        if any(
            p["rope_type"] not in ("default", "proportional")
            for p in self.text.rope_parameters.values()
            if p is not None
        ):
            raise ValueError("Gemma paged shared views require default/proportional RoPE")
        self.rotary = Gemma4TextRotaryEmbedding(self.text, device=self.device)
        self.workspace, self.wrappers = None, {}
        self._execution_depth, self._stream_id = 0, None
        self._finished = torch.cuda.Event()
        self._warned_gpu_inputs = False
        if self.kv_cache.dtype == torch.float32:
            logger.warning(
                "Gemma FP32 reference uses bounded Torch attention; FI requires FP16/BF16"
            )

    @contextmanager
    def execution(self):
        """Order pooled buffers across streams without waiting for GPU on CPU."""
        stream = torch.cuda.current_stream(self.device)
        if self._execution_depth:
            if stream.cuda_stream != self._stream_id:
                raise RuntimeError("Cannot switch CUDA streams within a Gemma forward")
            yield
            return
        if self._stream_id is not None and self._stream_id != stream.cuda_stream:
            stream.wait_event(self._finished)
        self._stream_id, self._execution_depth = stream.cuda_stream, 1
        try:
            yield
        finally:
            self._finished.record(stream)
            self._execution_depth = 0

    def _view(self, start, count, segments, query_start=None):
        n = sum(s.num_tokens for s in segments)
        return GemmaView(start, count, n - count if query_start is None else query_start, segments)

    def _block_segment(self, block, origin, extra_page=None, written=False):
        starts = block.page_starts + ([] if extra_page is None else [extra_page])
        pages = torch.tensor(starts, dtype=torch.int32, device="cpu") // self.page_size
        return Segment(
            pages, origin, page_size=self.page_size, length=block.num_tokens + int(written)
        )

    def prepare_metadata(self, batch):
        views, start = [], 0
        for req in batch.reqs:
            count = req.device_len - req.cached_len
            pages = (
                get_global_ctx().page_table[req.table_idx, : req.device_len : self.page_size]
                // self.page_size
            )
            views.append(
                self._view(
                    start,
                    count,
                    [Segment(pages, 0, page_size=self.page_size, length=req.device_len)],
                )
            )
            start += count
        batch.attn_metadata = GemmaMetadata(views, decode=batch.is_decode)
        if batch.mm_token_type_ids is None and all(
            hasattr(r, "input_ids") and r.input_ids.is_cpu for r in batch.reqs
        ):
            ids = torch.cat([r.input_ids[r.cached_len : r.device_len] for r in batch.reqs])
            self.prepare_inputs(batch, ids)

    def prepare(self, group, new_page_for_block, new_token_slots, graph_buffers=None):
        assert graph_buffers is None
        written = {id(b) for b in group.write_to}
        segments, views = {}, []
        for wi, context in enumerate(group):
            parts, offset = [], 0
            for block in context.cache_view:
                if id(block) not in segments:
                    segments[id(block)] = self._block_segment(
                        block, 0, new_page_for_block.get(id(block)), id(block) in written
                    )
                s = segments[id(block)]._replace(origin=offset)
                parts.append(s)
                offset += s.num_tokens
            qpos = offset - 1
            if not any(b is context.output_block for b in context.cache_view):
                block = context.output_block
                new_page = new_page_for_block.get(id(block))
                page = new_page if new_page is not None else block.page_starts[-1]
                parts.append(
                    Segment(
                        torch.tensor([page // self.page_size], dtype=torch.int32),
                        qpos,
                        block.num_tokens,
                        self.page_size,
                        1,
                        block.num_tokens % self.page_size,
                    )
                )
            views.append(self._view(wi, 1, parts, qpos))
        return GemmaMetadata(views, decode=True)

    def prepare_prefill_batch(self, specs, graph_buffers=None):
        assert graph_buffers is None
        views, start, cached = [], 0, {}
        for spec in specs:
            parts, offset = [], 0
            for block in spec.context:
                if id(block) not in cached:
                    cached[id(block)] = self._block_segment(block, 0)
                s = cached[id(block)]._replace(origin=offset)
                parts.append(s)
                offset += s.num_tokens
            pages = spec.self_page_starts // self.page_size
            parts.append(
                Segment(
                    pages,
                    offset,
                    page_size=self.page_size,
                    length=spec.self_prefix_len + spec.num_new,
                )
            )
            views.append(self._view(start, spec.num_new, parts))
            start += spec.num_new
        return GemmaMetadata(views)

    def prepare_inputs(self, batch, ids_cpu=None, types_cpu=None):
        """Normal session/Req callers retain CPU metadata instead of GPU round trips."""
        meta = batch.attn_metadata
        if types_cpu is not None:
            assert types_cpu.is_cpu
            types_cpu = types_cpu.flatten()
            if types_cpu.numel() != sum(v.count for v in meta.views):
                raise ValueError("Gemma token types and query rows differ")
        meta.image_spans = []
        for view in meta.views:
            groups = (
                get_block_sequence_ids_for_mask(
                    types_cpu[view.start : view.start + view.count][None], torch.device("cpu")
                )[0].tolist()
                if types_cpu is not None
                else [-1] * view.count
            )
            spans, offset = [], 0
            for group, members in groupby(groups):
                length = sum(1 for _ in members)
                spans.append((offset, offset + length, group < 0))
                offset += length
            meta.image_spans.append(spans)
        if ids_cpu is not None:
            assert ids_cpu.is_cpu
            ids_cpu = ids_cpu.flatten()
            image = ids_cpu == getattr(self.hf_config, "image_token_id", -1)
            placeholders = image.clone()
            for name in ("video_token_id", "audio_token_id"):
                token = getattr(self.hf_config, name, None)
                if token is not None:
                    placeholders |= ids_cpu == token
            meta.image_indices = image.nonzero().flatten().to(self.device, non_blocking=True)
            meta.placeholder_indices = (
                placeholders.nonzero().flatten().to(self.device, non_blocking=True)
            )

    def ensure_inputs(self, batch, need_images=False):
        meta = batch.attn_metadata
        if meta.image_spans is not None and (not need_images or meta.image_indices is not None):
            return
        types = batch.mm_token_type_ids
        if types is not None or need_images:
            # Compatibility for manual low-level Batch callers that supply ONLY
            # GPU inputs. Normal session/Req paths prepare these CPU values once.
            if not self._warned_gpu_inputs:
                logger.warning(
                    "Gemma low-level GPU-only metadata needs a host preparation; "
                    "use prepare_inputs with CPU IDs/types to avoid this transfer"
                )
                self._warned_gpu_inputs = True
            self.prepare_inputs(
                batch,
                batch.input_ids.detach().cpu() if need_images else None,
                types.detach().cpu() if types is not None else None,
            )
        else:
            self.prepare_inputs(batch)

    def _rope(self, x, positions, layer_type):
        cos, sin = self.rotary(x[None], positions[None], layer_type=layer_type)
        return apply_rotary_pos_emb(x[None], cos, sin, unsqueeze_dim=2)[0]

    def shift_keys(self, slots, position_shift):
        positions = torch.full_like(slots, position_shift)
        for layer_id, kind in enumerate(self.text.layer_types):
            pool = self.kv_cache.k_cache(layer_id).flatten(0, 1)
            pool.index_copy_(
                0, slots, self._rope(pool[slots].float(), positions, kind).to(pool.dtype)
            )

    def _part(self, view, segment, source, a, b, ka, kb, *, exact_start=False):
        """Use real pages; an exact unaligned start becomes a short page=1 edge."""
        p = segment.page_size
        lo, hi = (
            segment.first_offset + ka - segment.origin,
            segment.first_offset + kb - segment.origin,
        )
        if exact_start and lo % p:
            edge_end = min(hi, (lo // p + 1) * p)
            indices = segment.indices[lo // p : lo // p + 1] * p
            indices = indices + torch.arange(lo % p, lo % p + edge_end - lo, device=indices.device)
            yield AttentionPart(
                view.start + a - view.query_start,
                b - a,
                a - segment.origin + segment.local_start,
                indices,
                edge_end - lo,
                1,
                source,
                a,
                ka,
            )
            if edge_end == hi:
                return
            ka += edge_end - lo
            lo = edge_end
            source += 1
        base = lo // p * p
        indices = segment.indices[lo // p : (hi + p - 1) // p]
        yield AttentionPart(
            view.start + a - view.query_start,
            b - a,
            a - segment.origin + segment.local_start,
            indices,
            hi - base,
            p,
            source,
            a,
            segment.origin + base - segment.first_offset,
        )

    def _make_group(self, parts, decode, causal, left, custom_window, max_parts):
        def upload(t):
            return t.to(self.device, non_blocking=True)

        rows_cpu = torch.cat([torch.arange(p.row, p.row + p.count) for p in parts])
        positions_cpu = torch.cat([torch.arange(p.position, p.position + p.count) for p in parts])
        dest_cpu = torch.cat(
            [torch.arange(p.row, p.row + p.count) * max_parts + p.destination for p in parts]
        )
        index_parts = [p.indices.to(torch.int32) for p in parts]
        indices = (
            upload(torch.cat(index_parts))
            if all(t.is_cpu for t in index_parts)
            else torch.cat([upload(t) for t in index_parts])
        )
        qo = torch.tensor([0] + [p.count for p in parts], dtype=torch.int32).cumsum(0).int()
        kv = (
            torch.tensor([0] + [p.indices.numel() for p in parts], dtype=torch.int32)
            .cumsum(0)
            .int()
        )
        lens = torch.tensor([p.kv_len for p in parts], dtype=torch.int32)
        last = (lens - 1) % parts[0].page_size + 1
        custom_mask = None
        if custom_window is not None:
            masks = []
            for p in parts:
                # A historical block ends before the NEW query range. Both
                # dimensions are bounded by W (+one alignment page), not T.
                assert p.count < custom_window and p.kv_len < custom_window + p.page_size
                qp = torch.arange(p.query_global, p.query_global + p.count, device=self.device)
                kp = torch.arange(p.key_global, p.key_global + p.kv_len, device=self.device)
                mask = (kp[None] > qp[:, None] - custom_window).flatten()
                masks.append(mask)
            custom_mask = torch.cat(masks)
        return AttentionPlan(
            parts,
            decode,
            causal,
            left,
            custom_window,
            upload(rows_cpu),
            upload(positions_cpu),
            upload(dest_cpu),
            indices,
            qo,
            kv,
            last,
            lens,
            custom_mask,
        )

    def _build_plans(self, meta, batch, layer):
        self.ensure_inputs(batch)
        window = layer.sliding_window if layer.is_sliding else None
        max_parts = max(len(v.segments) for v in meta.views) * (2 if meta.decode else 1)
        buckets = {}

        def add(parts, decode, causal=False, left=-1, custom=None):
            for p in parts:
                key = (decode, p.page_size, causal, left, custom)
                buckets.setdefault(key, []).append(p)

        for vi, view in enumerate(meta.views):
            for source, segment in enumerate(view.segments):
                start, stop = segment.origin, segment.origin + segment.num_tokens
                if not segment.num_tokens:
                    continue
                if meta.decode:
                    assert view.count == 1
                    qa = view.query_start
                    ka = max(start, qa - window + 1) if window is not None else start
                    kb = min(stop, qa + 1)
                    if ka < kb:
                        add(
                            self._part(
                                view, segment, 2 * source, qa, qa + 1, ka, kb, exact_start=True
                            ),
                            True,
                        )
                    continue
                if stop <= view.query_start:
                    # All historical blocks share one call, even when their
                    # window offsets differ. No per-offset wrapper/bucket.
                    for a, b, ka, kb, _, _ in attention_ranges(
                        view.query_start,
                        view.query_start + view.count,
                        start,
                        stop,
                        causal=False,
                        window=window,
                    ):
                        add(self._part(view, segment, source, a, b, ka, kb), False, custom=window)
                    continue
                spans = (
                    meta.image_spans[vi]
                    if layer.is_sliding and self.text.use_bidirectional_attention == "vision"
                    else [(0, view.count, True)]
                )
                for lo, hi, causal in spans:
                    a, b = view.query_start + lo, view.query_start + hi
                    # This is the self block. Aligning KV's end with this query
                    # span preserves FI's native causal/window coordinates.
                    ka = max(start, a - window + 1) if window is not None else start
                    kb = min(stop, b)
                    if ka < kb:
                        add(
                            self._part(view, segment, source, a, b, ka, kb),
                            False,
                            causal=causal,
                            left=window - 1 if window is not None else -1,
                        )
        plans = []
        for (decode, p, causal, left, custom), parts in buckets.items():
            plan = self._make_group(parts, decode, causal, left, custom, max_parts)
            if self.kv_cache.dtype != torch.float32:
                from flashinfer import (
                    BatchDecodeWithPagedKVCacheWrapper,
                    BatchPrefillWithPagedKVCacheWrapper,
                )

                key = (layer.layer_type, decode, p, causal, left, custom)
                if key not in self.wrappers:
                    if self.workspace is None:
                        self.workspace = torch.empty(
                            128 * 1024**2, dtype=torch.uint8, device=self.device
                        )
                    wrapper = (
                        BatchDecodeWithPagedKVCacheWrapper(
                            self.workspace,
                            kv_layout="NHD",
                            # Keep the same arithmetic family as prefill. The
                            # Qwen GQA>=4 heuristic switched Gemma's GQA2 layers
                            # to CUDA cores and regressed two learned L2 gates.
                            use_tensor_cores=True,
                            backend="fa2",
                        )
                        if decode
                        else BatchPrefillWithPagedKVCacheWrapper(
                            self.workspace, kv_layout="NHD", backend="fa2"
                        )
                    )
                    self.wrappers[key] = WrapperSlot(wrapper, self.workspace)
                plan.slot = self.wrappers[key]
            plans.append(plan)
        return plans, max_parts

    def _reference(self, query, plan, k_pool, v_pool):
        """FP32 tests only; bounded query tiles and the SAME visibility."""
        outs, lses, offset = [], [], 0
        for part in plan.parts:
            p = part.page_size
            indices = part.indices.to(self.device)
            slots = (indices[:, None] * p + torch.arange(p, device=self.device)).flatten()[
                : part.kv_len
            ]
            key, value = k_pool.flatten(0, 1)[slots.long()], v_pool.flatten(0, 1)[slots.long()]
            repeats = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeats, 1).transpose(0, 1)
            value = value.repeat_interleave(repeats, 1).transpose(0, 1)
            for lo in range(0, part.count, 64):
                hi = min(lo + 64, part.count)
                qp = torch.arange(lo, hi, device=self.device) + part.kv_len - part.count
                kp = torch.arange(part.kv_len, device=self.device)
                allowed = torch.ones((hi - lo, part.kv_len), dtype=torch.bool, device=self.device)
                if plan.causal:
                    allowed &= kp[None] <= qp[:, None]
                if plan.window_left >= 0:
                    allowed &= kp[None] >= qp[:, None] - plan.window_left
                if plan.custom_window is not None:
                    allowed &= kp[None] + part.key_global > (
                        torch.arange(lo, hi, device=self.device)[:, None]
                        + part.query_global
                        - plan.custom_window
                    )
                scores = query[offset + lo : offset + hi].transpose(0, 1) @ key.transpose(1, 2)
                scores.masked_fill_(~allowed, -torch.inf)
                outs.append((scores.softmax(-1) @ value).transpose(0, 1))
                lses.append(scores.logsumexp(-1).transpose(0, 1))
            offset += part.count
        return torch.cat(outs), torch.cat(lses)

    def forward(self, q, k, v, layer_id, batch, *, layer):
        if not self._execution_depth:
            with self.execution():
                return self.forward(q, k, v, layer_id, batch, layer=layer)
        meta = batch.attn_metadata
        if layer.layer_type not in meta.plans:
            meta.plans[layer.layer_type] = self._build_plans(meta, batch, layer)
        plans, max_parts = meta.plans[layer.layer_type]
        stream = torch.cuda.current_stream(self.device)
        if self.workspace is not None:
            self.workspace.record_stream(stream)
        key = self._rope(k, batch.positions, layer.layer_type)
        self.kv_cache.store_kv(
            key.flatten(1).contiguous(), v.flatten(1).contiguous(), batch.out_loc, layer_id
        )
        kv = tuple(pool(layer_id) for pool in (self.kv_cache.k_cache, self.kv_cache.v_cache))
        for tensor in kv:
            tensor.record_stream(stream)
        values = q.new_zeros(q.shape[0] * max_parts, *q.shape[1:])
        scores = torch.full(values.shape[:2], -torch.inf, device=self.device, dtype=torch.float32)
        for plan in plans:
            # A retained batch may be replayed on another stream. Keep its
            # borrowed GPU tables alive until that stream finishes reading.
            for tensor in (
                plan.rows,
                plan.positions,
                plan.destinations,
                plan.indices,
                plan.custom_mask,
            ):
                if tensor is not None:
                    tensor.record_stream(stream)
            query = self._rope(q[plan.rows], plan.positions, layer.layer_type)
            if plan.slot is None:
                out, lse = self._reference(query, plan, *kv)
            else:
                plan.slot.bind(plan, q.shape[1], k.shape[1], k.shape[2], q.dtype)
                cache = (
                    kv
                    if plan.page_size == self.page_size
                    else tuple(t.flatten(0, 1).unsqueeze(1) for t in kv)
                )
                out, lse = plan.slot.wrapper.run(query, cache, return_lse=True)
            values[plan.destinations], scores[plan.destinations] = out, lse
        values = values.view(q.shape[0], max_parts, *q.shape[1:])
        scores = scores.view(q.shape[0], max_parts, q.shape[1])
        if max_parts == 1:
            return values[:, 0]
        if q.dtype == torch.float32:
            return (values * scores.softmax(1)[..., None]).sum(1)
        from flashinfer import merge_states

        return merge_states(values, scores)[0]

    def init_capture_graph(self, max_seq_len, bs_list):
        if bs_list:
            raise ValueError("Gemma 4 currently supports eager execution only")

    def prepare_for_capture(self, batch):
        raise ValueError("Gemma 4 currently supports eager execution only")

    prepare_for_replay = prepare_for_capture
