"""Fixed-capacity FlashInfer decode metadata for the existing shared attention.

Real lengths/segment membership change through tables, not graph keys. Each
worker owns D main segment slots and one implicit-self slot. Inactive slots are
masked before the unchanged log-sum-exp merge.
"""

import torch
import torch.nn.functional as F

from .attention_metadata import page_numbers, upload_page_indices


class SharedDecodeAttentionBuffers:
    def __init__(self, attention, workers, depth, max_pages, dummy_slots):
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        from .attention import SharedCacheAttnMetadata

        self.attention, self.workers, self.depth = attention, workers, depth
        self.dummy_slots = dummy_slots
        self.dummy_cpu = dummy_slots.cpu().tolist()
        self.dummy_parts = [[slot // attention.page_size] for slot in self.dummy_cpu]
        self.main_count = workers * depth
        self.max_pages = max_pages
        device = attention.device
        self.main_indptr = torch.zeros(self.main_count + 1, dtype=torch.int32, device=device)
        self.main_indices = torch.zeros(max_pages, dtype=torch.int32, device=device)
        self.main_last = torch.ones(self.main_count, dtype=torch.int32, device=device)
        self.aux_indptr = torch.arange(workers + 1, dtype=torch.int32, device=device)
        self.aux_indices = dummy_slots.to(torch.int32).clone()
        self.aux_last = torch.ones(workers, dtype=torch.int32, device=device)

        def wrapper(indptr, indices, last):
            return BatchDecodeWithPagedKVCacheWrapper(
                attention._workspace, kv_layout='NHD', use_cuda_graph=True,
                use_tensor_cores=attention.num_qo_heads // attention.num_kv_heads >= 4,
                paged_kv_indptr_buffer=indptr, paged_kv_indices_buffer=indices,
                paged_kv_last_page_len_buffer=last, backend='fa2')
        self.main = wrapper(self.main_indptr, self.main_indices, self.main_last)
        self.aux = wrapper(self.aux_indptr, self.aux_indices, self.aux_last)
        w = torch.arange(workers, device=device)
        slots = torch.arange(depth, device=device)
        self.meta = SharedCacheAttnMetadata(
            shared_cache_op=attention,
            sub_worker=torch.cat([w.repeat_interleave(depth), w]),
            sub_loc=torch.zeros(self.main_count + workers, dtype=torch.int64, device=device),
            pad_slot=torch.cat([(w[:, None] * (depth + 1) + slots).flatten(), w * (depth + 1) + depth]),
            num_workers=workers, max_segments=depth + 1,
            n_main=self.main_count, n_aux=workers,
            graph_buffers=self)
        self.valid = torch.zeros(self.main_count + workers, dtype=torch.bool, device=device)
        self.event = torch.cuda.Event()
        self.event.record()
        self.refs = []

    def workspace_requirements(self):
        # FA2 graph-mode bounds use the FIXED slot/query count, dimensions and
        # device occupancy, not the active requests or current KV lengths.
        # Decode has one query per slot, including all padded context slots.
        a, result = self.attention, []
        for wrapper, count, page in ((self.main, self.main_count, a.page_size),
                                     (self.aux, self.workers, 1)):
            size = wrapper.workspace_size(
                indptr=torch.arange(count+1, dtype=torch.int32),
                indices=torch.zeros(count, device=a.device, dtype=torch.int32),
                last_page_len=torch.ones(count, dtype=torch.int32),
                seq_lens=torch.ones(count, dtype=torch.int32),
                num_qo_heads=a.num_qo_heads, num_kv_heads=a.num_kv_heads,
                head_dim=a.head_dim, page_size=page,
                q_data_type=a.dtype, kv_data_type=a.dtype)
            result.append((wrapper, *size))
        return result

    def plan(self, main_workers, main_locs, main_slots, page_parts, page_counts,
             seq_lengths, last_pages, aux_workers, aux_locs, aux_slots):
        a, w, d = self.attention, self.workers, self.depth
        # The slot geometry is fixed even when a worker has zero contexts.
        parts = [self.dummy_parts[i // d] for i in range(self.main_count)]
        counts = [1] * self.main_count
        lengths = [1] * self.main_count
        last = [1] * self.main_count
        loc = [0] * (self.main_count + w)
        valid = [False] * (self.main_count + w)
        for worker, offset, slot, pages, count, length, last_page in zip(
                main_workers, main_locs, main_slots, page_parts, page_counts, seq_lengths, last_pages):
            if slot >= d:
                raise ValueError('Shared attention exceeds graph segment capacity')
            i = worker * d + slot
            parts[i], counts[i], lengths[i], last[i] = pages, count, length, last_page
            loc[i], valid[i] = offset, True
        if sum(counts) > self.max_pages:
            raise ValueError('Shared attention exceeds graph page-index capacity')
        aux_indices = list(self.dummy_cpu)
        for worker, offset, token_slot in zip(aux_workers, aux_locs, aux_slots):
            aux_indices[worker] = token_slot
            loc[self.main_count + worker] = offset
            valid[self.main_count + worker] = True
        self.event.synchronize()  # FlashInfer reuses its pinned planning workspace
        self.refs = []

        def host(values, dtype=torch.int32):
            tensor = torch.tensor(values, dtype=dtype, pin_memory=True)
            self.refs.append(tensor)
            return tensor
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        indices = upload_page_indices(parts, a.device, self.refs)
        aux_gpu_indices = upload_page_indices([aux_indices], a.device, self.refs)
        common = dict(num_qo_heads=a.num_qo_heads, num_kv_heads=a.num_kv_heads,
                      head_dim=a.head_dim, pos_encoding_mode='NONE',
                      data_type=a.dtype, q_data_type=a.dtype, kv_data_type=a.dtype,
                      non_blocking=True)
        self.main.plan(indptr=host(offsets), indices=indices, last_page_len=host(last),
                       seq_lens=host(lengths), page_size=a.page_size, **common)
        self.aux.plan(indptr=host(list(range(w + 1))), indices=aux_gpu_indices,
                      last_page_len=host([1] * w), seq_lens=host([1] * w), page_size=1, **common)
        self.meta.sub_loc.copy_(host(loc, torch.int64), non_blocking=True)
        self.valid.copy_(host(valid, torch.bool), non_blocking=True)
        self.event.record()
        return self.meta


class SharedPrefillAttentionBuffers:
    """Fixed query-row/request capacities; dynamic lengths remain metadata.

    Context slots are request-major; self is one slot/request. Only REAL query
    lengths enter the planner: padding must not alter FlashInfer's split-KV
    work estimate. Capture is primed at capacity; eager calls use real slices.
    All scatter destinations (including padding) form a permutation, so a dummy
    row cannot race with a real row while writing the merge scratch.
    """

    def __init__(self, attention, workers, depth, rows, max_pages, dummy_slots):
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper
        from .attention import SharedCacheAttnMetadata

        self.attention, self.workers, self.depth, self.rows = attention, workers, depth, rows
        self.max_pages, self.ctx_rows = max_pages, rows * depth
        self.use_3d = attention.mrope_section is not None
        self.dummy_parts = [[slot // attention.page_size] for slot in dummy_slots.cpu().tolist()]
        device, total = attention.device, rows * (depth + 1)

        def wrapper(count):
            return BatchPrefillWithPagedKVCacheWrapper(
                attention._workspace, kv_layout='NHD', use_cuda_graph=True, backend='fa2',
                qo_indptr_buf=torch.zeros(count+1, dtype=torch.int32, device=device),
                paged_kv_indptr_buf=torch.zeros(count+1, dtype=torch.int32, device=device),
                paged_kv_indices_buf=torch.zeros(max_pages, dtype=torch.int32, device=device),
                paged_kv_last_page_len_buf=torch.ones(count, dtype=torch.int32, device=device))

        self.main, self.aux = wrapper(workers * depth), wrapper(workers)
        self.used_ctx_rows, self.used_self_rows = self.ctx_rows, rows
        self.meta = SharedCacheAttnMetadata(shared_cache_op=attention,
            sub_worker=torch.zeros(total, dtype=torch.int64, device=device),
            sub_loc=torch.zeros((3, total) if self.use_3d else (total,), dtype=torch.int64, device=device),
            pad_slot=torch.zeros(total, dtype=torch.int64, device=device),
            num_workers=rows, max_segments=depth+1, n_ctx_rows=self.ctx_rows,
            last_indices=torch.zeros(workers, dtype=torch.int64, device=device),
            phase='prefill_batch', graph_buffers=self)
        self.meta.sub_worker[self.ctx_rows:] = torch.arange(rows, device=device)
        self.meta.pad_slot[self.ctx_rows:] = torch.arange(rows, device=device) * (depth+1) + depth
        self.valid = torch.zeros(total, dtype=torch.bool, device=device)
        self._all_slots = torch.arange(total)
        self.event = torch.cuda.Event()
        self.event.record()
        self.refs = []

    def workspace_requirements(self):
        # In FA2's default graph plan, tile/padded-batch bounds depend on the
        # FIXED total row capacity, request count and model/hardware dimensions,
        # not the current lengths. Query the public allocator estimator once.
        a, result = self.attention, []
        for wrapper, count, rows, causal in (
            (self.main, self.workers*self.depth, self.ctx_rows, False),
            (self.aux, self.workers, self.rows, True)):
            qo = torch.zeros(count+1, dtype=torch.int32)
            qo[-1] = rows
            size = wrapper.workspace_size(qo_indptr=qo,
                paged_kv_indptr=torch.arange(count+1, dtype=torch.int32),
                paged_kv_indices=torch.zeros(count, device=a.device, dtype=torch.int32),
                paged_kv_last_page_len=torch.ones(count, dtype=torch.int32),
                seq_lens=torch.ones(count, dtype=torch.int32), causal=causal,
                num_qo_heads=a.num_qo_heads, num_kv_heads=a.num_kv_heads, head_dim_qk=a.head_dim,
                page_size=a.page_size, q_data_type=a.dtype, kv_data_type=a.dtype)
            result.append((wrapper, *size))
        return result

    def run(self, wrapper, q, kv, used_rows):
        # Python shape checks see actual rows in eager. During capture these
        # counts equal capacity, so replay retains the full-sized addresses;
        # the native plan's device row counts/masks limit actual GPU work.
        if used_rows == 0:
            return q.new_zeros(q.shape[:-1] + (kv[1].shape[-1],)), torch.full(
                q.shape[:2], -5.0e4, dtype=torch.float32, device=q.device)
        out, lse = wrapper.run(q[:used_rows], kv, return_lse=True)
        padding = q.shape[0]-used_rows
        if padding:
            out = F.pad(out, (0, 0, 0, 0, 0, padding))
            lse = F.pad(lse, (0, 0, 0, padding), value=-5.0e4)
        return out, lse

    def plan(self, specs):
        from .attention import _rel_positions

        a, w, d, r = self.attention, self.workers, self.depth, self.rows
        if len(specs) > w or sum(s.num_new for s in specs) > r or any(len(s.context) > d for s in specs):
            raise ValueError('Shared prefill exceeds graph request/row/segment capacity')
        self.event.synchronize()  # protects FlashInfer's reused pinned plan workspace
        self.refs = []
        ctx_q, self_q = [0] * (w*d), [0] * w
        page_snapshot = {}
        ctx_pages = [self.dummy_parts[i // d] for i in range(w*d)]
        self_pages = list(self.dummy_parts)
        ctx_lengths, self_lengths = [1] * (w*d), [1] * w
        gather, slots, locations = [], [], []
        self_loc = torch.zeros((3, r) if self.use_3d else (r,), dtype=torch.int64)
        self_slots = torch.arange(r) * (d+1) + d
        last_rows, offset = [0] * w, 0
        for worker, spec in enumerate(specs):
            n = spec.num_new
            source = torch.arange(offset, offset+n)
            rel = _rel_positions(spec.mrope_rel, n, self.use_3d)
            own_span = spec.self_prefix_len if spec.self_prefix_span is None else spec.self_prefix_span
            position = own_span + sum(b.mrope_span for b in spec.context)
            for slot, block in enumerate(spec.context):
                index = worker*d + slot
                ctx_q[index], ctx_lengths[index] = n, block.num_tokens
                ctx_pages[index] = page_numbers(block, page_snapshot)
                gather.append(source)
                slots.append(source * (d+1) + slot)
                locations.append(rel + position)
                position -= block.mrope_span
            self_q[worker] = n
            self_lengths[worker] = spec.self_prefix_len + n
            self_pages[worker] = [start // a.page_size for start in spec.self_page_starts.cpu().tolist()]
            self_loc[..., offset:offset+n] = rel + own_span
            # Retain the old order: contexts, then self, then zero-weight slots.
            self_slots[offset:offset+n] = source * (d+1) + len(spec.context)
            last_rows[worker], offset = offset+n-1, offset+n

        n_ctx = sum(ctx_q)

        def upload(destination, value):
            host = torch.as_tensor(value, dtype=destination.dtype, device='cpu').pin_memory()
            self.refs.append(host)
            destination.copy_(host, non_blocking=True)

        for wrapper, lengths, pages, qlens, causal in (
            (self.main, ctx_lengths, ctx_pages, ctx_q, False),
            (self.aux, self_lengths, self_pages, self_q, True)):
            counts = [len(p) for p in pages]
            if sum(counts) > self.max_pages:
                raise ValueError('Shared prefill exceeds graph page-index capacity')
            host = [torch.tensor(x, dtype=torch.int32, pin_memory=True) for x in (
                [0, *qlens], [0, *counts],
                [n - (count-1)*a.page_size for n, count in zip(lengths, counts)], lengths)]
            host[0].cumsum_(0)
            host[1].cumsum_(0)
            indices = upload_page_indices(pages, a.device, self.refs)
            self.refs.extend(host)
            wrapper.plan(qo_indptr=host[0], paged_kv_indptr=host[1], paged_kv_indices=indices,
                paged_kv_last_page_len=host[2], seq_lens=host[3], causal=causal,
                num_qo_heads=a.num_qo_heads, num_kv_heads=a.num_kv_heads, head_dim_qk=a.head_dim,
                page_size=a.page_size, pos_encoding_mode='NONE', q_data_type=a.dtype,
                kv_data_type=a.dtype, non_blocking=True,
                max_token_per_sequence=max(qlens, default=0))
        active_slots = torch.cat(slots) if slots else torch.empty(0, dtype=torch.int64)
        # Vectorized complement, not a Python scan over rows*depth padded slots.
        unused = torch.ones(r*(d+1), dtype=torch.bool)
        unused[active_slots] = False
        unused[self_slots] = False
        permutation = torch.cat([active_slots, self._all_slots[unused]])
        assert permutation.numel() == self.ctx_rows
        gather_rows = torch.zeros(self.ctx_rows, dtype=torch.int64)
        ctx_loc = torch.zeros((3, self.ctx_rows) if self.use_3d else (self.ctx_rows,), dtype=torch.int64)
        if gather:
            gather_rows[:n_ctx] = torch.cat(gather)
            ctx_loc[..., :n_ctx] = torch.cat(locations, dim=-1)
        valid = torch.zeros(self.ctx_rows+r, dtype=torch.bool)
        valid[:n_ctx] = True
        valid[self.ctx_rows:self.ctx_rows+offset] = True
        upload(self.meta.sub_worker[:self.ctx_rows], gather_rows)
        upload(self.meta.sub_loc, torch.cat([ctx_loc, self_loc], dim=-1))
        upload(self.meta.pad_slot, torch.cat([permutation, self_slots]))
        upload(self.meta.last_indices, last_rows)
        upload(self.valid, valid)
        self.used_ctx_rows, self.used_self_rows = n_ctx, offset
        self.event.record()
        return self.meta
