"""Fixed-capacity FlashInfer decode metadata for the existing shared attention.

Real lengths/segment membership change through tables, not graph keys. Each
worker owns D main segment slots and one implicit-self slot. Inactive slots are
masked before the unchanged log-sum-exp merge.
"""

import torch


class SharedDecodeAttentionBuffers:
    def __init__(self, attention, workers, depth, max_pages, dummy_slots):
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        from .attention import SharedCacheAttnMetadata

        self.attention, self.workers, self.depth = attention, workers, depth
        self.dummy_slots = dummy_slots
        self.dummy_cpu = dummy_slots.cpu().tolist()
        dummy_pages = dummy_slots // attention.page_size
        self.dummy_parts = [dummy_pages[i:i + 1] for i in range(workers)]
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
        indices = torch.cat(parts).int()
        self.refs.append(indices)
        common = dict(num_qo_heads=a.num_qo_heads, num_kv_heads=a.num_kv_heads,
                      head_dim=a.head_dim, pos_encoding_mode='NONE',
                      data_type=a.dtype, q_data_type=a.dtype, kv_data_type=a.dtype,
                      non_blocking=True)
        self.main.plan(indptr=host(offsets), indices=indices, last_page_len=host(last),
                       seq_lens=host(lengths), page_size=a.page_size, **common)
        self.aux.plan(indptr=host(list(range(w + 1))), indices=host(aux_indices),
                      last_page_len=host([1] * w), seq_lens=host([1] * w), page_size=1, **common)
        self.meta.sub_loc.copy_(host(loc, torch.int64), non_blocking=True)
        self.valid.copy_(host(valid, torch.bool), non_blocking=True)
        self.event.record()
        return self.meta
