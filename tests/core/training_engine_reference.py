"""Torch replacements for half-only FlashInfer operations and independent cache provenance."""

import math

topology = None
import torch
import torch.nn.functional as F


def norm(x, weight, eps=1e-6, out=None, plus_one=False, **kw):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    y = (y * (weight.float() + int(plus_one))).to(x.dtype)
    if out is not None:
        out.copy_(y)
        return out
    return y


def fused_norm(x, residual, weight, eps=1e-6, plus_one=False, **kw):
    residual.add_(x)
    x.copy_(norm(residual, weight, eps, plus_one=plus_one))


def silu_mul(x, out=None, **kw):
    gate, up = x.chunk(2, -1)
    y = F.silu(gate) * up
    if out is not None:
        out.copy_(y)
        return out
    return y


def merge_states(v, s):
    maximum = s.amax(1, keepdim=True)
    weights = torch.exp2(s - maximum)
    denominator = weights.sum(1)
    y = (v.float() * weights[..., None]).sum(1) / denominator[..., None]
    return y.to(v.dtype), maximum[:, 0] + denominator.log2()


class Paged:
    def __init__(self, workspace, **kw):
        self._int_workspace_buffer = torch.empty(16, dtype=torch.uint8, device=workspace.device)

    def run(self, q, paged_kv_cache, return_lse=False, **kw):
        if topology is not None:
            topology.observe(self)
        k_cache, v_cache = paged_kv_cache
        outputs, lses = [], []
        for i, (start, end) in enumerate(zip(self.qptr[:-1], self.qptr[1:])):
            ids = self.indices[self.kptr[i] : self.kptr[i + 1]].long()
            n = (len(ids) - 1) * self.page_size + self.last[i]
            k = k_cache[ids].flatten(0, 1)[:n]
            v = v_cache[ids].flatten(0, 1)[:n]
            repeats = q.shape[1] // k.shape[1]
            k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
            scores = (
                torch.matmul(q[start:end].transpose(0, 1).float(), k.permute(1, 2, 0).float())
                * self.scale
            )
            if self.causal:
                mask = torch.arange(n, device=q.device)[None, :] > torch.arange(
                    end - start, device=q.device
                )[:, None] + n - (end - start)
                scores.masked_fill_(mask[None], -torch.inf)
            output = torch.matmul(scores.softmax(-1), v.transpose(0, 1).float()).transpose(0, 1)
            outputs.append(output.to(q.dtype))
            lses.append(scores.logsumexp(-1).transpose(0, 1) / math.log(2))
        output, lse = torch.cat(outputs), torch.cat(lses)
        return (output, lse) if return_lse else output


class Decode(Paged):
    def plan(
        self, indptr, indices, last_page_len, num_qo_heads, num_kv_heads, head_dim, page_size, **kw
    ):
        assert kw.get("pos_encoding_mode", "NONE") == "NONE"
        self.kptr, self.last = indptr.tolist(), last_page_len.tolist()
        self.indices, self.page_size = indices, page_size
        self.qptr = list(range(len(self.kptr)))
        self.scale, self.causal = kw.get("sm_scale") or head_dim**-0.5, False


class Prefill(Paged):
    def plan(
        self,
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        page_size,
        causal=False,
        **kw,
    ):
        assert kw.get("pos_encoding_mode", "NONE") == "NONE"
        self.qptr, self.kptr = qo_indptr.tolist(), paged_kv_indptr.tolist()
        self.indices, self.last = paged_kv_indices, paged_kv_last_page_len.tolist()
        self.page_size, self.scale, self.causal = (
            page_size,
            kw.get("sm_scale") or head_dim_qk**-0.5,
            causal,
        )


def install():
    from functools import partial

    import flashinfer

    flashinfer.rmsnorm = norm
    flashinfer.gemma_rmsnorm = partial(norm, plus_one=True)
    flashinfer.fused_add_rmsnorm = fused_norm
    flashinfer.gemma_fused_add_rmsnorm = partial(fused_norm, plus_one=True)
    flashinfer.silu_and_mul = silu_mul
    flashinfer.merge_states = merge_states
    flashinfer.BatchDecodeWithPagedKVCacheWrapper = Decode
    flashinfer.BatchPrefillWithPagedKVCacheWrapper = Prefill


def check():
    device = "cuda"
    p = Prefill(torch.empty(1, device=device))
    p.plan(
        torch.tensor([0, 2]),
        torch.tensor([0, 1]),
        torch.tensor([0], device=device),
        torch.tensor([4]),
        1,
        1,
        1,
        4,
        causal=True,
    )
    out, lse = p.run(
        torch.zeros(2, 1, 1, device=device),
        (
            torch.zeros(1, 4, 1, 1, device=device),
            torch.tensor([0.0, 2.0, 4.0, 6.0], device=device).reshape(1, 4, 1, 1),
        ),
        True,
    )
    torch.testing.assert_close(out.flatten(), torch.tensor([2.0, 3.0], device=device))
    torch.testing.assert_close(lse.flatten(), torch.tensor([math.log2(3), 2.0], device=device))
    out, lse = merge_states(
        torch.tensor([0.0, 4.0], device=device).reshape(1, 2, 1, 1),
        torch.tensor([0.0, math.log2(3)], device=device).reshape(1, 2, 1),
    )
    torch.testing.assert_close(out.flatten(), torch.tensor([3.0], device=device))
    torch.testing.assert_close(lse.flatten(), torch.tensor([2.0], device=device))
    print("FP32 attention adapter causal-mask and base-2 LSE checks passed", flush=True)


from torch.overrides import TorchFunctionMode


class DoubleMath(TorchFunctionMode):
    """Promote explicit FP32 compute casts; enter only after loading checkpoint tensors."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if func is torch.Tensor.float:
            return args[0].double()
        if func is torch.Tensor.to:
            args = tuple(torch.float64 if x is torch.float32 else x for x in args)
        if kwargs.get("dtype") is torch.float32:
            kwargs["dtype"] = torch.float64
        return func(*args, **kwargs)


class RuntimeHistory:
    """Integer provenance from actual cache plans, independent of Trace compilation."""

    def __init__(self, session):
        self.slots, self.attention, self.contents, self.tails, self.gdn = {}, {}, {}, {}, {}
        original_copy, original_shift = session._copy_slots, session._shift_keys_
        original_free, original_merge, original_append = (
            session.free_block,
            session.merge_blocks,
            session.append_block,
        )

        def copy_slots(sources, destination, destination_start):
            values = [self.slots[s] for s in sources.tolist()]
            destinations = destination.token_slots_tensor().tolist()[
                destination_start : destination_start + len(values)
            ]
            result = original_copy(sources, destination, destination_start)
            self.slots.update(zip(destinations, values, strict=True))
            return result

        def shift_keys(*, block, start, num_tokens, position_shift):
            slots = block.token_slots_tensor().tolist()[start : start + num_tokens]
            result = original_shift(
                block=block, start=start, num_tokens=num_tokens, position_shift=position_shift
            )
            for slot in slots:
                row, pos = self.slots[slot]
                self.slots[slot] = row, pos + position_shift
            return result

        def free(block):
            slots = block.token_slots_tensor().tolist()
            result = original_free(block)
            for slot in slots:
                self.slots.pop(slot, None)
            self.contents.pop(block.block_id, None)
            self.tails.pop(block.block_id, None)
            return result

        def merge(operation, left, right):
            a, b = (self.contents.get(block.block_id, ()) for block in (left, right))
            tail = self.tails.get((right if b else left).block_id, (-1,) * 4)
            result = operation(left, right)
            self.contents[result.block_id], self.tails[result.block_id] = a + b, tail
            return result

        session._copy_slots, session._shift_keys_, session.free_block = copy_slots, shift_keys, free
        session.merge_blocks = lambda left, right: merge(original_merge, left, right)
        session.append_block = lambda left, right: merge(original_append, left, right)

    @staticmethod
    def positions(value):
        if value.ndim == 2:
            assert (
                value.shape[0] == 3
                and torch.equal(value[0], value[1])
                and torch.equal(value[0], value[2])
            )
            value = value[0]
        return value.tolist()

    def begin(self, batch, rows, chains, targets, lengths):
        self.batch, self.rows, self.seen = batch, rows, set()
        self.qpos = self.positions(
            batch.positions if batch.mrope_positions is None else batch.mrope_positions
        )
        self.slots.update(
            (slot, (row, pos))
            for slot, row, pos in zip(batch.out_loc.tolist(), rows, self.qpos, strict=True)
        )
        cursor, pending = 0, []
        for chain, target, length in zip(chains, targets, lengths, strict=True):
            path, tail = (), (-1,) * 4
            for block in chain:
                sources = self.contents.get(block.block_id, ())
                path += sources
                if sources:
                    tail = self.tails[block.block_id]
            writes = rows[cursor : cursor + length]
            for row in writes:
                path += (row,)
                tail = (*tail[1:], row)
                self.gdn[row] = path, tail
            pending.append(
                (target.block_id, self.contents.get(target.block_id, ()) + tuple(writes), tail)
            )
            cursor += length
        assert cursor == len(rows)
        for block, sources, tail in pending:
            self.contents[block], self.tails[block] = sources, tail

    def observe(self, wrapper):
        if id(wrapper) in self.seen:
            return
        self.seen.add(id(wrapper))
        meta = self.batch.attn_metadata
        sc = getattr(meta, "shared_cache_op", None)
        if sc is None:
            rows, positions, segments, offset = self.rows, self.qpos, [0] * len(self.rows), 0
        else:
            rows = [self.rows[i] for i in meta.sub_worker.tolist()]
            positions = self.positions(meta.sub_loc)
            segments = [i % meta.max_segments for i in meta.pad_slot.tolist()]
            if meta.phase == "prefill_batch":
                assert wrapper in (sc.prefill_ctx_wrapper, sc.prefill_self_wrapper)
                offset = 0 if wrapper is sc.prefill_ctx_wrapper else meta.n_ctx_rows
            else:
                assert wrapper in (sc.wrapper, sc.aux_wrapper)
                offset = 0 if wrapper is sc.wrapper else meta.n_main
        indices = wrapper.indices.tolist()
        for i, last in enumerate(wrapper.last):
            pages = indices[wrapper.kptr[i] : wrapper.kptr[i + 1]]
            length = (len(pages) - 1) * wrapper.page_size + last
            slots = [
                page * wrapper.page_size + j for page in pages for j in range(wrapper.page_size)
            ][:length]
            sources = tuple(self.slots[slot] for slot in slots)
            start, end = wrapper.qptr[i : i + 2]
            for j in range(start, end):
                visible = length - (end - start) + j - start + 1 if wrapper.causal else length
                index = offset + j
                key = rows[index], segments[index]
                assert key not in self.attention
                self.attention[key] = positions[index], sources, visible

    def assert_replay(self, replay):
        from minisgl.training.gdn import GDNReplay

        order = replay.order.tolist()
        keys = set()
        for packed, reads in enumerate(replay.reads):
            for segment, (version, lo, hi, position) in enumerate(reads):
                key = order[packed], segment
                keys.add(key)
                wanted = tuple(
                    (order[start + j], offset + j)
                    for start, end, offset in replay.source_spans(version, lo, hi)
                    for j in range(end - start)
                )
                actual_position, sources, visible = self.attention[key]
                assert actual_position == position, (key, actual_position, position)
                assert sources[:visible] == wanted, (key, "attention provenance")
        assert keys == set(self.attention)
        for mode in ("expanded", "factorized"):
            plan = GDNReplay.from_trace(replay.trace, replay.order, segment_size=8, read_mode=mode)
            nodes = [()] + [(row,) for row in order]
            for level in plan.levels:
                nodes.extend(nodes[left] + nodes[right] for left, right in level.tolist())
            for packed, path in enumerate(plan.reads.tolist()):
                sources = sum((nodes[i] for i in reversed(path) if i), ())
                expected, tail = self.gdn[order[packed]]
                assert sources == expected, (order[packed], "GDN DAG")
                assert (
                    tuple(order[i] if i >= 0 else -1 for i in plan.conv_sources[packed].tolist())
                    == tail
                )
            levels, reads, cached, (factors, lookup) = plan.partitions
            total = sum(len(level[2]) for level in levels) + len(cached)
            dense, short = [None] * total, {}
            for index, source in enumerate(cached.tolist(), total - len(cached)):
                dense[index] = (order[source],)

            def child(size, source):
                return (
                    dense[source]
                    if not size
                    else ((order[source],) if size == 1 else tuple(order[i] for i in source))
                )

            for ls, rs, parents, left, right in levels:
                for p, a, b in zip(parents.tolist(), left.tolist(), right.tolist(), strict=True):
                    dense[p] = child(ls, a) + child(rs, b)
            column_count = sum(sources.numel() for _, _, sources in factors) + bool(factors)
            column_sources, factor_sources = [None] * column_count, {}
            if column_count:
                column_sources[0] = ()  # The executable factor buffers reserve zero padding.
            for rows, columns, sources in factors:
                for row, cols, src in zip(
                    rows.tolist(), columns.tolist(), sources.tolist(), strict=True
                ):
                    assert row not in factor_sources, (mode, row, "duplicate factor scale row")
                    factor_sources[row] = tuple(order[i] for i in src)
                    for offset, (column, source) in enumerate(
                        zip(cols, factor_sources[row], strict=True)
                    ):
                        assert 0 < column < column_count, (mode, column, "factor column bounds")
                        assert column_sources[column] is None, (mode, column, "factor overwrite")
                        column_sources[column] = (row, offset, source)
            assert all(source is not None for source in column_sources), "unwritten factor column"
            assert set(factor_sources) == set(range(len(lookup))), "factor scale row coverage"
            for row, columns in enumerate(lookup.tolist()):
                assert all(0 <= column < column_count for column in columns), "lookup bounds"
                terms = tuple(column_sources[column] for column in columns if column)
                assert terms == tuple(
                    (row, offset, source) for offset, source in enumerate(factor_sources[row])
                ), (mode, row, "factor lookup/column provenance")
                short[row] = tuple(source for _, _, source in terms)
            paths = [[] for _ in order]
            for groups in reads:
                for kind, (rows, indices) in enumerate(groups):
                    for row, source in zip(rows.tolist(), indices.tolist(), strict=True):
                        paths[row].append(
                            (order[source],)
                            if kind == 0
                            else dense[source] if kind == 1 else short[source]
                        )
            for packed, path in enumerate(paths):
                assert sum(reversed(path), ()) == self.gdn[order[packed]][0], (
                    order[packed],
                    "GDN executable partitions",
                    mode,
                )
        return {"attention_segments": len(keys), "gdn_rows": len(self.gdn), "partition_modes": 2}
