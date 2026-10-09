from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from itertools import groupby

import torch
import torch.nn.functional as F

from .trace import Trace


@dataclass
class GDNReplay:
    conv_sources: torch.Tensor
    levels: list[torch.Tensor]
    reads: torch.Tensor
    segment_size: int = 8
    read_mode: str = "expanded"
    cache_reused: bool = True
    read_backend: str = "torch"

    @cached_property
    def read_paths(self):
        reads = self.partitions[1]
        paths = torch.zeros(len(self.reads), len(reads), 2, dtype=torch.long)
        for depth, groups in enumerate(reads):
            for kind, (rows, indices) in enumerate(groups, 1):
                paths[rows.cpu(), depth, 0] = kind
                paths[rows.cpu(), depth, 1] = indices.cpu()
        return paths.to(self.reads.device)

    @cached_property
    def partitions(self):
        n, device = len(self.reads), self.reads.device
        if self.segment_size < 1:
            raise ValueError("Segment size must be positive")
        if self.read_mode not in {"expanded", "factorized"}:
            raise ValueError("Read mode must be expanded or factorized")

        def tensor(values):
            return torch.tensor(values, dtype=torch.long, device=device)

        children = [level.cpu().tolist() for level in self.levels]
        pairs = [pair for level in children for pair in level]
        segments = [()] + [(i,) for i in range(n)]
        for left, right in pairs:
            if left <= 0 or right <= 0:
                raise ValueError("Summary children must be nonempty")
            a, b = segments[left], segments[right]
            segments.append(
                a + b
                if a is not None and b is not None and len(a) + len(b) <= self.segment_size
                else None
            )
        retained = [i for i in range(n + 1, len(segments)) if segments[i] is None]
        uses = Counter()
        for node in retained:
            for i in pairs[node - n - 1]:
                uses[i] += 1
                if self.cache_reused and uses[i] == 3 and i > n and segments[i] is not None:
                    retained.append(i)
        dense = {i: row for row, i in enumerate(retained)}
        cached = {
            i: len(dense) + row
            for row, i in enumerate(
                sorted(i for i, count in uses.items() if self.cache_reused and i <= n and count > 2)
            )
        }
        table = {**dense, **cached}

        def child(node):
            if node in table:
                return 0, table[node]
            rows = segments[node]
            return len(rows), rows[0] if len(rows) == 1 else rows

        levels, start = [], n + 1
        for level in children:
            groups = {}
            for row, (left, right) in enumerate(level, start):
                if row in dense:
                    left_size, left = child(left)
                    right_size, right = child(right)
                    groups.setdefault((left_size, right_size), []).append((dense[row], left, right))
            for kinds, rows in groups.items():
                levels.append((*kinds, *(tensor(values) for values in zip(*rows))))
            start += len(level)
        reads = []
        paths = []
        short = {}
        for path in self.reads.cpu().tolist():
            expanded = []
            for node in path:
                if node in dense:
                    expanded.append((0, dense[node]))
                elif node > n and self.read_mode == "factorized":
                    if node not in short:
                        short[node] = len(short)
                    expanded.append((2, short[node]))
                elif node:
                    expanded.extend((1, row) for row in reversed(segments[node]))
            paths.append(expanded)
        for depth in range(max(1, max(map(len, paths)))):
            groups = []
            for kind in (1, 0, 2):
                pairs = [
                    (row, path[depth][1])
                    for row, path in enumerate(paths)
                    if depth < len(path) and path[depth][0] == kind
                ]
                groups.append(
                    (
                        tensor([row for row, _ in pairs]),
                        tensor([i for _, i in pairs]),
                    )
                )
            reads.append(groups)
        width = max((len(segments[node]) for node in short), default=0)
        lookup, groups, cursor = [[0] * width for _ in short], {}, 1
        for node, row in short.items():
            sources = segments[node]
            columns = tuple(range(cursor, cursor + len(sources)))
            lookup[row][: len(sources)] = columns
            groups.setdefault(len(sources), []).append((row, columns, sources))
            cursor += len(sources)
        factors = [tuple(tensor(values) for values in zip(*rows)) for rows in groups.values()]
        return (
            levels,
            reads,
            tensor([i - 1 for i in cached]),
            (factors, tensor(lookup).reshape(len(short), width)),
        )

    @classmethod
    def from_trace(
        cls,
        trace: Trace,
        order,
        device="cpu",
        conv_kernel_size=4,
        segment_size=8,
        read_mode="expanded",
        cache_reused=True,
        read_backend="torch",
    ):
        """Compile shared snapshot summaries and original convolution windows."""
        if read_backend not in {"torch", "triton"}:
            raise ValueError("Read backend must be torch or triton")
        n = len(trace.tokens)
        order = torch.as_tensor(order).cpu().tolist()
        if not n or sorted(order) != list(range(n)) or conv_kernel_size < 1:
            raise ValueError("Expected a nonempty trace, a token permutation, and positive width")
        inverse = {row: packed for packed, row in enumerate(order)}
        nodes, intern, heights = [], {}, [0] * (n + 1)

        def join(left, right):
            if not left or not right:
                return left or right
            pair = (left, right)
            if pair not in intern:
                intern[pair] = n + 1 + len(nodes)
                nodes.append(pair)
                heights.append(max(heights[left], heights[right]) + 1)
            return intern[pair]

        def append(forest, node):
            forest = list(forest)
            while forest and heights[forest[-1]] == heights[node]:
                node = join(forest.pop(), node)
            return (*forest, node)

        def summarize(forest):
            root = 0
            for node in forest:
                root = join(root, node)
            return root

        # Each block holds (token count, summary forest, raw-QKV convolution tail).
        empty = (0, (), (-1,) * conv_kernel_size)
        blocks, windows, paths = {}, [None] * n, [None] * n
        update_index = 0
        for step, rows in groupby(enumerate(trace.tokens), key=lambda x: x[1].step):
            rows = list(rows)
            while update_index < len(trace.updates) and trace.updates[update_index].step <= step:
                update = trace.updates[update_index]
                count, forest, tail = empty
                if update.refs is not None:
                    if update.sources:
                        raise ValueError("Mutation cannot contain both sources and refs")
                    root = 0
                    for block, length in update.refs:
                        state = blocks.get(block, empty)
                        if block < 0 or length < 0 or length != state[0]:
                            raise ValueError("Mutation must reference a live block snapshot")
                        count += length
                        root = join(root, summarize(state[1]))
                        if length:
                            tail = state[2]
                    forest = (root,) if root else ()
                else:
                    for source in update.sources:
                        if source < 0 or source >= rows[0][0] or windows[source] is None:
                            raise ValueError("Mutation refers to an unavailable token")
                        forest = append(forest, inverse[source] + 1)
                    count = len(update.sources)
                    tail = windows[update.sources[-1]] if count else empty[2]
                blocks[update.block] = (count, forest, tail)
                update_index += 1
            kinds = {token.kind for _, token in rows}
            if len(kinds) != 1 or not kinds <= {"prefill", "decode"}:
                raise ValueError("A step must contain either prefills or decodes")
            decode = rows[0][1].kind == "decode"
            written = {token.block for _, token in rows}
            if decode and len(written) != len(rows):
                raise ValueError("Decode needs one writer per block")
            before = {block: blocks.get(block, empty) for _, t in rows for block, _ in t.view}
            pending = {}
            for row, token in rows:
                count, forest, _ = blocks.get(token.block, empty)
                if count != token.position:
                    raise ValueError("Token position does not match the live block")
                path, tail = [], empty[2]
                for block, length in token.view:
                    length -= int(block in written) if decode else int(block == token.block)
                    read_blocks = blocks if not decode and block == token.block else before
                    state = read_blocks.get(block, empty)
                    if length != state[0]:
                        raise ValueError("GDN view must reference the block's pre-update prefix")
                    path.extend(state[1])
                    if length:
                        tail = state[2]
                leaf = inverse[row] + 1
                windows[row] = (*tail[1:], row)
                paths[row] = (*path, leaf)
                state = (count + 1, append(forest, leaf), windows[row])
                if decode:
                    pending[token.block] = state
                else:
                    blocks[token.block] = state
            blocks.update(pending)

        remap = {i: i for i in range(n + 1)}
        levels = []
        groups = {}
        for i in range(n + 1, len(heights)):
            groups.setdefault(heights[i], []).append(i)
        for height in sorted(groups):
            group = groups[height]
            levels.append(
                torch.tensor(
                    [[remap[a], remap[b]] for i in group for a, b in [nodes[i - n - 1]]],
                    device=device,
                )
            )
            for i in group:
                remap[i] = len(remap)
        depth = max(map(len, paths))
        reads = [[remap[i] for i in reversed(paths[row])] for row in order]
        reads = [path + [0] * (depth - len(path)) for path in reads]
        conv = [[inverse[i] if i >= 0 else -1 for i in windows[row]] for row in order]
        return cls(
            torch.tensor(conv, device=device),
            levels,
            torch.tensor(reads, device=device),
            segment_size,
            read_mode,
            cache_reused,
            read_backend,
        )


def _segment_factors(key, value, alpha, beta):
    """HF-style triangular summary with products preserving zero-decay gradients."""
    key, value, alpha, beta = (x.transpose(1, 2) for x in (key, value, alpha, beta))
    positions = torch.arange(key.shape[-2], device=key.device)
    lower = positions[:, None] > positions[None, :]
    decay = torch.where(lower, alpha.unsqueeze(-1), 1).cumprod(-2)
    prefix = alpha.cumprod(-1)
    system = ((beta.unsqueeze(-1) * key) @ key.transpose(-1, -2)) * decay
    rhs = beta.unsqueeze(-1) * torch.cat((value, prefix.unsqueeze(-1) * key), -1)
    solved = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
    values, keys = solved.split((value.shape[-1], key.shape[-1]), -1)
    tail = decay[..., -1, :, None] * key
    return prefix[..., -1], -keys.transpose(1, 2), values.transpose(1, 2), tail.transpose(1, 2)


def _segment_matrices(key, value, alpha, beta):
    scale, u, z, w = _segment_factors(key, value, alpha, beta)
    identity = torch.eye(key.shape[-1], dtype=key.dtype, device=key.device)
    w = w.transpose(1, 2)
    return scale[..., None, None] * identity + u.permute(0, 2, 3, 1) @ w, z.permute(0, 2, 3, 1) @ w


class _Delta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, alpha, beta, replay, save):
        n, heads, dk = key.shape
        dv = value.shape[-1]
        levels, reads, cached, (factors, lookup) = replay.partitions
        total = sum(len(level[2]) for level in levels) + len(cached)
        a = key.new_empty(total, heads, dk, dk)
        b = value.new_empty(total, heads, dv, dk)
        identity = torch.eye(dk, dtype=key.dtype, device=key.device) if total else None

        def matrices(indices, leaf):
            if not leaf:
                return a[indices], b[indices]
            if leaf > 1:
                return _segment_matrices(*(x[indices] for x in (key, value, alpha, beta)))
            k, v = key[indices], value[indices]
            al, be = alpha[indices, :, None, None], beta[indices, :, None, None]
            return al * (identity - be * k[..., :, None] * k[..., None, :]), be * v[
                ..., :, None
            ] * k[..., None, :]

        if len(cached):
            a[-len(cached) :], b[-len(cached) :] = matrices(cached, True)
        for left_leaf, right_leaf, parents, left, right in levels:
            al, bl = matrices(left, left_leaf)
            ar, br = matrices(right, right_leaf)
            a[parents] = al @ ar
            b[parents] = bl @ ar + br
            del al, bl, ar, br
        columns = sum(sources.numel() for _, _, sources in factors) + bool(factors)
        scale = alpha.new_empty(len(lookup), heads)
        fu = key.new_zeros(columns, heads, dk)
        fz = value.new_zeros(columns, heads, dv)
        fw = torch.zeros_like(fu)
        for rows, columns, sources in factors:
            ss, uu, zz, ww = _segment_factors(*(x[sources] for x in (key, value, alpha, beta)))
            scale[rows], fu[columns], fz[columns], fw[columns] = ss, uu, zz, ww
            del ss, uu, zz, ww
        if replay.read_backend == "triton":
            from .gdn_kernel import read_paths

            output, vectors = read_paths(
                query,
                key,
                value,
                alpha,
                beta,
                a,
                b,
                scale,
                fu,
                fz,
                fw,
                lookup,
                replay.read_paths,
                save,
            )
        else:
            r = query * dk**-0.5
            output = torch.zeros_like(value)
            vectors = []
            for (rows, indices), (dense_rows, dense_indices), (short_rows, short_indices) in reads:
                if save:
                    vectors.append(r)
                next_r = r.clone()
                if len(rows):
                    k, v = key[indices], value[indices]
                    al, be = alpha[indices, :, None], beta[indices, :, None]
                    kr = (k * r[rows]).sum(-1, keepdim=True)
                    output[rows] += be * v * kr
                    next_r[rows] = al * (r[rows] - be * k * kr)
                if len(dense_rows):
                    rr = r[dense_rows, ..., None]
                    output[dense_rows] += (b[dense_indices] @ rr).squeeze(-1)
                    next_r[dense_rows] = (a[dense_indices] @ rr).squeeze(-1)
                if len(short_rows):
                    columns = lookup[short_indices]
                    rr = r[short_rows]
                    s = (fw[columns] * rr[:, None]).sum(-1)
                    output[short_rows] += (fz[columns] * s[..., None]).sum(1)
                    next_r[short_rows] = scale[short_indices, :, None] * rr + (
                        fu[columns] * s[..., None]
                    ).sum(1)
                    del rr, s
                r = next_r
            vectors = torch.stack(vectors) if save else None
        if save:
            ctx.save_for_backward(
                key,
                value,
                alpha,
                beta,
                a,
                b,
                vectors,
                cached,
                scale,
                fu,
                fz,
                fw,
                lookup,
                *(indices for level in levels for indices in level[2:]),
                *(indices for group in factors for indices in group),
                *(indices for groups in reads for group in groups for indices in group),
            )
            ctx.level_kinds = [(left, right) for left, right, *_ in levels]
            ctx.factor_groups = len(factors)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if torch.is_grad_enabled():
            raise RuntimeError("GDN replay supports first-order gradients only")
        key, value, alpha, beta, a, b, vectors, cached, scale, fu, fz, fw, lookup, *indices = (
            ctx.saved_tensors
        )
        indices = iter(indices)
        levels = [
            (left, right, next(indices), next(indices), next(indices))
            for left, right in ctx.level_kinds
        ]
        factors = [(next(indices), next(indices), next(indices)) for _ in range(ctx.factor_groups)]
        reads = [
            ((rows, ids), (dense_rows, dense_ids), (short_rows, short_ids))
            for rows, ids, dense_rows, dense_ids, short_rows, short_ids in zip(*([indices] * 6))
        ]
        da, db = torch.zeros_like(a), torch.zeros_like(b)
        ds, du, dz, dw = map(torch.zeros_like, (scale, fu, fz, fw))
        grad_key, grad_value, grad_alpha, grad_beta = map(
            torch.zeros_like, (key, value, alpha, beta)
        )
        dr = torch.zeros_like(vectors[0])
        for depth in range(len(reads) - 1, -1, -1):
            (rows, indices), (dense_rows, dense_indices), (short_rows, short_indices) = reads[depth]
            r = vectors[depth]
            next_dr = dr.clone()
            if len(rows):
                k, v, rr, u, g = key[indices], value[indices], r[rows], dr[rows], grad_output[rows]
                al, be = alpha[indices, :, None], beta[indices, :, None]
                kr, uk, gv = ((x * y).sum(-1, keepdim=True) for x, y in ((k, rr), (u, k), (g, v)))
                grad_value.index_add_(0, indices, be * kr * g)
                grad_alpha.index_add_(0, indices, (u * (rr - be * k * kr)).sum(-1))
                grad_beta.index_add_(0, indices, (kr * (gv - al * uk)).squeeze(-1))
                grad_key.index_add_(0, indices, be * ((gv - al * uk) * rr - al * kr * u))
                next_dr[rows] = al * u + be * (gv - al * uk) * k
            if len(dense_rows):
                rr, u, g = (x[dense_rows, ..., None] for x in (r, dr, grad_output))
                db.index_add_(0, dense_indices, g @ rr.transpose(-1, -2))
                da.index_add_(0, dense_indices, u @ rr.transpose(-1, -2))
                next_dr[dense_rows] = (
                    b[dense_indices].transpose(-1, -2) @ g + a[dense_indices].transpose(-1, -2) @ u
                ).squeeze(-1)
            if len(short_rows):
                columns = lookup[short_indices]
                rr, u, g = (x[short_rows] for x in (r, dr, grad_output))
                t = (fz[columns] * g[:, None]).sum(-1) + (fu[columns] * u[:, None]).sum(-1)
                w = fw[columns]
                s = (w * rr[:, None]).sum(-1)
                ds.index_add_(0, short_indices, (u * rr).sum(-1))
                du.index_add_(0, columns.flatten(), (u[:, None] * s[..., None]).flatten(0, 1))
                dz.index_add_(0, columns.flatten(), (g[:, None] * s[..., None]).flatten(0, 1))
                dw.index_add_(0, columns.flatten(), (rr[:, None] * t[..., None]).flatten(0, 1))
                next_dr[short_rows] = scale[short_indices, :, None] * u + (w * t[..., None]).sum(1)
                del rr, u, g, w, s, t
            dr = next_dr

        for rows, columns, sources in factors:
            with torch.enable_grad():
                inputs = tuple(
                    x[sources].detach().requires_grad_() for x in (key, value, alpha, beta)
                )
                gradients = torch.autograd.grad(
                    _segment_factors(*inputs),
                    inputs,
                    (ds[rows], du[columns], dz[columns], dw[columns]),
                )
            for target, gradient in zip((grad_key, grad_value, grad_alpha, grad_beta), gradients):
                target.index_add_(0, sources.flatten(), gradient.flatten(0, 1))
            del inputs, gradients, gradient
        del ds, du, dz, dw

        identity = torch.eye(key.shape[-1], dtype=key.dtype, device=key.device) if len(a) else None

        def matrices(indices, leaf):
            if not leaf:
                return a[indices], b[indices]
            if leaf > 1:
                return _segment_matrices(*(x[indices] for x in (key, value, alpha, beta)))
            k, v = key[indices], value[indices]
            al, be = alpha[indices, :, None, None], beta[indices, :, None, None]
            return al * (identity - be * k[..., :, None] * k[..., None, :]), be * v[
                ..., :, None
            ] * k[..., None, :]

        def accumulate(indices, leaf, ga, gb):
            if not leaf:
                da.index_add_(0, indices, ga)
                db.index_add_(0, indices, gb)
                return
            if leaf > 1:
                with torch.enable_grad():
                    inputs = tuple(
                        x[indices].detach().requires_grad_() for x in (key, value, alpha, beta)
                    )
                    gradients = torch.autograd.grad(_segment_matrices(*inputs), inputs, (ga, gb))
                for target, gradient in zip(
                    (grad_key, grad_value, grad_alpha, grad_beta), gradients
                ):
                    target.index_add_(0, indices.flatten(), gradient.flatten(0, 1))
                return
            k, v = key[indices], value[indices]
            al, be = alpha[indices], beta[indices]
            ak = (ga @ k.unsqueeze(-1)).squeeze(-1)
            bk = (gb @ k.unsqueeze(-1)).squeeze(-1)
            kak = (k * ak).sum(-1)
            grad_alpha.index_add_(0, indices, ga.diagonal(dim1=-2, dim2=-1).sum(-1) - be * kak)
            grad_beta.index_add_(0, indices, (v * bk).sum(-1) - al * kak)
            grad_key.index_add_(
                0,
                indices,
                -(al * be).unsqueeze(-1)
                * (ak + (ga.transpose(-1, -2) @ k.unsqueeze(-1)).squeeze(-1))
                + be.unsqueeze(-1) * (gb.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1),
            )
            grad_value.index_add_(0, indices, be.unsqueeze(-1) * bk)

        for left_leaf, right_leaf, parents, left, right in reversed(levels):
            ga, gb = da[parents], db[parents]
            ar, _ = matrices(right, right_leaf)
            del _
            accumulate(left, left_leaf, ga @ ar.transpose(-1, -2), gb @ ar.transpose(-1, -2))
            del ar
            al, bl = matrices(left, left_leaf)
            accumulate(right, right_leaf, al.transpose(-1, -2) @ ga + bl.transpose(-1, -2) @ gb, gb)
            del al, bl, ga, gb
        if len(cached):
            accumulate(cached, True, da[-len(cached) :], db[-len(cached) :])
        grad_query = dr * key.shape[-1] ** -0.5
        return grad_query, grad_key, grad_value, grad_alpha, grad_beta, None, None


def delta(query, key, value, alpha, beta, replay: GDNReplay):
    """First-order Torch replay with bounded short-segment reconstruction.

    T dense summaries use O(T H d^2); vectors use O(N D H d) at execution depth D.
    Factorized reads add O(P H d) for P cached factor columns. Traces store indices.
    """
    if replay.read_backend not in {"torch", "triton"}:
        raise ValueError("Read backend must be torch or triton")
    if replay.read_backend == "triton" and not query.is_cuda:
        raise ValueError("Triton GDN reads require CUDA FP32 inputs")
    inputs = tuple(x.float() for x in (query, key, value, alpha, beta))
    query, key, value, alpha, beta = inputs
    query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    save = torch.is_grad_enabled() and any(x.requires_grad for x in inputs)
    return _Delta.apply(query, key, value, alpha, beta, replay, save)


def gdn_forward(module, hidden_states, replay: GDNReplay):
    if hidden_states.shape[0] != 1 or module.activation not in {"silu", "swish"}:
        raise ValueError("GDN replay requires one packed sequence and SiLU activation")
    x = hidden_states[0]
    raw = module.in_proj_qkv(x)
    padded = torch.cat((torch.zeros_like(raw[:1]), raw))
    window = padded[replay.conv_sources + 1]
    weight = module.conv1d.weight[:, 0].T
    mixed = (window.float() * weight.float()).sum(1)
    if module.conv1d.bias is not None:
        mixed = mixed + module.conv1d.bias.float()
    mixed = F.silu(mixed).to(raw.dtype)
    q, k, v = mixed.split((module.key_dim, module.key_dim, module.value_dim), dim=-1)
    q = q.view(len(x), module.num_k_heads, module.head_k_dim)
    k = k.view_as(q)
    v = v.view(len(x), module.num_v_heads, module.head_v_dim)
    groups = module.num_v_heads // module.num_k_heads
    q, k = (t.repeat_interleave(groups, dim=1) for t in (q, k))
    beta = module.in_proj_b(x).sigmoid()
    g = -module.A_log.float().exp() * F.softplus(
        module.in_proj_a(x).float() + module.dt_bias.float()
    )
    core = delta(q, k, v, g.exp(), beta, replay).to(raw.dtype)
    z = module.in_proj_z(x).view_as(core)
    core = module.norm(core, z).reshape(len(x), module.value_dim)
    return module.out_proj(core)[None]
