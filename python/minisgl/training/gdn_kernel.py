from typing import Any

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _read_paths(
    Query,
    Key,
    Value,
    Alpha,
    Beta,
    A,
    B,
    Scale,
    U,
    Z,
    W,
    Lookup,
    Paths,
    Out,
    Vectors,
    N: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    DEPTH: tl.constexpr,
    WIDTH: tl.constexpr,
    SAVE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    k, v = tl.arange(0, BK), tl.arange(0, BV)
    r = tl.load(Query + (row * H + head) * DK + k, k < DK, other=0.0) * DK**-0.5
    output = tl.full((BV,), 0, tl.float32)
    for depth in range(DEPTH):
        if SAVE:
            offset = ((depth.to(tl.int64) * N + row) * H + head) * DK
            tl.store(Vectors + offset + k, r, k < DK)
        kind = tl.load(Paths + (row * DEPTH + depth) * 2)
        index = tl.load(Paths + (row * DEPTH + depth) * 2 + 1)
        if kind == 1:
            key = tl.load(Key + (index * H + head) * DK + k, k < DK, other=0.0)
            value = tl.load(Value + (index * H + head) * DV + v, v < DV, other=0.0)
            alpha = tl.load(Alpha + index * H + head)
            beta = tl.load(Beta + index * H + head)
            kr = tl.sum(key * r, 0)
            output += beta * value * kr
            r = alpha * (r - beta * key * kr)
        elif kind == 2:
            b = tl.load(
                B + (index * H + head) * DV * DK + v[:, None] * DK + k[None, :],
                (v[:, None] < DV) & (k[None, :] < DK),
                other=0.0,
            )
            output += tl.sum(b * r[None, :], 1)
            a = tl.load(
                A + (index * H + head) * DK * DK + k[:, None] * DK + k[None, :],
                (k[:, None] < DK) & (k[None, :] < DK),
                other=0.0,
            )
            r = tl.sum(a * r[None, :], 1)
        elif kind == 3:
            if WIDTH:
                slot = tl.arange(0, BS)
                column = tl.load(Lookup + index * WIDTH + slot, slot < WIDTH, other=0)
                w = tl.load(
                    W + (column[:, None] * H + head) * DK + k[None, :],
                    (column[:, None] != 0) & (k[None, :] < DK),
                    other=0.0,
                )
                s = tl.sum(w * r[None, :], 1)
                z = tl.load(
                    Z + (column[:, None] * H + head) * DV + v[None, :],
                    (column[:, None] != 0) & (v[None, :] < DV),
                    other=0.0,
                )
                output += tl.sum(z * s[:, None], 0)
                u = tl.load(
                    U + (column[:, None] * H + head) * DK + k[None, :],
                    (column[:, None] != 0) & (k[None, :] < DK),
                    other=0.0,
                )
                r = tl.load(Scale + index * H + head) * r + tl.sum(u * s[:, None], 0)
    tl.store(Out + (row * H + head) * DV + v, output, v < DV)


def read_paths(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor,
    u: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
    lookup: torch.Tensor,
    paths: torch.Tensor,
    save: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, heads, dk = query.shape
    dv, depth = value.shape[-1], paths.shape[1]
    if not query.is_cuda or query.dtype != torch.float32:
        raise ValueError("Triton GDN reads require CUDA FP32 inputs")
    output = torch.empty_like(value, memory_format=torch.contiguous_format)
    vectors = query.new_empty((depth, n, heads, dk) if save else (0,))
    with torch.cuda.device(query.device):
        _read_paths[(n, heads)](
            *(x.contiguous() for x in (query, key, value, alpha, beta)),
            a,
            b,
            scale,
            u,
            z,
            w,
            lookup,
            paths,
            output,
            vectors,
            n,
            heads,
            dk,
            dv,
            depth,
            lookup.shape[1],
            save,
            tr.next_power_of_2(dk),
            tr.next_power_of_2(dv),
            tr.next_power_of_2(max(1, lookup.shape[1])),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output, vectors


CHUNK, COLUMNS, QUERIES = 64, 32, 32
LAUNCH: dict[str, tuple[str, int, int]] = {
    "prepare": ("tf32x3", 8, 1),
    "states": ("tf32x3", 4, 1),
    "read": ("tf32x3", 4, 2),
    "read_state": ("ieee", 4, 1),
    "read_query": ("tf32x3", 8, 1),
    "states_backward": ("tf32x3", 8, 1),
    "columns_backward": ("tf32x3", 8, 2),
    "prepare_backward": ("ieee", 8, 2),
}


def _launch(name: str, device: torch.device) -> dict[str, Any]:
    precision, warps, stages = LAUNCH[name]
    if torch.cuda.get_device_capability(device) < (8, 0):
        precision, stages = "ieee", 1
    return {"PRECISION": precision, "num_warps": warps, "num_stages": stages}


@tr.jit
def _mm(a, b, PRECISION: tl.constexpr):
    return tl.dot(a, b, input_precision=PRECISION)


@tr.jit
def _rows(Pointer, slot, head, H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    d = tl.arange(0, BD)
    return tl.load(Pointer + (slot[:, None] * H + head) * D + d[None, :], d[None, :] < D, other=0.0)


@tr.jit
def _tile(Pointer, chunk, head, c, H: tl.constexpr, DW: tl.constexpr, BK: tl.constexpr):
    k = tl.arange(0, BK)
    return Pointer + ((chunk * H + head) * BK + k[:, None]) * DW + c[None, :]


@tr.jit
def _decay_rows(Decay, chunk, head, row, valid, H: tl.constexpr, BT: tl.constexpr):
    t = tl.arange(0, BT)
    pointer = Decay + ((chunk * H + head) * BT + row[:, None]) * BT + t[None, :]
    return tl.load(pointer, valid[:, None], other=0.0)


@tr.jit
def _queries(
    Query,
    Positions,
    Prefix,
    Decay,
    start,
    end,
    chunk,
    head,
    H: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BQ: tl.constexpr,
):
    k = tl.arange(0, BK)
    read = start + tl.arange(0, BQ)
    valid = read < end
    read = read.to(tl.int64)
    row = tl.load(Positions + read, valid, other=1) - 1
    query = tl.load(Query + (read[:, None] * H + head) * BK + k[None, :], valid[:, None], other=0.0)
    top = tl.load(Prefix + (chunk * BT + row) * H + head, valid, other=0.0)
    decay = _decay_rows(Decay, chunk, head, row, valid, H, BT)
    return read, valid, query, row, top, decay


@tr.jit
def _chunk_decay(Alpha, Segments, Prefix, Decay, H: tl.constexpr, BT: tl.constexpr):
    chunk = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    t = tl.arange(0, BT)
    slot = chunk * BT + t
    alpha = tl.load(Alpha + slot * H + head)
    start = tl.load(Segments + slot)
    later = t[None, :] > t[:, None]
    transposed = tl.cumprod(tl.where(later, alpha[None, :], 1.0), 1)
    inside = (t[None, :] >= t[:, None]) & (start[None, :] == start[:, None])
    transposed = tl.where(inside, transposed, 0.0)
    first = t[:, None] == start[None, :]
    prefix = tl.sum(tl.where(first, alpha[:, None] * transposed, 0.0), 0)
    tl.store(Prefix + slot * H + head, prefix)
    tl.store(Decay + ((chunk * H + head) * BT + t[None, :]) * BT + t[:, None], transposed)


@tr.jit
def _chunk_prepare(
    Key,
    Value,
    Beta,
    Prefix,
    Decay,
    W,
    U,
    T,
    H: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    PRECISION: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    t, v = tl.arange(0, BT), tl.arange(0, BV)
    slot = chunk * BT + t
    key = _rows(Key, slot, head, H, BK, BK)
    beta = tl.load(Beta + slot * H + head)
    prefix = tl.load(Prefix + slot * H + head)
    decay = tl.load(Decay + ((chunk * H + head) * BT + t[:, None]) * BT + t[None, :])
    lower = t[:, None] > t[None, :]
    a = -beta[:, None] * _mm(key, tl.trans(key), PRECISION) * tl.where(lower, decay, 0.0)
    for index in range(1, BT):
        row = tl.sum(tl.where(t[:, None] == index, a, 0.0), 0)
        row += tl.sum(row[:, None] * a, 0)
        a = tl.where(t[:, None] == index, row[None, :], a)
    a += tl.where(t[:, None] == t[None, :], 1.0, 0.0)
    tl.store(T + ((chunk * H + head) * BT + t[:, None]) * BT + t[None, :], a)
    w = _mm(a, (beta * prefix)[:, None] * key, PRECISION)
    tl.store(W + (slot[:, None] * H + head) * BK + tl.arange(0, BK)[None, :], w)
    u = _mm(a, beta[:, None] * _rows(Value, slot, head, H, DV, BV), PRECISION)
    tl.store(U + (slot[:, None] * H + head) * DV + v[None, :], u, v[None, :] < DV)


@tr.jit
def _chunk_states(
    Key,
    Prefix,
    Decay,
    W,
    U,
    Bounds,
    State,
    New,
    H: tl.constexpr,
    DV: tl.constexpr,
    DW: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    PRECISION: tl.constexpr,
):
    version = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    t, k = tl.arange(0, BT), tl.arange(0, BK)
    c = tl.program_id(2) * BC + tl.arange(0, BC)
    columns = c[None, :] < DW
    begin = tl.load(Bounds + version).to(tl.int64)
    end = tl.load(Bounds + version + 1).to(tl.int64)
    state = tl.where(k[:, None] == c[None, :], 1.0, 0.0)
    for chunk in range(begin, end):
        slot = chunk * BT + t
        tl.store(_tile(State, chunk, head, c, H, DW, BK), state, columns)
        last = tl.load(Prefix + (chunk * BT + BT - 1) * H + head)
        tail = tl.load(Decay + ((chunk * H + head) * BT + BT - 1) * BT + t)
        u = tl.load(
            U + (slot[:, None] * H + head) * DV + c[None, :] - BK,
            (c[None, :] >= BK) & columns,
            other=0.0,
        )
        new = u - _mm(_rows(W, slot, head, H, BK, BK), state, PRECISION)
        tl.store(New + (slot[:, None] * H + head) * DW + c[None, :], new, columns)
        key = _rows(Key, slot, head, H, BK, BK) * tail[:, None]
        state = state * last + _mm(tl.trans(key), new, PRECISION)


@tr.jit
def _chunk_read(
    Query,
    Positions,
    Tiles,
    Key,
    Prefix,
    Decay,
    State,
    New,
    Out,
    Scores,
    H: tl.constexpr,
    DW: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    BQ: tl.constexpr,
    SAVE: tl.constexpr,
    PRECISION: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    t = tl.arange(0, BT)
    chunk = tl.load(Tiles + tile * 3).to(tl.int64)
    start, end = tl.load(Tiles + tile * 3 + 1), tl.load(Tiles + tile * 3 + 2)
    slot = chunk * BT + t
    read, valid, query, row, top, decay = _queries(
        Query, Positions, Prefix, Decay, start, end, chunk, head, H, BT, BK, BQ
    )
    score = _mm(query, tl.trans(_rows(Key, slot, head, H, BK, BK)), PRECISION)
    score = tl.where(t[None, :] <= row[:, None], score, 0.0)
    if SAVE:
        tl.store(Scores + (read[:, None] * H + head) * BT + t[None, :], score, valid[:, None])
    weight = score * decay
    query *= top[:, None]
    for offset in range(0, DW, BC):
        c = offset + tl.arange(0, BC)
        columns = c[None, :] < DW
        state = tl.load(_tile(State, chunk, head, c, H, DW, BK), columns, other=0.0)
        new = tl.load(New + (slot[:, None] * H + head) * DW + c[None, :], columns, other=0.0)
        out = _mm(query, state, PRECISION) + _mm(weight, new, PRECISION)
        tl.store(Out + (read[:, None] * H + head) * DW + c[None, :], out, valid[:, None] & columns)


@tr.jit
def _chunk_read_state(
    Query,
    Scores,
    Grad,
    Positions,
    Targets,
    Offsets,
    Prefix,
    Decay,
    DState,
    DNew,
    H: tl.constexpr,
    DW: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    BQ: tl.constexpr,
    PRECISION: tl.constexpr,
):
    target = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    t = tl.arange(0, BT)
    c = tl.program_id(2) * BC + tl.arange(0, BC)
    columns = c[None, :] < DW
    chunk = tl.load(Targets + target).to(tl.int64)
    begin, end = tl.load(Offsets + target), tl.load(Offsets + target + 1)
    slot = chunk * BT + t
    grad_state = tl.zeros((BK, BC), tl.float32)
    grad_new = tl.zeros((BT, BC), tl.float32)
    for start in range(begin, end, BQ):
        read, valid, query, _, top, decay = _queries(
            Query, Positions, Prefix, Decay, start, end, chunk, head, H, BT, BK, BQ
        )
        score = tl.load(
            Scores + (read[:, None] * H + head) * BT + t[None, :], valid[:, None], other=0.0
        )
        grad = tl.load(
            Grad + (read[:, None] * H + head) * DW + c[None, :], valid[:, None] & columns, other=0.0
        )
        grad_state += _mm(tl.trans(query * top[:, None]), grad, PRECISION)
        grad_new += _mm(tl.trans(score * decay), grad, PRECISION)
    pointer = _tile(DState, chunk, head, c, H, DW, BK)
    tl.store(pointer, tl.load(pointer, columns) + grad_state, columns)
    pointer = DNew + (slot[:, None] * H + head) * DW + c[None, :]
    tl.store(pointer, tl.load(pointer, columns) + grad_new, columns)


@tr.jit
def _chunk_read_query(
    Query,
    Scores,
    Grad,
    Positions,
    Targets,
    Offsets,
    Key,
    Prefix,
    Decay,
    State,
    New,
    DQuery,
    DKey,
    DPrefix,
    DDecay,
    H: tl.constexpr,
    DW: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    BQ: tl.constexpr,
    PRECISION: tl.constexpr,
):
    target = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    t, k = tl.arange(0, BT), tl.arange(0, BK)
    chunk = tl.load(Targets + target).to(tl.int64)
    begin, end = tl.load(Offsets + target), tl.load(Offsets + target + 1)
    slot = chunk * BT + t
    key = _rows(Key, slot, head, H, BK, BK)
    grad_key = tl.zeros((BT, BK), tl.float32)
    grad_prefix = tl.zeros((BT,), tl.float32)
    grad_decay = tl.zeros((BT, BT), tl.float32)
    for start in range(begin, end, BQ):
        read, valid, query, row, top, decay = _queries(
            Query, Positions, Prefix, Decay, start, end, chunk, head, H, BT, BK, BQ
        )
        grad_query = tl.zeros((BQ, BK), tl.float32)
        grad_score = tl.zeros((BQ, BT), tl.float32)
        for offset in range(0, DW, BC):
            c = offset + tl.arange(0, BC)
            columns = c[None, :] < DW
            grad = tl.load(
                Grad + (read[:, None] * H + head) * DW + c[None, :],
                valid[:, None] & columns,
                other=0.0,
            )
            state = tl.load(_tile(State, chunk, head, c, H, DW, BK), columns, other=0.0)
            new = tl.load(New + (slot[:, None] * H + head) * DW + c[None, :], columns, other=0.0)
            grad_query += _mm(grad, tl.trans(state), PRECISION)
            grad_score += _mm(grad, tl.trans(new), PRECISION)
        score = tl.load(
            Scores + (read[:, None] * H + head) * BT + t[None, :], valid[:, None], other=0.0
        )
        hit = (t[None, :] == row[:, None]) & valid[:, None]
        own = tl.sum(query * grad_query, 1)
        grad_prefix += tl.sum(tl.where(hit, own[:, None], 0.0), 0)
        selector = tl.trans(tl.where(hit, 1.0, 0.0))
        grad_decay += tl.dot(selector, grad_score * score, input_precision="ieee")
        grad_query *= top[:, None]
        weight = decay * grad_score
        grad_query += _mm(weight, key, PRECISION)
        grad_key += _mm(tl.trans(weight), query, PRECISION)
        tl.store(DQuery + (read[:, None] * H + head) * BK + k[None, :], grad_query, valid[:, None])
    pointer = DKey + (slot[:, None] * H + head) * BK + k[None, :]
    tl.store(pointer, tl.load(pointer) + grad_key)
    pointer = DPrefix + slot * H + head
    tl.store(pointer, tl.load(pointer) + grad_prefix)
    pointer = DDecay + ((chunk * H + head) * BT + t[:, None]) * BT + t[None, :]
    tl.store(pointer, tl.load(pointer) + grad_decay)


@tr.jit
def _chunk_states_backward(
    Key,
    Prefix,
    Decay,
    W,
    Bounds,
    DState,
    DNew,
    H: tl.constexpr,
    DW: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    PRECISION: tl.constexpr,
):
    version = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    t = tl.arange(0, BT)
    c = tl.program_id(2) * BC + tl.arange(0, BC)
    columns = c[None, :] < DW
    begin = tl.load(Bounds + version).to(tl.int64)
    end = tl.load(Bounds + version + 1).to(tl.int64)
    grad = tl.zeros((BK, BC), tl.float32)
    for index in range(begin, end):
        chunk = end - 1 - index + begin
        slot = chunk * BT + t
        pointer = _tile(DState, chunk, head, c, H, DW, BK)
        reads = tl.load(pointer, columns, other=0.0)
        tl.store(pointer, grad, columns)
        last = tl.load(Prefix + (chunk * BT + BT - 1) * H + head)
        tail = tl.load(Decay + ((chunk * H + head) * BT + BT - 1) * BT + t)
        key = _rows(Key, slot, head, H, BK, BK) * tail[:, None]
        pointer = DNew + (slot[:, None] * H + head) * DW + c[None, :]
        grad_new = tl.load(pointer, columns, other=0.0) + _mm(key, grad, PRECISION)
        tl.store(pointer, grad_new, columns)
        transposed = tl.trans(_rows(W, slot, head, H, BK, BK))
        grad = grad * last - _mm(transposed, grad_new, PRECISION) + reads


@tr.jit
def _chunk_columns_backward(
    Key,
    Decay,
    GradW,
    State,
    New,
    DState,
    DNew,
    DKey,
    DPrefix,
    DDecay,
    H: tl.constexpr,
    DW: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    PRECISION: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    t, k = tl.arange(0, BT), tl.arange(0, BK)
    slot = chunk * BT + t
    bottom = ((chunk * H + head) * BT + BT - 1) * BT + t
    tail = tl.load(Decay + bottom)
    grad_w = tl.zeros((BT, BK), tl.float32)
    grad_tail = tl.zeros((BT, BK), tl.float32)
    grad_last = tl.zeros((BK,), tl.float32)
    for offset in range(0, DW, BC):
        c = offset + tl.arange(0, BC)
        columns = c[None, :] < DW
        state = tl.load(_tile(State, chunk, head, c, H, DW, BK), columns, other=0.0)
        grad_state = tl.load(_tile(DState, chunk, head, c, H, DW, BK), columns, other=0.0)
        new = tl.load(New + (slot[:, None] * H + head) * DW + c[None, :], columns, other=0.0)
        grad_new = tl.load(DNew + (slot[:, None] * H + head) * DW + c[None, :], columns, other=0.0)
        grad_w -= _mm(grad_new, tl.trans(state), PRECISION)
        grad_tail += _mm(new, tl.trans(grad_state), PRECISION)
        grad_last += tl.sum(state * grad_state, 1)
    through = tl.sum(grad_tail * _rows(Key, slot, head, H, BK, BK), 1)
    tl.store(DDecay + bottom, tl.load(DDecay + bottom) + through)
    pointer = DPrefix + (chunk * BT + BT - 1) * H + head
    tl.store(pointer, tl.load(pointer) + tl.sum(grad_last, 0))
    pointer = (slot[:, None] * H + head) * BK + k[None, :]
    tl.store(DKey + pointer, tl.load(DKey + pointer) + grad_tail * tail[:, None])
    tl.store(GradW + pointer, grad_w)


@tr.jit
def _chunk_prepare_backward(
    Key,
    Value,
    Beta,
    Segments,
    Prefix,
    Decay,
    T,
    GradW,
    DNew,
    DKey,
    DPrefix,
    DDecay,
    DValue,
    DBeta,
    DAlpha,
    H: tl.constexpr,
    DV: tl.constexpr,
    DW: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BS: tl.constexpr,
    PRECISION: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    t = tl.arange(0, BT)
    slot = chunk * BT + t
    start = tl.load(Segments + slot)
    beta = tl.load(Beta + slot * H + head)
    prefix = tl.load(Prefix + slot * H + head)
    gate = beta * prefix
    square = ((chunk * H + head) * BT + t[:, None]) * BT + t[None, :]
    decay = tl.load(Decay + square)
    transposed = tl.load(T + ((chunk * H + head) * BT + t[None, :]) * BT + t[:, None])
    gram = tl.zeros((BT, BT), tl.float32)
    grad_inverse = tl.zeros((BT, BT), tl.float32)
    projected = tl.zeros((BT,), tl.float32)
    for offset in range(0, BK, BS):
        d = offset + tl.arange(0, BS)
        key = tl.load(Key + (slot[:, None] * H + head) * BK + d[None, :])
        grad_w = tl.load(GradW + (slot[:, None] * H + head) * BK + d[None, :])
        grad_x = _mm(transposed, grad_w, PRECISION)
        gram += _mm(key, tl.trans(key), PRECISION)
        grad_inverse += _mm(grad_w, tl.trans(gate[:, None] * key), PRECISION)
        projected += tl.sum(key * grad_x, 1)
    grad_beta = prefix * projected
    for offset in range(0, BV, BS):
        d = offset + tl.arange(0, BS)
        columns = d[None, :] < DV
        value = tl.load(Value + (slot[:, None] * H + head) * DV + d[None, :], columns, other=0.0)
        grad_u = tl.load(
            DNew + (slot[:, None] * H + head) * DW + BK + d[None, :], columns, other=0.0
        )
        grad_y = _mm(transposed, grad_u, PRECISION)
        grad_inverse += _mm(grad_u, tl.trans(beta[:, None] * value), PRECISION)
        grad_beta += tl.sum(value * grad_y, 1)
        tl.store(
            DValue + (slot[:, None] * H + head) * DV + d[None, :], beta[:, None] * grad_y, columns
        )
    lower = t[:, None] > t[None, :]
    chain = _mm(_mm(transposed, grad_inverse, PRECISION), transposed, PRECISION)
    grad_a = tl.where(lower, -chain, 0.0)
    factor = grad_a * gram * decay
    mixed = grad_a * decay * beta[:, None]
    for offset in range(0, BK, BS):
        d = offset + tl.arange(0, BS)
        pointer = DKey + (slot[:, None] * H + head) * BK + d[None, :]
        key = tl.load(Key + (slot[:, None] * H + head) * BK + d[None, :])
        grad_w = tl.load(GradW + (slot[:, None] * H + head) * BK + d[None, :])
        update = gate[:, None] * _mm(transposed, grad_w, PRECISION)
        update += _mm(mixed, key, PRECISION) + _mm(tl.trans(mixed), key, PRECISION)
        tl.store(pointer, tl.load(pointer) + update)
    grad_beta += tl.sum(factor, 1)
    grad_prefix = tl.load(DPrefix + slot * H + head) + beta * projected
    grad_decay = tl.load(DDecay + square) + grad_a * gram * beta[:, None]
    grad_decay = tl.where(start[:, None] == start[None, :], grad_decay, 0.0)
    above = t[:, None] >= 1
    shifted = tl.load(Decay + square - BT, above, other=0.0)
    previous = tl.load(Prefix + (slot - 1) * H + head, t > start, other=1.0)
    paths = tl.dot(grad_decay, tl.trans(shifted), input_precision="ieee")
    grad_alpha = tl.sum(decay * paths, 0) + previous * tl.sum(grad_prefix[:, None] * decay, 0)
    tl.store(DBeta + slot * H + head, grad_beta)
    tl.store(DAlpha + slot * H + head, grad_alpha)


def _check(*tensors: torch.Tensor) -> None:
    if any(not x.is_cuda or x.dtype != torch.float32 or not x.is_contiguous() for x in tensors):
        raise ValueError("Chunked GDN reads require contiguous CUDA FP32 inputs")


def _check_index(*tensors: torch.Tensor) -> None:
    if any(not x.is_cuda or x.dtype != torch.int32 or not x.is_contiguous() for x in tensors):
        raise ValueError("Chunked GDN plans require contiguous CUDA int32 metadata")


def chunk_states(
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    bounds: torch.Tensor,
    segments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(key, value, alpha, beta)
    _check_index(bounds, segments)
    slots, heads, width = key.shape
    dv = value.shape[-1]
    chunks, columns = slots // CHUNK, width + dv
    prefix, w, u = torch.empty_like(alpha), torch.empty_like(key), torch.empty_like(value)
    decay = key.new_empty(chunks, heads, CHUNK, CHUNK)
    inverse = key.new_empty(chunks, heads, CHUNK, CHUNK)
    state = key.new_empty(chunks, heads, width, columns)
    new = key.new_empty(slots, heads, columns)
    with torch.cuda.device(key.device):
        _chunk_decay[(chunks, heads)](alpha, segments, prefix, decay, heads, CHUNK, num_warps=8)
        _chunk_prepare[(chunks, heads)](
            key,
            value,
            beta,
            prefix,
            decay,
            w,
            u,
            inverse,
            heads,
            dv,
            CHUNK,
            width,
            tr.next_power_of_2(max(dv, 16)),
            **_launch("prepare", key.device),
        )
        _chunk_states[(len(bounds) - 1, heads, tr.cdiv(columns, COLUMNS))](
            key,
            prefix,
            decay,
            w,
            u,
            bounds,
            state,
            new,
            heads,
            dv,
            columns,
            CHUNK,
            width,
            COLUMNS,
            **_launch("states", key.device),
        )
    return prefix, decay, w, inverse, state, new


def chunk_read(
    query: torch.Tensor,
    positions: torch.Tensor,
    tiles: torch.Tensor,
    key: torch.Tensor,
    prefix: torch.Tensor,
    decay: torch.Tensor,
    state: torch.Tensor,
    new: torch.Tensor,
    save: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check(query, key, prefix, decay, state, new)
    _check_index(positions, tiles)
    reads, heads, width = query.shape
    columns = state.shape[-1]
    output = query.new_empty(reads, heads, columns)
    scores = query.new_empty((reads, heads, CHUNK) if save else (0,))
    with torch.cuda.device(query.device):
        _chunk_read[(len(tiles), heads)](
            query,
            positions,
            tiles,
            key,
            prefix,
            decay,
            state,
            new,
            output,
            scores,
            heads,
            columns,
            CHUNK,
            width,
            COLUMNS,
            QUERIES,
            save,
            **_launch("read", query.device),
        )
    return output, scores


def chunk_read_backward(
    query: torch.Tensor,
    scores: torch.Tensor,
    grad: torch.Tensor,
    positions: torch.Tensor,
    targets: torch.Tensor,
    offsets: torch.Tensor,
    key: torch.Tensor,
    prefix: torch.Tensor,
    decay: torch.Tensor,
    state: torch.Tensor,
    new: torch.Tensor,
    grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    _check(query, scores, grad, key, prefix, decay, state, new, *grads)
    _check_index(positions, targets, offsets)
    grad_state, grad_new, grad_key, grad_prefix, grad_decay = grads
    reads, heads, width = query.shape
    columns = state.shape[-1]
    grad_query = torch.empty_like(query)
    with torch.cuda.device(query.device):
        _chunk_read_state[(len(targets), heads, tr.cdiv(columns, COLUMNS))](
            query,
            scores,
            grad,
            positions,
            targets,
            offsets,
            prefix,
            decay,
            grad_state,
            grad_new,
            heads,
            columns,
            CHUNK,
            width,
            COLUMNS,
            QUERIES,
            **_launch("read_state", query.device),
        )
        _chunk_read_query[(len(targets), heads)](
            query,
            scores,
            grad,
            positions,
            targets,
            offsets,
            key,
            prefix,
            decay,
            state,
            new,
            grad_query,
            grad_key,
            grad_prefix,
            grad_decay,
            heads,
            columns,
            CHUNK,
            width,
            COLUMNS,
            QUERIES,
            **_launch("read_query", query.device),
        )
    return grad_query


def chunk_states_backward(
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    prefix: torch.Tensor,
    decay: torch.Tensor,
    w: torch.Tensor,
    inverse: torch.Tensor,
    state: torch.Tensor,
    new: torch.Tensor,
    bounds: torch.Tensor,
    segments: torch.Tensor,
    grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(key, value, beta, prefix, decay, w, inverse, state, new, *grads)
    _check_index(bounds, segments)
    grad_state, grad_new, grad_key, grad_prefix, grad_decay = grads
    slots, heads, width = key.shape
    dv, columns = value.shape[-1], state.shape[-1]
    grad_w = torch.empty_like(key)
    grad_value, grad_beta, grad_alpha = map(torch.empty_like, (value, beta, prefix))
    with torch.cuda.device(key.device):
        _chunk_states_backward[(len(bounds) - 1, heads, tr.cdiv(columns, COLUMNS))](
            key,
            prefix,
            decay,
            w,
            bounds,
            grad_state,
            grad_new,
            heads,
            columns,
            CHUNK,
            width,
            COLUMNS,
            **_launch("states_backward", key.device),
        )
        _chunk_columns_backward[(slots // CHUNK, heads)](
            key,
            decay,
            grad_w,
            state,
            new,
            grad_state,
            grad_new,
            grad_key,
            grad_prefix,
            grad_decay,
            heads,
            columns,
            CHUNK,
            width,
            COLUMNS,
            **_launch("columns_backward", key.device),
        )
        _chunk_prepare_backward[(slots // CHUNK, heads)](
            key,
            value,
            beta,
            segments,
            prefix,
            decay,
            inverse,
            grad_w,
            grad_new,
            grad_key,
            grad_prefix,
            grad_decay,
            grad_value,
            grad_beta,
            grad_alpha,
            heads,
            dv,
            columns,
            CHUNK,
            width,
            tr.next_power_of_2(max(dv, 16)),
            min(width, 64),
            **_launch("prepare_backward", key.device),
        )
    return grad_key, grad_value, grad_alpha, grad_beta
