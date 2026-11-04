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


def read_paths(query, key, value, alpha, beta, a, b, scale, u, z, w, lookup, paths, save):
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


@tr.jit
def _mm(a, b):
    return tl.dot(a, b, input_precision="ieee")


@tr.jit
def _rows(Pointer, slot, head, H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    d = tl.arange(0, BD)
    return tl.load(Pointer + (slot[:, None] * H + head) * D + d[None, :], d[None, :] < D, other=0.0)


@tr.jit
def _tile(Pointer, chunk, head, c, H: tl.constexpr, DW: tl.constexpr, BK: tl.constexpr):
    k = tl.arange(0, BK)
    return Pointer + ((chunk * H + head) * BK + k[:, None]) * DW + c[None, :]


@tr.jit
def _queries(
    Query,
    Positions,
    start,
    end,
    head,
    cum,
    H: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BQ: tl.constexpr,
):
    t, k = tl.arange(0, BT), tl.arange(0, BK)
    read = start + tl.arange(0, BQ)
    valid = read < end
    read = read.to(tl.int64)
    position = tl.load(Positions + read, valid, other=1)
    query = tl.load(Query + (read[:, None] * H + head) * BK + k[None, :], valid[:, None], other=0.0)
    mask = t[None, :] < position[:, None]
    top = tl.sum(tl.where(t[None, :] == position[:, None] - 1, cum[None, :], 0.0), 1)
    decay = tl.where(mask, tl.exp(tl.where(mask, top[:, None] - cum[None, :], 0.0)), 0.0)
    return read, valid, query, position, top, decay


@tr.jit
def _chunk_prepare(
    Key,
    Value,
    LogAlpha,
    Beta,
    Cum,
    W,
    U,
    T,
    H: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    t, v = tl.arange(0, BT), tl.arange(0, BV)
    slot = chunk * BT + t
    key = _rows(Key, slot, head, H, BK, BK)
    beta = tl.load(Beta + slot * H + head)
    cum = tl.cumsum(tl.load(LogAlpha + slot * H + head), 0)
    tl.store(Cum + slot * H + head, cum)
    lower = t[:, None] > t[None, :]
    decay = tl.where(lower, tl.exp(tl.where(lower, cum[:, None] - cum[None, :], 0.0)), 0.0)
    a = -beta[:, None] * _mm(key, tl.trans(key)) * decay
    for index in range(1, BT):
        row = tl.sum(tl.where(t[:, None] == index, a, 0.0), 0)
        row += tl.sum(row[:, None] * a, 0)
        a = tl.where(t[:, None] == index, row[None, :], a)
    a += tl.where(t[:, None] == t[None, :], 1.0, 0.0)
    tl.store(T + ((chunk * H + head) * BT + t[:, None]) * BT + t[None, :], a)
    w = _mm(a, (beta * tl.exp(cum))[:, None] * key)
    tl.store(W + (slot[:, None] * H + head) * BK + tl.arange(0, BK)[None, :], w)
    u = _mm(a, beta[:, None] * _rows(Value, slot, head, H, DV, BV))
    tl.store(U + (slot[:, None] * H + head) * DV + v[None, :], u, v[None, :] < DV)


@tr.jit
def _chunk_states(
    Key,
    Cum,
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
        cum = tl.load(Cum + slot * H + head)
        last = tl.load(Cum + (chunk * BT + BT - 1) * H + head)
        u = tl.load(
            U + (slot[:, None] * H + head) * DV + c[None, :] - BK,
            (c[None, :] >= BK) & columns,
            other=0.0,
        )
        new = u - _mm(_rows(W, slot, head, H, BK, BK), state)
        tl.store(New + (slot[:, None] * H + head) * DW + c[None, :], new, columns)
        key = _rows(Key, slot, head, H, BK, BK) * tl.exp(last - cum)[:, None]
        state = state * tl.exp(last) + _mm(tl.trans(key), new)


@tr.jit
def _chunk_read(
    Query,
    Positions,
    Tiles,
    Key,
    Cum,
    State,
    New,
    Out,
    Weights,
    H: tl.constexpr,
    DW: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    BQ: tl.constexpr,
    SAVE: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    t = tl.arange(0, BT)
    chunk = tl.load(Tiles + tile * 3).to(tl.int64)
    start, end = tl.load(Tiles + tile * 3 + 1), tl.load(Tiles + tile * 3 + 2)
    slot = chunk * BT + t
    cum = tl.load(Cum + slot * H + head)
    read, valid, query, _, top, decay = _queries(
        Query, Positions, start, end, head, cum, H, BT, BK, BQ
    )
    weight = _mm(query, tl.trans(_rows(Key, slot, head, H, BK, BK))) * decay
    if SAVE:
        tl.store(Weights + (read[:, None] * H + head) * BT + t[None, :], weight, valid[:, None])
    query *= tl.exp(top)[:, None]
    for offset in range(0, DW, BC):
        c = offset + tl.arange(0, BC)
        columns = c[None, :] < DW
        state = tl.load(_tile(State, chunk, head, c, H, DW, BK), columns, other=0.0)
        new = tl.load(New + (slot[:, None] * H + head) * DW + c[None, :], columns, other=0.0)
        out = _mm(query, state) + _mm(weight, new)
        tl.store(Out + (read[:, None] * H + head) * DW + c[None, :], out, valid[:, None] & columns)


@tr.jit
def _chunk_read_state(
    Query,
    Weights,
    Grad,
    Positions,
    Targets,
    Offsets,
    Cum,
    DState,
    DNew,
    H: tl.constexpr,
    DW: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    BQ: tl.constexpr,
):
    target = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    t = tl.arange(0, BT)
    c = tl.program_id(2) * BC + tl.arange(0, BC)
    columns = c[None, :] < DW
    chunk = tl.load(Targets + target).to(tl.int64)
    begin, end = tl.load(Offsets + target), tl.load(Offsets + target + 1)
    slot = chunk * BT + t
    cum = tl.load(Cum + slot * H + head)
    grad_state = tl.zeros((BK, BC), tl.float32)
    grad_new = tl.zeros((BT, BC), tl.float32)
    for start in range(begin, end, BQ):
        read, valid, query, _, top, _ = _queries(
            Query, Positions, start, end, head, cum, H, BT, BK, BQ
        )
        weight = tl.load(
            Weights + (read[:, None] * H + head) * BT + t[None, :], valid[:, None], other=0.0
        )
        grad = tl.load(
            Grad + (read[:, None] * H + head) * DW + c[None, :], valid[:, None] & columns, other=0.0
        )
        grad_state += _mm(tl.trans(query * tl.exp(top)[:, None]), grad)
        grad_new += _mm(tl.trans(weight), grad)
    pointer = _tile(DState, chunk, head, c, H, DW, BK)
    tl.store(pointer, tl.load(pointer, columns) + grad_state, columns)
    pointer = DNew + (slot[:, None] * H + head) * DW + c[None, :]
    tl.store(pointer, tl.load(pointer, columns) + grad_new, columns)


@tr.jit
def _chunk_read_query(
    Query,
    Weights,
    Grad,
    Positions,
    Targets,
    Offsets,
    Key,
    Cum,
    State,
    New,
    DQuery,
    DKey,
    DCum,
    H: tl.constexpr,
    DW: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    BQ: tl.constexpr,
):
    target = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    t, k = tl.arange(0, BT), tl.arange(0, BK)
    chunk = tl.load(Targets + target).to(tl.int64)
    begin, end = tl.load(Offsets + target), tl.load(Offsets + target + 1)
    slot = chunk * BT + t
    key = _rows(Key, slot, head, H, BK, BK)
    cum = tl.load(Cum + slot * H + head)
    grad_key = tl.zeros((BT, BK), tl.float32)
    grad_cum = tl.zeros((BT,), tl.float32)
    for start in range(begin, end, BQ):
        read, valid, query, position, top, decay = _queries(
            Query, Positions, start, end, head, cum, H, BT, BK, BQ
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
            grad_query += _mm(grad, tl.trans(state))
            grad_score += _mm(grad, tl.trans(new))
        grad_query *= tl.exp(top)[:, None]
        part = grad_score * tl.load(
            Weights + (read[:, None] * H + head) * BT + t[None, :], valid[:, None], other=0.0
        )
        weight = decay * grad_score
        own = tl.sum(query * grad_query, 1) + tl.sum(part, 1)
        grad_cum += tl.sum(tl.where(t[None, :] == position[:, None] - 1, own[:, None], 0.0), 0)
        grad_cum -= tl.sum(part, 0)
        grad_query += _mm(weight, key)
        grad_key += _mm(tl.trans(weight), query)
        tl.store(DQuery + (read[:, None] * H + head) * BK + k[None, :], grad_query, valid[:, None])
    pointer = DKey + (slot[:, None] * H + head) * BK + k[None, :]
    tl.store(pointer, tl.load(pointer) + grad_key)
    tl.store(DCum + slot * H + head, tl.load(DCum + slot * H + head) + grad_cum)


@tr.jit
def _chunk_states_backward(
    Key,
    Cum,
    W,
    Bounds,
    DState,
    DNew,
    H: tl.constexpr,
    DW: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
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
        cum = tl.load(Cum + slot * H + head)
        last = tl.load(Cum + (chunk * BT + BT - 1) * H + head)
        key = _rows(Key, slot, head, H, BK, BK) * tl.exp(last - cum)[:, None]
        pointer = DNew + (slot[:, None] * H + head) * DW + c[None, :]
        grad_new = tl.load(pointer, columns, other=0.0) + _mm(key, grad)
        tl.store(pointer, grad_new, columns)
        grad = (
            grad * tl.exp(last) - _mm(tl.trans(_rows(W, slot, head, H, BK, BK)), grad_new) + reads
        )


@tr.jit
def _chunk_columns_backward(
    Key,
    Cum,
    GradW,
    State,
    New,
    DState,
    DNew,
    DKey,
    DCum,
    H: tl.constexpr,
    DW: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    t, k = tl.arange(0, BT), tl.arange(0, BK)
    slot = chunk * BT + t
    cum = tl.load(Cum + slot * H + head)
    last = tl.sum(tl.where(t == BT - 1, cum, 0.0), 0)
    tail = tl.exp(last - cum)
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
        grad_w -= _mm(grad_new, tl.trans(state))
        grad_tail += _mm(new, tl.trans(grad_state))
        grad_last += tl.sum(state * grad_state, 1)
    through = tl.sum(grad_tail * _rows(Key, slot, head, H, BK, BK), 1) * tail
    grad_last = tl.sum(grad_last, 0) * tl.exp(last) + tl.sum(through, 0)
    grad_cum = tl.load(DCum + slot * H + head) - through + tl.where(t == BT - 1, grad_last, 0.0)
    tl.store(DCum + slot * H + head, grad_cum)
    pointer = (slot[:, None] * H + head) * BK + k[None, :]
    tl.store(DKey + pointer, tl.load(DKey + pointer) + grad_tail * tail[:, None])
    tl.store(GradW + pointer, grad_w)


@tr.jit
def _chunk_prepare_backward(
    Key,
    Value,
    Beta,
    Cum,
    T,
    GradW,
    DNew,
    DKey,
    DCum,
    DValue,
    DBeta,
    DLogAlpha,
    H: tl.constexpr,
    DV: tl.constexpr,
    COLUMNS: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BS: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    t = tl.arange(0, BT)
    slot = chunk * BT + t
    beta = tl.load(Beta + slot * H + head)
    cum = tl.load(Cum + slot * H + head)
    gate = beta * tl.exp(cum)
    transposed = tl.load(T + ((chunk * H + head) * BT + t[None, :]) * BT + t[:, None])
    gram = tl.zeros((BT, BT), tl.float32)
    grad_inverse = tl.zeros((BT, BT), tl.float32)
    projected = tl.zeros((BT,), tl.float32)
    for offset in range(0, BK, BS):
        d = offset + tl.arange(0, BS)
        pointer = DKey + (slot[:, None] * H + head) * BK + d[None, :]
        key = tl.load(Key + (slot[:, None] * H + head) * BK + d[None, :])
        grad_w = tl.load(GradW + (slot[:, None] * H + head) * BK + d[None, :])
        grad_x = _mm(transposed, grad_w)
        gram += _mm(key, tl.trans(key))
        grad_inverse += _mm(grad_w, tl.trans(gate[:, None] * key))
        projected += tl.sum(key * grad_x, 1)
        tl.store(pointer, tl.load(pointer) + gate[:, None] * grad_x)
    grad_beta = tl.exp(cum) * projected
    for offset in range(0, BV, BS):
        d = offset + tl.arange(0, BS)
        columns = d[None, :] < DV
        value = tl.load(Value + (slot[:, None] * H + head) * DV + d[None, :], columns, other=0.0)
        grad_u = tl.load(
            DNew + (slot[:, None] * H + head) * COLUMNS + BK + d[None, :], columns, other=0.0
        )
        grad_y = _mm(transposed, grad_u)
        grad_inverse += _mm(grad_u, tl.trans(beta[:, None] * value))
        grad_beta += tl.sum(value * grad_y, 1)
        tl.store(
            DValue + (slot[:, None] * H + head) * DV + d[None, :], beta[:, None] * grad_y, columns
        )
    lower = t[:, None] > t[None, :]
    decay = tl.where(lower, tl.exp(tl.where(lower, cum[:, None] - cum[None, :], 0.0)), 0.0)
    grad_a = tl.where(lower, -_mm(_mm(transposed, grad_inverse), transposed), 0.0)
    factor = grad_a * gram * decay
    weighted = factor * beta[:, None]
    mixed = grad_a * decay * beta[:, None]
    for offset in range(0, BK, BS):
        d = offset + tl.arange(0, BS)
        pointer = DKey + (slot[:, None] * H + head) * BK + d[None, :]
        key = tl.load(Key + (slot[:, None] * H + head) * BK + d[None, :])
        tl.store(pointer, tl.load(pointer) + _mm(mixed, key) + _mm(tl.trans(mixed), key))
    grad_beta += tl.sum(factor, 1)
    grad_cum = tl.sum(weighted, 1) - tl.sum(weighted, 0) + gate * projected
    grad_cum += tl.load(DCum + slot * H + head)
    grad_log_alpha = tl.sum(grad_cum, 0) - tl.cumsum(grad_cum, 0) + grad_cum
    tl.store(DBeta + slot * H + head, grad_beta)
    tl.store(DLogAlpha + slot * H + head, grad_log_alpha)


def _check(*tensors):
    if any(not x.is_cuda or x.dtype != torch.float32 for x in tensors):
        raise ValueError("Chunked GDN reads require CUDA FP32 inputs")


def chunk_states(key, value, log_alpha, beta, bounds):
    _check(key, value, log_alpha, beta)
    slots, heads, width = key.shape
    dv = value.shape[-1]
    chunks, columns = slots // CHUNK, width + dv
    cum, w, u = torch.empty_like(log_alpha), torch.empty_like(key), torch.empty_like(value)
    inverse = key.new_empty(chunks, heads, CHUNK, CHUNK)
    state = key.new_empty(chunks, heads, width, columns)
    new = key.new_empty(slots, heads, columns)
    with torch.cuda.device(key.device):
        _chunk_prepare[(chunks, heads)](
            key,
            value,
            log_alpha,
            beta,
            cum,
            w,
            u,
            inverse,
            heads,
            dv,
            CHUNK,
            width,
            tr.next_power_of_2(max(dv, 16)),
            num_warps=4,
            num_stages=1,
        )
        _chunk_states[(len(bounds) - 1, heads, tr.cdiv(columns, COLUMNS))](
            key,
            cum,
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
            num_warps=4,
            num_stages=1,
        )
    return cum, w, inverse, state, new


def chunk_read(query, positions, tiles, key, cum, state, new, save):
    reads, heads, width = query.shape
    columns = state.shape[-1]
    output = query.new_empty(reads, heads, columns)
    weights = query.new_empty((reads, heads, CHUNK) if save else (0,))
    with torch.cuda.device(query.device):
        _chunk_read[(len(tiles), heads)](
            query,
            positions,
            tiles,
            key,
            cum,
            state,
            new,
            output,
            weights,
            heads,
            columns,
            CHUNK,
            width,
            COLUMNS,
            QUERIES,
            save,
            num_warps=4,
            num_stages=1,
        )
    return output, weights


def chunk_read_backward(
    query, weights, grad, positions, targets, offsets, key, cum, state, new, grads
):
    grad_state, grad_new, grad_key, grad_cum = grads
    reads, heads, width = query.shape
    columns = state.shape[-1]
    grad_query = torch.empty_like(query)
    with torch.cuda.device(query.device):
        _chunk_read_state[(len(targets), heads, tr.cdiv(columns, COLUMNS))](
            query,
            weights,
            grad,
            positions,
            targets,
            offsets,
            cum,
            grad_state,
            grad_new,
            heads,
            columns,
            CHUNK,
            width,
            COLUMNS,
            QUERIES,
            num_warps=4,
            num_stages=1,
        )
        _chunk_read_query[(len(targets), heads)](
            query,
            weights,
            grad,
            positions,
            targets,
            offsets,
            key,
            cum,
            state,
            new,
            grad_query,
            grad_key,
            grad_cum,
            heads,
            columns,
            CHUNK,
            width,
            COLUMNS,
            QUERIES,
            num_warps=8,
            num_stages=1,
        )
    return grad_query


def chunk_states_backward(key, value, beta, cum, w, inverse, state, new, bounds, grads):
    grad_state, grad_new, grad_key, grad_cum = grads
    slots, heads, width = key.shape
    dv, columns = value.shape[-1], state.shape[-1]
    grad_w = torch.empty_like(key)
    grad_value, grad_beta, grad_log_alpha = map(torch.empty_like, (value, beta, cum))
    with torch.cuda.device(key.device):
        _chunk_states_backward[(len(bounds) - 1, heads, tr.cdiv(columns, COLUMNS))](
            key,
            cum,
            w,
            bounds,
            grad_state,
            grad_new,
            heads,
            columns,
            CHUNK,
            width,
            COLUMNS,
            num_warps=4,
            num_stages=1,
        )
        _chunk_columns_backward[(slots // CHUNK, heads)](
            key,
            cum,
            grad_w,
            state,
            new,
            grad_state,
            grad_new,
            grad_key,
            grad_cum,
            heads,
            columns,
            CHUNK,
            width,
            COLUMNS,
            num_warps=8,
            num_stages=1,
        )
        _chunk_prepare_backward[(slots // CHUNK, heads)](
            key,
            value,
            beta,
            cum,
            inverse,
            grad_w,
            grad_new,
            grad_key,
            grad_cum,
            grad_value,
            grad_beta,
            grad_log_alpha,
            heads,
            dv,
            columns,
            CHUNK,
            width,
            tr.next_power_of_2(max(dv, 16)),
            min(width, 64),
            num_warps=8,
            num_stages=1,
        )
    return grad_key, grad_value, grad_log_alpha, grad_beta
