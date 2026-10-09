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
    row, head = tl.program_id(0), tl.program_id(1)
    k, v = tl.arange(0, BK), tl.arange(0, BV)
    r = tl.load(Query + (row * H + head) * DK + k, k < DK, other=0.0) * DK**-0.5
    output = tl.full((BV,), 0, tl.float32)
    for depth in range(DEPTH):
        if SAVE:
            tl.store(Vectors + ((depth * N + row) * H + head) * DK + k, r, k < DK)
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
