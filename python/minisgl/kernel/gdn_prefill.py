"""Ragged prefill: use installed FLA kernels; scan only our affine summary here."""

import torch
import triton as tr
import triton.language as tl


def chunk_gdn(q, k, v, g, beta, initial, cu, chunk_indices, chunk_offsets,
              output_final_state=False, *, skip_empty_states=False):
    """With skip_empty_states, empty final-state entries are unspecified/unused."""
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.utils.constant import RCP_LN2
    from .gdn_fla_guards import kernels

    # Match FLA's public input_guard before entering lower-level kernels.
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)
    b, t, h, dk = k.shape
    hv, dv = v.shape[-2:]
    (cs, cs_body), (kk, kk_body), (wu, wu_body), (hs, hs_body), (o, o_body) = kernels()
    n, chunks = len(cu)-1, len(chunk_indices)
    meta = dict(T=t, N=n, BT=64, IS_VARLEN=True)
    gc = torch.empty_like(g, dtype=torch.float32)  # FLA cumsum's default output dtype.
    cs[(chunks, b*hv)](chunk_offsets, g, gc, RCP_LN2, cu, chunk_indices,
                       B=b, H=hv, REVERSE=False, BODY=cs_body, **meta)
    coefficients = k.new_zeros(b, t, hv, 64)  # FLA writes only the lower triangle.
    beta = beta.float()
    kk[(chunks, b*hv)](chunk_offsets, k, gc, beta, coefficients, cu, chunk_indices,
                       H=h, HV=hv, K=dk, BC=16, BODY=kk_body, **meta)
    w, u = k.new_empty(b, t, hv, dk), torch.empty_like(v)
    wu[(chunks, b*hv)](chunk_offsets, k, v, beta, w, u, coefficients, gc, cu, chunk_indices,
                       H=h, HV=hv, K=dk, V=dv, BK=64, BV=64, BODY=wu_body, **meta)
    # Only actual chunks are read by O, and H writes each before O runs.
    # Unused scratch can remain uninitialized; no capacity-wide memset needed.
    states = k.new_empty(b, chunks, hv, dk, dv)
    values = torch.empty_like(v)
    final = initial.new_empty(initial.shape) if output_final_state else None
    # Preserve final=initial for empty sequences by default. Joint capture
    # opts out: inactive final entries have no consumer/write pointer.
    hs[lambda m: (tr.cdiv(dv, m['BV']), n*hv)](
        k, u, w, values, gc, states, initial, final, cu, chunk_offsets,
        T=t, H=h, HV=hv, K=dk, V=dv, BT=64, STATE_V_FIRST=False,
        SKIP_EMPTY=skip_empty_states or final is None, BODY=hs_body)
    out = torch.empty_like(v)
    o[lambda m: (tr.cdiv(dv, m['BV']), chunks, b*hv)](
        chunk_offsets, q, k, values, states, gc, out, cu, chunk_indices, dk**-.5,
        T=t, N=n, H=h, HV=hv, K=dk, V=dv, BT=64, STATE_V_FIRST=False, BODY=o_body)
    out = out.masked_fill(torch.arange(t, device=q.device)[None, :, None, None] >= cu[-1], 0)
    return out, final


@tr.jit
def _affine_scan(Read, Write, Key, Value, Alpha, Beta, Cu,
                 H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                 BK: tl.constexpr, R: tl.constexpr, ONE_TOKEN: tl.constexpr, STATE: tl.constexpr):
    worker, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    oa, ob = tl.load(Write + worker * 3), tl.load(Write + worker * 3 + 1)
    if (oa == 0) & (ob == 0):
        return
    rows = tile * R + tl.arange(0, R)
    columns = tl.arange(0, BK)
    is_a = rows < DK
    local_row = tl.where(is_a, rows, rows - DK)
    nrows = tl.where(is_a, DK, DV)
    pa, pb = tl.load(Read + worker * 3), tl.load(Read + worker * 3 + 1)
    pointer = tl.where(is_a, pa, pb).to(tl.pointer_type(STATE))
    offset = (head * nrows[:, None] + local_row[:, None]) * DK + columns[None, :]
    mask = (rows[:, None] < DK + DV) & (columns[None, :] < DK)
    present = tl.where(is_a, pa != 0, pb != 0)
    state = tl.load(pointer[:, None] + offset, mask & present[:, None], other=0).to(tl.float32)
    identity = (local_row[:, None] == columns[None, :]) & is_a[:, None]
    state = tl.where(present[:, None], state, identity.to(tl.float32))
    if ONE_TOKEN:
        start, end = worker, worker + 1
    else:
        start, end = tl.load(Cu + worker), tl.load(Cu + worker + 1)
    for t in range(start, end):
        key = tl.load(Key + (t * H + head) * DK + columns, columns < DK, other=0)
        value = tl.load(Value + (t * H + head) * DV + local_row,
                        (rows < DK + DV) & ~is_a, other=0)
        alpha, beta = tl.load(Alpha + t * H + head), tl.load(Beta + t * H + head)
        # FP32 GEMV accumulation: independent strided partial sums, FMA within
        # each lane, then a tree reduction. Avoid materializing rounded products
        # before summation (the latter drifts from the existing GEMV recurrence).
        lanes = tl.arange(0, 8)
        partial = tl.full((R, 8), 0, tl.float32)
        for group in tl.static_range(BK // 8):
            indices = lanes + group * 8
            matrix = tl.gather(state, tl.broadcast_to(indices[None, :], (R, 8)), 1)
            vector = tl.gather(key, indices, 0)
            partial = tl.fma(matrix, vector[None, :], partial)
        # Fix the reduction tree explicitly: tl.sum can choose a different
        # within-thread tree when the compiler changes the matrix row layout.
        even, odd = tl.split(partial.reshape(R, 4, 2))
        e0, e1 = tl.split(even.reshape(R, 2, 2))
        o0, o1 = tl.split(odd.reshape(R, 2, 2))
        p0, p4 = tl.split(e0)
        p2, p6 = tl.split(e1)
        p1, p5 = tl.split(o0)
        p3, p7 = tl.split(o1)
        dot = ((p0 + p4) + (p2 + p6)) + ((p1 + p5) + (p3 + p7))
        # Explicit scalar RN instructions retain the eager Torch boundaries;
        # packed f32x2 lowering can otherwise contract the final multiply/subtract.
        state = tl.inline_asm_elementwise(
            "{ .reg .f32 lhs, ab, ak, erase, bv, write; "
            "mul.rn.f32 lhs, $1, $2; mul.rn.f32 ab, $2, $3; "
            "mul.rn.f32 ak, ab, $4; mul.rn.f32 erase, ak, $5; "
            "sub.rn.f32 lhs, lhs, erase; mul.rn.f32 bv, $3, $6; "
            "mul.rn.f32 write, bv, $5; add.rn.f32 $0, lhs, write; }",
            constraints="=f,f,f,f,f,f,f",
            args=[state, alpha, beta, dot[:, None], key[None, :], value[:, None]],
            dtype=tl.float32, is_pure=True, pack=1)
    destination = tl.where(is_a, oa, ob).to(tl.pointer_type(STATE))
    enabled = tl.where(is_a, oa != 0, ob != 0)
    tl.store(destination[:, None] + offset, state, mask & enabled[:, None])


def capture_affine_scan(read, write, key, value, alpha, beta, cu=None, *, l2norm_eps=1e-6,
                        state_dtype=torch.float32):
    """Pointer update; absent cu means one token per worker (decode)."""
    key = key.float()
    key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + l2norm_eps)
    value, alpha, beta = value.float().contiguous(), alpha.float().contiguous(), beta.float().contiguous()
    _, h, dk = key.shape
    dv = value.shape[-1]
    _affine_scan[(read.shape[0], h, tr.cdiv(dk + dv, 8))](
        read, write, key.contiguous(), value, alpha, beta, cu,
        h, dk, dv, max(8, tr.next_power_of_2(dk)), 8, cu is None,
        tl.bfloat16 if state_dtype == torch.bfloat16 else tl.float32,
        num_warps=2, enable_fp_fusion=False)
