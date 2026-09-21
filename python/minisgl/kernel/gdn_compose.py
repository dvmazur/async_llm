"""Active-node FP32 compose for block-owned affine summaries.

Adapted from planned/gdn_kernels.py compose_first/compose_level (cuda-graphs,
922df78). Keep its IEEE dot and level-count guard, but read existing A/B
pointer tables instead of requiring a pool. Missing A is identity, missing B
is zero; an absent per-layer state is NOT an inactive trie node.
"""

import triton as tr
import triton.language as tl


@tr.jit
def _compose_first(Pointers, Counts, Out, Initial, Sinks, Workers, Empty,
                   H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                   N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if Initial is not None:
        # Empty/padded rows have no producing node. Other rows are written
        # exactly once by their terminal producer, so no competing memset.
        if tl.load(Empty + row):
            tl.store(Initial + row * N + offsets, 0., offsets < N)
    if row < tl.load(Counts):
        address = tl.load(Pointers + row * 2 + 1)
        b = address.to(tl.pointer_type(tl.float32))
        values = tl.load(b + offsets, (address != 0) & (offsets < N), other=0.)
        tl.store(Out + row * N + offsets, values, offsets < N)
        if Initial is not None:
            dst = (offsets // (DV * DK)) * DK * DV + (offsets % DK) * DV + (offsets // DK) % DV
            for i in range(tl.load(Sinks + row), tl.load(Sinks + row + 1)):
                worker = tl.load(Workers + i)
                tl.store(Initial + worker * N + dst, values, offsets < N)


@tr.jit(do_not_specialize=["LEVEL"])
def _compose_level(Pointers, Parents, Counts, Previous, Out, Initial, Sinks, Workers, LEVEL,
                   H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile, group = tl.program_id(0), tl.program_id(1)
    row, head = group // H, group % H
    # The graph launches a fixed capacity grid. Padding does no state loads,
    # matrix arithmetic or stores, even when a whole level is empty.
    if row < tl.load(Counts + LEVEL):
        parent = tl.load(Parents + row)
        a_address = tl.load(Pointers + row * 2)
        b_address = tl.load(Pointers + row * 2 + 1)
        a = a_address.to(tl.pointer_type(tl.float32)) + head * DK * DK
        b = b_address.to(tl.pointer_type(tl.float32)) + head * DV * DK
        previous = Previous + (parent * H + head) * DV * DK
        m = (tile // tl.cdiv(DK, BN)) * BM + tl.arange(0, BM)
        n = (tile % tl.cdiv(DK, BN)) * BN + tl.arange(0, BN)
        offsets = m[:, None] * DK + n[None, :]
        mask = (m[:, None] < DV) & (n[None, :] < DK)
        if a_address != 0:
            ks = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), tl.float32)
            for start in range(0, DK, BK):
                k = start + ks
                p = tl.load(previous + m[:, None] * DK + k[None, :],
                            (m[:, None] < DV) & (k[None, :] < DK), other=0.)
                av = tl.load(a + k[:, None] * DK + n[None, :],
                             (k[:, None] < DK) & (n[None, :] < DK), other=0.)
                acc += tl.dot(p, av, input_precision="ieee")
        else:
            acc = tl.load(previous + offsets, mask, other=0.)
        value = acc + tl.load(b + offsets, mask & (b_address != 0), other=0.)
        tl.store(Out + (row * H + head) * DV * DK + offsets, value, mask)
        if Initial is not None:
            for i in range(tl.load(Sinks + row), tl.load(Sinks + row + 1)):
                worker = tl.load(Workers + i)
                dst = Initial + (worker * H + head) * DK * DV + n[None, :] * DV + m[:, None]
                tl.store(dst, value, mask)


def compose_first(pointers, counts, out, initial=None, sinks=None, workers=None, empty=None):
    """Write only real first-level nodes, in block [W,H,Dv,Dk] layout."""
    n = out.numel() // out.shape[0]
    _, h, dv, dk = out.shape
    _compose_first[(out.shape[0], tr.cdiv(n, 512))](
        pointers, counts, out, initial, sinks, workers, empty, h, dk, dv, n, 512)


def compose_level(pointers, parents, counts, level, previous, out,
                  initial=None, sinks=None, sink_workers=None):
    """Out[node] = Previous[parent[node]] @ A[node] + B[node].

    Previous and out must not overlap: nodes can share parents and CUDA thread
    blocks have no global barrier within this kernel. Adjacent levels are
    separate ordered launches; the caller supplies ping-pong frontier buffers.
    """
    workers, h, dv, dk = out.shape
    _compose_level[(tr.cdiv(dv, 64) * tr.cdiv(dk, 32), workers * h)](
        pointers, parents, counts, previous, out, initial, sinks, sink_workers,
        level, h, dk, dv, 64, 32, 16)
