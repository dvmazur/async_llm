# Recurrence arithmetic adapted from FLA 0.5.2 fused_recurrent.py.
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Bounded, slot-addressed GDN kernels for the planned runtime.

Compose publishes terminal states directly. Decode recurrence and affine
capture share a kernel and never allocate/store an unused final recurrent S.
All writes use the existing affine pool; no fusion-specific output pool.
"""
import os
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

FAST_EXP = tl.constexpr(os.environ.get("FLA_USE_FAST_OPS", "0") == "1")


@triton.jit
def recurrence_exp(x):
    # Match installed FLA's selected math without making FLA import mandatory.
    if FAST_EXP:
        return libdevice.fast_expf(x.to(tl.float32))
    return tl.exp(x.to(tl.float32))


@triton.jit
def capture_row_sum(x,ROWS:tl.constexpr,KD:tl.constexpr):
    # Old pointer capture assigns one value/lane, reduces within each warp,
    # then reduces warp partials. Keep that tree in the fused CTA layout;
    # a normal tl.sum picks a different tree when recurrence uses eight rows.
    cols=tl.arange(0,KD)
    for shift in tl.static_range(min(5,triton.next_power_of_2(KD).bit_length()-1)-1,-1,-1):
        x+=tl.gather(x,tl.broadcast_to((cols^(1<<shift))[None,:],(ROWS,KD)),axis=1)
    for shift in tl.static_range(triton.next_power_of_2(KD).bit_length()-2,4,-1):
        x+=tl.gather(x,tl.broadcast_to((cols^(1<<shift))[None,:],(ROWS,KD)),axis=1)
    return tl.sum(tl.where(cols[None,:]==0,x,0.),axis=1)


@triton.jit(do_not_specialize=["LAYER"])
def compose_first(
    pool, slots, counts, terminal_nodes, sink_offsets, sink_workers,
    frontier, initial, LAYER, SLOTS: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    off = tile * BLOCK + tl.arange(0, BLOCK)
    mask = off < D * D
    # Roots and inactive rows have no producing trie node. Initialize them
    # here without a separate zero kernel or touching persistent state.
    if tl.load(terminal_nodes + row) < 0:
        tl.store(initial + (row * H + head) * D * D + off, 0., mask)
    if row < tl.load(counts):
        slot = tl.load(slots + row).to(tl.int64)
        base = ((LAYER.to(tl.int64) * SLOTS + slot) * 2 * H + H + head) * D * D
        value = tl.load(pool + base + off, mask, other=0.)
        tl.store(frontier + (row * H + head) * D * D + off, value, mask)
        lo, hi = tl.load(sink_offsets + row), tl.load(sink_offsets + row + 1)
        for index in range(lo, hi):
            worker = tl.load(sink_workers + index)
            tl.store(initial + (worker * H + head) * D * D + off, value, mask)


@triton.jit(do_not_specialize=["LAYER", "DEPTH"])
def compose_level(
    pool, slots, parents, counts, sink_offsets, sink_workers, previous, output, initial,
    LAYER, DEPTH, WIDTH: tl.constexpr,
    SLOTS: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    tile, group = tl.program_id(0), tl.program_id(1)
    row, head = group // H, group % H
    if row < tl.load(counts + DEPTH):
        node = DEPTH * WIDTH + row
        slot = tl.load(slots + node).to(tl.int64)
        parent = tl.load(parents + node)
        base = ((LAYER.to(tl.int64) * SLOTS + slot) * 2 * H + head) * D * D
        m = (tile // tl.cdiv(D, BN)) * BM + tl.arange(0, BM)
        n = (tile % tl.cdiv(D, BN)) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for start in range(0, D, BK):
            k = start + ks
            p = tl.load(previous + (parent * H + head) * D * D + m[:, None] * D + k[None, :],
                        (m[:, None] < D) & (k[None, :] < D), other=0.)
            a = tl.load(pool + base + k[:, None] * D + n[None, :],
                        (k[:, None] < D) & (n[None, :] < D), other=0.)
            acc += tl.dot(p, a, input_precision="ieee")
        mask = (m[:, None] < D) & (n[None, :] < D)
        offsets = m[:, None] * D + n[None, :]
        value = acc + tl.load(pool + base + H * D * D + offsets, mask, other=0.)
        tl.store(output + (row * H + head) * D * D + offsets, value, mask)
        lo, hi = tl.load(sink_offsets + node), tl.load(sink_offsets + node + 1)
        for index in range(lo, hi):
            worker = tl.load(sink_workers + index)
            tl.store(initial + (worker * H + head) * D * D + offsets, value, mask)


@triton.jit(do_not_specialize=["LAYER"])
def recurrent_capture(
    q, k, v, g, beta, alpha, initial, pool, writes, fresh, active, output,
    conv_pool, conv_window,
    LAYER, SLOTS: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
    D: tl.constexpr, KD: tl.constexpr, ROWS: tl.constexpr,
    HAS_CONV: tl.constexpr, C: tl.constexpr, CK: tl.constexpr,
    QSTRIDE: tl.constexpr, KSTRIDE: tl.constexpr, VSTRIDE: tl.constexpr,
    prefill_recipe, PREFILL_SPECIAL: tl.constexpr,
):
    row_tile, group = tl.program_id(0), tl.program_id(1)
    worker, head = group // HV, group % HV
    rows = row_tile * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, KD)
    mask = (rows[:, None] < D) & (cols[None, :] < D)
    recipe = 2
    if PREFILL_SPECIAL:
        recipe = tl.load(prefill_recipe)
        if recipe == 0:
            return
    if tl.load(active + worker):
        khead = head // (HV // H)
        raw_k = tl.load(k + worker * KSTRIDE + khead * D + cols, cols < D, other=0.).to(tl.float32)
        raw_q = tl.load(q + worker * QSTRIDE + khead * D + cols, cols < D, other=0.).to(tl.float32)
        value = tl.load(v + worker * VSTRIDE + head * D + rows, rows < D, other=0.).to(tl.float32)
        kval = raw_k / tl.sqrt(tl.sum(raw_k * raw_k) + 1.e-6)
        qval = (raw_q / tl.sqrt(tl.sum(raw_q * raw_q) + 1.e-6)) * (D ** -0.5)
        gate = tl.load(g + group).to(tl.float32)
        bet = tl.load(beta + group).to(tl.float32)
        if recipe != 1:
            # Retain FLA's recurrence exp and capture's separately supplied alpha.
            old_s = tl.load(initial + group * D * D + rows[:, None] * D + cols[None, :], mask, other=0.)
            decayed = old_s * recurrence_exp(gate)
            innovation = bet * (value - tl.sum(decayed * kval[None, :], axis=1))
            new_s = decayed + innovation[:, None] * kval[None, :]
            result = tl.sum(new_s * qval[None, :], axis=1)
            tl.store(output + group * D + rows, result.to(output.dtype.element_ty), rows < D)
        # Final S is intentionally not materialized: no successor cache consumer.
        # All compose readers have completed. These independent full-row updates
        # can safely overwrite their own A/B rows in the existing pool.
        slot = tl.load(writes + worker).to(tl.int64)
        is_fresh = tl.load(fresh + worker)
        base = ((LAYER.to(tl.int64) * SLOTS + slot) * 2 * HV + head) * D * D
        offsets = rows[:, None] * D + cols[None, :]
        A = tl.load(pool + base + offsets, mask & ~is_fresh, other=0.)
        A += tl.where(is_fresh & (rows[:, None] == cols[None, :]), 1., 0.)
        B = tl.load(pool + base + HV * D * D + offsets, mask & ~is_fresh, other=0.)
        # Preserve capture's rsqrt convention instead of silently equating it
        # with recurrence's divide-by-sqrt normalization.
        capture_k = raw_k * tl.rsqrt(tl.sum(raw_k * raw_k) + 1.e-6)
        if PREFILL_SPECIAL:
            a = libdevice.exp(gate)
        else:
            a = tl.load(alpha + group).to(tl.float32)
        ak = capture_row_sum(A * capture_k[None, :],ROWS,KD)
        bk = capture_row_sum(B * capture_k[None, :],ROWS,KD)
        new_A = a * A - a * bet * ak[:, None] * capture_k[None, :]
        new_B = a * B - a * bet * bk[:, None] * capture_k[None, :] + bet * value[:, None] * capture_k[None, :]
        tl.store(pool + base + offsets, new_A, mask)
        tl.store(pool + base + HV * D * D + offsets, new_B, mask)
        if HAS_CONV:
            # Conv read has already completed for the entire phase. Each CTA
            # publishes a disjoint piece of its worker's existing raw-window
            # workspace; no extra buffer or standalone store kernel is needed.
            cta = head * tl.cdiv(D, ROWS) + row_tile
            for start in range(cta * 128, C * CK, HV * tl.cdiv(D, ROWS) * 128):
                offset = start + tl.arange(0, 128)
                raw = tl.load(conv_window + worker * C * CK + offset, offset < C*CK, other=0.)
                tl.store(conv_pool + (LAYER.to(tl.int64)*SLOTS + slot)*C*CK + offset,
                         raw, offset < C*CK)
    else:
        if not PREFILL_SPECIAL:
            tl.store(output + group * D + rows, 0., rows < D)
