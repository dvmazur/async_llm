# Adapted from FLA 0.5.2 ops/gated_delta_rule/fused_recurrent.py.
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""FLA one-token, v-first recurrence with a pointer per initial-state row.

Only initial-state addressing differs from FLA's corresponding specialization:
FP32 arithmetic, q/k normalization, reductions, launch tiling, and packed output
are retained. No new cache and no changes to which state may be reused.
"""
import triton
import triton.language as tl

from fla.ops.utils.op import exp


@triton.jit
def gdn_recurrent_pointer_kernel(
    q, k, v, g, beta, state_ptrs, o, ht,
    scale: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k, mask_v = o_k < K, o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    parent = tl.load(state_ptrs + i_n).to(tl.pointer_type(tl.float32))
    p_h0 = parent + i_hv * K * V + o_v[:, None] * K + o_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    b_q = tl.load(q + (i_n * H + i_h) * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(k + (i_n * H + i_h) * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(v + i_nh * V + o_v, mask=mask_v, other=0).to(tl.float32)
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale
    b_beta = tl.load(beta + i_nh).to(tl.float32)
    b_g = tl.load(g + i_nh).to(tl.float32)
    b_h *= exp(b_g)
    b_v = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(o + i_nh * V + o_v, b_o.to(o.dtype.element_ty), mask=mask_v)
    p_ht = ht + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h, mask=mask_h)
