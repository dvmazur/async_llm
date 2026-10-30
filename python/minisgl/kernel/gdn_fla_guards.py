"""Capacity guards calling installed FLA JIT bodies, without vendored math.

The original launch configuration candidates are reused by local autotuners.
Nothing patches FLA's functions, global caches, or site-packages. Bind lazily
so importing the no-FLA runtime does not require FLA to be installed.
"""
from functools import cache

import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=['T'])
def _fla_guard_cumsum(Count, s, o, scale, cu_seqlens, chunk_indices, T, N: tl.constexpr,
            B: tl.constexpr, H: tl.constexpr, BT: tl.constexpr,
            REVERSE: tl.constexpr, IS_VARLEN: tl.constexpr, BODY: tl.constexpr):
    if tl.program_id(0) < tl.load(Count + N):
        BODY(s, o, scale, cu_seqlens, chunk_indices, T, B, H, BT,
             REVERSE, True, IS_VARLEN, False)


@tr.jit(do_not_specialize=['T'])
def _fla_guard_kkt(Count, k, g, beta, A, cu_seqlens, chunk_indices, T, N: tl.constexpr,
         H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
         BC: tl.constexpr, BK: tl.constexpr, IS_VARLEN: tl.constexpr, BODY: tl.constexpr):
    if tl.program_id(0) < tl.load(Count + N):
        BODY(k, g, beta, A, cu_seqlens, chunk_indices, T,
             H, HV, K, BT, BC, BK, True, IS_VARLEN)


@tr.jit(do_not_specialize=['T'])
def _fla_guard_wu(Count, k, v, beta, w, u, A, g, cu_seqlens, chunk_indices, T, N: tl.constexpr,
        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
        IS_VARLEN: tl.constexpr, BODY: tl.constexpr):
    if tl.program_id(0) < tl.load(Count + N):
        BODY(k, v, beta, w, u, A, g, cu_seqlens, chunk_indices, T,
             H, HV, K, V, BT, BK, BV, True, IS_VARLEN)


@tr.jit(do_not_specialize=['T'])
def _fla_guard_h(k, v, w, v_new, g, h, h0, ht, cu_seqlens, chunk_offsets, T,
        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BT: tl.constexpr, BV: tl.constexpr, STATE_V_FIRST: tl.constexpr,
        SKIP_EMPTY: tl.constexpr, BODY: tl.constexpr):
    worker = tl.program_id(1) // HV
    active = tl.load(cu_seqlens + worker + 1) > tl.load(cu_seqlens + worker)
    if active or not SKIP_EMPTY:
        BODY(k, v, w, v_new, g, None, h, h0, ht, cu_seqlens, chunk_offsets, T,
             H, HV, K, V, BT, BV, True, False, True, ht is not None, True, STATE_V_FIRST, True)


@tr.jit(do_not_specialize=['T'])
def _fla_guard_o(Count, q, k, v, h, g, o, cu_seqlens, chunk_indices, scale, T, N: tl.constexpr,
        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
        STATE_V_FIRST: tl.constexpr, BODY: tl.constexpr):
    if tl.program_id(1) < tl.load(Count + N):
        BODY(q, k, v, h, g, None, o, cu_seqlens, chunk_indices, scale, T,
             H, HV, K, V, BT, BK, BV, True, False, STATE_V_FIRST, True)


@cache
def kernels():
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    from fla.ops.common.chunk_o import chunk_fwd_kernel_o
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
    from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd_kernel
    from fla.ops.utils.cumsum import chunk_local_cumsum_scalar_kernel
    from fla.utils import autotune_cache_kwargs
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    result = []
    for original, guard in (
        (chunk_local_cumsum_scalar_kernel, _fla_guard_cumsum),
        (chunk_gated_delta_rule_fwd_kkt_solve_kernel, _fla_guard_kkt),
        (recompute_w_u_fwd_kernel, _fla_guard_wu),
        (chunk_gated_delta_rule_fwd_kernel_h_blockdim64, _fla_guard_h),
        (chunk_fwd_kernel_o, _fla_guard_o),
    ):
        body, tuner = original, None
        while not isinstance(body, JITFunction):
            if isinstance(body, Autotuner):
                tuner = body
            body = body.fn
        if tuner is None or tuner.reset_to_zero or tuner.restore_value or tuner.early_config_prune:
            raise RuntimeError('Unsupported FLA launch contract for guarded GDN')
        # Independent cache: count VALUES must not specialize the graph/kernel.
        # Use FLA's supported tiles/warp counts, including architecture limits.
        launch = tr.autotune(configs=tuner.configs, key=tuner.keys, **autotune_cache_kwargs)(guard)
        result.append((launch, body))
    return tuple(result)
