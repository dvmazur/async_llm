from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.kernel.gdn_conv import causal_conv1d_silu
from minisgl.kernel.gdn_norm import gated_rmsnorm
from minisgl.layers import BaseOP, LinearReplicated
from minisgl.utils import nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Req

    from .config import ModelConfig


_AFFINE_CAPTURE_MAX_PADDING_RATIO = float(
    os.environ.get("MINISGL_GDN_AFFINE_CAPTURE_MAX_PADDING_RATIO", "0.0")
)

# Optional fast Gated DeltaNet kernels (flash-linear-attention, Triton).  When
# present they replace the pure-torch chunk/recurrent scans below — a large speedup
# for GDN prefill (and a modest one for decode).  Absent -> pure-torch fallback.
try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _fla_chunk,
    )
    from fla.ops.gated_delta_rule import (
        fused_recurrent_gated_delta_rule as _fla_recurrent,
    )
except Exception as exc:  # pragma: no cover - depends on the optional GPU backend
    _fla_chunk = None
    _fla_recurrent = None
    warnings.warn(
        "Flash Linear Attention kernels are unavailable; Qwen3.5 GDN is falling back "
        f"to the much slower pure-PyTorch implementation ({type(exc).__name__}: {exc}).",
        RuntimeWarning,
        stacklevel=2,
    )


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked delta-rule for prefill. Inputs are (B, T, H, D); returns
    (core_attn_out (B, T, H, Dv), final_recurrent_state (B, H, Dk, Dv)).

    ``initial_state`` (B, H, Dk, Dv), HF convention, seeds the recurrence — used by
    async-reasoning to thread a prior block's composed state into this prefill."""
    initial_dtype = query.dtype
    query = _l2norm(query, eps=1e-6)
    key = _l2norm(key, eps=1e-6)
    query, key, value, beta, g = (
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    )

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = (
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    )
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    if initial_state is None:
        last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
    else:
        last_recurrent_state = initial_state.to(value)
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-step (per-token) delta-rule for decode. Inputs are (B, 1, H, D);
    ``initial_state`` is (B, H, Dk, Dv). Returns (out (B, 1, H, Dv), new_state)."""
    initial_dtype = query.dtype
    query = _l2norm(query, eps=1e-6)
    key = _l2norm(key, eps=1e-6)
    query, key, value, beta, g = (
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    )

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = initial_state.to(value)

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# ---- dispatchers: fast fla kernels when available, else pure-torch ----
# Both take (B, T, H, D) inputs and l2-norm q/k internally.  State convention
# is [B,H,Dk,Dv] by default and [B,H,Dv,Dk] when state_v_first=True; input and
# returned final state always use the same convention.


def _chunk_delta(
    query,
    key,
    value,
    g,
    beta,
    initial_state=None,
    *,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
):
    if _fla_chunk is not None:
        return _fla_chunk(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
    initial_state_hf = (
        initial_state.transpose(-1, -2)
        if state_v_first and initial_state is not None
        else initial_state
    )
    if cu_seqlens is not None:
        # FLA's varlen representation is [1, sum(T_i), H, D] plus offsets and
        # one initial state per sequence.  Preserve the same API in the optional
        # pure-PyTorch fallback by evaluating those independent spans in order.
        assert query.shape[0] == 1, "varlen GDN expects a flattened batch dimension of one"
        offsets = (
            cu_seqlens_cpu.tolist()
            if cu_seqlens_cpu is not None
            else cu_seqlens.detach().cpu().tolist()
        )
        outputs = []
        final_states = []
        for worker, (start, end) in enumerate(zip(offsets, offsets[1:])):
            state = (
                None
                if initial_state_hf is None
                else initial_state_hf[worker : worker + 1]
            )
            output, final_state = _chunk_gated_delta_rule(
                query[:, start:end],
                key[:, start:end],
                value[:, start:end],
                g[:, start:end],
                beta[:, start:end],
                initial_state=state,
            )
            outputs.append(output)
            final_states.append(final_state)
        output = torch.cat(outputs, dim=1)
        final_state_hf = torch.cat(final_states, dim=0)
        final_state = final_state_hf.transpose(-1, -2) if state_v_first else final_state_hf
        return output, final_state
    output, final_state_hf = _chunk_gated_delta_rule(
        query, key, value, g, beta, initial_state=initial_state_hf
    )
    final_state = final_state_hf.transpose(-1, -2) if state_v_first else final_state_hf
    return output, final_state


def _capture_affine_summary_fla(
    ar,
    lin_idx: int,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    workers=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
) -> bool:
    """Capture a whole block summary with one augmented FLA recurrence.

    ``state=[A_hat; B_hat]`` and ``value=[0; v]`` make the ordinary GDN
    recurrence update the two affine components together. Returns ``False``
    when FLA or the shared-cache scan-state interface is unavailable, allowing
    the exact token-loop implementation to remain the portable fallback.
    """
    if (
        _fla_chunk is None
        or not key.is_cuda
        or not hasattr(ar, "affine_scan_initial_state")
        or not hasattr(ar, "store_affine_scan_state")
    ):
        return False
    d_k = key.shape[-1]
    d_v = value.shape[-1]
    num_heads = key.shape[-2]
    initial_state = ar.affine_scan_initial_state(
        lin_idx,
        num_heads=num_heads,
        d_k=d_k,
        d_v=d_v,
        workers=workers,
    )
    augmented_value = torch.cat(
        [value.new_zeros(*value.shape[:-1], d_k), value], dim=-1
    )
    _, final_state = _chunk_delta(
        key,
        key,
        augmented_value,
        g,
        beta,
        initial_state=initial_state,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    ar.store_affine_scan_state(lin_idx, final_state, d_k=d_k, workers=workers)
    return True


def _core_and_capture_affine_fla(
    ar,
    lin_idx: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    core_initial_state: Optional[torch.Tensor],
    workers=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
) -> Optional[torch.Tensor]:
    """Run the real GDN state and its block-affine summary in one FLA scan.

    The old fast capture path launched two scans over the same token rows:
    ``[S]`` with value ``v`` for model output, then ``[A; B]`` with value
    ``[0; v]`` for the reusable block summary.  A delta-rule scan updates state
    rows independently and shares ``k/g/beta``, so one state
    ``[S; A; B]`` with value ``[v; 0; v]`` is exactly the same recurrence.

    Returns only the model-output component.  ``None`` means that FLA is not
    available, in which case callers retain the portable two-stage fallback.
    """
    if (
        _fla_chunk is None
        or not key.is_cuda
        or not hasattr(ar, "affine_scan_initial_state")
        or not hasattr(ar, "store_affine_scan_state")
    ):
        return None

    d_k = key.shape[-1]
    d_v = value.shape[-1]
    num_heads = key.shape[-2]
    affine_initial_state = ar.affine_scan_initial_state(
        lin_idx,
        num_heads=num_heads,
        d_k=d_k,
        d_v=d_v,
        workers=workers,
    )
    if core_initial_state is None:
        core_initial_state = affine_initial_state.new_zeros(
            affine_initial_state.shape[0], num_heads, d_v, d_k
        )
    combined_initial_state = torch.cat(
        (core_initial_state, affine_initial_state), dim=-2
    )
    combined_value = torch.cat(
        (
            value,
            value.new_zeros(*value.shape[:-1], d_k),
            value,
        ),
        dim=-1,
    )
    combined_output, combined_final_state = _chunk_delta(
        query,
        key,
        combined_value,
        g,
        beta,
        initial_state=combined_initial_state,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    ar.store_affine_scan_state(
        lin_idx,
        combined_final_state[..., d_v:, :],
        d_k=d_k,
        workers=workers,
    )
    return combined_output[..., :d_v]


def _recurrent_delta(query, key, value, g, beta, initial_state, *, state_v_first=False):
    if isinstance(initial_state, list):
        if (_fla_recurrent is not None and query.is_cuda
                and state_v_first and query.shape[1] == 1):
            from minisgl.kernel.gdn_recurrent import recurrent_gdn_pointer

            return recurrent_gdn_pointer(query, key, value, g, beta, initial_state)
        # Portable/reference paths retain the exact dense recurrence and layout.
        initial_state = torch.cat(initial_state, dim=0)
    if _fla_recurrent is not None:
        return _fla_recurrent(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            state_v_first=state_v_first,
        )
    initial_state_hf = initial_state.transpose(-1, -2) if state_v_first else initial_state
    output, final_state_hf = _recurrent_gated_delta_rule(
        query, key, value, g, beta, initial_state_hf
    )
    final_state = final_state_hf.transpose(-1, -2) if state_v_first else final_state_hf
    return output, final_state


# ============================================================================
# Modules
# ============================================================================


def _gdn_gates_eager(a, b, A_log, dt_bias, *, beta_fp32=False):
    """Prefill rounds beta to b's dtype; fused-reference decode keeps FP32."""
    beta = b.float().sigmoid() if beta_fp32 else b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    return beta, g


# Compile a tensor-only function, not a bound method: all GDN layers share the
# graph despite having different parameter objects. Dynamic row counts cover
# decode and flattened/ragged prefill. Compilation is lazy; CPU stays eager.
# Distinct dtype/rank/stride/inference-mode combinations still need separate
# graphs. Allow those variants locally (including tiny-model regression tests)
# without changing Dynamo's global limit or silently falling back to eager.
_compiled_gdn_gates = torch.compile(
    _gdn_gates_eager, fullgraph=True, dynamic=True, recompile_limit=32
)


class _Conv1d(BaseOP):
    """Depthwise causal conv1d holder; weight key matches checkpoint `conv1d.weight`."""

    def __init__(self, conv_dim: int, kernel_size: int):
        self.weight = torch.empty(conv_dim, 1, kernel_size)


class _GatedRMSNorm(BaseOP):
    """RMSNorm over the last dim, then multiply by SiLU(gate). Plain (no +1) weight,
    matching Qwen3.5 `linear_attn.norm`. Computed in fp32 for stability."""

    def __init__(self, size: int, eps: float):
        self.weight = torch.empty(size)
        self._eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return gated_rmsnorm(x,self.weight,gate,self._eps)


class Qwen3_5GatedDeltaNet(BaseOP):
    """Qwen3.5 linear-attention (Gated DeltaNet) layer.

    Unlike Qwen3-Next, Qwen3.5 keeps the projections separate (`in_proj_qkv/z/a/b`),
    so no head-interleaving reordering is needed.
    """

    def __init__(self, config: ModelConfig, linear_idx: int):
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = config.linear_key_dim
        self.value_dim = config.linear_value_dim
        self.conv_dim = config.linear_conv_dim
        self.conv_kernel = config.linear_conv_kernel_dim
        self._n_rep = self.num_v_heads // self.num_k_heads
        self._lin_idx = linear_idx

        self.in_proj_qkv = LinearReplicated(config.hidden_size, self.conv_dim, has_bias=False)
        self.in_proj_z = LinearReplicated(config.hidden_size, self.value_dim, has_bias=False)
        self.in_proj_a = LinearReplicated(config.hidden_size, self.num_v_heads, has_bias=False)
        self.in_proj_b = LinearReplicated(config.hidden_size, self.num_v_heads, has_bias=False)
        self.conv1d = _Conv1d(self.conv_dim, self.conv_kernel)
        self.A_log = torch.empty(self.num_v_heads)
        self.dt_bias = torch.empty(self.num_v_heads)
        self.norm = _GatedRMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = LinearReplicated(self.value_dim, config.hidden_size, has_bias=False)

    def _gates(self, a: torch.Tensor, b: torch.Tensor, *, beta_fp32=False) -> Tuple[torch.Tensor, torch.Tensor]:
        impl = _compiled_gdn_gates if a.is_cuda else _gdn_gates_eager
        return impl(a, b, self.A_log, self.dt_bias, beta_fp32=beta_fp32)

    def _split_heads(self, qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # qkv: (..., conv_dim) -> q/k (..., num_k_heads, head_k_dim),
        # v (..., num_v_heads, head_v_dim).  Keeping arbitrary leading dims lets
        # the async-prefill path retain a real request batch instead of flattening
        # it into a sequence of one-request kernel launches.
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(*q.shape[:-1], self.num_k_heads, self.head_k_dim)
        k = k.reshape(*k.shape[:-1], self.num_k_heads, self.head_k_dim)
        v = v.reshape(*v.shape[:-1], self.num_v_heads, self.head_v_dim)
        if self._n_rep > 1:
            q = q.repeat_interleave(self._n_rep, dim=-2)
            k = k.repeat_interleave(self._n_rep, dim=-2)
        return q, k, v

    @nvtx_annotate("LinearAttn")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        # Async-reasoning shared-cache path: compose the initial recurrent state
        # from the worker's block chain and capture per-token affine updates.
        if ctx.gdn_ar is not None:
            if batch.is_mixed:
                return self._forward_ar_mixed(x, ctx.gdn_ar)
            if batch.is_prefill:
                return self._forward_ar_prefill(x, ctx.gdn_ar)
            return self._forward_ar_decode(x, ctx.gdn_ar)
        # Normal serving path: per-request state pool indexed by table_idx.
        gdn = ctx.gdn_state
        assert gdn is not None, "GDNStatePool not initialized for hybrid model"
        if not batch.is_prefill:
            return self._forward_decode(x, batch.reqs, gdn)
        if not batch.is_mixed:
            return self._forward_prefill(x, batch.reqs, gdn)
        # A mixed batch is two row segments: the leading extend reqs rebuild their
        # recurrent state by scan, the trailing `num_decode` reqs advance theirs by one
        # step.  Both index the state pool by `table_idx` and the segments hold disjoint
        # reqs, so neither can read state the other is writing.
        num_extend = batch.num_prefill
        split = sum(req.extend_len for req in batch.reqs[:num_extend])
        return torch.cat(
            [
                self._forward_prefill(x[:split], batch.reqs[:num_extend], gdn),
                self._forward_decode(x[split:], batch.reqs[num_extend:], gdn),
            ]
        )

    # --- async-reasoning prefill: one block per request, compose prior + capture ---
    def _project_ar(self, x):
        return tuple(proj.forward(x) for proj in (
            self.in_proj_qkv, self.in_proj_z, self.in_proj_a, self.in_proj_b
        ))

    def _ar_output(self, core, project_output=True):
        rows = core.reshape(-1, self.value_dim)
        return self.out_proj.forward(rows) if project_output else rows

    def _forward_ar_mixed(self, x, ar) -> torch.Tensor:
        """Share GEMMs while preserving prefill-before-decode state updates."""
        projected = self._project_ar(x)
        split = ar.split
        prefill = self._forward_ar_prefill(
            x[:split], ar.prefill,
            projected=tuple(t[:split] for t in projected), project_output=False,
        )
        # Compose decode initial states only AFTER the prefill has captured
        # this layer's A/B and conv state. Keep the whole decode group together.
        decode = self._forward_ar_decode(
            x[split:], ar.decode,
            projected=tuple(t[split:] for t in projected), project_output=False,
        )
        return self.out_proj.forward(torch.cat((prefill, decode), dim=0))

    def _forward_ar_prefill(self, x, ar, *, projected=None, project_output=True) -> torch.Tensor:
        lin = self._lin_idx
        prior_conv = ar.prior_conv_states(lin)  # (R, conv_dim, k) | None
        # fp32 initial state: the delta-rule kernels upcast to fp32 anyway, and
        # fp32 composition avoids bf16 error compounding across long chains.
        initial_state = ar.compose_initial_recurrent_state(
            lin, dtype=torch.float32, state_v_first=True
        )  # (R,H,dv,dk)|None
        segments = ar.prefill_segments
        if segments is None or len(segments) == 1:
            return self._forward_ar_prefill_one(
                x, ar, 0, prior_conv, initial_state,
                projected=projected, project_output=project_output,
            )

        assert segments and all(length > 0 for length in segments), (
            "prefill_segments must contain positive lengths"
        )
        assert sum(segments) == x.shape[0], "prefill_segments do not cover the batch"

        # The scheduler commonly presents many jobs with exactly the same number
        # of rows (most importantly, batched one-token extends).  Preserve that
        # request dimension through projections, convolution, FLA and affine
        # capture.  Previously this path executed the complete GDN layer once per
        # request, turning e.g. batch 64 into 64 GEMVs and 64 recurrent launches.
        if segments[0] == 1 and all(length == 1 for length in segments):
            return self._forward_ar_prefill_equal_length(
                x, ar, len(segments), segments[0], prior_conv, initial_state,
                projected=projected, project_output=project_output,
            )

        return self._forward_ar_prefill_varlen(
            x, ar, segments, prior_conv, initial_state,
            projected=projected, project_output=project_output,
        )

    def _forward_ar_prefill_one(
        self, x, ar, w: int, prior_conv, initial_state, *, projected=None, project_output=True
    ) -> torch.Tensor:
        """One request's prefill rows, reading/writing worker slot *w*."""
        qkv, z, a, b = self._project_ar(x) if projected is None else projected
        return self._forward_ar_prefill_one_projected(
            qkv, z, a, b, ar, w, prior_conv, initial_state, project_output=project_output
        )

    def _forward_ar_prefill_one_projected(
        self, qkv, z, a, b, ar, w: int, prior_conv, initial_state, *, project_output=True
    ) -> torch.Tensor:
        """Reference suffix for one request after the projections are available."""
        lin = self._lin_idx
        length = qkv.shape[0]
        qkv2, new_conv_state = self._forward_ar_prefill_conv_one(qkv, w, prior_conv)

        q, kk, v = self._split_heads(qkv2)  # each (L, num_v_heads, d)
        beta, g = self._gates(a, b)
        worker_initial_state = (
            None if initial_state is None else initial_state[w : w + 1]
        )
        core = None
        if length > 1:
            core = _core_and_capture_affine_fla(
                ar,
                lin,
                q.unsqueeze(0),
                kk.unsqueeze(0),
                v.unsqueeze(0),
                g.unsqueeze(0),
                beta.unsqueeze(0),
                core_initial_state=worker_initial_state,
                workers=[w],
            )
        if core is None:
            core, _ = _chunk_delta(
                q.unsqueeze(0),
                kk.unsqueeze(0),
                v.unsqueeze(0),
                g.unsqueeze(0),
                beta.unsqueeze(0),
                initial_state=worker_initial_state,
                state_v_first=True,
            )
            ar.capture_token_affines(
                lin,
                kk.unsqueeze(0),
                v.unsqueeze(0),
                g.exp().unsqueeze(0),
                beta.unsqueeze(0),
                workers=[w],
            )
        ar.set_conv_states(lin, new_conv_state, workers=[w])

        core = core.reshape(length, self.num_v_heads, self.head_v_dim)
        core = self.norm.forward(core, z.reshape(length, self.num_v_heads, self.head_v_dim))
        return self._ar_output(core, project_output)

    def _forward_ar_prefill_conv_one(self, qkv, w: int, prior_conv):
        """Causal convolution for one varlen span; returns rows and its new window."""
        prior = None if prior_conv is None else prior_conv[w : w + 1]
        rows, window = causal_conv1d_silu(qkv.unsqueeze(0), self.conv1d.weight, prior)
        return rows.squeeze(0), window

    def _forward_ar_prefill_varlen(
        self, x, ar, segments, prior_conv, initial_state, *, projected=None, project_output=True
    ) -> torch.Tensor:
        """Run ragged request scans in one FLA call using cumulative offsets."""
        lin = self._lin_idx
        total = x.shape[0]
        qkv, z, a, b = self._project_ar(x) if projected is None else projected

        convolved = []
        conv_states = []
        offset = 0
        for worker, length in enumerate(segments):
            rows, state = self._forward_ar_prefill_conv_one(
                qkv[offset : offset + length], worker, prior_conv
            )
            convolved.append(rows)
            conv_states.append(state)
            offset += length
        qkv2 = torch.cat(convolved, dim=0)
        q, kk, v = self._split_heads(qkv2)
        beta, g = self._gates(a, b)
        core = _core_and_capture_affine_fla(
            ar,
            lin,
            q.unsqueeze(0),
            kk.unsqueeze(0),
            v.unsqueeze(0),
            g.unsqueeze(0),
            beta.unsqueeze(0),
            core_initial_state=initial_state,
            cu_seqlens=ar.prefill_cu_seqlens,
            cu_seqlens_cpu=ar.prefill_cu_seqlens_cpu,
        )
        if core is not None:
            ar.set_conv_states(lin, torch.cat(conv_states, dim=0))
            core = core.reshape(total, self.num_v_heads, self.head_v_dim)
            core = self.norm.forward(core, z.reshape(total, self.num_v_heads, self.head_v_dim))
            return self._ar_output(core, project_output)

        core, _ = _chunk_delta(
            q.unsqueeze(0),
            kk.unsqueeze(0),
            v.unsqueeze(0),
            g.unsqueeze(0),
            beta.unsqueeze(0),
            initial_state=initial_state,
            state_v_first=True,
            cu_seqlens=ar.prefill_cu_seqlens,
            cu_seqlens_cpu=ar.prefill_cu_seqlens_cpu,
        )

        # Portable no-FLA fallback: fold mutable-block summaries together when
        # padding does not inflate
        # the work excessively. Padding is an exact affine identity update
        # (alpha=1, beta=0), but every padded row still consumes GPU work. For a
        # highly ragged frontier, keep the old per-request capture while retaining
        # the batched projections, convolution and recurrent scan above.
        key_spans = []
        value_spans = []
        alpha_spans = []
        beta_spans = []
        offset = 0
        alpha = g.exp()
        for length in segments:
            span = slice(offset, offset + length)
            key_spans.append(kk[span])
            value_spans.append(v[span])
            alpha_spans.append(alpha[span])
            beta_spans.append(beta[span])
            offset += length
        padded_rows = len(segments) * max(segments)
        actual_rows = sum(segments)
        if padded_rows <= _AFFINE_CAPTURE_MAX_PADDING_RATIO * actual_rows:
            ar.capture_token_affines(
                lin,
                torch.nn.utils.rnn.pad_sequence(key_spans, batch_first=True),
                torch.nn.utils.rnn.pad_sequence(value_spans, batch_first=True),
                torch.nn.utils.rnn.pad_sequence(
                    alpha_spans, batch_first=True, padding_value=1.0
                ),
                torch.nn.utils.rnn.pad_sequence(beta_spans, batch_first=True),
            )
        else:
            for worker, (key_span, value_span, alpha_span, beta_span) in enumerate(
                zip(key_spans, value_spans, alpha_spans, beta_spans)
            ):
                ar.capture_token_affines(
                    lin,
                    key_span.unsqueeze(0),
                    value_span.unsqueeze(0),
                    alpha_span.unsqueeze(0),
                    beta_span.unsqueeze(0),
                    workers=[worker],
                )
        ar.set_conv_states(lin, torch.cat(conv_states, dim=0))

        core = core.reshape(total, self.num_v_heads, self.head_v_dim)
        core = self.norm.forward(core, z.reshape(total, self.num_v_heads, self.head_v_dim))
        return self._ar_output(core, project_output)

    def _forward_ar_prefill_equal_length(
        self, x, ar, workers: int, length: int, prior_conv, initial_state,
        *, projected=None, project_output=True,
    ) -> torch.Tensor:
        """Batched async prefill for ``workers`` sequences of the same length."""
        lin = self._lin_idx
        k = self.conv_kernel

        # One GEMM per projection over all request rows, rather than one GEMV/GEMM
        # per request.  The flattened scheduler layout is request-major, so view
        # restores [W, L, ...] without a copy.
        qkv, z, a, b = self._project_ar(x) if projected is None else projected
        qkv = qkv.reshape(workers, length, self.conv_dim)
        z = z.reshape(workers, length, self.value_dim)
        a = a.reshape(workers, length, self.num_v_heads)
        b = b.reshape(workers, length, self.num_v_heads)

        qkv2, new_conv_state = causal_conv1d_silu(qkv, self.conv1d.weight, prior_conv)

        q, kk, v = self._split_heads(qkv2)
        beta, g = self._gates(a, b)
        affine_captured = False
        if length == 1:
            if initial_state is None:
                initial_state = torch.zeros(
                    workers,
                    self.num_v_heads,
                    self.head_v_dim,
                    self.head_k_dim,
                    device=x.device,
                    dtype=torch.float32,
                )
            core, _ = _recurrent_delta(
                q,
                kk,
                v,
                g,
                beta,
                initial_state,
                state_v_first=True,
            )
        else:  # Kept for direct tests; production dispatch currently uses L=1 only.
            core = _core_and_capture_affine_fla(
                ar,
                lin,
                q,
                kk,
                v,
                g,
                beta,
                core_initial_state=initial_state,
            )
            affine_captured = core is not None
            if core is None:
                core, _ = _chunk_delta(
                    q,
                    kk,
                    v,
                    g,
                    beta,
                    initial_state=initial_state,
                    state_v_first=True,
                )

        if not affine_captured:
            ar.capture_token_affines(lin, kk, v, g.exp(), beta)
        ar.set_conv_states(lin, new_conv_state)

        core = core.reshape(workers * length, self.num_v_heads, self.head_v_dim)
        z = z.reshape(workers * length, self.num_v_heads, self.head_v_dim)
        core = self.norm.forward(core, z)
        return self._ar_output(core, project_output)

    # --- async-reasoning decode: W workers, one token each, batched ---
    def _forward_ar_decode(self, x, ar, *, projected=None, project_output=True) -> torch.Tensor:
        lin = self._lin_idx
        k = self.conv_kernel
        n = x.shape[0]  # num workers

        qkv, z, a, b = self._project_ar(x) if projected is None else projected

        prior_conv = ar.prior_conv_states(lin)
        qkv2, new_conv_state = causal_conv1d_silu(qkv.unsqueeze(1), self.conv1d.weight, prior_conv)
        qkv2 = qkv2.squeeze(1)

        q, kk, v = self._split_heads(qkv2)  # (W, num_v_heads, d)
        beta, g = self._gates(a, b, beta_fp32=True)
        successor_ticket = None
        # The low-precision no-FLA fallback normalizes q/k before upcasting,
        # unlike affine capture's FP32 normalization. Its returned state must
        # not replace recomposition across tokens. Preserve that legacy path.
        successor_compatible = _fla_recurrent is not None or kk.dtype == torch.float32
        if successor_compatible and hasattr(ar, "begin_decode_state"):
            initial_state, successor_ticket = ar.begin_decode_state(
                lin, materialize=(_fla_recurrent is None or not x.is_cuda)
            )
        else:
            initial_state = ar.compose_initial_recurrent_state(
                lin, dtype=torch.float32, state_v_first=True
            )  # (W,H,dv,dk)|None
        if initial_state is None:
            initial_state = torch.zeros(
                n,
                self.num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
                device=x.device,
                dtype=torch.float32,
            )
        core, final_state = _recurrent_delta(
            q.unsqueeze(1),
            kk.unsqueeze(1),
            v.unsqueeze(1),
            g.unsqueeze(1),
            beta.unsqueeze(1),
            initial_state,
            state_v_first=True,
        )

        ar.capture_token_affines(
            lin, kk.unsqueeze(1), v.unsqueeze(1), g.exp().unsqueeze(1), beta.unsqueeze(1)
        )
        ar.set_conv_states(lin, new_conv_state)

        if successor_ticket is not None:
            ar.finish_decode_state(lin, final_state, successor_ticket)

        core = core.reshape(n, self.num_v_heads, self.head_v_dim)
        core = self.norm.forward(core, z.reshape(n, self.num_v_heads, self.head_v_dim))
        return self._ar_output(core, project_output)

    # --- prefill: one chunked pass per request, write final state to pool ---
    def _forward_prefill(self, x, reqs: List[Req], gdn) -> torch.Tensor:
        out = torch.empty(
            x.shape[0], self.out_proj.full_output_size, dtype=x.dtype, device=x.device
        )
        k = self.conv_kernel
        offset = 0
        for req in reqs:
            assert req.cached_len == 0, (
                "Qwen3.5 hybrid prefill does not support prefix-cache reuse for linear "
                "layers yet; run with prefix caching disabled."
            )
            length = req.extend_len
            seg = x[offset : offset + length]
            offset += length

            qkv = self.in_proj_qkv.forward(seg)  # (L, conv_dim)
            z = self.in_proj_z.forward(seg)  # (L, value_dim)
            a = self.in_proj_a.forward(seg)
            b = self.in_proj_b.forward(seg)

            qkv, conv_state = causal_conv1d_silu(qkv.unsqueeze(0), self.conv1d.weight)
            qkv = qkv.squeeze(0)
            gdn.conv_state[self._lin_idx, req.table_idx] = conv_state.squeeze(0)

            q, kk, v = self._split_heads(qkv)
            beta, g = self._gates(a, b)
            core, rec_state = _chunk_delta(
                q.unsqueeze(0), kk.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0), beta.unsqueeze(0)
            )
            gdn.recurrent_state[self._lin_idx, req.table_idx] = rec_state.squeeze(0).float()

            core = core.reshape(length, self.num_v_heads, self.head_v_dim)
            core = self.norm.forward(core, z.reshape(length, self.num_v_heads, self.head_v_dim))
            out[offset - length : offset] = self.out_proj.forward(
                core.reshape(length, self.value_dim)
            )
        return out

    # --- decode: single recurrent step, batched across requests ---
    def _forward_decode(self, x, reqs: List[Req], gdn) -> torch.Tensor:
        table_idx = torch.tensor([req.table_idx for req in reqs], device=x.device, dtype=torch.long)
        n = x.shape[0]
        qkv = self.in_proj_qkv.forward(x)  # (N, conv_dim)
        z = self.in_proj_z.forward(x)
        a = self.in_proj_a.forward(x)
        b = self.in_proj_b.forward(x)

        # causal conv update: append new token, roll the conv window
        conv_state = gdn.conv_state[self._lin_idx, table_idx]  # (N, conv_dim, k)
        qkv, new_conv_state = causal_conv1d_silu(qkv.unsqueeze(1), self.conv1d.weight, conv_state)
        gdn.conv_state[self._lin_idx, table_idx] = new_conv_state
        qkv = qkv.squeeze(1)

        q, kk, v = self._split_heads(qkv)
        beta, g = self._gates(a, b, beta_fp32=True)
        rec_state = gdn.recurrent_state[self._lin_idx, table_idx]  # (N, num_v_heads, Dk, Dv)
        core, new_state = _recurrent_delta(
            q.unsqueeze(1),
            kk.unsqueeze(1),
            v.unsqueeze(1),
            g.unsqueeze(1),
            beta.unsqueeze(1),
            rec_state,
        )
        gdn.recurrent_state[self._lin_idx, table_idx] = new_state.float()

        core = core.reshape(n, self.num_v_heads, self.head_v_dim)
        core = self.norm.forward(core, z.reshape(n, self.num_v_heads, self.head_v_dim))
        return self.out_proj.forward(core.reshape(n, self.value_dim))


__all__ = ["Qwen3_5GatedDeltaNet"]
