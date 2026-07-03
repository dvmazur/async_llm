from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, LinearReplicated
from minisgl.utils import nvtx_annotate

if TYPE_CHECKING:
    from .config import ModelConfig

# Optional fast Gated DeltaNet kernels (flash-linear-attention, Triton).  When
# present they replace the pure-torch chunk/recurrent scans below — a large speedup
# for GDN prefill (and a modest one for decode).  Absent -> pure-torch fallback.
try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _fla_chunk,
        fused_recurrent_gated_delta_rule as _fla_recurrent,
    )
except Exception:  # pragma: no cover - fla is optional
    _fla_chunk = None
    _fla_recurrent = None


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
# Both take (B, T, H, D) inputs and l2-norm q/k internally; return
# (core (B, T, H, Dv), final_state (B, H, Dk, Dv)).


def _chunk_delta(query, key, value, g, beta, initial_state=None):
    if _fla_chunk is not None:
        return _fla_chunk(
            query, key, value, g=g, beta=beta, initial_state=initial_state,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
    return _chunk_gated_delta_rule(query, key, value, g, beta, initial_state=initial_state)


def _recurrent_delta(query, key, value, g, beta, initial_state):
    if _fla_recurrent is not None:
        return _fla_recurrent(
            query, key, value, g=g, beta=beta, initial_state=initial_state,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
    return _recurrent_gated_delta_rule(query, key, value, g, beta, initial_state)


# ============================================================================
# Modules
# ============================================================================


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
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self._eps)
        x = self.weight.to(torch.float32) * x
        x = x * F.silu(gate.to(torch.float32))
        return x.to(dtype)


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

    def _gates(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        return beta, g

    def _split_heads(
        self, qkv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # qkv: (T, conv_dim) -> q/k (T, num_k_heads, head_k_dim), v (T, num_v_heads, head_v_dim)
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(q.shape[0], self.num_k_heads, self.head_k_dim)
        k = k.reshape(k.shape[0], self.num_k_heads, self.head_k_dim)
        v = v.reshape(v.shape[0], self.num_v_heads, self.head_v_dim)
        if self._n_rep > 1:
            q = q.repeat_interleave(self._n_rep, dim=1)
            k = k.repeat_interleave(self._n_rep, dim=1)
        return q, k, v

    @nvtx_annotate("LinearAttn")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        # Async-reasoning shared-cache path: compose the initial recurrent state
        # from the worker's block chain and capture per-token affine updates.
        if ctx.gdn_ar is not None:
            if batch.is_prefill:
                return self._forward_ar_prefill(x, ctx.gdn_ar)
            return self._forward_ar_decode(x, ctx.gdn_ar)
        # Normal serving path: per-request state pool indexed by table_idx.
        gdn = ctx.gdn_state
        assert gdn is not None, "GDNStatePool not initialized for hybrid model"
        if batch.is_prefill:
            return self._forward_prefill(x, batch, gdn)
        return self._forward_decode(x, batch, gdn)

    # --- async-reasoning prefill: single-worker block, compose prior + capture ---
    def _forward_ar_prefill(self, x, ar) -> torch.Tensor:
        lin = self._lin_idx
        k = self.conv_kernel
        length = x.shape[0]

        qkv = self.in_proj_qkv.forward(x)  # (L, conv_dim)
        z = self.in_proj_z.forward(x)
        a = self.in_proj_a.forward(x)
        b = self.in_proj_b.forward(x)

        conv_in = qkv.transpose(0, 1).unsqueeze(0)  # (1, conv_dim, L)
        prior_conv = ar.prior_conv_states(lin)  # (1, conv_dim, k) or None
        if prior_conv is not None:
            ctx_tail = prior_conv[..., -(k - 1):]  # (1, conv_dim, k-1)
            full_input = torch.cat([ctx_tail, conv_in], dim=-1)  # (1, conv_dim, k-1+L)
            conv_out = F.silu(
                F.conv1d(full_input, self.conv1d.weight, groups=self.conv_dim, padding=k - 1)
            )
            qkv2 = conv_out[..., k - 1 : k - 1 + length]
            new_conv_state = full_input[..., -k:]
        else:
            conv_out = F.conv1d(
                conv_in, self.conv1d.weight, groups=self.conv_dim, padding=k - 1
            )[..., :length]
            qkv2 = F.silu(conv_out)
            pad = k - length
            new_conv_state = F.pad(conv_in, (pad, 0)) if pad >= 0 else conv_in[..., -k:]
        qkv2 = qkv2.squeeze(0).transpose(0, 1)  # (L, conv_dim)

        q, kk, v = self._split_heads(qkv2)  # each (L, num_v_heads, d)
        beta, g = self._gates(a, b)
        # fp32 initial state: the delta-rule kernels upcast to fp32 anyway, and
        # fp32 composition avoids bf16 error compounding across long chains.
        initial_state = ar.compose_initial_recurrent_state(lin, dtype=torch.float32)  # (1,H,dk,dv)|None
        core, _ = _chunk_delta(
            q.unsqueeze(0), kk.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0), beta.unsqueeze(0),
            initial_state=initial_state,
        )

        ar.capture_token_affines(
            lin, kk.unsqueeze(0), v.unsqueeze(0), g.exp().unsqueeze(0), beta.unsqueeze(0)
        )
        ar.set_conv_states(lin, new_conv_state)

        core = core.reshape(length, self.num_v_heads, self.head_v_dim)
        core = self.norm.forward(core, z.reshape(length, self.num_v_heads, self.head_v_dim))
        return self.out_proj.forward(core.reshape(length, self.value_dim))

    # --- async-reasoning decode: W workers, one token each, batched ---
    def _forward_ar_decode(self, x, ar) -> torch.Tensor:
        lin = self._lin_idx
        k = self.conv_kernel
        n = x.shape[0]  # num workers

        qkv = self.in_proj_qkv.forward(x)  # (W, conv_dim)
        z = self.in_proj_z.forward(x)
        a = self.in_proj_a.forward(x)
        b = self.in_proj_b.forward(x)

        prior_conv = ar.prior_conv_states(lin)
        if prior_conv is None:
            prior_conv = torch.zeros(n, self.conv_dim, k, device=x.device, dtype=qkv.dtype)
        conv_in = torch.cat([prior_conv, qkv.unsqueeze(-1)], dim=-1)  # (W, conv_dim, k+1)
        new_conv_state = conv_in[..., -k:]
        conv_out = F.conv1d(conv_in, self.conv1d.weight, groups=self.conv_dim, padding=0)
        qkv2 = F.silu(conv_out[..., -1:]).squeeze(-1)  # (W, conv_dim)

        q, kk, v = self._split_heads(qkv2)  # (W, num_v_heads, d)
        beta, g = self._gates(a, b)
        initial_state = ar.compose_initial_recurrent_state(lin, dtype=torch.float32)  # (W,H,dk,dv)|None
        if initial_state is None:
            initial_state = torch.zeros(
                n, self.num_v_heads, self.head_k_dim, self.head_v_dim,
                device=x.device, dtype=torch.float32,
            )
        core, _ = _recurrent_delta(
            q.unsqueeze(1), kk.unsqueeze(1), v.unsqueeze(1),
            g.unsqueeze(1), beta.unsqueeze(1), initial_state,
        )

        ar.capture_token_affines(
            lin, kk.unsqueeze(1), v.unsqueeze(1), g.exp().unsqueeze(1), beta.unsqueeze(1)
        )
        ar.set_conv_states(lin, new_conv_state)

        core = core.reshape(n, self.num_v_heads, self.head_v_dim)
        core = self.norm.forward(core, z.reshape(n, self.num_v_heads, self.head_v_dim))
        return self.out_proj.forward(core.reshape(n, self.value_dim))

    # --- prefill: one chunked pass per request, write final state to pool ---
    def _forward_prefill(self, x, batch, gdn) -> torch.Tensor:
        out = torch.empty(x.shape[0], self.out_proj.full_output_size, dtype=x.dtype, device=x.device)
        k = self.conv_kernel
        offset = 0
        for req in batch.reqs:
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

            conv_in = qkv.transpose(0, 1).unsqueeze(0)  # (1, conv_dim, L)
            conv_out = F.conv1d(
                conv_in, self.conv1d.weight, groups=self.conv_dim, padding=k - 1
            )[..., :length]
            qkv = F.silu(conv_out).squeeze(0).transpose(0, 1)  # (L, conv_dim)

            # conv state: last `k` input columns, left-padded if the sequence is shorter
            pad = k - length
            conv_state = F.pad(conv_in, (pad, 0)) if pad >= 0 else conv_in[..., -k:]
            gdn.conv_state[self._lin_idx, req.table_idx] = conv_state.squeeze(0)

            q, kk, v = self._split_heads(qkv)
            beta, g = self._gates(a, b)
            core, rec_state = _chunk_delta(
                q.unsqueeze(0), kk.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0), beta.unsqueeze(0)
            )
            gdn.recurrent_state[self._lin_idx, req.table_idx] = rec_state.squeeze(0).float()

            core = core.reshape(length, self.num_v_heads, self.head_v_dim)
            core = self.norm.forward(core, z.reshape(length, self.num_v_heads, self.head_v_dim))
            out[offset - length : offset] = self.out_proj.forward(core.reshape(length, self.value_dim))
        return out

    # --- decode: single recurrent step, batched across requests ---
    def _forward_decode(self, x, batch, gdn) -> torch.Tensor:
        table_idx = torch.tensor(
            [req.table_idx for req in batch.reqs], device=x.device, dtype=torch.long
        )
        n = x.shape[0]
        qkv = self.in_proj_qkv.forward(x)  # (N, conv_dim)
        z = self.in_proj_z.forward(x)
        a = self.in_proj_a.forward(x)
        b = self.in_proj_b.forward(x)

        # causal conv update: append new token, roll the conv window
        conv_state = gdn.conv_state[self._lin_idx, table_idx]  # (N, conv_dim, k)
        conv_in = torch.cat([conv_state, qkv.unsqueeze(-1)], dim=-1)  # (N, conv_dim, k+1)
        gdn.conv_state[self._lin_idx, table_idx] = conv_in[..., -self.conv_kernel :]
        conv_out = F.conv1d(conv_in, self.conv1d.weight, groups=self.conv_dim, padding=0)
        qkv = F.silu(conv_out[..., -1:]).squeeze(-1)  # (N, conv_dim)

        q, kk, v = self._split_heads(qkv)
        beta, g = self._gates(a, b)
        rec_state = gdn.recurrent_state[self._lin_idx, table_idx]  # (N, num_v_heads, Dk, Dv)
        core, new_state = _recurrent_delta(
            q.unsqueeze(1), kk.unsqueeze(1), v.unsqueeze(1),
            g.unsqueeze(1), beta.unsqueeze(1), rec_state,
        )
        gdn.recurrent_state[self._lin_idx, table_idx] = new_state.float()

        core = core.reshape(n, self.num_v_heads, self.head_v_dim)
        core = self.norm.forward(core, z.reshape(n, self.num_v_heads, self.head_v_dim))
        return self.out_proj.forward(core.reshape(n, self.value_dim))


__all__ = ["Qwen3_5GatedDeltaNet"]
