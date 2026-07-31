from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, LinearOProj, LinearReplicated, RMSNorm
from minisgl.utils import nvtx_annotate

if TYPE_CHECKING:
    from .config import ModelConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Qwen3_5Attention(BaseOP):
    """Qwen3.5 full-attention layer: GQA + per-head q/k RMSNorm, partial (neox) RoPE,
    and a sigmoid output gate (q_proj is 2x width, the second half is the gate).

    The fused ``qkv_proj`` is produced by the standard weight merge of the checkpoint's
    separate ``q_proj`` (2x), ``k_proj`` and ``v_proj``. (tp=1 only for v1.)
    """

    def __init__(self, config: ModelConfig, kv_idx: int):
        self.num_qo_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.qo_dim = self.num_qo_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self._kv_idx = kv_idx
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        self._rope_base = config.rotary_config.base
        self._mrope_section = config.mrope_section  # None for text-only
        self._inv_freq: torch.Tensor | None = None

        # qkv_proj output layout: [q+gate (2*qo_dim) | k (kv_dim) | v (kv_dim)]
        self.qkv_proj = LinearReplicated(
            config.hidden_size, 2 * self.qo_dim + 2 * self.kv_dim, has_bias=False
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = LinearOProj(self.qo_dim, config.hidden_size, has_bias=False)

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: (T, num_heads, head_dim); rotate the first `rotary_dim` dims (neox style).
        if self._inv_freq is None:
            self._inv_freq = 1.0 / (
                self._rope_base
                ** (
                    torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=x.device)
                    / self.rotary_dim
                )
            )
        freqs = positions.float()[:, None] * self._inv_freq[None, :]  # (T, rotary_dim/2)
        return self._apply_from_freqs(x, freqs)

    def _apply_mrope(self, x: torch.Tensor, mrope_positions: torch.Tensor) -> torch.Tensor:
        """Interleaved mRoPE: ``mrope_positions`` is ``[3, T]`` (temporal/height/width).
        Frequency slots are interleaved across the 3 axes per ``mrope_section``."""
        self._ensure_inv_freq(x.device)
        p = mrope_positions.to(x.device).float()  # (3, T)
        freqs3 = p[:, :, None] * self._inv_freq[None, None, :]  # (3, T, rotary_dim/2)
        freqs = freqs3[0].clone()  # start from all-T
        for dim, offset in ((1, 1), (2, 2)):  # H, W
            length = self._mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs[..., idx] = freqs3[dim][..., idx]
        return self._apply_from_freqs(x, freqs)

    def _ensure_inv_freq(self, device) -> None:
        if self._inv_freq is None:
            self._inv_freq = 1.0 / (
                self._rope_base
                ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device) / self.rotary_dim)
            )

    def _apply_from_freqs(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, rotary_dim)
        cos = emb.cos().to(x.dtype)[:, None, :]
        sin = emb.sin().to(x.dtype)[:, None, :]
        x_rot, x_pass = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        x_rot = x_rot * cos + _rotate_half(x_rot) * sin
        return torch.cat((x_rot, x_pass), dim=-1)

    @nvtx_annotate("FullAttn")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        qkv = self.qkv_proj.forward(x)
        del x
        q_gate, k, v = qkv.split([2 * self.qo_dim, self.kv_dim, self.kv_dim], dim=-1)

        q_gate = q_gate.view(-1, self.num_qo_heads, 2 * self.head_dim)
        q = q_gate[..., : self.head_dim].contiguous()
        gate = q_gate[..., self.head_dim :].contiguous()
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.contiguous()  # detach from the fused qkv slice for the kv-cache store

        q = self.q_norm.forward(q)
        k = self.k_norm.forward(k)

        # Async-reasoning shared-cache path: the op stores block-relative KV and
        # applies partial RoPE itself, so pass q/k through un-rotated (mirrors
        # layers.attention.AttentionLayer).  Otherwise use the normal backend.
        sc_op = getattr(ctx.batch.attn_metadata, "shared_cache_op", None)
        if sc_op is not None:
            o = sc_op.forward(
                q.reshape(-1, self.qo_dim), k.reshape(-1, self.kv_dim), v, self._kv_idx, ctx.batch
            )
        else:
            mrope = ctx.batch.mrope_positions
            if mrope is not None:
                q = self._apply_mrope(q, mrope)
                k = self._apply_mrope(k, mrope)
            else:
                q = self._apply_rope(q, ctx.batch.positions)
                k = self._apply_rope(k, ctx.batch.positions)
            o = ctx.attn_backend.forward(q, k.reshape(-1, self.kv_dim), v, self._kv_idx, ctx.batch)

        o = o.view(-1, self.num_qo_heads, self.head_dim) * torch.sigmoid(gate)
        return self.o_proj.forward(o.reshape(-1, self.qo_dim))


__all__ = ["Qwen3_5Attention"]
