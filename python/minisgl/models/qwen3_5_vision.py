"""Qwen3.5 vision tower (ViT) — a pure-torch port of transformers 5.12.1
``Qwen3_5VisionModel`` (which itself reuses Qwen3-VL's vision stack).

Input  : ``pixel_values [N, C*T*P*P]`` (from ``Qwen2VLImageProcessor``), ``grid_thw [n,3]``.
Output : ``image_embeds [N // merge**2, out_hidden_size]`` (== LM hidden), ready to
         scatter into the token embeddings at ``image_token_id`` positions.

No deepstack (Qwen3.5 removed it): a single ``merger`` and one injection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import torch
import torch.nn.functional as F
from minisgl.layers import BaseOP, LinearReplicated, OPList
from minisgl.utils import nvtx_annotate

if TYPE_CHECKING:
    from .config import VisionConfig


# ----------------------------------------------------------------------------
# vision_utils helpers (ported from transformers.vision_utils, pure functions)
# ----------------------------------------------------------------------------


def vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        0, dtype=torch.int32
    )
    return F.pad(cu, (1, 0), value=0)


def vision_position_ids(grid_thw: torch.Tensor, merge: int) -> torch.Tensor:
    """(row, col) ids per patch, in 2x2 merge-block order. Returns [N, 2] long."""
    device = grid_thw.device
    out: List[torch.Tensor] = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        hpos = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
        hpos = hpos.reshape(h // merge, merge, w // merge, merge).transpose(1, 2).flatten()
        wpos = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
        wpos = wpos.reshape(h // merge, merge, w // merge, merge).transpose(1, 2).flatten()
        out.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(out, dim=0)


def vision_bilinear(
    grid_thw: torch.Tensor, side: int, merge: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bilinear interpolation of the learned [side*side] pos-embed onto each image grid.
    Returns (indices [4, N] long, weights [4, N] float), in 2x2 merge-block order."""
    device = grid_thw.device
    idx_parts: List[List[torch.Tensor]] = [[] for _ in range(4)]
    w_parts: List[List[torch.Tensor]] = [[] for _ in range(4)]
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)
        h_floor, w_floor = h_grid.int(), w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac, w_frac = h_grid - h_floor, w_grid - w_floor
        hfo, hco = h_floor * side, h_ceil * side
        corner_idx = [
            (hfo[:, None] + w_floor[None, :]).flatten(),
            (hfo[:, None] + w_ceil[None, :]).flatten(),
            (hco[:, None] + w_floor[None, :]).flatten(),
            (hco[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_w = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]
        h_idx = torch.arange(h, device=device).view(h // merge, merge)
        w_idx = torch.arange(w, device=device).view(w // merge, merge)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten().repeat(t)
        for i in range(4):
            idx_parts[i].append(corner_idx[i][reorder])
            w_parts[i].append(corner_w[i][reorder])
    indices = torch.stack([torch.cat(p) for p in idx_parts])
    weights = torch.stack([torch.cat(p) for p in w_parts])
    return indices, weights


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# ----------------------------------------------------------------------------
# Small building blocks
# ----------------------------------------------------------------------------


class _LayerNorm(BaseOP):
    def __init__(self, size: int, eps: float = 1e-6):
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)
        self._eps = eps
        self._size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self._size,), self.weight, self.bias, self._eps)


class _PatchEmbedProj(BaseOP):
    """Holds the Conv3d weight (out, C, T, P, P) + bias; kernel==stride so it's a linear."""

    def __init__(self, out_dim: int, in_ch: int, tps: int, ps: int):
        self.weight = torch.empty(out_dim, in_ch, tps, ps, ps)
        self.bias = torch.empty(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.weight.shape[0]
        return F.linear(x.to(self.weight.dtype), self.weight.reshape(out, -1), self.bias)


class _PatchEmbed(BaseOP):
    def __init__(self, cfg: VisionConfig):
        self.proj = _PatchEmbedProj(cfg.hidden_size, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj.forward(x)


class _PosEmbed(BaseOP):
    def __init__(self, num_pos: int, dim: int):
        self.weight = torch.empty(num_pos, dim)


class _VisionAttention(BaseOP):
    def __init__(self, cfg: VisionConfig):
        self.qkv = LinearReplicated(cfg.hidden_size, cfg.hidden_size * 3, has_bias=True)
        self.proj = LinearReplicated(cfg.hidden_size, cfg.hidden_size, has_bias=True)
        self._num_heads = cfg.num_heads
        self._head_dim = cfg.hidden_size // cfg.num_heads

    def forward(self, x, cos, sin, seg_lens: List[int]) -> torch.Tensor:
        n = x.shape[0]
        H, D = self._num_heads, self._head_dim
        q, k, v = self.qkv.forward(x).reshape(n, 3, H, D).permute(1, 0, 2, 3).unbind(0)  # each (n,H,D)
        c = cos.unsqueeze(1)  # (n,1,D)
        s = sin.unsqueeze(1)
        q = (q * c) + (_rotate_half(q) * s)
        k = (k * c) + (_rotate_half(k) * s)
        outs: List[torch.Tensor] = []
        off = 0
        for L in seg_lens:
            sl = slice(off, off + L)
            qs = q[sl].transpose(0, 1).unsqueeze(0)  # (1,H,L,D)
            ks = k[sl].transpose(0, 1).unsqueeze(0)
            vs = v[sl].transpose(0, 1).unsqueeze(0)
            o = F.scaled_dot_product_attention(qs, ks, vs)  # scale = 1/sqrt(D) default
            outs.append(o.squeeze(0).transpose(0, 1))  # (L,H,D)
            off += L
        o = torch.cat(outs, 0).reshape(n, -1)
        return self.proj.forward(o)


class _VisionMLP(BaseOP):
    def __init__(self, cfg: VisionConfig):
        self.linear_fc1 = LinearReplicated(cfg.hidden_size, cfg.intermediate_size, has_bias=True)
        self.linear_fc2 = LinearReplicated(cfg.intermediate_size, cfg.hidden_size, has_bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gelu_pytorch_tanh
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x), approximate="tanh"))


class _VisionBlock(BaseOP):
    def __init__(self, cfg: VisionConfig):
        self.norm1 = _LayerNorm(cfg.hidden_size)
        self.norm2 = _LayerNorm(cfg.hidden_size)
        self.attn = _VisionAttention(cfg)
        self.mlp = _VisionMLP(cfg)

    def forward(self, x, cos, sin, seg_lens: List[int]) -> torch.Tensor:
        x = x + self.attn.forward(self.norm1.forward(x), cos, sin, seg_lens)
        x = x + self.mlp.forward(self.norm2.forward(x))
        return x


class _PatchMerger(BaseOP):
    def __init__(self, cfg: VisionConfig):
        merged = cfg.hidden_size * (cfg.spatial_merge_size**2)
        self.norm = _LayerNorm(cfg.hidden_size)  # pre-shuffle norm on hidden_size
        self.linear_fc1 = LinearReplicated(merged, merged, has_bias=True)
        self.linear_fc2 = LinearReplicated(merged, cfg.out_hidden_size, has_bias=True)
        self._merged = merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm.forward(x).view(-1, self._merged)
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x)))  # exact gelu


class Qwen3_5VisionModel(BaseOP):
    def __init__(self, cfg: VisionConfig):
        self.patch_embed = _PatchEmbed(cfg)
        self.pos_embed = _PosEmbed(cfg.num_position_embeddings, cfg.hidden_size)
        self.blocks = OPList([_VisionBlock(cfg) for _ in range(cfg.depth)])
        self.merger = _PatchMerger(cfg)
        self._merge = cfg.spatial_merge_size
        self._side = int(cfg.num_position_embeddings**0.5)
        self._num_heads = cfg.num_heads
        self._head_dim = cfg.hidden_size // cfg.num_heads
        self._rope_dim = self._head_dim // 2  # 2D rotary applied over head_dim//2 then doubled
        self._inv_freq: torch.Tensor | None = None

    @nvtx_annotate("VisionTower")
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        dev = pixel_values.device
        grid_thw = grid_thw.to(dev)
        bi, bw = vision_bilinear(grid_thw, self._side, self._merge)
        pos_ids = vision_position_ids(grid_thw, self._merge)  # (N,2)
        cu = vision_cu_seqlens(grid_thw)
        seg_lens = (cu[1:] - cu[:-1]).tolist()

        x = self.patch_embed.forward(pixel_values)  # (N, hidden)
        pos_embeds = (self.pos_embed.weight[bi] * bw[:, :, None]).sum(0)  # (N, hidden)
        x = x + pos_embeds.to(x.dtype)

        # 2D vision rotary: freqs (N, rope_dim) -> cos/sin over full head_dim
        if self._inv_freq is None:
            d = self._rope_dim
            self._inv_freq = 1.0 / (
                10000.0 ** (torch.arange(0, d, 2, dtype=torch.float32, device=dev) / d)
            )
        rot = (pos_ids[..., None].float() * self._inv_freq).flatten(1)  # (N, rope_dim)
        emb = torch.cat((rot, rot), dim=-1)  # (N, head_dim)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)

        for blk in self.blocks.op_list:
            x = blk.forward(x, cos, sin, seg_lens)
        return self.merger.forward(x)  # (N // merge^2, out_hidden)


__all__ = ["Qwen3_5VisionModel"]
