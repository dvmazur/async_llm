"""
CPU-only edge-case suite for ``apply_rope_correction``.

Run::

    uv run pytest tests/core/test_rope_correction_kernel.py -v

Complements ``tests/core/test_shared_cache.py::TestApplyRopeCorrection`` with
dtype, boundary, and batched-multi-block checks. No GPU, no engine, no HF.
"""

from __future__ import annotations

import pytest
import torch

from minisgl.shared_cache import apply_rope_correction


def _build_cos_sin_cache(head_dim: int, max_pos: int, base: float = 10000.0) -> torch.Tensor:
    """Build the [max_pos, head_dim] cache used by minisgl.layers.rotary.

    First half cols are cos(freqs), second half are sin(freqs).
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    t = torch.arange(max_pos, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _reference_rope(
    keys: torch.Tensor, positions: torch.Tensor, base: float = 10000.0
) -> torch.Tensor:
    """HF-style RoPE applied to *keys* at *positions* — the ground truth.

    keys: ``[N, num_heads, head_dim]`` (any float dtype, computed in fp32 here)
    positions: ``[N]`` integer rotations (may be negative)
    """
    head_dim = keys.shape[-1]
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)  # [N, half]
    cos = freqs.cos().unsqueeze(1)
    sin = freqs.sin().unsqueeze(1)
    k1 = keys.float()[..., :half]
    k2 = keys.float()[..., half:]
    return torch.cat([cos * k1 - sin * k2, cos * k2 + sin * k1], dim=-1)


# Per-dtype tolerance: dominated by the kernel's final down-cast from fp32.
_DTYPE_ATOL = [
    (torch.float32, 1e-5),
    (torch.float16, 5e-3),
    (torch.bfloat16, 3e-2),
]


@pytest.mark.parametrize("dtype,atol", _DTYPE_ATOL, ids=lambda v: str(v).split(".")[-1])
def test_rope_correction_dtype_matrix(dtype, atol):
    """Verify the kernel preserves dtype and matches reference per-dtype."""
    if isinstance(dtype, float):
        pytest.skip("malformed parametrize")

    head_dim, n_heads, n_tokens, max_pos = 64, 4, 8, 256
    torch.manual_seed(0)
    keys = torch.randn(n_tokens, n_heads, head_dim, dtype=dtype)
    corrections = torch.tensor([0, 5, -3, 17, -12, 42, 99, -50], dtype=torch.int64)
    cs = _build_cos_sin_cache(head_dim, max_pos)  # fp32 cache

    got = apply_rope_correction(keys, corrections, cs)
    expected = _reference_rope(keys, corrections).to(dtype)

    assert got.dtype == dtype, f"dtype not preserved: got {got.dtype}, expected {dtype}"
    assert got.shape == keys.shape
    max_diff = (got.float() - expected.float()).abs().max().item()
    assert torch.allclose(got.float(), expected.float(), atol=atol), (
        f"{dtype} max |diff| = {max_diff:.3e} > atol={atol:.3e}"
    )


def test_rope_correction_zero_delta_is_identity():
    """corrections = 0 → cos(0)=1, sin(0)=0 → output equals input."""
    head_dim = 64
    torch.manual_seed(1)
    keys = torch.randn(5, 2, head_dim)
    cs = _build_cos_sin_cache(head_dim, 128)
    out = apply_rope_correction(keys, torch.zeros(5, dtype=torch.int64), cs)
    assert torch.allclose(out, keys, atol=1e-6)


def test_rope_correction_max_position_boundary():
    """Corrections at +/-(max_pos - 1) — just inside the cache — must still
    match reference exactly. Larger absolute deltas would be clamped by the
    kernel and are intentionally out of scope here."""
    head_dim, max_pos = 64, 128
    torch.manual_seed(2)
    keys = torch.randn(2, 1, head_dim)
    corrections = torch.tensor([max_pos - 1, -(max_pos - 1)], dtype=torch.int64)
    cs = _build_cos_sin_cache(head_dim, max_pos)

    got = apply_rope_correction(keys, corrections, cs)
    expected = _reference_rope(keys, corrections)

    assert torch.allclose(got, expected, atol=1e-5), (
        f"max diff = {(got - expected).abs().max().item():.3e}"
    )


def test_rope_correction_large_negative_delta():
    """Exercise the sin(-θ) = -sin(θ) branch on a range of negative deltas."""
    head_dim, max_pos = 64, 256
    torch.manual_seed(3)
    keys = torch.randn(4, 2, head_dim)
    corrections = torch.tensor([-100, -50, -1, -64], dtype=torch.int64)
    cs = _build_cos_sin_cache(head_dim, max_pos)

    got = apply_rope_correction(keys, corrections, cs)
    expected = _reference_rope(keys, corrections)

    assert torch.allclose(got, expected, atol=1e-5), (
        f"max diff = {(got - expected).abs().max().item():.3e}"
    )


def test_rope_correction_empty_input():
    """Zero-length inputs must not crash and must return the same shape."""
    head_dim = 64
    keys = torch.zeros(0, 2, head_dim)
    corrections = torch.zeros(0, dtype=torch.int64)
    cs = _build_cos_sin_cache(head_dim, 64)

    out = apply_rope_correction(keys, corrections, cs)
    assert out.shape == keys.shape
    assert out.dtype == keys.dtype


def test_rope_correction_batched_multi_block():
    """Concatenated multi-block corrections in one call: each per-block slice
    must independently match its own reference RoPE."""
    head_dim, max_pos = 64, 256
    torch.manual_seed(4)

    block_corrs = [
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([-5, -4], dtype=torch.int64),
        torch.tensor([10, 11, 12, 13], dtype=torch.int64),
    ]
    flat_corr = torch.cat(block_corrs)
    n_total = int(flat_corr.numel())

    keys = torch.randn(n_total, 2, head_dim)
    cs = _build_cos_sin_cache(head_dim, max_pos)

    got = apply_rope_correction(keys, flat_corr, cs)
    expected = _reference_rope(keys, flat_corr)
    assert torch.allclose(got, expected, atol=1e-5), (
        f"batched max diff = {(got - expected).abs().max().item():.3e}"
    )

    # Each block slice independently agrees with reference on its own.
    offset = 0
    for corr in block_corrs:
        sl = slice(offset, offset + corr.numel())
        per_block_got = apply_rope_correction(keys[sl], corr, cs)
        per_block_expected = _reference_rope(keys[sl], corr)
        assert torch.allclose(per_block_got, per_block_expected, atol=1e-5), (
            f"per-block slice [{offset}:{offset + corr.numel()}] diverged"
        )
        # And the batched result on this slice matches the per-block call.
        assert torch.allclose(got[sl], per_block_got, atol=1e-5)
        offset += int(corr.numel())
