"""
CPU math tests for the Gated DeltaNet delta-rule kernels
(``minisgl.models.qwen3_5_delta._chunk_gated_delta_rule`` /
``_recurrent_gated_delta_rule``, the pure-torch reference used when ``fla`` is absent).

Checks the recurrence math directly:

* ``_l2norm`` produces unit rows;
* the recurrent kernel's final state equals the affine fold of the same
  ``(k, v, α, β)`` applied to the initial state (links the kernel to
  ``gdn_affine``);
* the per-token output is the query read-out ``(scale·q̂) · S_after``;
* the chunked prefill matches the sequential recurrence (state + outputs), for
  sequences that are and aren't a multiple of the chunk size, with and without
  an initial state;
* ``initial_state=None`` equals ``initial_state=0``.

An optional GPU test compares the ``fla`` kernels to the pure-torch reference
(skipped when CUDA or ``fla`` is unavailable).

Run::

    uv run pytest tests/core/test_gdn_delta.py -v
    # or, without pytest:
    .venv/bin/python tests/core/test_gdn_delta.py
"""

from __future__ import annotations

import minisgl.models.qwen3_5_delta as delta_module
import torch
from minisgl.models.qwen3_5_delta import (
    _chunk_delta,
    _chunk_gated_delta_rule,
    _fla_chunk,
    _fla_recurrent,
    _l2norm,
    _recurrent_delta,
    _recurrent_gated_delta_rule,
)
from minisgl.shared_cache.gdn_affine import (
    apply_gdn_affine,
    init_gdn_affine,
    update_affine_summary,
)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-30)


def _inputs(Bs=1, H=3, dk=8, dv=8, T=15, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(Bs, T, H, dk, generator=g, dtype=dtype)
    k = torch.randn(Bs, T, H, dk, generator=g, dtype=dtype)
    v = torch.randn(Bs, T, H, dv, generator=g, dtype=dtype)
    gate = -torch.rand(Bs, T, H, generator=g, dtype=dtype) * 2.0  # log-decay in (-2, 0]
    beta = torch.rand(Bs, T, H, generator=g, dtype=dtype)
    return q, k, v, gate, beta


# ---------------------------------------------------------------------------


def test_l2norm_unit_rows():
    x = torch.randn(4, 5, 16, dtype=torch.float64)
    n = _l2norm(x).norm(dim=-1)
    # eps makes it just under 1; well within tolerance for non-tiny vectors
    assert (n - 1.0).abs().max().item() < 1e-6


def test_recurrent_state_matches_affine_fold():
    """The recurrent kernel's final state == affine fold of the same tokens on S0.

    Both l2-norm the key; the affine fold uses that normed key and raw v,
    α=exp(g), β — exactly matching the kernel's state update.
    """
    Bs, H, dk, dv, T = 1, 3, 8, 8, 17
    q, k, v, g, beta = _inputs(Bs, H, dk, dv, T, seed=1)
    alpha = g.exp()
    S0_block = torch.randn(Bs, H, dv, dk)  # block convention [.,dv,dk]

    normed_k = _l2norm(k.float())
    A, B = init_gdn_affine(
        batch_size=Bs, num_heads=H, d_k=dk, d_v=dv, dtype=torch.float32, device="cpu"
    )
    for t in range(T):
        A, B = update_affine_summary(
            A_hat=A,
            B_hat=B,
            k=normed_k[:, t],
            v=v[:, t].float(),
            alpha=alpha[:, t],
            beta=beta[:, t],
        )
    S_affine = apply_gdn_affine(S0_block, A, B)  # [.,dv,dk]

    S0_hf = S0_block.transpose(-1, -2).contiguous()  # [.,dk,dv]
    _, S_delta = _recurrent_gated_delta_rule(q, k, v, g, beta, S0_hf)  # [.,dk,dv]
    assert _rel(S_delta, S_affine.transpose(-1, -2)) < 1e-4


def test_recurrent_output_is_query_readout():
    """Single-token output == (scale·q̂) read out against the post-update state."""
    Bs, H, dk, dv = 1, 3, 8, 8
    q, k, v, g, beta = _inputs(Bs, H, dk, dv, T=1, seed=2)
    S0_hf = torch.randn(Bs, H, dk, dv)
    out, S_after = _recurrent_gated_delta_rule(q, k, v, g, beta, S0_hf)  # out [.,1,H,dv]
    scale = dk**-0.5
    nq = _l2norm(q[:, 0].float()) * scale  # [.,H,dk]
    expected = (nq.unsqueeze(-1) * S_after).sum(-2)  # sum over dk -> [.,H,dv]
    assert _rel(out[:, 0], expected) < 1e-5


def test_chunk_matches_recurrent_no_initial_state():
    for T in (16, 20):  # multiple of chunk_size and not
        q, k, v, g, beta = _inputs(T=T, seed=10 + T)
        out_c, S_c = _chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=8, initial_state=None)
        out_r, S_r = _recurrent_gated_delta_rule(q, k, v, g, beta, torch.zeros(1, 3, 8, 8))
        assert _rel(S_c, S_r) < 1e-4, f"state mismatch T={T}"
        assert _rel(out_c, out_r) < 1e-4, f"output mismatch T={T}"


def test_chunk_matches_recurrent_with_initial_state():
    Bs, H, dk, dv, T = 1, 3, 8, 8, 20
    q, k, v, g, beta = _inputs(Bs, H, dk, dv, T, seed=42)
    S0 = torch.randn(Bs, H, dk, dv)
    out_c, S_c = _chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=8, initial_state=S0)
    out_r, S_r = _recurrent_gated_delta_rule(q, k, v, g, beta, S0)
    assert _rel(S_c, S_r) < 1e-4
    assert _rel(out_c, out_r) < 1e-4


def test_initial_state_none_equals_zero():
    q, k, v, g, beta = _inputs(T=12, seed=7)
    out_n, S_n = _chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=8, initial_state=None)
    out_z, S_z = _chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=8, initial_state=torch.zeros(1, 3, 8, 8)
    )
    assert _rel(S_n, S_z) < 1e-6
    assert _rel(out_n, out_z) < 1e-6


def test_v_first_dispatchers_preserve_puretorch_fallback(monkeypatch):
    """The no-FLA path accepts/returns [B,H,Dv,Dk] without a materialized transpose."""

    monkeypatch.setattr(delta_module, "_fla_chunk", None)
    monkeypatch.setattr(delta_module, "_fla_recurrent", None)
    Bs, H, dk, dv, T = 2, 3, 7, 5, 9
    q, k, v, g, beta = _inputs(Bs, H, dk, dv, T, seed=91)
    initial_v_first = torch.randn(Bs, H, dv, dk)
    initial_hf_view = initial_v_first.transpose(-1, -2)

    expected_chunk, expected_chunk_final = _chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=initial_hf_view
    )
    actual_chunk, actual_chunk_final = _chunk_delta(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_v_first,
        state_v_first=True,
    )
    torch.testing.assert_close(actual_chunk, expected_chunk)
    torch.testing.assert_close(actual_chunk_final, expected_chunk_final.transpose(-1, -2))

    expected_zero_chunk, expected_zero_final = _chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=None
    )
    actual_zero_chunk, actual_zero_final = _chunk_delta(
        q, k, v, g, beta, initial_state=None, state_v_first=True
    )
    torch.testing.assert_close(actual_zero_chunk, expected_zero_chunk)
    torch.testing.assert_close(actual_zero_final, expected_zero_final.transpose(-1, -2))

    expected_recurrent, expected_recurrent_final = _recurrent_gated_delta_rule(
        q[:, :1], k[:, :1], v[:, :1], g[:, :1], beta[:, :1], initial_hf_view
    )
    actual_recurrent, actual_recurrent_final = _recurrent_delta(
        q[:, :1],
        k[:, :1],
        v[:, :1],
        g[:, :1],
        beta[:, :1],
        initial_v_first,
        state_v_first=True,
    )
    torch.testing.assert_close(actual_recurrent, expected_recurrent)
    torch.testing.assert_close(actual_recurrent_final, expected_recurrent_final.transpose(-1, -2))


def test_fla_matches_puretorch_gpu():
    """Optional: the fla kernels agree with the pure-torch reference (bf16 tol)."""
    if _fla_chunk is None or _fla_recurrent is None or not torch.cuda.is_available():
        print("  [skip] fla or CUDA unavailable")
        return
    dev = "cuda"
    # Rectangular state makes a swapped K/V convention observable in the shape.
    Bs, H, dk, dv, T = 2, 16, 128, 64, 40
    q, k, v, g, beta = _inputs(Bs, H, dk, dv, T, seed=5, dtype=torch.bfloat16)
    q, k, v, beta = (t.to(dev) for t in (q, k, v, beta))
    g = g.float().to(dev)
    S0 = torch.randn(Bs, H, dk, dv, device=dev, dtype=torch.float32)

    c_pt, s_pt = _chunk_gated_delta_rule(q, k, v, g, beta, initial_state=S0)
    c_fla, s_fla = _fla_chunk(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=S0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert _rel(c_fla, c_pt) < 2e-2
    assert _rel(s_fla, s_pt) < 2e-2

    S0_v_first = S0.transpose(-1, -2).contiguous()
    c_fla_v_first, s_fla_v_first = _fla_chunk(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=S0_v_first,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        state_v_first=True,
    )
    assert _rel(c_fla_v_first, c_fla) < 2e-2
    assert _rel(s_fla_v_first.transpose(-1, -2), s_fla) < 2e-2

    q1, k1, v1, g1, b1 = (t[:, :1] for t in (q, k, v, g, beta))
    r_pt, sr_pt = _recurrent_gated_delta_rule(q1, k1, v1, g1, b1, S0)
    r_fla, sr_fla = _fla_recurrent(
        q1,
        k1,
        v1,
        g=g1,
        beta=b1,
        initial_state=S0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert _rel(r_fla, r_pt) < 2e-2
    assert _rel(sr_fla, sr_pt) < 2e-2

    r_fla_v_first, sr_fla_v_first = _fla_recurrent(
        q1,
        k1,
        v1,
        g=g1,
        beta=b1,
        initial_state=S0_v_first,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        state_v_first=True,
    )
    assert _rel(r_fla_v_first, r_fla) < 2e-2
    assert _rel(sr_fla_v_first.transpose(-1, -2), sr_fla) < 2e-2


if __name__ == "__main__":
    import sys

    fns = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
