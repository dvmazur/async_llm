"""
CPU-only math tests for the Gated DeltaNet affine cache
(``minisgl.shared_cache.gdn_affine``).

The affine summary ``(A_hat, B_hat)`` of a block of GDN tokens satisfies
``S_out = S_in @ A_hat + B_hat`` (block convention ``[B,H,d_v,d_k]``).  These
tests check the algebra directly, in float64, against dense references:

* ``init`` is (I, 0);
* ``apply`` is ``S@A + B`` (identity / zero behaviour);
* a single-token ``update`` equals the explicit ``A_t = αI − αβ kkᵀ``, ``B_t = β vkᵀ``;
* ``compose`` equals sequentially applying the two blocks (and is associative);
* folding per-token updates over a sequence and applying to ``S0`` reproduces the
  dense token-by-token GDN recurrence;
* splitting a sequence and composing the halves equals folding the whole.

Run::

    uv run pytest tests/core/test_gdn_affine.py -v
    # or, without pytest:
    .venv/bin/python tests/core/test_gdn_affine.py

No GPU, no engine, no HF.
"""

from __future__ import annotations

import torch

from minisgl.shared_cache.gdn_affine import (
    apply_gdn_affine,
    compose_gdn_affines,
    init_gdn_affine,
    update_affine_summary,
)

DEV = torch.device("cpu")
F64 = torch.float64


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).norm().item() / max(b.norm().item(), 1e-30)


def _rand_tokens(Bs, H, dk, dv, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(Bs, T, H, dk, generator=g, dtype=F64)
    v = torch.randn(Bs, T, H, dv, generator=g, dtype=F64)
    alpha = torch.rand(Bs, T, H, generator=g, dtype=F64) * 0.5 + 0.5  # (0.5, 1]
    beta = torch.rand(Bs, T, H, generator=g, dtype=F64)
    return k, v, alpha, beta


def _explicit_At_Bt(k, v, alpha, beta):
    """Dense per-token blocks: A_t = αI − αβ kkᵀ [.,dk,dk], B_t = β vkᵀ [.,dv,dk]."""
    dk = k.shape[-1]
    eye = torch.eye(dk, dtype=k.dtype)
    a = alpha[..., None, None]
    b = beta[..., None, None]
    kkT = k.unsqueeze(-1) * k.unsqueeze(-2)  # [.,dk,dk]
    A_t = a * eye - a * b * kkT
    B_t = b * v.unsqueeze(-1) * k.unsqueeze(-2)  # [.,dv,dk]
    return A_t, B_t


def _dense_step(S, k, v, alpha, beta):
    """One GDN recurrent step in block convention S [.,dv,dk]:
    S_t = αS − αβ (S k) kᵀ + β v kᵀ."""
    a = alpha[..., None, None]
    b = beta[..., None, None]
    Sk = (S * k.unsqueeze(-2)).sum(-1)  # S @ k -> [.,dv]
    erase = a * b * Sk.unsqueeze(-1) * k.unsqueeze(-2)
    write = b * v.unsqueeze(-1) * k.unsqueeze(-2)
    return a * S - erase + write


# ---------------------------------------------------------------------------


def test_init_is_identity_and_zero():
    A, B = init_gdn_affine(batch_size=2, num_heads=3, d_k=8, d_v=6, dtype=F64, device=DEV)
    assert tuple(A.shape) == (2, 3, 8, 8)
    assert tuple(B.shape) == (2, 3, 6, 8)
    eye = torch.eye(8, dtype=F64).expand(2, 3, 8, 8)
    assert _rel(A, eye) < 1e-12
    assert B.abs().max().item() == 0.0


def test_apply_identity_and_zero_state():
    A, B = init_gdn_affine(batch_size=2, num_heads=3, d_k=8, d_v=8, dtype=F64, device=DEV)
    S = torch.randn(2, 3, 8, 8, dtype=F64)
    # apply(S, I, 0) == S
    assert _rel(apply_gdn_affine(S, A, B), S) < 1e-12
    # apply(0, A, B) == B  for arbitrary (A, B)
    Ar = torch.randn(2, 3, 8, 8, dtype=F64)
    Br = torch.randn(2, 3, 8, 8, dtype=F64)
    out = apply_gdn_affine(torch.zeros_like(S), Ar, Br)
    assert _rel(out, Br) < 1e-12


def test_single_token_update_matches_explicit_blocks():
    k, v, alpha, beta = _rand_tokens(2, 3, 8, 8, T=1, seed=1)
    A0, B0 = init_gdn_affine(batch_size=2, num_heads=3, d_k=8, d_v=8, dtype=F64, device=DEV)
    A, B = update_affine_summary(
        A_hat=A0, B_hat=B0, k=k[:, 0], v=v[:, 0], alpha=alpha[:, 0], beta=beta[:, 0]
    )
    A_t, B_t = _explicit_At_Bt(k[:, 0], v[:, 0], alpha[:, 0], beta[:, 0])
    # from identity: A == A_t, B == B_t
    assert _rel(A, A_t) < 1e-11
    assert _rel(B, B_t) < 1e-11


def test_running_update_matches_compose_with_token_block():
    """update(A,B, token) == compose((A,B), (A_t, B_t))."""
    k, v, alpha, beta = _rand_tokens(2, 3, 8, 8, T=2, seed=2)
    A, B = init_gdn_affine(batch_size=2, num_heads=3, d_k=8, d_v=8, dtype=F64, device=DEV)
    # seed a non-trivial running (A, B) with the first token
    A, B = update_affine_summary(
        A_hat=A, B_hat=B, k=k[:, 0], v=v[:, 0], alpha=alpha[:, 0], beta=beta[:, 0]
    )
    # second token via update ...
    A_upd, B_upd = update_affine_summary(
        A_hat=A, B_hat=B, k=k[:, 1], v=v[:, 1], alpha=alpha[:, 1], beta=beta[:, 1]
    )
    # ... vs via compose with the explicit per-token block
    A_t, B_t = _explicit_At_Bt(k[:, 1], v[:, 1], alpha[:, 1], beta[:, 1])
    A_cmp, B_cmp = compose_gdn_affines(A_first=A, B_first=B, A_second=A_t, B_second=B_t)
    assert _rel(A_upd, A_cmp) < 1e-11
    assert _rel(B_upd, B_cmp) < 1e-11


def test_compose_equals_sequential_apply():
    A1 = torch.randn(2, 3, 8, 8, dtype=F64)
    B1 = torch.randn(2, 3, 8, 8, dtype=F64)
    A2 = torch.randn(2, 3, 8, 8, dtype=F64)
    B2 = torch.randn(2, 3, 8, 8, dtype=F64)
    S = torch.randn(2, 3, 8, 8, dtype=F64)
    seq = apply_gdn_affine(apply_gdn_affine(S, A1, B1), A2, B2)
    Ac, Bc = compose_gdn_affines(A_first=A1, B_first=B1, A_second=A2, B_second=B2)
    assert _rel(apply_gdn_affine(S, Ac, Bc), seq) < 1e-10


def test_compose_associativity():
    mats = [
        (torch.randn(1, 2, 6, 6, dtype=F64), torch.randn(1, 2, 6, 6, dtype=F64)) for _ in range(3)
    ]
    (A1, B1), (A2, B2), (A3, B3) = mats
    left = compose_gdn_affines(A_first=A1, B_first=B1, A_second=A2, B_second=B2)
    left = compose_gdn_affines(A_first=left[0], B_first=left[1], A_second=A3, B_second=B3)
    right = compose_gdn_affines(A_first=A2, B_first=B2, A_second=A3, B_second=B3)
    right = compose_gdn_affines(A_first=A1, B_first=B1, A_second=right[0], B_second=right[1])
    assert _rel(left[0], right[0]) < 1e-10
    assert _rel(left[1], right[1]) < 1e-10


def test_fold_matches_dense_recurrence():
    """Folding per-token affines then applying to S0 == dense GDN recurrence from S0."""
    Bs, H, dk, dv, T = 2, 3, 8, 8, 20
    k, v, alpha, beta = _rand_tokens(Bs, H, dk, dv, T, seed=3)
    S0 = torch.randn(Bs, H, dv, dk, dtype=F64)

    A, B = init_gdn_affine(batch_size=Bs, num_heads=H, d_k=dk, d_v=dv, dtype=F64, device=DEV)
    for t in range(T):
        A, B = update_affine_summary(
            A_hat=A, B_hat=B, k=k[:, t], v=v[:, t], alpha=alpha[:, t], beta=beta[:, t]
        )
    S_affine = apply_gdn_affine(S0, A, B)

    S = S0.clone()
    for t in range(T):
        S = _dense_step(S, k[:, t], v[:, t], alpha[:, t], beta[:, t])
    assert _rel(S_affine, S) < 1e-10
    # apply-to-zero gives exactly B_hat
    assert _rel(apply_gdn_affine(torch.zeros_like(S0), A, B), B) < 1e-12


def test_split_compose_equals_full_fold():
    Bs, H, dk, dv, T, m = 1, 2, 6, 6, 17, 7
    k, v, alpha, beta = _rand_tokens(Bs, H, dk, dv, T, seed=4)

    def fold(lo, hi):
        A, B = init_gdn_affine(batch_size=Bs, num_heads=H, d_k=dk, d_v=dv, dtype=F64, device=DEV)
        for t in range(lo, hi):
            A, B = update_affine_summary(
                A_hat=A, B_hat=B, k=k[:, t], v=v[:, t], alpha=alpha[:, t], beta=beta[:, t]
            )
        return A, B

    Af, Bf = fold(0, T)
    A1, B1 = fold(0, m)
    A2, B2 = fold(m, T)
    Ac, Bc = compose_gdn_affines(A_first=A1, B_first=B1, A_second=A2, B_second=B2)
    assert _rel(Ac, Af) < 1e-10
    assert _rel(Bc, Bf) < 1e-10


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
