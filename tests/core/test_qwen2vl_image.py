"""Verify the pure-torch Qwen2-VL image preprocessor against a transformers oracle.

Oracle ``tmp/qwen2vl_img_ref.npz`` is produced by the isolated env
(transformers 5.12.1) running ``Qwen2VLImageProcessor`` on two synthetic images:
an *aligned* one (size already a patch multiple -> no resize) and an *unaligned*
one (needs resize). The aligned case must match bit-close (only rescale/normalize/
patchify, which we replicate exactly); the resized case matches loosely because
torch bicubic only approximates PIL's.

Run::  .venv/bin/python tests/core/test_qwen2vl_image.py
"""

from __future__ import annotations

import os

import numpy as np
import torch
from minisgl.models.qwen2vl_image import preprocess_image

_REF = os.path.join(os.path.dirname(__file__), "..", "..", "tmp", "qwen2vl_img_ref.npz")
_KW = {
    "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
    "min_pixels": 56 * 56, "max_pixels": 256 * 256,
}


def _run(name: str):
    ref = np.load(_REF)
    pv, grid = preprocess_image(ref[f"{name}_img"], **_KW)
    return pv, grid, torch.tensor(ref[f"{name}_pv"]), torch.tensor(ref[f"{name}_grid"])


def test_aligned_matches_hf_exactly():
    if not os.path.exists(_REF):
        print("  [skip] tmp/qwen2vl_img_ref.npz missing")
        return
    pv, grid, pv_hf, grid_hf = _run("aligned")
    assert torch.equal(grid, grid_hf), (grid.tolist(), grid_hf.tolist())
    assert pv.shape == pv_hf.shape, (pv.shape, pv_hf.shape)
    max_abs = (pv - pv_hf).abs().max().item()
    assert max_abs < 1e-4, f"aligned max|Δ|={max_abs}"


def test_unaligned_grid_and_close():
    if not os.path.exists(_REF):
        print("  [skip] tmp/qwen2vl_img_ref.npz missing")
        return
    pv, grid, pv_hf, grid_hf = _run("unaligned")
    # grid + shape must be exact (smart_resize is deterministic integer math)
    assert torch.equal(grid, grid_hf), (grid.tolist(), grid_hf.tolist())
    assert pv.shape == pv_hf.shape, (pv.shape, pv_hf.shape)
    # values only approximate (torch vs PIL bicubic); require rough agreement
    rel = (pv - pv_hf).norm().item() / (pv_hf.norm().item() + 1e-9)
    assert rel < 0.15, f"unaligned relL2={rel}"


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
