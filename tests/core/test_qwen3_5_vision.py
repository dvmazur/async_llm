"""Tests for Qwen3.5 vision support.

Two self-contained CPU tests of the interleaved-mRoPE position logic
(``minisgl.models.qwen3_5_mrope.get_rope_index``) always run.

Two ground-truth tests are gated on an HF dump produced by
``tmp/vis_ref.py`` (in the isolated transformers-5.12.1 env) landing at
``tmp/vis_ref.npz`` plus the Qwen3.5-0.8B checkpoint being present:
  - ``get_rope_index`` matches HF exactly;
  - the ported vision tower matches HF's vision embeddings (fp32).

Run::

    uv run pytest tests/core/test_qwen3_5_vision.py -v
    # or standalone:
    HF_HOME=/mnt/LLM .venv/bin/python tests/core/test_qwen3_5_vision.py
"""

from __future__ import annotations

import glob
import os

import torch

from minisgl.models.qwen3_5_mrope import get_rope_index

_REF = os.path.join(os.path.dirname(__file__), "..", "..", "tmp", "vis_ref.npz")
_CKPT = glob.glob("/mnt/LLM/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/")


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-30)


# --- self-contained mRoPE position tests -----------------------------------


def test_get_rope_index_text_only():
    ids = torch.tensor([5, 6, 7, 8, 9], dtype=torch.long)
    pos = get_rope_index(ids, image_token_id=999, spatial_merge_size=2, image_grid_thw=None)
    assert tuple(pos.shape) == (3, 5)
    expected = torch.arange(5).view(1, -1).expand(3, -1)
    assert torch.equal(pos, expected)  # all 3 axes equal & incrementing


def test_get_rope_index_image_compression():
    img = 999
    # 2 text, then a 1x4x4 image (4 llm tokens after 2x2 merge), then 3 text
    ids = torch.tensor([1, 1] + [img] * 4 + [2, 2, 2], dtype=torch.long)
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    pos = get_rope_index(ids, image_token_id=img, spatial_merge_size=2, image_grid_thw=grid)
    # image occupies llm grid 1x2x2 starting at pos 2; text resumes at 2 + max(4,4)//2 = 4
    exp_t = torch.tensor([0, 1, 2, 2, 2, 2, 4, 5, 6])
    exp_h = torch.tensor([0, 1, 2, 2, 3, 3, 4, 5, 6])
    exp_w = torch.tensor([0, 1, 2, 3, 2, 3, 4, 5, 6])
    assert torch.equal(pos[0], exp_t)
    assert torch.equal(pos[1], exp_h)
    assert torch.equal(pos[2], exp_w)


# --- HF-ground-truth tests (gated) -----------------------------------------


def test_get_rope_index_matches_hf_dump():
    if not os.path.exists(_REF):
        print("  [skip] tmp/vis_ref.npz not found (run tmp/vis_ref.py)")
        return
    import numpy as np

    ref = np.load(_REF)
    ids = torch.tensor(ref["input_ids"], dtype=torch.long)
    grid = torch.tensor(ref["image_grid_thw"], dtype=torch.long)
    got = get_rope_index(ids, image_token_id=248056, spatial_merge_size=2, image_grid_thw=grid)
    assert torch.equal(got, torch.tensor(ref["rope_positions"], dtype=torch.long))


def test_vision_tower_matches_hf_dump():
    if not os.path.exists(_REF) or not _CKPT:
        print("  [skip] dump or checkpoint missing")
        return
    import numpy as np
    import safetensors
    from minisgl.distributed import set_tp_info

    try:
        set_tp_info(rank=0, size=1)
    except Exception:
        pass  # already set
    from minisgl.models import ModelConfig
    from minisgl.models.qwen3_5_vision import Qwen3_5VisionModel
    from minisgl.utils import cached_load_hf_config

    path = _CKPT[0]
    cfg = ModelConfig.from_hf(cached_load_hf_config(path))
    vis = Qwen3_5VisionModel(cfg.vision_config)  # cpu, fp32
    sd = {}
    keys = set(vis.state_dict().keys())
    for f in sorted(glob.glob(path + "*.safetensors")):
        with safetensors.safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if k.startswith("model.visual."):
                    name = k[len("model.visual.") :]
                    if name in keys:
                        sd[name] = sf.get_tensor(k).float()
    assert set(sd.keys()) == keys, keys - set(sd.keys())
    vis.load_state_dict(sd)

    ref = np.load(_REF)
    with torch.no_grad():
        out = vis.forward(
            torch.tensor(ref["pixel_values"], dtype=torch.float32),
            torch.tensor(ref["image_grid_thw"], dtype=torch.long),
        )
    rel = _rel(out, torch.tensor(ref["vision_embeds"], dtype=torch.float32))
    assert rel < 1e-4, f"vision tower relL2={rel:.2e}"


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
