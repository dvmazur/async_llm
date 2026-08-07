"""Tests for Qwen3.5 vision support.

Two CPU tests of the interleaved-mRoPE position logic
(``minisgl.models.qwen3_5_mrope.get_rope_index``) always run.

Three ground-truth tests are gated on the HF ``Qwen/Qwen3.5-0.8B``
checkpoint being loadable through ``transformers``; they run it on CUDA
if available (else CPU) in fp32 and compare against it directly:
  - ``get_rope_index`` matches HF exactly (one and two images);
  - the ported vision tower matches HF's vision embeddings (fp32).

Run::

    uv run pytest tests/core/test_qwen3_5_vision_encoder.py -v
    # or standalone:
    .venv/bin/python tests/core/test_qwen3_5_vision_encoder.py
"""

from __future__ import annotations

import functools

import pytest

import numpy as np
from PIL import Image
import torch

from transformers import AutoModelForImageTextToText, AutoProcessor

from minisgl.models.qwen3_5_mrope import get_rope_index
from minisgl.models import ModelConfig
from minisgl.models.qwen3_5_vision import Qwen3_5VisionModel


_MODEL_ID = "Qwen/Qwen3.5-0.8B"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@functools.lru_cache(maxsize=1)
def _make_hf_model():
    """HF reference model + processor."""
    model = AutoModelForImageTextToText.from_pretrained(_MODEL_ID, dtype=torch.float32, device_map=_DEVICE).eval()
    return model, AutoProcessor.from_pretrained(_MODEL_ID)


def _make_hf_inputs(processor, num_images: int):
    """Processor output for a prompt with ``num_images`` deterministic images."""

    rng = np.random.default_rng(42)
    x = np.linspace(0, 1, 96, dtype=np.float32)
    images = [
        np.broadcast_to(np.stack([255 * x, 255 * (1 - x), 255 * x], axis=-1).astype(np.uint8), (64, 96, 3))
    ] + [rng.integers(0, 256, (64, 96, 3), dtype=np.uint8) for _ in range(num_images - 1)]
    content = [{"type": "image", "image": Image.fromarray(images[i])} for i in range(0, num_images)]
    content.append({"type": "text", "text": "Describe the image."})
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


# --- mRoPE position tests ---------------------------------------------------


def test_get_rope_index_text_only():
    ids = torch.tensor([5, 6, 7, 8, 9], dtype=torch.long)
    pos = get_rope_index(ids, mm_token_type_ids=torch.zeros_like(ids), spatial_merge_size=2, image_grid_thw=None)
    assert tuple(pos.shape) == (3, 5)
    expected = torch.arange(5).view(1, -1).expand(3, -1)
    assert torch.equal(pos, expected)  # all 3 axes equal & incrementing


def test_get_rope_index_image_compression():
    img = 999
    # 2 text, then a 1x4x4 image (4 llm tokens after 2x2 merge), then 3 text
    ids = torch.tensor([1, 1] + [img] * 4 + [2, 2, 2], dtype=torch.long)
    grid = torch.tensor([1, 4, 4], dtype=torch.long)
    pos, _ = get_rope_index(ids, (ids==img).long(), spatial_merge_size=2, image_grid_thw=grid)
    # image occupies llm grid 1x2x2 starting at pos 2; text resumes at 2 + max(4,4)//2 = 4
    exp_t = torch.tensor([0, 1, 2, 2, 2, 2, 4, 5, 6])
    exp_h = torch.tensor([0, 1, 2, 2, 3, 3, 4, 5, 6])
    exp_w = torch.tensor([0, 1, 2, 3, 2, 3, 4, 5, 6])
    assert torch.equal(pos[0], exp_t)
    assert torch.equal(pos[1], exp_h)
    assert torch.equal(pos[2], exp_w)


# --- HF-ground-truth tests (gated) -----------------------------------------


@pytest.mark.parametrize("num_images", [1, 2])
def test_get_rope_index_matches_hf(num_images: int):
    model, processor = _make_hf_model()
    inputs = _make_hf_inputs(processor, num_images)
    grid = inputs["image_grid_thw"]
    assert grid.shape[0] == num_images
    got = get_rope_index(
        inputs["input_ids"][0],
        mm_token_type_ids=torch.eq(inputs["input_ids"][0], model.config.image_token_id).long(),
        spatial_merge_size=2,
        image_grid_thw=grid,
    )
    hf_inputs = inputs.to(_DEVICE)
    with torch.no_grad():
        expected, _ = model.model.get_rope_index(
            hf_inputs["input_ids"],
            image_grid_thw=hf_inputs["image_grid_thw"],
            mm_token_type_ids=hf_inputs["mm_token_type_ids"],
            attention_mask=hf_inputs.get("attention_mask"),
        )
    assert torch.equal(got.long().cpu(), expected.long().cpu())

def test_vision_tower_matches_hf():
    model, processor = _make_hf_model()
    hf_vis = model.model.visual
    cfg = ModelConfig.from_hf(model.config)
    vis = Qwen3_5VisionModel(cfg.vision_config)
    keys = set(vis.state_dict().keys())
    sd = {k: v.detach().float() for k, v in hf_vis.state_dict().items() if k in keys}
    assert set(sd.keys()) == keys, keys - set(sd.keys())
    vis.load_state_dict(sd)

    inputs = _make_hf_inputs(processor, 1)
    pixel_values = inputs["pixel_values"].float().to(_DEVICE)
    grid = inputs["image_grid_thw"].to(_DEVICE)
    with torch.no_grad():
        out = vis.forward(pixel_values, grid)
        ref = hf_vis(pixel_values, grid).pooler_output
    if isinstance(ref, tuple):  # (embeds, deepstack_features)
        ref = ref[0]
    rel = (out.float() - ref.float()).norm().item() / max(out.float().norm().item(), 1e-30)
    assert rel < 1e-4, f"vision tower relL2={rel:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
