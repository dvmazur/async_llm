"""Pure-torch Qwen2-VL / Qwen3.5 image preprocessing (no PIL / transformers needed).

Turns an ``H×W×C`` uint8 image into the ``(pixel_values, image_grid_thw)`` the
vision tower expects, replicating ``transformers`` ``Qwen2VLImageProcessor``:
smart-resize to a patch-aligned size, rescale to ``[0,1]``, channel-normalize,
then patchify into ``[grid_t·grid_h·grid_w, C·temporal·patch·patch]``.

The rescale/normalize/patchify path is bit-identical to transformers; the resize
step uses ``torch`` bicubic interpolation, which only *approximates* PIL's bicubic
(so an image whose size is already patch-aligned round-trips exactly, while a
resized one matches closely but not to the bit). Feeding a numpy/torch array
sidesteps needing an image-decoder library in the environment.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# CLIP-style defaults shared by Qwen2-VL / Qwen3.5.
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> Tuple[int, int]:
    """Round ``(height, width)`` to multiples of ``factor`` under a pixel budget,
    preserving aspect ratio — verbatim port of transformers' ``smart_resize``."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be < 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return max(h_bar, factor), max(w_bar, factor)


def preprocess_image(
    image,
    *,
    patch_size: int = 16,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
    min_pixels: int = 56 * 56,
    max_pixels: int = 256 * 256,
    image_mean: Sequence[float] = _MEAN,
    image_std: Sequence[float] = _STD,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess one ``H×W×C`` uint8 image (numpy array or tensor).

    Returns ``(pixel_values [N, C·T·P·P] float32, image_grid_thw [1, 3] long)``.
    """
    img = torch.as_tensor(np.asarray(image)).to(torch.float32)
    if img.ndim != 3 or img.shape[-1] not in (1, 3):
        raise ValueError(f"expected H×W×C image, got shape {tuple(img.shape)}")
    height, width, channel = img.shape
    factor = patch_size * merge_size
    rh, rw = smart_resize(height, width, factor, min_pixels, max_pixels)

    chw = img.permute(2, 0, 1)  # (C, H, W)
    if (rh, rw) != (height, width):
        # PIL uses antialiased bicubic for downscaling; torch approximates it.
        chw = F.interpolate(
            chw.unsqueeze(0), size=(rh, rw), mode="bicubic", align_corners=False, antialias=True
        ).squeeze(0)

    chw = chw / 255.0
    mean = torch.tensor(image_mean, dtype=torch.float32).view(channel, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).view(channel, 1, 1)
    chw = (chw - mean) / std

    # single frame -> pad up to temporal_patch_size by repeating the last frame
    frames = chw.unsqueeze(0)  # (1, C, rh, rw)
    if frames.shape[0] % temporal_patch_size != 0:
        pad = temporal_patch_size - frames.shape[0] % temporal_patch_size
        frames = torch.cat([frames, frames[-1:].expand(pad, -1, -1, -1)], dim=0)

    grid_t = frames.shape[0] // temporal_patch_size
    grid_h, grid_w = rh // patch_size, rw // patch_size
    patches = frames.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(
        grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
    )
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
    return flatten.contiguous(), grid_thw
