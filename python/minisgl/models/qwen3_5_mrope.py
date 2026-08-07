"""Qwen3.5 interleaved multimodal RoPE (mRoPE) position computation.

Ported from transformers 5.12.1 ``Qwen3_5Model.get_rope_index`` / ``get_vision_position_ids``,
specialized to a single 1-D token sequence (mini-sglang processes flat sequences).

Text runs get 1-D positions (all 3 axes equal); image tokens get (t, h, w) grid positions
that compress the sequence (an image advances the running position by ``max(h,w)//merge``,
not by its token count).  Video is not supported yet.
"""

from __future__ import annotations

import itertools
from typing import Optional

import torch


def _vision_position_ids_3d(
    start: int, t: int, h: int, w: int, merge: int, device
) -> torch.Tensor:
    """
    (3, t*h/merge*w/merge) temporal/height/width positions for one image, offset by `start`.
    Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
    """
    lt, lh, lw = t, h // merge, w // merge
    pt = torch.arange(lt, device=device)  # time_interval = 1
    pw = torch.arange(lw, device=device) + start
    ph = torch.arange(lh, device=device) + start
    pw = pw.repeat(lh * lt)
    ph = ph.repeat_interleave(lw).repeat(lt)
    pt = pt.repeat_interleave(lh * lw) + start
    return torch.stack([pt, ph, pw], dim=0)


def get_rope_index(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    spatial_merge_size: int,
    image_grid_thw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute interleaved-mRoPE positions ``[3, T]`` for a single sequence ``input_ids [T]``.
    Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
    """
    device = input_ids.device
    ids = input_ids.tolist()
    types = mm_token_type_ids
    grids = iter(image_grid_thw.tolist()) if image_grid_thw is not None else iter(())

    parts = []
    cur = 0
    for key, group in itertools.groupby(enumerate(types), lambda x: x[1]):
        g = list(group)
        n = g[-1][0] + 1 - g[0][0]
        if key == 0:  # text
            parts.append(torch.arange(n, device=device).view(1, -1).expand(3, -1) + cur)
            cur += n
        elif key in (1, 2):  # 1 = image, 2 = video
            t, h, w = (int(v) for v in next(grids))
            parts.append(_vision_position_ids_3d(cur, t, h, w, spatial_merge_size, device))
            cur += max(h, w) // spatial_merge_size
        else:
            raise NotImplementedError(f"Unexpected mm_token_type_ids entry == {key}")
    return torch.cat(parts, dim=1).reshape(3, -1)


__all__ = ["get_rope_index"]
