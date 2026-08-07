"""Qwen3.5 interleaved multimodal RoPE (mRoPE) position computation.

Copied from transformers 5.12.1 ``Qwen3_5Model.get_rope_index`` / ``get_vision_position_ids``
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
"""

from __future__ import annotations

import itertools

import torch


def get_vision_position_ids(
        start_position: int,
        grid_thw: list[int] | torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
):
    """
    Compute 3D positional indices for vision tokens derived from a single image or video input.

    The positions are generated from the input grid defined by temporal (T), height (H), and
    width (W) dimensions. Temporal and spatial dimensions can be downscaled according to the
    merge sizes used in the vision backbone. The resulting positions are offset by `start_position`.

    Args:
        start_position (`int`):
            Offset added to all computed positional indices.
        grid_thw (`Sequence[int]` or `torch.Tensor` of shape `(3,)`):
            The (T, H, W) grid representing the feature layout of the current image or video after patch embedding.
        temp_merge_size (`int`, *optional*):
            Factor by which the temporal dimension is reduced in the backbone. The temporal grid size is divided
            by this value. Defaults to 1.
        spatial_merge_size (`int`, *optional*):
            Factor by which the spatial dimensions (H and W) are reduced in the backbone. Both H and W are divided
            by this value. Defaults to 1.
        time_interval (`int`, *optional*):
            Spacing factor applied between consecutive temporal position indices.Defaults to 1.
        device (`str` or `torch.device`, *optional*):
            Device on which the resulting tensor is allocated. If `None`, uses the current default device.

    Returns:
        torch.LongTensor of shape (3, sequence_length):
            Positional indices for temporal, height, and width dimensions,
            flattened into sequence form and offset by `start_position`.
    """
    llm_grid_t, llm_grid_h, llm_grid_w = (
        grid_thw[0].item() // temp_merge_size,
        grid_thw[1].item() // spatial_merge_size,
        grid_thw[2].item() // spatial_merge_size,
    )

    position_temporal = torch.arange(llm_grid_t, device=device) * time_interval
    position_height = torch.arange(llm_grid_h, device=device) + start_position
    position_width = torch.arange(llm_grid_w, device=device) + start_position

    T_grid, H_grid, W_grid = torch.meshgrid(position_temporal, position_height, position_width, indexing="ij")
    vision_position_ids = torch.stack([T_grid, H_grid, W_grid], dim=0).reshape(3, -1)
    vision_position_ids[0] += start_position  # must be after time_interval multiply
    return vision_position_ids


def get_rope_index(
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        spatial_merge_size: int,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Difference from Qwen2VL/Qwen2.5VL's get_rope_index:
    - Since Qwen3.5 use timestamps to separate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split too.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`):
            Token type ids matching each modality to a different value in the input sequence, i.e. text (0), image (1), video (2).
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """

    # Separate video grid thw into multiple grids because timestamps are used to separate videos.
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }

    for batch_idx, current_input_ids in enumerate(input_ids):
        input_token_type = mm_token_type_ids[batch_idx]
        if attention_mask is not None:
            current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
            input_token_type = input_token_type[attention_mask[batch_idx].bool()]

        input_type_group = []
        for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            # text == 0
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
            # image == 1, video == 2
            else:
                grid_thw = next(grid_iters[modality_type])
                vision_position_ids = get_vision_position_ids(
                    current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                )
                llm_pos_ids_list.append(vision_position_ids)
                current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if attention_mask is not None:
            position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
        else:
            position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas

__all__ = ["get_rope_index"]
