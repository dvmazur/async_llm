from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .qwen3_5_attn import Qwen3_5Attention
from .qwen3_5_delta import Qwen3_5GatedDeltaNet
from .qwen3_5_mrope import get_rope_index
from .utils import GatedMLP

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, kv_idx: int, linear_idx: int):
        assert config.layer_types is not None
        if config.layer_types[layer_id] == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, linear_idx)
            self._is_linear = True
        else:
            self.self_attn = Qwen3_5Attention(config, kv_idx)
            self._is_linear = False
        self.mlp = GatedMLP(config)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None, mrope_positions: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        if self._is_linear:
            x = self.linear_attn.forward(x)
        else:
            x = self.self_attn.forward(x, mrope_positions)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        assert config.layer_types is not None
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        if config.is_multimodal:
            from .qwen3_5_vision import Qwen3_5VisionModel

            assert config.vision_config.out_hidden_size == config.hidden_size, (
                "vision out_hidden_size must equal LM hidden_size for the embed scatter"
            )
            self.visual = Qwen3_5VisionModel(config.vision_config)
        layers = []
        kv_idx = linear_idx = 0
        for layer_id in range(config.num_layers):
            layers.append(Qwen3_5DecoderLayer(config, layer_id, kv_idx, linear_idx))
            if config.layer_types[layer_id] == "linear_attention":
                linear_idx += 1
            else:
                kv_idx += 1
        self.layers = OPList(layers)
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        assert (pixel_values is None) == (image_grid_thw is None) == (mm_token_type_ids is None)
        mrope_positions: torch.Tensor | None = None
        if pixel_values is not None:
            assert mm_token_type_ids is not None
            image_embeds = self.visual.forward(pixel_values, image_grid_thw)  # (n_img, hidden)
            image_mask = mm_token_type_ids == 1  # 0 - text, 1 - image, 2 - video, etc
            x = x.clone()
            x[image_mask] = image_embeds.to(x.dtype)
            spatial_merge_size = self.config.vision_config.spatial_merge_size
            # TODO[jheuristic] proper batching support
            # TODO[jheuristic] post-update block positions override
            mrope_positions = get_rope_index(input_ids, mm_token_type_ids, spatial_merge_size, image_grid_thw)
            get_global_ctx().batch.mrope_span_override = int(mrope_positions.max().item()) + 1
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual, mrope_positions)
        return self.norm.forward(x, residual)[0]


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        output = self.model.forward(
            batch.input_ids,
            pixel_values=batch.pixel_values,
            image_grid_thw=batch.image_grid_thw,
            mm_token_type_ids=batch.mm_token_type_ids,
        )
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3_5ForCausalLM"]
