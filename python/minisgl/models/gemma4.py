"""Gemma 4 eager adapter: HF network, mini-sglang KV ownership/FP8.

MLP, norms, output head, vision, embedding scale, layer scalar and logits softcap
come from HF. Attention integrates the runtime's KV cache. No HF generation loop
or HF-owned persistent KV cache is used.
"""

from __future__ import annotations

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from torch import nn
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM as HFTextLM,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForConditionalGeneration as HFMultimodalLM,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextAttention,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
    Gemma4VisionRotaryEmbedding,
)

from .base import BaseLLMModel


class _FP8Linear(nn.Module):
    """Thin nn.Module binding to the existing mini-sglang FP8 kernel."""

    def __init__(self, weight, scales, bias=None):
        super().__init__()
        from minisgl.kernel.fp8 import validate_weight

        validate_weight(weight, scales)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("weight_scale_inv", scales)
        self.bias = bias

    def forward(self, x):
        from minisgl.kernel.fp8 import block_fp8_linear

        return block_fp8_linear(x, self.weight, self.weight_scale_inv, self.bias)


class Gemma4Attention(Gemma4TextAttention):
    # Adapted from Transformers Gemma4TextAttention.forward; see NOTICE.
    # Adaptation: keep the reference projections/norms and K=V semantics.
    # The runtime owns paged KV and uses block-local keys/query-relative RoPE;
    # HF DynamicCache and the reference's single-sequence mask are not used.
    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        shared_kv_states=None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = (
            self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
        )
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)
        ctx = get_global_ctx()
        output = ctx.attn_backend.forward(
            query_states[0], key_states[0], value_states[0], self.layer_idx, ctx.batch, layer=self
        )
        return self.o_proj(output.reshape(*input_shape, -1)), None


class Gemma4ForCausalLM(BaseLLMModel):
    def __init__(self, config):
        if get_tp_info().size != 1:
            raise ValueError("Gemma 4 eager adapter currently supports TP=1")
        self.config = config
        self._hf = (
            HFMultimodalLM(config.hf_config) if config.is_multimodal else HFTextLM(config.hf_config)
        )
        text_model = self._hf.model.language_model if config.is_multimodal else self._hf.model
        for i, layer in enumerate(text_model.layers):
            layer.self_attn = Gemma4Attention(text_model.config, i)
        self._hf.eval()

    @property
    def model(self):
        return self._hf.model

    def state_dict(self, *, prefix="", result=None):
        result = {} if result is None else result
        result.update(
            {(prefix + "." if prefix else "") + k: v for k, v in self._hf.state_dict().items()}
        )
        return result

    def load_state_dict(self, state_dict, **kwargs):
        for name, module in list(self._hf.named_modules()):
            weight = state_dict.get(name + ".weight")
            if (
                isinstance(module, nn.Linear)
                and weight is not None
                and weight.dtype == torch.float8_e4m3fn
            ):
                parent_name, _, leaf = name.rpartition(".")
                parent = self._hf.get_submodule(parent_name)
                setattr(
                    parent,
                    leaf,
                    _FP8Linear(
                        weight,
                        state_dict[name + ".weight_scale_inv"],
                        state_dict.get(name + ".bias"),
                    ),
                )
        if self.config.tie_word_embeddings:
            key = (
                "model.language_model.embed_tokens.weight"
                if self.config.is_multimodal
                else "model.embed_tokens.weight"
            )
            state_dict["lm_head.weight"] = state_dict[key]
        self._hf.load_state_dict(state_dict, strict=True, assign=True)
        self._hf.tie_weights()
        device = next(self._hf.parameters()).device
        # Nonpersistent reference constants are absent from safetensors. Recreate
        # them using the original constructors, never allocate a second model.
        for module in self._hf.modules():
            if isinstance(module, (Gemma4TextRotaryEmbedding, Gemma4VisionRotaryEmbedding)):
                fresh = type(module)(module.config, device=device)
                for name, value in fresh._buffers.items():
                    module._buffers[name] = value
            elif isinstance(module, Gemma4TextScaledWordEmbedding):
                module.embed_scale = torch.tensor(module.scalar_embed_scale, device=device)
        state_dict.clear()
        self._hf.eval()

    @torch.inference_mode()
    def encode_images(self, pixel_values, image_position_ids):
        features = self._hf.get_image_features(
            pixel_values=pixel_values, image_position_ids=image_position_ids, return_dict=True
        ).pooler_output
        return torch.cat(features, dim=0)

    @torch.inference_mode()
    def forward(self):
        with get_global_ctx().attn_backend.execution():
            return self._forward()

    def _forward(self):
        batch = get_global_ctx().batch
        indices = batch.attn_metadata.get_last_indices(batch.size).long()
        input_ids = batch.input_ids[None].long()
        image_embeds = batch.image_embeds
        if image_embeds is None and batch.pixel_values is not None:
            image_embeds = self.encode_images(batch.pixel_values, batch.image_position_ids)
        embeds = None
        if image_embeds is not None:
            get_global_ctx().attn_backend.ensure_inputs(batch, need_images=True)
            meta = batch.attn_metadata
            if meta.image_indices.numel() != image_embeds.shape[0]:
                raise ValueError("Image embeddings and placeholder token count differ")
            ids = input_ids.clone()
            ids[:, meta.placeholder_indices] = self.config.hf_config.text_config.pad_token_id
            embeds = self._hf.get_input_embeddings()(ids)
            embeds[0].index_copy_(0, meta.image_indices, image_embeds.to(embeds.dtype))
        text = self._hf.model.language_model if self.config.is_multimodal else self._hf.model
        # Delegate the mathematics to the original HF layers, but omit the
        # TextModel prelude: its single-sequence RoPE/masks/cache are not used
        # by our paged attention. PLE/shared-KV models are rejected by config.
        hidden = text.embed_tokens(input_ids) if embeds is None else embeds
        for layer in text.layers[: text.config.num_hidden_layers]:
            hidden = layer(hidden)
        hidden = text.norm(hidden)
        logits = self._hf.lm_head(hidden[:, indices])
        # Same order/dtype as HF Gemma4ForConditionalGeneration.forward, v5.14.1.
        cap = text.config.final_logit_softcapping
        if cap is not None:
            logits = (logits / cap).tanh() * cap
        return logits[0]
