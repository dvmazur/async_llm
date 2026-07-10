from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class VisionConfig:
    """Qwen3.5 vision-tower config (ViT). Present only for the multimodal path."""

    depth: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    out_hidden_size: int
    num_position_embeddings: int
    hidden_act: str


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    # Qwen3.5-style hybrid (Gated DeltaNet) extensions; defaults keep other models unchanged.
    partial_rotary_factor: float = 1.0
    attn_output_gate: bool = False
    layer_types: tuple[str, ...] | None = None
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 0
    # Multimodal (Qwen3.5 vision) extensions; None/absent for text-only models.
    mrope_section: tuple[int, ...] | None = None
    vision_config: VisionConfig | None = None
    image_token_id: int = -1
    video_token_id: int = -1
    vision_start_token_id: int = -1
    vision_end_token_id: int = -1

    @property
    def is_multimodal(self) -> bool:
        return self.vision_config is not None

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type

    @property
    def is_hybrid(self) -> bool:
        """True only for models with real linear-attention layers (e.g. Qwen3.5).

        Note: recent transformers adds an all-``full_attention`` ``layer_types`` to plain
        Qwen2/Qwen3 configs, so presence of ``layer_types`` alone is not sufficient.
        """
        return self.layer_types is not None and "linear_attention" in self.layer_types

    @property
    def num_kv_layers(self) -> int:
        """Number of layers that own a real KV cache (full-attention layers)."""
        if self.layer_types is None:
            return self.num_layers
        return sum(t == "full_attention" for t in self.layer_types)

    @property
    def num_linear_layers(self) -> int:
        if self.layer_types is None:
            return 0
        return sum(t == "linear_attention" for t in self.layer_types)

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_conv_dim(self) -> int:
        return 2 * self.linear_key_dim + self.linear_value_dim

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        top = config  # original (multimodal) config, before swapping to text_config
        if hasattr(config, "text_config") and config.text_config is not None:
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Rope: Qwen3.5 nests it under `rope_parameters`; Llama/Qwen use a direct
        # `rope_theta`; Mistral keeps it inside the `rope_scaling` dict.
        rope_params = getattr(config, "rope_parameters", None)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        if rope_params is not None:
            rope_scaling = None
            rope_theta = getattr(rope_params, "rope_theta")
            partial_rotary_factor = getattr(
                rope_params, "partial_rotary_factor", partial_rotary_factor
            )
        else:
            rope_scaling = getattr(config, "rope_scaling", None)
            rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]

        # Hybrid linear-attention (Gated DeltaNet) layout, present only on Qwen3.5.
        layer_types = getattr(config, "layer_types", None)
        layer_types = tuple(layer_types) if layer_types is not None else None

        # Interleaved mRoPE section (Qwen3.5), inside rope_parameters.
        mrope_section = None
        if rope_params is not None:
            ms = getattr(rope_params, "mrope_section", None)
            mrope_section = tuple(ms) if ms is not None else None

        # Vision tower + multimodal token ids (from the top-level multimodal config).
        vc = getattr(top, "vision_config", None)
        vision_config = None
        if vc is not None:
            vision_config = VisionConfig(
                depth=vc.depth,
                hidden_size=vc.hidden_size,
                num_heads=vc.num_heads,
                intermediate_size=vc.intermediate_size,
                in_channels=getattr(vc, "in_channels", 3),
                patch_size=vc.patch_size,
                temporal_patch_size=vc.temporal_patch_size,
                spatial_merge_size=vc.spatial_merge_size,
                out_hidden_size=vc.out_hidden_size,
                num_position_embeddings=vc.num_position_embeddings,
                hidden_act=getattr(vc, "hidden_act", "gelu_pytorch_tanh"),
            )

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            partial_rotary_factor=partial_rotary_factor,
            attn_output_gate=getattr(config, "attn_output_gate", False),
            layer_types=layer_types,
            linear_num_key_heads=getattr(config, "linear_num_key_heads", 0),
            linear_num_value_heads=getattr(config, "linear_num_value_heads", 0),
            linear_key_head_dim=getattr(config, "linear_key_head_dim", 0),
            linear_value_head_dim=getattr(config, "linear_value_head_dim", 0),
            linear_conv_kernel_dim=getattr(config, "linear_conv_kernel_dim", 0),
            mrope_section=mrope_section,
            vision_config=vision_config,
            image_token_id=getattr(top, "image_token_id", -1),
            video_token_id=getattr(top, "video_token_id", -1),
            vision_start_token_id=getattr(top, "vision_start_token_id", -1),
            vision_end_token_id=getattr(top, "vision_end_token_id", -1),
        )
