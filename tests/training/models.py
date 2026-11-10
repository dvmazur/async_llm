from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from transformers import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)
from transformers.models.qwen3_5 import modeling_qwen3_5


def tiny_model():
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def tiny_hybrid():
    torch.manual_seed(7)
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=6,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
        },
    )
    config._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(config).eval()
    with torch.no_grad():
        model.model.layers[0].linear_attn.A_log.fill_(-1)
    return model


def torch_hybrid():
    with patch.multiple(
        modeling_qwen3_5,
        create=True,  # Older optional FLA constructor hooks disappeared in HF 5.19.
        FusedRMSNormGated=None,
        chunk_gated_delta_rule=None,
        fused_recurrent_gated_delta_rule=None,
    ):
        return tiny_hybrid()


def tiny_image_model():
    torch.manual_seed(71)
    config = Qwen3_5Config(
        image_token_id=29,
        vision_start_token_id=30,
        vision_end_token_id=31,
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 6,
            "linear_conv_kernel_dim": 4,
            "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
        },
    )
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    model = Qwen3_5ForConditionalGeneration(config).train()
    with torch.no_grad():
        model.model.language_model.layers[0].linear_attn.A_log.fill_(-1)
    return model


def four_layer_hybrid(device):
    config = tiny_hybrid().config
    config.num_hidden_layers = 4
    config.layer_types = ["linear_attention", "full_attention"] * 2
    model = Qwen3_5ForCausalLM(config).to(device).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.A_log.fill_(-1)
    return model


def live_llm(reference):
    """Represent inference's fused leaf tensors without initializing a GPU engine."""
    from minisgl.llm import AsyncLLM

    parameters = dict(reference.named_parameters())
    groups = {name: [name] for name in parameters}
    for name in parameters:
        for projections, fused in ((("q", "k", "v"), "qkv"), (("gate", "up"), "gate_up")):
            part = f".{projections[0]}_proj."
            if part in name:
                sources = [name.replace(part, f".{projection}_proj.") for projection in projections]
                for source in sources:
                    groups.pop(source)
                groups[name.replace(part, f".{fused}_proj.")] = sources
    weights = {
        name: (
            parameters[sources[0]].detach().clone()
            if len(sources) == 1
            else torch.cat([parameters[source].detach() for source in sources])
        )
        for name, sources in groups.items()
    }
    llm = object.__new__(AsyncLLM)
    llm.engine = NS(
        model=NS(state_dict=lambda: weights),
        config=NS(hf_config=reference.config, tp_info=NS(size=1), quantization=None),
    )
    llm.async_engine = NS(has_work=False)
    return llm, weights, groups
