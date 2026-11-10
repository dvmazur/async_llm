"""Small mapping from the HF Gemma 4 config to mini-sglang's common fields."""

from .config import ModelConfig, RotaryConfig


def from_gemma4(config) -> ModelConfig:
    text = getattr(config, "text_config", config)
    # 31B is dense and has neither PLE nor cross-layer KV sharing. Reject those
    # other Gemma sizes instead of silently implementing a different network.
    if (text.enable_moe_block or text.hidden_size_per_layer_input
            or text.num_kv_shared_layers or text.use_bidirectional_attention == "all"):
        raise ValueError("Gemma 4 currently requires dense layers without PLE or KV sharing")
    quant = getattr(config, "quantization_config", None)
    if quant:
        from .fp8_config import validate_fp8_config
        validate_fp8_config(quant)
    rope = text.rope_parameters["sliding_attention"]
    layers = text.per_layer_config
    shapes = tuple((layer.num_key_value_heads, layer.head_dim) for layer in layers)
    first = layers[0]
    return ModelConfig(
        num_layers=text.num_hidden_layers, num_qo_heads=text.num_attention_heads,
        num_kv_heads=first.num_key_value_heads, head_dim=first.head_dim,
        hidden_size=text.hidden_size, vocab_size=text.vocab_size,
        intermediate_size=first.intermediate_size, rms_norm_eps=text.rms_norm_eps,
        rotary_config=RotaryConfig(first.head_dim, first.head_dim, text.max_position_embeddings,
                                  rope["rope_theta"], None),
        hidden_act=text.hidden_activation, tie_word_embeddings=text.tie_word_embeddings,
        num_experts=0, num_experts_per_tok=0, moe_intermediate_size=0,
        shared_expert_intermediate_size=0, norm_topk_prob=False,
        model_type="gemma4_text", architectures=config.architectures or ["Gemma4ForCausalLM"],
        layer_types=tuple(text.layer_types), vision_config=getattr(config, "vision_config", None),
        image_token_id=getattr(config, "image_token_id", -1),
        video_token_id=getattr(config, "video_token_id", -1),
        hf_config=config, kv_layer_shapes=shapes,
    )
