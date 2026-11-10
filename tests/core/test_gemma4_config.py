"""Gemma counterparts of Qwen config/loading tests; no full checkpoint needed."""
import pytest

from minisgl.models.fp8_config import validate_fp8_config


def block_config():
    return dict(quant_method="compressed-tensors", format="float-quantized",
                quantization_status="compressed", config_groups={"FP8_BLOCK": dict(
                    targets=["Linear"], output_activations=None,
                    weights=dict(num_bits=8, type="float", symmetric=True, strategy="block",
                                 block_structure=[128, 128], dynamic=False),
                    input_activations=dict(num_bits=8, type="float", symmetric=True,
                                           strategy="group", group_size=128, dynamic=True))})


def test_fp8_formats():
    validate_fp8_config(block_config())
    validate_fp8_config(dict(quant_method="fp8", fmt="e4m3", activation_scheme="dynamic",
                             weight_block_size=[128, 128]))


@pytest.mark.parametrize("part,key,value", [
    ("weights", "strategy", "channel"), ("weights", "block_structure", [64, 128]),
    ("weights", "num_bits", 4), ("weights", "type", "int"),
    ("weights", "symmetric", False), ("weights", "dynamic", True),
    ("input_activations", "dynamic", False), ("input_activations", "group_size", 64),
    ("input_activations", "strategy", "token"),
])
def test_rejects_different_quantization(part, key, value):
    config = block_config()
    config["config_groups"]["FP8_BLOCK"][part][key] = value
    with pytest.raises(ValueError, match="serialized"):
        validate_fp8_config(config)


@pytest.mark.parametrize("field,value", [
    ("format", "pack-quantized"), ("quantization_status", "initialized"),
    ("kv_cache_scheme", {"num_bits": 8}), ("transform_config", {"rotation": True}),
    ("config_groups", {}),
])
def test_rejects_other_storage(field, value):
    config = block_config()
    config[field] = value
    with pytest.raises(ValueError, match="serialized"):
        validate_fp8_config(config)


def tiny_config():
    from transformers import Gemma4TextConfig
    return Gemma4TextConfig(hidden_size=256, intermediate_size=512, vocab_size=512,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            num_global_key_value_heads=1, head_dim=128, global_head_dim=256,
                            attention_k_eq_v=True,
                            layer_types=["sliding_attention", "full_attention"],
                            num_kv_shared_layers=0, hidden_size_per_layer_input=0,
                            enable_moe_block=False, max_position_embeddings=4096)


def test_gemma_shapes_and_config_preserved():
    from minisgl.models.config import ModelConfig
    hf = tiny_config()
    c = ModelConfig.from_hf(hf)
    assert c.hf_config is hf
    assert c.is_gemma4 and not c.is_hybrid and not c.is_moe
    assert c.num_kv_layers == 2
    assert c.kv_layer_shapes == ((2, 128), (1, 256))
    assert c.hidden_act == "gelu_pytorch_tanh"


@pytest.mark.parametrize("name,value", [("enable_moe_block", True),
                                      ("hidden_size_per_layer_input", 32),
                                      ("num_kv_shared_layers", 1)])
def test_unsupported_gemma_variants_fail_explicitly(name, value):
    from minisgl.models.config import ModelConfig
    config = tiny_config()
    setattr(config, name, value)
    with pytest.raises(ValueError, match="dense layers"):
        ModelConfig.from_hf(config)


def test_ct_loads_scales_without_inverting_or_quantizing(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file
    import minisgl.distributed.info as info
    from minisgl.distributed import DistributedInfo
    from minisgl.models.weight import load_weight
    config = tiny_config()
    config.quantization_config = block_config()
    config.save_pretrained(tmp_path)
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 1))
    weight = torch.arange(128 * 256).reshape(128, 256).remainder(31).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.125, 0.25]])
    prefix = "model.layers.0.self_attn.o_proj"
    save_file({prefix + ".weight": weight, prefix + ".weight_scale": scales},
              tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="explicit"):
        list(load_weight(str(tmp_path), torch.device("cpu")))
    loaded = dict(load_weight(str(tmp_path), torch.device("cpu"), quantization="fp8"))
    assert torch.equal(loaded[prefix + ".weight"].view(torch.uint8), weight.view(torch.uint8))
    assert torch.equal(loaded[prefix + ".weight_scale_inv"], scales)


def test_gemma_does_not_silently_ignore_tensor_parallelism(monkeypatch):
    from minisgl.models.config import ModelConfig
    from minisgl.models.gemma4 import Gemma4ForCausalLM
    from minisgl.distributed import DistributedInfo
    import minisgl.distributed.info as info
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 2))
    with pytest.raises(ValueError, match="TP=1"):
        Gemma4ForCausalLM(ModelConfig.from_hf(tiny_config()))
