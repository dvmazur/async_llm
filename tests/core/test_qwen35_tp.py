"""TP shard boundaries must preserve GDN groups and serialized FP8 scales."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedInfo
from minisgl.models.weight import _shard_tensor


def _gdn_config():
    return SimpleNamespace(
        linear_key_dim=256, linear_value_dim=768,
        linear_num_key_heads=2, linear_num_value_heads=6,
        linear_key_head_dim=128, linear_value_head_dim=128,
    )


@pytest.mark.parametrize("tensor_name", ["in_proj_qkv.weight", "conv1d.weight"])
def test_packed_gdn_shards_reconstruct_q_k_v(tensor_name):
    config = _gdn_config()
    width = 4 if tensor_name.startswith("conv") else 16
    weight = torch.arange(1280 * width).reshape(1280, width).float()
    key = "model.layers.0.linear_attn." + tensor_name
    shards = [_shard_tensor(key, weight, r, 2, config) for r in range(2)]
    split = [s.split([128, 128, 384]) for s in shards]
    rebuilt = torch.cat([torch.cat([split[r][part] for r in range(2)]) for part in range(3)])
    torch.testing.assert_close(rebuilt, weight, rtol=0, atol=0)
    if not tensor_name.startswith("conv"):
        x = torch.randn(7, width)
        local = [F.linear(x, s).split([128, 128, 384], dim=-1) for s in shards]
        actual = torch.cat(
            [torch.cat([local[r][p] for r in range(2)], dim=-1) for p in range(3)], dim=-1
        )
        torch.testing.assert_close(actual, F.linear(x, weight))


def test_gdn_fp8_scales_follow_weight_blocks():
    config = _gdn_config()
    scales = torch.arange(20).reshape(10, 2).float()
    expanded = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    prefix = "model.layers.0.linear_attn.in_proj_qkv.weight"
    for rank in range(2):
        w = _shard_tensor(prefix, expanded, rank, 2, config)
        s = _shard_tensor(prefix + "_scale_inv", scales, rank, 2, config)
        torch.testing.assert_close(
            w, s.repeat_interleave(128, 0).repeat_interleave(128, 1), rtol=0, atol=0
        )


def test_gdn_output_projection_reduces_partial_results():
    config = _gdn_config()
    weight, x = torch.randn(128, 768), torch.randn(3, 768)
    actual = sum(
        F.linear(
            x.chunk(2, dim=-1)[r],
            _shard_tensor("model.layers.0.linear_attn.out_proj.weight", weight, r, 2, config),
        )
        for r in range(2)
    )
    torch.testing.assert_close(actual, F.linear(x, weight), rtol=1e-4, atol=1e-4)


def test_hybrid_module_shapes_are_local(monkeypatch):
    import minisgl.distributed.info as info
    from minisgl.models.config import ModelConfig
    from minisgl.models.qwen3_5_attn import Qwen3_5Attention
    from minisgl.models.qwen3_5_delta import Qwen3_5GatedDeltaNet

    # Use the public 27B dimensions without downloading any checkpoint.
    from transformers import Qwen3_5Config

    c = ModelConfig.from_hf(
        Qwen3_5Config(
            text_config={
                "hidden_size": 5120,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "intermediate_size": 17408,
                "num_hidden_layers": 4,
            }
        )
    )
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 2))
    with torch.device("meta"):
        attention, gdn = Qwen3_5Attention(c, 0), Qwen3_5GatedDeltaNet(c, 0)
    assert (attention.num_qo_heads, attention.num_kv_heads) == (12, 2)
    assert attention.qkv_proj.weight.shape == (7168, 5120)
    assert attention.o_proj.weight.shape == (5120, 3072)
    assert (gdn.num_k_heads, gdn.num_v_heads) == (8, 24)
    assert gdn.in_proj_qkv.weight.shape == (5120, 5120)
    assert gdn.out_proj.weight.shape == (5120, 3072)
