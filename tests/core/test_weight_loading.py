"""Checkpoint loading must work without large pinned host allocations."""

import minisgl.distributed.info as dist_info
import pytest
import torch
from minisgl.distributed import DistributedInfo
from minisgl.models.weight import load_weight
from safetensors.torch import save_file
from transformers import Qwen2Config


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "device_type",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
        ),
    ],
)
def test_load_checkpoint_without_pinned_memory(tmp_path, monkeypatch, rank, device_type):
    Qwen2Config(
        vocab_size=6,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        architectures=["Qwen2ForCausalLM"],
    ).save_pretrained(tmp_path)
    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(rank, 2))

    attention = "model.layers.0.self_attn."
    mlp = "model.layers.0.mlp."
    specs = {
        "model.embed_tokens.weight": ((6, 8), torch.bfloat16),
        "model.norm.weight": ((8,), torch.float32),
        attention + "q_proj.weight": ((8, 8), torch.bfloat16),
        attention + "k_proj.weight": ((4, 8), torch.bfloat16),
        mlp + "gate_proj.weight": ((6, 8), torch.float32),
        attention + "v_proj.weight": ((4, 8), torch.bfloat16),
        mlp + "up_proj.weight": ((6, 8), torch.float32),
        attention + "o_proj.weight": ((8, 8), torch.bfloat16),
        mlp + "down_proj.weight": ((8, 6), torch.float32),
    }
    tensors = {
        name: (torch.arange(torch.Size(shape).numel()).reshape(shape) + index * 8).to(dtype)
        for index, (name, (shape, dtype)) in enumerate(specs.items())
    }
    # Both fusion groups span files, exercising the streaming merge buffers.
    items = list(tensors.items())
    save_file(dict(items[:5]), tmp_path / "model-00001-of-00002.safetensors")
    save_file(dict(items[5:]), tmp_path / "model-00002-of-00002.safetensors")

    empty = torch.empty

    def reject_pinned_allocation(*args, **kwargs):
        if kwargs.get("pin_memory", False):
            raise AssertionError("Weight loading must not require pinned host allocations")
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", reject_pinned_allocation)
    device = torch.device("cuda:0" if device_type == "cuda" else "cpu")
    loaded = dict(load_weight(str(tmp_path), device))

    expected = {
        "model.embed_tokens.weight": tensors["model.embed_tokens.weight"][
            rank * 3 : (rank + 1) * 3
        ],
        "model.norm.weight": tensors["model.norm.weight"],
        attention
        + "qkv_proj.weight": torch.cat(
            [
                tensors[attention + "q_proj.weight"][rank * 4 : (rank + 1) * 4],
                tensors[attention + "k_proj.weight"][rank * 2 : (rank + 1) * 2],
                tensors[attention + "v_proj.weight"][rank * 2 : (rank + 1) * 2],
            ]
        ),
        mlp
        + "gate_up_proj.weight": torch.cat(
            [
                tensors[mlp + "gate_proj.weight"][rank * 3 : (rank + 1) * 3],
                tensors[mlp + "up_proj.weight"][rank * 3 : (rank + 1) * 3],
            ]
        ),
        attention
        + "o_proj.weight": tensors[attention + "o_proj.weight"][:, rank * 4 : (rank + 1) * 4],
        mlp + "down_proj.weight": tensors[mlp + "down_proj.weight"][:, rank * 3 : (rank + 1) * 3],
    }
    assert loaded.keys() == expected.keys()
    for name, reference in expected.items():
        assert loaded[name].device == device
        assert loaded[name].dtype == reference.dtype
        assert loaded[name].is_contiguous()
        torch.testing.assert_close(loaded[name].cpu(), reference, rtol=0, atol=0)
