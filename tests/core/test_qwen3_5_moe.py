from __future__ import annotations

from dataclasses import replace

import minisgl.core as core
import minisgl.distributed.info as dist_info
import torch
import torch.nn.functional as F
from minisgl.core import Context
from minisgl.distributed import DistributedInfo
from minisgl.models.config import ModelConfig
from minisgl.models.qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5ForCausalLM, Qwen3_5Model
from minisgl.models.qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeMLP,
    Qwen3_5MoeModel,
)
from minisgl.models.register import get_model_class
from minisgl.models.utils import GatedMLP
from minisgl.models.weight import load_weight
from safetensors.torch import save_file
from transformers import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock as HFQwen3_5MoeSparseMoeBlock,
)


def _tiny_hf_config() -> Qwen3_5MoeConfig:
    return Qwen3_5MoeConfig(
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "hidden_act": "silu",
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": False,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 24,
            "num_experts_per_tok": 2,
            "num_experts": 4,
            "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 0],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 64,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 64,
        },
    )


class _TorchMoeBackend:
    """Small, deterministic reference for the fused backend interface."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        assert activation == "silu"
        assert not apply_router_weight_on_input
        routing = torch.softmax(gating_output, dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(routing, topk, dim=-1)
        if renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(hidden_states.dtype)

        output = torch.zeros_like(hidden_states)
        for token_idx in range(hidden_states.shape[0]):
            for slot in range(topk):
                expert_idx = int(indices[token_idx, slot])
                gate, up = F.linear(hidden_states[token_idx], w1[expert_idx]).chunk(2)
                expert_output = F.linear(F.silu(gate) * up, w2[expert_idx])
                output[token_idx] += weights[token_idx, slot] * expert_output
        # The production fused backend writes the routed output into its input.
        hidden_states.copy_(output)
        return hidden_states


def _set_single_rank_globals(monkeypatch) -> None:
    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
    ctx = Context(page_size=1)
    ctx.moe_backend = _TorchMoeBackend()
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)


def test_qwen3_5_moe_config_and_registry(monkeypatch):
    _set_single_rank_globals(monkeypatch)
    config = ModelConfig.from_hf(_tiny_hf_config())

    assert config.model_type == "qwen3_5_moe_text"
    assert config.architectures == ["Qwen3_5MoeForConditionalGeneration"]
    assert config.intermediate_size == 0
    assert config.num_experts == 4
    assert config.num_experts_per_tok == 2
    assert config.moe_intermediate_size == 16
    assert config.shared_expert_intermediate_size == 24
    assert config.norm_topk_prob is True

    with torch.device("meta"):
        model = get_model_class(config.architectures[0], config)
    assert isinstance(model, Qwen3_5MoeForCausalLM)
    assert issubclass(Qwen3_5MoeForCausalLM, Qwen3_5ForCausalLM)
    assert issubclass(Qwen3_5MoeModel, Qwen3_5Model)
    assert issubclass(Qwen3_5MoeDecoderLayer, Qwen3_5DecoderLayer)
    assert all(isinstance(layer.mlp, Qwen3_5MoeMLP) for layer in model.model.layers.op_list)
    state = model.state_dict()
    assert state["model.layers.0.mlp.experts.gate_up_proj"].shape == (4, 32, 32)
    assert state["model.layers.0.mlp.experts.down_proj"].shape == (4, 32, 16)
    assert state["model.layers.0.mlp.shared_expert.gate_up_proj.weight"].shape == (48, 32)
    assert state["model.layers.0.mlp.shared_expert_gate.weight"].shape == (1, 32)

    dense_config = replace(
        config,
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5_text",
        intermediate_size=64,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
    )
    with torch.device("meta"):
        dense_model = get_model_class(dense_config.architectures[0], dense_config)
    assert type(dense_model) is Qwen3_5ForCausalLM
    assert all(isinstance(layer.mlp, GatedMLP) for layer in dense_model.model.layers.op_list)


def test_qwen3_5_moe_mlp_matches_transformers(monkeypatch):
    _set_single_rank_globals(monkeypatch)
    hf_config = _tiny_hf_config().text_config
    config = ModelConfig.from_hf(hf_config)

    torch.manual_seed(7)
    reference = HFQwen3_5MoeSparseMoeBlock(hf_config)
    for parameter in reference.parameters():
        parameter.data.normal_(mean=0.0, std=0.1)

    actual = Qwen3_5MoeMLP(config)
    ref_state = reference.state_dict()
    actual.load_state_dict(
        {
            "gate.weight": ref_state["gate.weight"].clone(),
            "experts.gate_up_proj": ref_state["experts.gate_up_proj"].clone(),
            "experts.down_proj": ref_state["experts.down_proj"].clone(),
            "shared_expert.gate_up_proj.weight": torch.cat(
                [
                    ref_state["shared_expert.gate_proj.weight"],
                    ref_state["shared_expert.up_proj.weight"],
                ],
                dim=0,
            ),
            "shared_expert.down_proj.weight": ref_state[
                "shared_expert.down_proj.weight"
            ].clone(),
            "shared_expert_gate.weight": ref_state["shared_expert_gate.weight"].clone(),
        }
    )
    actual.shared_expert.act_fn = lambda x: F.silu(x.chunk(2, dim=-1)[0]) * x.chunk(
        2, dim=-1
    )[1]

    hidden_states = torch.randn(1, 5, config.hidden_size)
    with torch.no_grad():
        expected = reference(hidden_states)
        got = actual.forward(hidden_states[0]).unsqueeze(0)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_qwen3_5_moe_packed_checkpoint_loading(tmp_path, monkeypatch):
    _set_single_rank_globals(monkeypatch)
    hf_config = _tiny_hf_config()
    hf_config.save_pretrained(tmp_path)

    prefix = "model.language_model.layers.0.mlp"
    tensors = {
        f"{prefix}.gate.weight": torch.randn(4, 32),
        f"{prefix}.experts.gate_up_proj": torch.randn(4, 32, 32),
        f"{prefix}.experts.down_proj": torch.randn(4, 32, 16),
        f"{prefix}.shared_expert.gate_proj.weight": torch.randn(24, 32),
        f"{prefix}.shared_expert.up_proj.weight": torch.randn(24, 32),
        f"{prefix}.shared_expert.down_proj.weight": torch.randn(32, 24),
        f"{prefix}.shared_expert_gate.weight": torch.randn(1, 32),
        "mtp.layers.0.mlp.gate.weight": torch.randn(4, 32),
    }
    save_file(tensors, tmp_path / "model.safetensors")

    loaded = dict(load_weight(str(tmp_path), torch.device("cpu")))
    runtime_prefix = "model.layers.0.mlp"
    torch.testing.assert_close(
        loaded[f"{runtime_prefix}.experts.gate_up_proj"],
        tensors[f"{prefix}.experts.gate_up_proj"],
    )
    torch.testing.assert_close(
        loaded[f"{runtime_prefix}.experts.down_proj"],
        tensors[f"{prefix}.experts.down_proj"],
    )
    torch.testing.assert_close(
        loaded[f"{runtime_prefix}.shared_expert.gate_up_proj.weight"],
        torch.cat(
            [
                tensors[f"{prefix}.shared_expert.gate_proj.weight"],
                tensors[f"{prefix}.shared_expert.up_proj.weight"],
            ],
            dim=0,
        ),
    )
    assert not any(name.startswith("mtp.") for name in loaded)


def test_qwen3_5_moe_packed_checkpoint_loading_tp(tmp_path, monkeypatch):
    hf_config = _tiny_hf_config()
    hf_config.save_pretrained(tmp_path)
    prefix = "model.language_model.layers.0.mlp"
    gate_up = torch.arange(4 * 32 * 32, dtype=torch.float32).view(4, 32, 32)
    down = torch.arange(4 * 32 * 16, dtype=torch.float32).view(4, 32, 16)
    save_file(
        {
            f"{prefix}.experts.gate_up_proj": gate_up,
            f"{prefix}.experts.down_proj": down,
        },
        tmp_path / "model.safetensors",
    )

    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(1, 2))
    loaded = dict(load_weight(str(tmp_path), torch.device("cpu")))
    runtime_prefix = "model.layers.0.mlp.experts"
    gate, up = gate_up.chunk(2, dim=1)
    expected_gate_up = torch.cat(
        (gate.chunk(2, dim=1)[1], up.chunk(2, dim=1)[1]), dim=1
    )
    torch.testing.assert_close(
        loaded[f"{runtime_prefix}.gate_up_proj"], expected_gate_up
    )
    torch.testing.assert_close(
        loaded[f"{runtime_prefix}.down_proj"], down.chunk(2, dim=2)[1]
    )
