from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.layers import BaseOP, LinearReplicated, MoELayer
from minisgl.utils import nvtx_annotate

from .qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5ForCausalLM, Qwen3_5Model
from .utils import GatedMLP

if TYPE_CHECKING:
    from .config import ModelConfig


def _shared_expert_gate(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Keep the scalar gate FP32 through the expert sum, as in SGLang.

    A BF16 Linear followed by BF16 sigmoid loses two intermediate roundings.
    This is intentionally eager arithmetic, not a new fused implementation.
    """
    return (x.float() * weight.float()).sum(-1, keepdim=True).sigmoid()


class Qwen3_5MoeMLP(BaseOP):
    """Qwen3.5-MoE routed experts plus gated shared expert."""

    def __init__(self, config: ModelConfig):
        assert config.num_experts > 0
        assert config.num_experts_per_tok > 0
        assert config.moe_intermediate_size > 0
        assert config.shared_expert_intermediate_size > 0
        self.gate = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=False,
        )
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            activation=config.hidden_act,
        )
        self.shared_expert = GatedMLP(
            config,
            intermediate_size=config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = LinearReplicated(
            config.hidden_size,
            1,
            has_bias=False,
        )

    @nvtx_annotate("MoE")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = x.shape
        hidden_states = x.view(-1, hidden_dim)
        # The fused MoE backend may reuse ``hidden_states`` as its output buffer.
        # Compute every shared-expert input before dispatching routed experts.
        shared = self.shared_expert.forward(hidden_states)
        shared_gate = _shared_expert_gate(hidden_states, self.shared_expert_gate.weight)
        router_logits = self.gate.forward(hidden_states)
        routed = self.experts.forward(hidden_states, router_logits)
        return (routed.float() + shared_gate * shared.float()).to(x.dtype).view(num_tokens, hidden_dim)


class Qwen3_5MoeDecoderLayer(Qwen3_5DecoderLayer):
    mlp_cls = Qwen3_5MoeMLP


class Qwen3_5MoeModel(Qwen3_5Model):
    decoder_layer_cls = Qwen3_5MoeDecoderLayer


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    model_cls = Qwen3_5MoeModel


__all__ = [
    "Qwen3_5MoeMLP",
    "Qwen3_5MoeDecoderLayer",
    "Qwen3_5MoeModel",
    "Qwen3_5MoeForCausalLM",
]
