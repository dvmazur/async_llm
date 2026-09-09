import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        intermediate_size_per_partition = div_even(intermediate_size, tp_size)
        self.gate_up_proj = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
        )
        self.down_proj = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
        )
        self.gate_up_proj_scale_inv = None
        self.down_proj_scale_inv = None

    def load_state_dict(self, state_dict, *, prefix="", _internal=False):
        key = f"{prefix}.gate_up_proj" if prefix else "gate_up_proj"
        if state_dict[key].dtype == torch.float8_e4m3fn:
            for name in ("gate_up_proj", "down_proj"):
                key = f"{prefix}.{name}" if prefix else name
                value, scale = state_dict[key], state_dict[key + "_scale_inv"]
                e, n, k = value.shape
                if (value.shape != getattr(self, name).shape or value.dtype != torch.float8_e4m3fn
                        or n % 128 or k % 128 or scale.shape != (e, n // 128, k // 128)
                        or scale.dtype != torch.float32):
                    raise ValueError(f"Invalid block-FP8 experts/scales: {key}")
                setattr(self, name, torch.empty_like(value, device="meta"))
                setattr(self, name + "_scale_inv", torch.empty_like(scale, device="meta"))
        super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        ctx = get_global_ctx()
        final_hidden_states = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            **({"w1_scale": self.gate_up_proj_scale_inv, "w2_scale": self.down_proj_scale_inv}
               if self.gate_up_proj_scale_inv is not None else {}),
        )
        if self.tp_size > 1:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)
        return final_hidden_states
