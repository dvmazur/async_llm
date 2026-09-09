from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


class GDNStatePool:
    """Per-request recurrent state for Qwen3.5 Gated DeltaNet (linear-attention) layers.

    Unlike the token-addressed KV cache, each linear-attention layer keeps a fixed-size
    state per running request, indexed by ``Req.table_idx`` (stable across decode steps):

    - ``conv_state``: rolling input window for the depthwise causal conv1d,
      shape ``(num_linear_layers, max_running_req, conv_dim, conv_kernel)`` (matches the
      reference ``causal_conv1d_update`` window).
    - ``recurrent_state``: the delta-rule SSM accumulator,
      shape ``(num_linear_layers, max_running_req, num_v_heads, head_k_dim, head_v_dim)``.

    ``max_running_req`` here already includes the +1 dummy slot used during padding.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        max_running_req: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        num_lin = model_config.num_linear_layers
        conv_dim = model_config.linear_conv_dim
        conv_state_len = model_config.linear_conv_kernel_dim
        self._device = device
        self.conv_state = torch.zeros(
            (num_lin, max_running_req, conv_dim, conv_state_len),
            device=device,
            dtype=dtype,
        )
        # The recurrent accumulator is kept in fp32 to match the reference math.
        self.recurrent_state = torch.zeros(
            (
                num_lin,
                max_running_req,
                model_config.linear_num_value_heads,
                model_config.linear_key_head_dim,
                model_config.linear_value_head_dim,
            ),
            device=device,
            dtype=torch.float32,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def storage_bytes(self) -> int:
        """Bytes reserved by the recurrent and convolution state tensors."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.recurrent_state, self.conv_state)
        )

    def reset(self, lin_idx: int, table_idx: int) -> None:
        """Clear a single request's state for one linear layer (fresh prefill)."""
        self.conv_state[lin_idx, table_idx].zero_()
        self.recurrent_state[lin_idx, table_idx].zero_()
