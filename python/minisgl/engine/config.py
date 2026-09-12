from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from transformers import GenerationConfig

from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config
from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    # Shared decode uses explicit graph sizes; topology stays device metadata.
    shared_cuda_graph_max_depth: int = 16
    # Explicit full shared-prefill token-row capacities; None keeps prefill eager.
    # cuda_graph_max_bs=0 disables both phases, including these profiles.
    shared_cuda_graph_prefill_rows: List[int] | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # Default query-row budget per shared-cache prefill forward; None disables chunking.
    # Read by SharedCacheSession, not by the engine itself.
    max_prefill_rows: int | None = None
    # Opt-in loading of serialized dynamic block-FP8 checkpoints. None keeps BF16/FP16.
    quantization: str | None = None
    generation_config: GenerationConfig = None
    distributed_addr: str = "tcp://127.0.0.1:2333"

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def _default_generation_config(self) -> GenerationConfig:
        """The configured generation config, or the model's own.

        Cached because resolving it hits the HF hub, and the shared-cache paths
        ask for default sampling params once per request per forward.
        """
        try:
            return GenerationConfig.from_pretrained(self.model_path)
        except OSError:  # missing file
            return GenerationConfig()

    def get_default_sampling_params(self):  # not a property to avoid accidental in-place changes
        generation_config = (
            self.generation_config
            if self.generation_config is not None
            else self._default_generation_config
        )
        return SamplingParams.from_hf(generation_config)

    @cached_property
    def model_config(self) -> ModelConfig:
        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len
