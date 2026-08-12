from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch
from transformers import GenerationConfig

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool, GDNStatePool
    from minisgl.moe import BaseMoeBackend
    from minisgl.shared_cache.gdn import SharedCacheGDN


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0

    @classmethod
    def from_hf(cls, generation_config: GenerationConfig):
        temperature = getattr(generation_config, "temperature", 1)
        top_k = getattr(generation_config, "top_k", None)
        top_p = getattr(generation_config, "top_p", None)
        return cls(temperature=temperature if getattr(generation_config, "do_sample", True) else 0.0,
                   top_k=top_k if top_k is not None else -1,
                   top_p=top_p if top_p is not None else 1.0,
                   ignore_eos=bool(getattr(generation_config, "eos_token_id", None)))


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)
    # Multimodal (Qwen3.5 vision): pixel_values [N, C*T*P*P] + image_grid_thw [n,3] feed the
    # vision tower; mm_token_type_ids [T] 0 - text, 1 - image, 2 - video. None for text-only.
    pixel_values: "torch.Tensor | None" = field(default=None, init=False)
    image_grid_thw: "torch.Tensor | None" = field(default=None, init=False)
    mm_token_type_ids: "torch.Tensor | None" = field(default=None, init=False)
    # Vision-tower output [n_img_tokens, hidden] in place of pixel_values, for callers
    # that ran the tower themselves (chunked prefill: run once, slice per chunk).
    image_embeds: "torch.Tensor | None" = field(default=None, init=False)
    mrope_span_override: int | None = field(default=None, init=False)
    mrope_positions: "torch.Tensor | None" = field(default=None, init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    # Recurrent state for hybrid models (Qwen3.5 Gated DeltaNet); None otherwise.
    gdn_state: GDNStatePool | None = field(default=None, init=False)
    # Async-reasoning GDN composer; set by SharedCacheSession during a shared-cache
    # forward, None on the normal serving path (GDN layers then use gdn_state).
    gdn_ar: "SharedCacheGDN | None" = field(default=None, init=False)
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
