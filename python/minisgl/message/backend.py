from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int


@dataclass
class SharedCacheCreateBlockBackendMsg(BaseBackendMsg):
    uid: int


@dataclass
class SharedCachePrefillBackendMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32
    # Existing block ids the new block should attend to during prefill, in
    # view order (the new block is appended after them).
    context: Optional[List[str]] = None


@dataclass
class SharedCacheDecodeBackendMsg(BaseBackendMsg):
    uid: int
    cache_structure: List[List[str]]
    write_to: List[str]
    max_tokens: int
    sampling_params: SamplingParams
    # Per-worker first input token (e.g. the last token reported by a previous
    # generate call).  When None, the scheduler samples a seed from the last
    # prefilled block's logits and reports it as the first streamed chunk.
    first_tokens: Optional[List[int]] = None


@dataclass
class SharedCacheDeleteBackendMsg(BaseBackendMsg):
    block_id: str
