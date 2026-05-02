from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minisgl.core import SamplingParams

from .frontend import SharedCacheBlockReply, SharedCacheDecodeReply  # noqa: F401 — kept in globals() for decoder
from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int


@dataclass
class SharedCacheCreateBlockMsg(BaseTokenizerMsg):
    uid: int


@dataclass
class SharedCachePrefillMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, str]]


@dataclass
class SharedCacheDecodeMsg(BaseTokenizerMsg):
    uid: int
    cache_structure: List[List[str]]
    write_to: List[str]
    max_tokens: int
    sampling_params: SamplingParams


@dataclass
class SharedCacheDeleteMsg(BaseTokenizerMsg):
    block_id: str
