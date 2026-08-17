from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import Batch

if TYPE_CHECKING:
    from minisgl.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]


def resolve_mixed_batch(*, enabled: bool, is_hybrid: bool) -> bool:
    """Whether mixed prefill+decode forwards are safe for this model.

    NOTE: hybrid models dispatch their linear-attention layers on ``batch.is_prefill`` and
    rebuild the recurrent state from scratch, so a decode req inside an extend batch would
    be computed wrongly -- and wrongly *silently*, with no shape or assertion failure to
    catch it.  Keep them on alternating batches.
    """
    return enabled and not is_hybrid


def mix_batches(prefill: Batch | None, decode: Batch | None) -> Batch | None:
    """Fuse a prefill and a decode batch into one mixed batch.

    The decode reqs are appended after the extend reqs: the rest of the system reads a
    mixed batch as an extend batch whose trailing reqs have extend_len == 1, and only
    `num_decode` marks where the decode segment starts.
    """
    if prefill is None or decode is None:
        return prefill or decode
    return Batch(reqs=prefill.reqs + decode.reqs, phase="prefill", num_decode=decode.size)
