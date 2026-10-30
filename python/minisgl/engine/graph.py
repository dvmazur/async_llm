from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_max_bs == 0:
        return []
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    return [bs for bs in [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
            if bs <= cuda_graph_max_bs]


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        prefill_graph_rows: List[int] | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        self.replay_count = 0
        self.prefill_replay_count = 0
        self.prefill_graph_rows = sorted(set(prefill_graph_rows or [])) if self.max_graph_bs else []
        try:
            self._capture_graphs(max_seq_len, vocab_size, model)
        except BaseException:
            # A later profile can fail after earlier captures succeeded.
            # Destroy those graph handles before releasing their buffers/NCCL.
            self.destroy_cuda_graphs()
            raise

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        self.graph_map: Dict[int | tuple[str, int], torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.buffer.set_batch(batch)
            graph = self._capture_one(batch, model, self.buffer.logits[:bs], pool)
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        for rows in reversed(self.prefill_graph_rows):
            logger.info_rank0(f"Capturing shared prefill graph: token rows = {rows}")
            batch, logits = self.attn_backend.prefill_capture_batch(rows)
            self.graph_map['prefill', rows] = self._capture_one(batch, model, logits, pool)

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def _capture_one(self, batch, model, logits, pool):
        graph = torch.cuda.CUDAGraph()
        try:
            self.attn_backend.prepare_for_capture(batch)
            with get_global_ctx().forward_batch(batch):
                logits.copy_(model.forward())
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    logits.copy_(model.forward())
        finally:
            if finish := getattr(self.attn_backend, "finish_capture", None):
                finish(batch)
        return graph

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        if batch.is_prefill and not batch.is_mixed and self.prefill_graph_rows:
            buffers = getattr(batch.attn_metadata, 'graph_buffers', None)
            return buffers is not None and ('prefill', buffers.rows) in self.graph_map
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        if batch.is_prefill:
            rows = batch.attn_metadata.graph_buffers.rows
            self.attn_backend.prepare_for_replay(batch)
            self.graph_map['prefill', rows].replay()
            self.prefill_replay_count += 1
            return self.attn_backend.prefill_inputs.logits[:batch.size]
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        self.replay_count += 1
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        if batch.is_prefill:
            batch.padded_reqs = batch.reqs
            return
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        self.graph_map.clear()
        gc.collect()
        if destroy := getattr(self.attn_backend, "destroy_capture_graph", None):
            destroy()
