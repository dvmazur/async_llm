from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, NamedTuple, NoReturn, Optional, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    SharedCacheBlockReply,
    SharedCacheCreateBlockBackendMsg,
    SharedCacheDecodeBackendMsg,
    SharedCacheDecodeReply,
    SharedCacheDeleteBackendMsg,
    SharedCachePrefillBackendMsg,
    UserMsg,
)
from minisgl.shared_cache import SharedBlock, SharedCacheSession, WorkerGroup
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class _PendingScDecode:
    uid: int
    group: WorkerGroup
    next_ids: torch.Tensor  # [num_workers] CPU int32 — input for the next decode step
    remaining: int
    sampling_params: SamplingParams


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or 0)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # shared cache session (None when shared_cache_page_budget == 0)
        sc_budget = config.shared_cache_page_budget
        self.sc_session: Optional[SharedCacheSession] = (
            SharedCacheSession(self.engine, max_pages=sc_budget) if sc_budget > 0 else None
        )
        self._sc_block_registry: Dict[str, SharedBlock] = {}
        self._sc_prefill_logits: Dict[str, torch.Tensor] = {}  # [1, vocab] per prefilled block
        self._pending_sc_decodes: List[_PendingScDecode] = []

        # initialize other managers; normal requests use pages after the SC slice
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            page_offset=sc_budget,
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or bool(self._pending_sc_decodes)
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)

        # Run one SC decode step only when the engine is free this iteration.
        # After _process_last_data, the previous batch's GPU work is complete.
        if ongoing_data is None and self._pending_sc_decodes:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                self._step_sc_decodes()

        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or bool(self._pending_sc_decodes)
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

        if ongoing_data is None and self._pending_sc_decodes:
            self._step_sc_decodes()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        elif isinstance(msg, SharedCacheCreateBlockBackendMsg):
            self._sc_handle_create(msg)
        elif isinstance(msg, SharedCachePrefillBackendMsg):
            self._sc_handle_prefill(msg)
        elif isinstance(msg, SharedCacheDecodeBackendMsg):
            self._sc_handle_decode(msg)
        elif isinstance(msg, SharedCacheDeleteBackendMsg):
            self._sc_handle_delete(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    # ------------------------------------------------------------------
    # Shared cache handlers
    # ------------------------------------------------------------------

    def _sc_check(self) -> bool:
        if self.sc_session is None:
            logger.error("Received shared-cache message but shared_cache_page_budget == 0")
            return False
        return True

    def _sc_handle_create(self, msg: SharedCacheCreateBlockBackendMsg) -> None:
        if not self._sc_check():
            return
        block_id = str(uuid.uuid4())
        self._sc_block_registry[block_id] = self.sc_session.create_block()  # type: ignore[union-attr]
        self.send_sc_reply(SharedCacheBlockReply(uid=msg.uid, block_id=block_id))

    def _sc_handle_prefill(self, msg: SharedCachePrefillBackendMsg) -> None:
        if not self._sc_check():
            return
        block_id = str(uuid.uuid4())
        block = self.sc_session.create_block()  # type: ignore[union-attr]
        context = [self._sc_block_registry[bid] for bid in (msg.context or [])]
        logits = self.sc_session.prefill_block(block, msg.input_ids, context=context)  # type: ignore[union-attr]
        self._sc_block_registry[block_id] = block
        self._sc_prefill_logits[block_id] = logits  # shape [1, vocab_size]
        self.send_sc_reply(SharedCacheBlockReply(uid=msg.uid, block_id=block_id))

    def _sc_handle_decode(self, msg: SharedCacheDecodeBackendMsg) -> None:
        if not self._sc_check():
            return
        cache_structure = [
            [self._sc_block_registry[bid] for bid in worker_seq]
            for worker_seq in msg.cache_structure
        ]
        write_to = [self._sc_block_registry[bid] for bid in msg.write_to]
        group = WorkerGroup(cache_structure=cache_structure, write_to=write_to)

        if msg.first_tokens is not None:
            # Client-provided continuation tokens (e.g. the last token reported
            # by a previous generate call) — no re-seeding, so the transcript
            # in the cache stays continuous across calls.
            assert len(msg.first_tokens) == len(msg.write_to), (
                f"first_tokens has {len(msg.first_tokens)} entries "
                f"for {len(msg.write_to)} workers"
            )
            first_ids: List[int] = [int(t) for t in msg.first_tokens]
        else:
            # Seed first input token for each worker from the last prefilled
            # block in its sequence, and report the seeds as the first chunk
            # so the client transcript matches the cache content.
            first_ids = []
            for worker_seq in msg.cache_structure:
                logits: Optional[torch.Tensor] = None
                for bid in reversed(worker_seq):
                    if bid in self._sc_prefill_logits:
                        logits = self._sc_prefill_logits[bid]
                        break
                assert logits is not None, f"No prefill logits found for a worker in decode group {msg.uid}"
                first_ids.append(int(self._sample_from_logits(logits, msg.sampling_params).item()))
            self.send_sc_reply(
                SharedCacheDecodeReply(uid=msg.uid, worker_tokens=first_ids, finished=False)
            )

        self._pending_sc_decodes.append(
            _PendingScDecode(
                uid=msg.uid,
                group=group,
                next_ids=torch.tensor(first_ids, dtype=torch.int32),
                remaining=msg.max_tokens,
                sampling_params=msg.sampling_params,
            )
        )

    def _sc_handle_delete(self, msg: SharedCacheDeleteBackendMsg) -> None:
        block = self._sc_block_registry.pop(msg.block_id, None)
        self._sc_prefill_logits.pop(msg.block_id, None)
        if block is not None and self.sc_session is not None:
            pages = block.clear()
            if pages:
                self.sc_session._free_pages_back(
                    torch.tensor(pages, dtype=torch.int32, device=self.device)
                )

    def _step_sc_decodes(self) -> None:
        still_pending: List[_PendingScDecode] = []
        for pending in self._pending_sc_decodes:
            logits = self.sc_session.decode_step(pending.group, pending.next_ids)  # type: ignore[union-attr]
            next_ids = self._sample_from_logits(logits, pending.sampling_params)
            pending.next_ids = next_ids.cpu().to(torch.int32)
            pending.remaining -= 1
            finished = pending.remaining <= 0
            self.send_sc_reply(
                SharedCacheDecodeReply(
                    uid=pending.uid,
                    worker_tokens=pending.next_ids.tolist(),
                    finished=finished,
                )
            )
            if not finished:
                still_pending.append(pending)
        self._pending_sc_decodes = still_pending

    def _sample_from_logits(self, logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Sample one token per row from logits of shape [N, vocab_size], returning [N]."""
        logits = logits.squeeze(0) if logits.dim() == 2 and logits.shape[0] == 1 else logits
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if params.temperature == 0.0:
            return logits.argmax(dim=-1)
        scaled = logits / params.temperature
        if 0 < params.top_k < scaled.shape[-1]:
            top_k_vals = torch.topk(scaled, params.top_k, dim=-1).values
            scaled = scaled.masked_fill(scaled < top_k_vals[..., -1:], float("-inf"))
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
