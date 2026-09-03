#!/usr/bin/env python3
"""Headless 16-client v48-like async shared-cache benchmark.

Each client owns independent prompt/history/image-surrogate blocks and three
concurrent streams.  Descriptor, thinker and probe tails grow in the same tick;
thinker reads descriptor and probe reads both, reproducing v48's same-step cache
dependencies without Minetest or vision preprocessing.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import importlib.util
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch


MODEL = "/workspace/models/Qwen3.6-35B-A3B"
RESULTS = Path("/workspace/results/v48_multiclient")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--clients", type=int, default=16)
    parser.add_argument("--warmup-ticks", type=int, default=2)
    parser.add_argument("--measured-ticks", type=int, default=8)
    parser.add_argument("--max-running-req", type=int, default=64)
    return parser.parse_args()


@dataclass
class ClientBlocks:
    prompt: object
    history: object
    image: object
    descriptor_prefix: object
    descriptor_tail: object
    mailbox: object
    thinker_prefix: object
    thinker_tail: object
    probe_tail: object

    def all(self) -> list[object]:
        return [
            self.prompt,
            self.history,
            self.image,
            self.descriptor_prefix,
            self.descriptor_tail,
            self.mailbox,
            self.thinker_prefix,
            self.thinker_tail,
            self.probe_tail,
        ]


class ForwardCounter:
    def __init__(self, session):
        self.session = session
        self.decode_batches: list[int] = []
        self.prefill_batches: list[int] = []
        self._decode = session.decode_step
        self._prefill = session.prefill_batch

        def decode(group, input_ids):
            self.decode_batches.append(int(group.num_workers))
            return self._decode(group, input_ids)

        def prefill(jobs):
            self.prefill_batches.append(len(jobs))
            return self._prefill(jobs)

        session.decode_step = decode
        session.prefill_batch = prefill

    def snapshot(self) -> tuple[int, int]:
        return len(self.prefill_batches), len(self.decode_batches)

    def restore(self) -> None:
        self.session.decode_step = self._decode
        self.session.prefill_batch = self._prefill


def token_ids(tokenizer, text: str, length: int) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"tokenizer produced no ids for {text!r}")
    repeated = (ids * ((length + len(ids) - 1) // len(ids)))[:length]
    return torch.tensor(repeated, dtype=torch.int32)


async def create_clients(llm, count: int) -> list[ClientBlocks]:
    clients = []
    for _ in range(count):
        blocks = await asyncio.gather(*(llm.create_block() for _ in range(9)))
        clients.append(ClientBlocks(*blocks))
    return clients


async def setup_clients(llm, clients: list[ClientBlocks]) -> None:
    prompt_ids = token_ids(llm.tokenizer, "Minecraft navigation policy. ", 64)
    history_ids = token_ids(llm.tokenizer, "Previous observations and actions. ", 24)
    image_ids = token_ids(llm.tokenizer, "Current visual observation embedding surrogate. ", 48)
    descriptor_ids = token_ids(llm.tokenizer, "Describe only decision-relevant visual change. ", 16)
    mailbox_ids = token_ids(llm.tokenizer, "Descriptor mailbox for the actor. ", 12)
    thinker_ids = token_ids(llm.tokenizer, "Reason about the next embodied action. ", 16)

    await asyncio.gather(
        *(
            llm.prefill_block(prompt_ids, write_to=client.prompt, return_logits=False)
            for client in clients
        )
    )
    await asyncio.gather(
        *(
            llm.prefill_block(
                history_ids,
                cache_view=[client.prompt],
                write_to=client.history,
                return_logits=False,
            )
            for client in clients
        )
    )
    await asyncio.gather(
        *(
            llm.prefill_block(
                image_ids,
                cache_view=[client.prompt, client.history],
                write_to=client.image,
                return_logits=False,
            )
            for client in clients
        )
    )

    static_jobs = []
    for client in clients:
        base = [client.prompt, client.history, client.image]
        static_jobs.extend(
            [
                llm.prefill_block(
                    descriptor_ids,
                    cache_view=base,
                    write_to=client.descriptor_prefix,
                    return_logits=False,
                ),
                llm.prefill_block(
                    mailbox_ids,
                    cache_view=[*base, client.descriptor_prefix],
                    write_to=client.mailbox,
                    return_logits=False,
                ),
                llm.prefill_block(
                    thinker_ids,
                    cache_view=base,
                    write_to=client.thinker_prefix,
                    return_logits=False,
                ),
            ]
        )
    # Mailbox depends on descriptor_prefix for each client. The scheduler keeps
    # conflicting requests for the following prefill tick while still batching
    # independent clients together.
    await asyncio.gather(*static_jobs)


async def one_v48_tick(llm, clients: list[ClientBlocks], tick: int) -> None:
    descriptor_token = torch.tensor([1000 + tick], dtype=torch.int32)
    thinker_token = torch.tensor([2000 + tick], dtype=torch.int32)
    probe_token = torch.tensor([3000 + tick], dtype=torch.int32)
    jobs = []
    for client in clients:
        base = [client.prompt, client.history, client.image]
        descriptor_chain = [*base, client.descriptor_prefix, client.descriptor_tail]
        thinker_chain = [
            *base,
            client.descriptor_prefix,
            client.descriptor_tail,
            client.mailbox,
            client.thinker_prefix,
            client.thinker_tail,
        ]
        probe_chain = [*thinker_chain, client.probe_tail]
        jobs.extend(
            [
                llm(descriptor_token, cache_view=descriptor_chain, return_logits=False),
                llm(thinker_token, cache_view=thinker_chain, return_logits=False),
                llm(probe_token, cache_view=probe_chain, return_logits=False),
            ]
        )
    await asyncio.gather(*jobs)


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


async def run(args: argparse.Namespace) -> dict[str, object]:
    from minisgl.llm import AsyncLLM

    llm = AsyncLLM(
        MODEL,
        dtype=torch.bfloat16,
        max_running_req=args.max_running_req,
        memory_ratio=0.80,
        max_seq_len_override=2048,
        num_page_override=32768,
        max_prefill_rows=2048,
        attention_backend="fi",
    )
    counter = ForwardCounter(llm.async_engine.session)
    clients: list[ClientBlocks] = []
    setup_seconds = None
    tick_times = []
    try:
        started = time.perf_counter()
        clients = await create_clients(llm, args.clients)
        await setup_clients(llm, clients)
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - started

        for tick in range(args.warmup_ticks):
            await one_v48_tick(llm, clients, tick)
        torch.cuda.synchronize()
        measured_start = counter.snapshot()
        for tick in range(args.measured_ticks):
            torch.cuda.synchronize()
            started = time.perf_counter()
            await one_v48_tick(llm, clients, args.warmup_ticks + tick)
            torch.cuda.synchronize()
            tick_times.append(time.perf_counter() - started)

        prefill_start, decode_start = measured_start
        measured_prefill_batches = counter.prefill_batches[prefill_start:]
        measured_decode_batches = counter.decode_batches[decode_start:]
        cache = getattr(llm.async_engine.session.sc_gdn, "compose_state_cache", None)
        cache_stats = None
        if cache is not None:
            cache_stats = {
                **cache.stats,
                "resident_entries": cache.resident_entries,
                "resident_bytes": cache.resident_bytes,
                "max_bytes": cache.max_bytes,
            }
        measured_seconds = sum(tick_times)
        measured_tokens = args.clients * 3 * args.measured_ticks
        return {
            "setup_seconds": setup_seconds,
            "tick_seconds": tick_times,
            "tick_seconds_distribution": distribution(tick_times),
            "measured_seconds": measured_seconds,
            "measured_decode_tokens": measured_tokens,
            "aggregate_decode_tokens_per_second": measured_tokens / measured_seconds,
            "measured_decode_forward_count": len(measured_decode_batches),
            "measured_decode_batch_sizes": measured_decode_batches,
            "mean_decode_batch_size": statistics.fmean(measured_decode_batches),
            "min_decode_batch_size": min(measured_decode_batches),
            "max_decode_batch_size": max(measured_decode_batches),
            "unexpected_measured_prefill_batches": measured_prefill_batches,
            "all_prefill_batch_sizes": counter.prefill_batches,
            "all_decode_batch_sizes": counter.decode_batches,
            "compose_cache_stats": cache_stats,
            "gdn_storage_bytes": getattr(llm.async_engine.session.sc_gdn, "gdn_storage_bytes", None),
            "compose_cache_ratio": getattr(
                llm.async_engine.session.sc_gdn, "compose_cache_ratio", None
            ),
        }
    finally:
        if clients:
            await asyncio.gather(*(llm.free_block(block) for client in clients for block in client.all()))
        counter.restore()
        await llm.close()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    expected_python = (repo / "python").resolve()
    resolved = importlib.util.find_spec("minisgl.llm")
    if resolved is None or resolved.origin is None:
        raise RuntimeError("cannot import minisgl.llm")
    origin = Path(resolved.origin).resolve()
    if not origin.is_relative_to(expected_python):
        raise RuntimeError(f"wrong minisgl import: {origin}, expected under {expected_python}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(
        "CONFIG",
        json.dumps(
            {
                "variant": args.variant,
                "repo": str(repo),
                "clients": args.clients,
                "warmup_ticks": args.warmup_ticks,
                "measured_ticks": args.measured_ticks,
                "minisgl": str(origin),
            }
        ),
        flush=True,
    )
    result = asyncio.run(run(args))
    output = {
        "schema_version": 1,
        "timestamp_utc": stamp,
        "variant": args.variant,
        "repo": str(repo),
        "minisgl": str(origin),
        "model": MODEL,
        "clients": args.clients,
        "streams_per_client": 3,
        "warmup_ticks": args.warmup_ticks,
        "measured_ticks": args.measured_ticks,
        "max_running_req": args.max_running_req,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "fla_core": importlib.metadata.version("fla-core"),
        "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        **result,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{args.variant}_{stamp}.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print("RESULT", json.dumps(output), flush=True)
    print("RESULT_PATH", path, flush=True)


if __name__ == "__main__":
    main()
