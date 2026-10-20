"""Concurrent thinker, writer, and live-cache probe with colored CLI output.

The probe appends only its question to [probe_prompt, thinker, writer].
Three coroutines share one engine; GPU forwards run in scheduler order.
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import torch
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext
from transformers import AutoTokenizer

from .engine import (
    ModeSwitchProbe,
    build_async_llm,
    encode,
    ends_with_double_newline,
    single_token_id,
    vocab_id_or_none,
)

# Default math problem from the original AsyncReasoning notebook.
DEFAULT_PROBLEM = "Calculate x - x^2 + x^3 for x = 5, 6, 7, 8. Return all 4 answers in \\boxed{ }."

DEFAULT_MODEL = os.environ.get("MINISGL_DEMO_MODEL", "Qwen/Qwen3.5-27B")
DEFAULT_MAX_STEPS = 1024
DEFAULT_MAX_ANSWER_TOKENS = 1024
DEFAULT_PROBE_PERIOD = 20
DEFAULT_MEMORY_RATIO = 0.85
DEFAULT_PAGE_SIZE = 1


@dataclass
class DemoConfig:
    """Everything the demo needs to run, with sane defaults baked in."""

    model: str = DEFAULT_MODEL
    problem: str = DEFAULT_PROBLEM
    max_steps: int = DEFAULT_MAX_STEPS
    max_answer_tokens: int = DEFAULT_MAX_ANSWER_TOKENS
    probe_period: int = DEFAULT_PROBE_PERIOD
    memory_ratio: float = DEFAULT_MEMORY_RATIO
    page_size: int = DEFAULT_PAGE_SIZE


class Prompting:
    """Prompt fragments + the mode-switching probe text.

    Based on AsyncReasoning's ``AsyncReasoningPrompting``.
    """

    def __init__(self, problem: str):
        self.input_prompt = f"<|im_start|>user\n{problem}\n"
        self.thinker_output_prefix = "<|im_end|>\n<|im_start|>assistant\n<think>"
        self.writer_output_prefix = " ... [SYSTEM: thoughts will continue here]\n</think>\n"
        self.mode_switching_prompt = (
            "<|im_start|>user\n"
            "You are an AI assistant that can think and write responses concurrently, "
            "and you must decide whether or not you should pause writing and think more.\n"
            "Read the current partial thoughts and response below, then decide whether "
            "you can continue writing the response without pausing (yes/no):\n"
            ' - Answer "yes" if your thoughts have enough information to write the next '
            "response paragraph, even if the full task is not solved yet.\n"
            ' - Answer "no" if your thoughts aren\'t enough to write the next response '
            "paragraph, i.e. if your response ran out of thoughts.\n"
        )
        self.mode_switching_question = (
            "...\n\nWait, are my current thoughts enough to write the next paragraph "
            "or formula? (yes/no): "
        )
        self.yes_token = "yes"
        self.no_token = "no"


# ANSI colours keyed by role.
_C = {
    "thinker": "\033[2;36m",  # dim cyan
    "writer": "\033[1;32m",  # bold green
    "state": "\033[1;33m",  # bold yellow
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def _ansi(text: str, *keys: str) -> str:
    return "".join(_C[k] for k in keys) + text + _C["reset"]


def _print_header(text: str) -> None:
    bar = "─" * 64
    print(f"\n{bar}")
    print(_ansi(f"  {text}", "state"))
    print(f"{bar}\n", flush=True)


def _print_state_change(text: str) -> None:
    print(_ansi(f"\n  [{text}]", "state"), flush=True)


def _stream_token(text: str, role: str) -> None:
    sys.stdout.write(_ansi(text, role))
    sys.stdout.flush()


async def _run_loop(
    config: DemoConfig,
    llm: AsyncLLM,
    tokenizer: AutoTokenizer,
    prompting: Prompting,
) -> tuple[List[int], List[int]]:
    if min(config.max_steps, config.max_answer_tokens, config.probe_period) < 1:
        raise ValueError("Token budgets and probe period must be positive")
    forbidden = [
        i
        for i in (vocab_id_or_none(tokenizer, t) for t in ("<|im_start|>", "<|endoftext|>"))
        if i is not None
    ]
    end_think = vocab_id_or_none(tokenizer, "</think>")
    eos = tokenizer.eos_token_id
    writer_forbidden = forbidden + ([end_think] if end_think is not None else [])
    probe = ModeSwitchProbe(
        llm,
        tokenizer,
        prompting.mode_switching_prompt,
        prompting.mode_switching_question,
        prompting.yes_token,
        prompting.no_token,
    )
    done, thinking_done = asyncio.Event(), asyncio.Event()
    writer_may_run, probe_due = asyncio.Event(), asyncio.Event()
    thinker_tokens, writer_tokens = [], []
    blocks, tasks = [], []
    start = time.perf_counter()
    try:

        async def prefill() -> None:
            print("Prefilling blocks...")
            for _ in range(3):
                blocks.append(await llm.create_block())
            prompt_blk, thinker_blk, writer_blk = blocks
            for text, view in (
                (prompting.input_prompt, [prompt_blk]),
                (prompting.thinker_output_prefix, [prompt_blk, thinker_blk]),
                (prompting.writer_output_prefix, [prompt_blk, thinker_blk, writer_blk]),
            ):
                await llm.forward(encode(text, tokenizer), view, return_logits=False)

        tasks = [asyncio.create_task(prefill())]
        await asyncio.shield(tasks[0])
        prompt_blk, thinker_blk, writer_blk = blocks
        # Feed the header newline once; extra blank lines can end Qwen3.5's thoughts early.
        thinker_ctx = AsyncContext(
            cache_view=[prompt_blk, thinker_blk], next_input_id=single_token_id("\n", tokenizer)
        )
        writer_ctx = AsyncContext(
            cache_view=[prompt_blk, thinker_blk, writer_blk],
            next_input_id=single_token_id("\n\n", tokenizer),
        )

        async def thinker_coro() -> None:
            reason = "thinker hit token limit"
            try:
                async for token in llm.async_generate(
                    thinker_ctx, forbid_ids=forbidden, max_steps=config.max_steps
                ):
                    if done.is_set():
                        break
                    if token in (end_think, eos):
                        reason = "thinker finished"
                        break
                    thinker_tokens.append(token)
                    _stream_token(tokenizer.decode([token]), "thinker")
                    if len(thinker_tokens) % config.probe_period == 0 or ends_with_double_newline(
                        thinker_tokens, tokenizer
                    ):
                        probe_due.set()
                # Commit the final budget-limited thought before releasing the writer.
                if not done.is_set() and thinker_ctx.next_input_id not in (None, end_think, eos):
                    await llm.forward(cache_view=thinker_ctx, return_logits=False)
                if not done.is_set():
                    _print_state_change(f"{reason} -> writer continues")
            except BaseException:
                done.set()
                raise
            finally:
                thinking_done.set()
                writer_may_run.set()
                probe_due.set()

        async def writer_coro() -> None:
            try:
                await writer_may_run.wait()
                if done.is_set():
                    return
                async for token in llm.async_generate(
                    writer_ctx, forbid_ids=writer_forbidden, max_steps=config.max_answer_tokens
                ):
                    if done.is_set():
                        break
                    if token == eos:
                        _print_state_change("writer hit EOS -- finished")
                        break
                    writer_tokens.append(token)
                    _stream_token(tokenizer.decode([token]), "writer")
                    if len(writer_tokens) >= config.max_answer_tokens:
                        _print_state_change("writer hit token limit -- finished")
                        break
                    if not thinking_done.is_set():
                        if ends_with_double_newline(writer_tokens, tokenizer):
                            _print_state_change("writer end-of-step -> thinker leads")
                            writer_may_run.clear()
                        # The sampled token stays pending while the writer is paused.
                        await writer_may_run.wait()
                    if done.is_set():
                        break
            finally:
                done.set()
                writer_may_run.set()
                probe_due.set()

        async def probe_coro() -> None:
            try:
                while True:
                    await probe_due.wait()
                    probe_due.clear()
                    if done.is_set() or thinking_done.is_set():
                        return
                    should_write, yes_logit, no_logit = await probe.check_continue_writing(
                        thinker_blk, writer_blk
                    )
                    # A result requested before thinking ended must not park the writer now.
                    if done.is_set() or thinking_done.is_set():
                        return
                    changed = should_write != writer_may_run.is_set()
                    if should_write:
                        writer_may_run.set()
                    else:
                        writer_may_run.clear()
                    if changed:
                        state = "thinker_and_writer" if should_write else "thinker_only"
                        _print_state_change(
                            f"probe yes={yes_logit:.2f} no={no_logit:.2f} -> {state}"
                        )
            except BaseException:
                done.set()
                writer_may_run.set()
                raise

        _print_header("Generation")
        print(_ansi("  thinker (dim cyan) | writer (bold green)\n", "dim"))
        tasks = [asyncio.create_task(coro()) for coro in (thinker_coro, writer_coro, probe_coro)]
        # Keep cancellation from reaching an engine request before its GPU work has drained.
        results = await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
        for result in results:
            if isinstance(result, BaseException):
                raise result
    finally:
        done.set()
        writer_may_run.set()
        probe_due.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await probe.close()
        for block in reversed(blocks):
            await llm.free_block(block)

    _print_header("Final")
    for label, tokens, role in (
        ("Thinker", thinker_tokens, "thinker"),
        ("Writer", writer_tokens, "writer"),
    ):
        print(_ansi(f"  {label}:", role))
        print(_ansi(tokenizer.decode(tokens, skip_special_tokens=True), role))
    _print_state_change(
        f"{len(thinker_tokens)} thought tokens; {len(writer_tokens)} answer tokens; "
        f"{time.perf_counter() - start:.1f}s"
    )
    return thinker_tokens, writer_tokens


async def _run_demo(
    config: DemoConfig, llm: AsyncLLM, prompting: Prompting
) -> tuple[List[int], List[int]]:
    tokenizer = llm.tokenizer
    try:
        return await _run_loop(config, llm, tokenizer, prompting)
    finally:
        await llm.close()


def run(config: DemoConfig) -> None:
    """Run the demo end-to-end against a freshly-built AsyncLLM."""
    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(2)

    _print_header("Async Thoughts Demo (minisgl)")
    print(f"  model    : {config.model}")
    print(f"  problem  : {config.problem}")
    print(f"  token limits: {config.max_steps} thoughts, {config.max_answer_tokens} answer")

    print("\nLoading tokenizer & engine...")
    llm = build_async_llm(
        config.model, memory_ratio=config.memory_ratio, page_size=config.page_size
    )
    prompting = Prompting(config.problem)

    try:
        asyncio.run(_run_demo(config, llm, prompting))
    finally:
        gc.collect()
        torch.cuda.empty_cache()
