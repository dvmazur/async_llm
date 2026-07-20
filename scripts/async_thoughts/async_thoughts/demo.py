"""Prompting, display scaffolding, and the async thinker/writer/probe coroutines.

Cache layout (mirrors AsyncReasoning's AsyncReasoningCache):

  input_prompt block  : chat-formatted user prompt ending with <|im_end|>.
  thinker_output block: pre-filled with "<|im_start|>assistant\\n<think>\\n",
                        grows as the thinker decodes.
  writer_output block : pre-filled with " ... [SYSTEM: thoughts will continue
                        here]\\n</think>\\n", grows as the writer decodes.
                        This block is prefilled IN CONTEXT of the prompt +
                        thinker prefix (matters numerically -- prefilling
                        standalone gives the writer prefix the wrong attention
                        outputs at deeper layers).

Concurrency (the ASYNC_SCHED_DESIGN.md user API): everything below is custom
scaffolding over the unified ``AsyncLLM.forward`` method — conditional
prefill for the block setup, a decode-mode custom-generate loop per stream
(client-side greedy pick over the raw logits), and a throwaway prefill for
the probe (``ModeSwitchProbe``, which caches the static probe-prompt
encodings and — on standard-attention models — its prefix KV, so each probe
only prefills the growing thinker/writer tail in context of it).  Thinker
and writer are independent coroutines; the scheduler
batches whichever streams are live into one forward per tick.  The
mode-switching probe runs as a third coroutine, signalled at the thinker's
cadence, so the thinker never stalls while the probe prefill is in flight —
unlike the old lock-step demo, which serialized the probe against decoding.
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
from dataclasses import dataclass
from typing import AsyncIterator, List

import torch
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext, CacheBlock
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

# Qwen3-32B at bf16 fits on a single 80 GiB GPU and keeps the writer alive;
# smaller Qwen3 models starve the writer because the probe margin is too small.
#
# Qwen3.5 (hybrid Gated-DeltaNet) models are also supported — their
# linear-attention layers compose via the GDN affine cache (see
# minisgl.shared_cache.gdn).  Even small Qwen3.5 variants (e.g.
# Qwen/Qwen3.5-0.8B) keep the writer active, so they make handy quick demos;
# for Qwen/Qwen3.5-27B lower --memory-ratio (~0.4) so the forward has headroom.
DEFAULT_MODEL = os.environ.get("MINISGL_DEMO_MODEL", "Qwen/Qwen3-32B")
DEFAULT_MAX_STEPS = 800
DEFAULT_PROBE_PERIOD = 30
DEFAULT_MEMORY_RATIO = 0.9
DEFAULT_PAGE_SIZE = 1


@dataclass
class DemoConfig:
    """Everything the demo needs to run, with sane defaults baked in."""

    model: str = DEFAULT_MODEL
    problem: str = DEFAULT_PROBLEM
    max_steps: int = DEFAULT_MAX_STEPS
    probe_period: int = DEFAULT_PROBE_PERIOD
    memory_ratio: float = DEFAULT_MEMORY_RATIO
    page_size: int = DEFAULT_PAGE_SIZE


class Prompting:
    """Prompt fragments + the mode-switching probe text.

    Verbatim port of AsyncReasoning's ``AsyncReasoningPrompting``.
    """

    def __init__(self, problem: str):
        self.input_prompt = f"<|im_start|>user\n{problem}\n"
        self.thinker_output_prefix = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
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


def _tokens_with_pending(block: CacheBlock, ctx: AsyncContext) -> List[int]:
    """The stream's full token history: KV-backed tokens + the pending one."""
    tokens = list(block.token_ids)
    if ctx.next_input_id is not None:
        tokens.append(ctx.next_input_id)
    return tokens


async def _forward_generate(
    llm: AsyncLLM, ctx: AsyncContext, forbid_ids: List[int]
) -> AsyncIterator[int]:
    """Custom greedy generate over ``AsyncLLM.forward`` (decode mode).

    Each step feeds the context's pending token and picks the next one
    client-side: mask the forbidden ids, argmax, re-seed ``next_input_id``.
    Breaking out is clean — the pending token survives, so a later loop over
    the same context resumes where it left off.
    """
    while True:
        out = await llm.forward(cache_view=ctx)
        logits = out.logits  # raw row; ours to mutate
        if forbid_ids:
            logits[forbid_ids] = float("-inf")
        token = int(logits.argmax())
        ctx.next_input_id = token
        yield token


async def _run_loop(
    config: DemoConfig,
    llm: AsyncLLM,
    tokenizer: AutoTokenizer,
    prompting: Prompting,
) -> tuple[List[int], List[int]]:
    # Forbidden token ids: never let the streams emit boundary markers.
    # Match AR's two forbidden sets, gracefully skipping any markers the
    # tokenizer doesn't have (e.g. for non-Qwen models without <think>).
    writer_forbid_names = ["</think>", "<|im_start|>", "<|endoftext|>"]
    thinker_forbid_names = ["</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    writer_forbid_ids = [
        i for i in (vocab_id_or_none(tokenizer, n) for n in writer_forbid_names) if i is not None
    ]
    thinker_forbid_ids = [
        i for i in (vocab_id_or_none(tokenizer, n) for n in thinker_forbid_names) if i is not None
    ]

    probe = ModeSwitchProbe(
        llm,
        tokenizer,
        prompting.mode_switching_prompt,
        prompting.mode_switching_question,
        yes_token=prompting.yes_token,
        no_token=prompting.no_token,
    )
    nn_id = single_token_id("\n\n", tokenizer)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1

    # Prefill the three persistent blocks via ``forward`` (conditional
    # prefill: write_to is the last block of the view).  thinker/writer
    # prefixes are prefilled IN CONTEXT of their prefix chain (matches AR;
    # see module doc).
    print("Prefilling blocks...")
    prompt_blk, thinker_blk, writer_blk = await asyncio.gather(
        llm.create_block(), llm.create_block(), llm.create_block()
    )
    await llm.forward(
        encode(prompting.input_prompt, tokenizer),
        write_to=prompt_blk,
        return_logits=False,
    )
    await llm.forward(
        encode(prompting.thinker_output_prefix, tokenizer),
        [prompt_blk, thinker_blk],
        return_logits=False,
    )
    await llm.forward(
        encode(prompting.writer_output_prefix, tokenizer),
        [prompt_blk, thinker_blk, writer_blk],
        return_logits=False,
    )

    # Per-agent contexts.  The "\n\n" separator that AR appends but does NOT
    # prefill seeds each stream's first decode input.
    thinker_ctx = AsyncContext(cache_view=[prompt_blk, thinker_blk], next_input_id=nn_id)
    writer_ctx = AsyncContext(cache_view=[prompt_blk, thinker_blk, writer_blk], next_input_id=nn_id)

    done = asyncio.Event()  # EOS / max-steps: everyone shuts down
    writer_may_run = asyncio.Event()  # starts cleared: thinker leads
    probe_due = asyncio.Event()  # thinker signals the probe's cadence

    async def thinker_coro() -> None:
        steps = 0
        try:
            async for token in _forward_generate(llm, thinker_ctx, thinker_forbid_ids):
                _stream_token(tokenizer.decode([token]), "thinker")
                steps += 1
                if done.is_set() or steps >= config.max_steps:
                    if steps >= config.max_steps:
                        _print_state_change("thinker hit max steps -- terminating")
                    break
                if steps % config.probe_period == 0 or ends_with_double_newline(
                    _tokens_with_pending(thinker_blk, thinker_ctx), tokenizer
                ):
                    probe_due.set()
        finally:
            # Unpark everyone so the gather can finish.
            done.set()
            probe_due.set()
            writer_may_run.set()

    async def writer_coro() -> None:
        while not done.is_set():
            await writer_may_run.wait()  # park until the probe says "go"
            if done.is_set():
                return

            async for token in _forward_generate(llm, writer_ctx, writer_forbid_ids):
                _stream_token(tokenizer.decode([token]), "writer")
                if token == eos_id:
                    _print_state_change("writer hit EOS -- terminating")
                    done.set()
                    return
                if done.is_set():
                    return
                # End of a paragraph: hand control back to the thinker; the
                # probe re-arms writing when the thoughts catch up.
                if token == nn_id or ends_with_double_newline(
                    _tokens_with_pending(writer_blk, writer_ctx), tokenizer
                ):
                    _print_state_change("writer end-of-step -> thinker leads")
                    writer_may_run.clear()
                    break
                if not writer_may_run.is_set():  # probe parked us mid-paragraph
                    break

    async def probe_coro() -> None:
        while not done.is_set():
            await probe_due.wait()
            probe_due.clear()
            if done.is_set():
                return
            should_write, yes_logit, no_logit = await probe.check_continue_writing(
                _tokens_with_pending(thinker_blk, thinker_ctx),
                _tokens_with_pending(writer_blk, writer_ctx),
            )
            if done.is_set():
                return
            changed = should_write != writer_may_run.is_set()
            if should_write:
                writer_may_run.set()
            else:
                writer_may_run.clear()
            if changed:
                state = "thinker_and_writer" if should_write else "thinker_only"
                _print_state_change(f"probe yes={yes_logit:.2f} no={no_logit:.2f} -> {state}")

    _print_header("Generation")
    print(_ansi("  thinker (dim cyan) | writer (bold green)\n", "dim"))

    await asyncio.gather(thinker_coro(), writer_coro(), probe_coro())

    # Final output.
    _print_header("Final")
    print(_ansi("  Thinker:", "thinker"))
    thinker_tokens = _tokens_with_pending(thinker_blk, thinker_ctx)
    print(_ansi(tokenizer.decode(thinker_tokens, skip_special_tokens=True), "thinker"))
    print()
    print(_ansi("  Writer:", "writer"))
    writer_tokens = _tokens_with_pending(writer_blk, writer_ctx)
    print(_ansi(tokenizer.decode(writer_tokens, skip_special_tokens=True), "writer"))
    print()

    await probe.close()
    for blk in (prompt_blk, thinker_blk, writer_blk):
        await llm.free_block(blk)

    return thinker_tokens, writer_tokens


async def _run_demo(
    config: DemoConfig, llm: AsyncLLM, prompting: Prompting
) -> tuple[List[int], List[int]]:
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
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
    print(f"  max steps: {config.max_steps}")

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
