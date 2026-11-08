"""A conditional ordinary-generation baseline for short, EOS-terminated Llama drafts."""

from __future__ import annotations

import asyncio
import gc
import json
import os
from contextlib import aclosing, nullcontext, redirect_stdout
from dataclasses import asdict, dataclass
from typing import Sequence

from minisgl.core import SamplingParams
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM

from minisgl_examples.training.recording import (
    Samples,
    build_training_example,
    generate_with_targets,
)

MAX_SHORT_DRAFT = 16


@dataclass
class ShortRetryResult:
    """Content IDs exclude sampled EOS. Both phases spend the same writer allowance."""

    draft_ids: list[int]
    draft_stop_id: int | None
    retry_ids: list[int] | None = None
    retry_stop_id: int | None = None

    @property
    def source(self) -> str:
        return "async_draft" if self.retry_ids is None else "native_retry"

    @property
    def final_ids(self) -> list[int]:
        return self.draft_ids if self.retry_ids is None else self.retry_ids

    @property
    def sampled_tokens(self) -> int:
        return (
            len(self.draft_ids)
            + int(self.draft_stop_id is not None)
            + len(self.retry_ids or ())
            + int(self.retry_stop_id is not None)
        )


class DraftStopObserver:
    """Observe writer EOS as it is sampled by the demo's prompt/thinker/writer view."""

    def __init__(self, llm: AsyncLLM):
        self.llm = llm
        self.writer_stop_id = None

    def __getattr__(self, name):
        return getattr(self.llm, name)

    @property
    def async_engine(self):
        return self.llm.async_engine

    @async_engine.setter
    def async_engine(self, engine):
        self.llm.async_engine = engine

    async def async_generate(self, context, **kwargs):
        writer = len(context.cache_view) == 3
        async with aclosing(self.llm.async_generate(context, **kwargs)) as stream:
            async for result in stream:
                token = result[0] if isinstance(result, tuple) else result
                if writer and token in self.llm.eos_token_ids:
                    self.writer_stop_id = int(token)
                yield result


def retry_allowance(draft_length: int, draft_stop_id: int | None, writer_budget: int) -> int:
    """Return the remaining sample allowance for one eligible retry, counting draft EOS."""
    samples = draft_length + int(draft_stop_id is not None)
    if draft_length < 0 or writer_budget < 1 or samples > writer_budget:
        raise ValueError("Draft samples must fit the positive writer budget")
    if draft_stop_id is None or draft_length > MAX_SHORT_DRAFT:
        return 0
    return writer_budget - samples


async def generate_native_answer(
    llm: AsyncLLM, prompt_ids: Sequence[int], budget: int, *, samples: Samples | None = None
) -> tuple[list[int], int | None]:
    """Greedy ordinary generation on a private block, starting from the final prompt token."""
    if not len(prompt_ids) or budget < 1:
        raise ValueError("Native generation needs a nonempty prompt and a positive sample budget")
    cancelled = asyncio.Event()

    async def decode():
        block = await llm.create_block()
        output = []
        try:
            if len(prompt_ids) > 1:
                await llm.forward(prompt_ids[:-1], write_to=block, return_logits=False)
            context = AsyncContext(cache_view=[block], next_input_id=int(prompt_ids[-1]))
            if cancelled.is_set():
                return output, None
            async with aclosing(
                generate_with_targets(
                    llm,
                    context,
                    samples,
                    sampling_params=SamplingParams(temperature=0.0, max_tokens=budget),
                    max_steps=budget,
                )
            ) as stream:
                async for token in stream:
                    if token in llm.eos_token_ids:
                        return output, int(token)
                    output.append(int(token))
                    if cancelled.is_set():
                        break
            return output, None
        finally:
            await llm.free_block(block)

    task = asyncio.create_task(decode())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Drain the in-flight model request before releasing any of its KV.
        cancelled.set()
        await asyncio.gather(task, return_exceptions=True)
        raise


async def retry_short_answer(
    llm: AsyncLLM,
    *,
    prompt_ids: Sequence[int],
    draft_ids: Sequence[int],
    draft_stop_id: int | None,
    max_answer_tokens: int,
    samples: Samples | None = None,
) -> ShortRetryResult:
    """Retry a short EOS draft once. A capped or immediately empty retry is still the output."""
    if draft_stop_id is not None and draft_stop_id not in llm.eos_token_ids:
        raise ValueError("draft_stop_id must be a sampled checkpoint EOS")
    if any(token in llm.eos_token_ids for token in draft_ids):
        raise ValueError("draft_ids contain EOS; supply the stop ID separately")
    result = ShortRetryResult(list(draft_ids), draft_stop_id)
    remaining = retry_allowance(len(draft_ids), draft_stop_id, max_answer_tokens)
    if remaining:
        result.retry_ids, result.retry_stop_id = await generate_native_answer(
            llm, prompt_ids, remaining, samples=samples
        )
    return result


def _example(recorded, samples, problem, thoughts, answer):
    record = build_training_example(recorded, samples)
    record["trace"] = asdict(record["trace"])
    return {"prompt": problem, "thinker_tokens": thoughts, "writer_tokens": answer, **record}


async def _run(config, llm):
    from minisgl_examples.demos.async_thoughts.demo import LlamaPrompting, _run_loop

    try:
        if llm.config.model_config.model_type != "llama":
            raise ValueError("Short-answer native retry requires a Llama checkpoint")
        if config.trace_output is not None and config.trace_output.exists():
            raise FileExistsError(config.trace_output)
        prompting = LlamaPrompting(config.problem, llm.tokenizer, protocol=config.llama_protocol)
        prompt_ids = llm.tokenizer.encode(prompting.input_prompt, add_special_tokens=False)
        observer = DraftStopObserver(llm)
        recording = config.trace_output is not None
        draft_samples = {} if recording else None
        with RecordedAsyncLLM(observer) if recording else nullcontext(observer) as draft_llm:
            # This CLI owns stdout. The public API above never redirects another caller's streams.
            with open(os.devnull, "w") as stream, redirect_stdout(stream):
                thoughts, draft = await _run_loop(
                    config, draft_llm, llm.tokenizer, prompting, draft_samples
                )
            if recording:
                draft_trace = _example(draft_llm, draft_samples, config.problem, thoughts, draft)

        native_samples = {} if recording else None
        with RecordedAsyncLLM(llm) if recording else nullcontext(llm) as native_llm:
            outcome = await retry_short_answer(
                native_llm,
                prompt_ids=prompt_ids,
                draft_ids=draft,
                draft_stop_id=observer.writer_stop_id,
                max_answer_tokens=config.max_answer_tokens,
                samples=native_samples,
            )
            if recording and outcome.retry_ids is not None:
                native_trace = _example(
                    native_llm, native_samples, config.problem, [], outcome.retry_ids
                )

        if recording:
            # The standard examples array contains the selected output only; retain the discarded
            # draft and its SFT labels separately so both attempts can be replayed and audited.
            provenance = {
                "protocol": config.llama_protocol,
                "source": outcome.source,
                "writer_budget": config.max_answer_tokens,
                "sampled_tokens": outcome.sampled_tokens,
                "max_short_draft": MAX_SHORT_DRAFT,
                **asdict(outcome),
            }
            chosen = draft_trace
            if outcome.retry_ids is not None:
                chosen = native_trace
                provenance["discarded_draft"] = draft_trace
            record = {
                "model": str(config.model),
                "examples": [chosen],
                "short_answer_retry": provenance,
            }
            config.trace_output.parent.mkdir(parents=True, exist_ok=True)
            with config.trace_output.open("x", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(
            f"Selected {outcome.source} ({outcome.sampled_tokens}/{config.max_answer_tokens} writer samples)"
        )
        print(llm.tokenizer.decode(outcome.final_ids, skip_special_tokens=True))
        return outcome
    finally:
        await llm.close()


def run_short_retry(config) -> ShortRetryResult:
    """Run the opt-in buffered baseline in a CLI process with a dedicated Llama engine."""
    import torch

    from minisgl_examples.demos.async_thoughts.engine import build_async_llm

    if not torch.cuda.is_available():
        raise RuntimeError("Short-answer retry requires CUDA")
    llm = build_async_llm(
        config.model,
        memory_ratio=config.memory_ratio,
        page_size=config.page_size,
        max_context_len=config.max_context_len,
    )
    try:
        return asyncio.run(_run(config, llm))
    finally:
        gc.collect()
        torch.cuda.empty_cache()
