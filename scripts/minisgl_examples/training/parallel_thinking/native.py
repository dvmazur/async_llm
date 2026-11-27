"""Ordinary single-block Qwen chat generation for comparison with Hogwild."""

from collections.abc import Sequence
from contextlib import aclosing
from typing import Any

from minisgl.core import SamplingParams
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext

from minisgl_examples.training.parallel_thinking.rollout import extract_boxed_answer


async def run_native(
    llm: AsyncLLM,
    *,
    problem: str,
    max_steps: int,
    sampling: SamplingParams,
    eos_token_ids: Sequence[int],
    max_context_tokens: int,
    capture_trace: bool = False,
) -> dict[str, Any]:
    """Decode one ordinary xhigh Qwen response, counting each sampled token including EOS."""
    if max_steps < 1 or max_context_tokens < 1:
        raise ValueError("The generation budget and context limit must be positive")
    tokenizer = llm.tokenizer
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="xhigh",
    )
    prompt = tokenizer.encode(rendered, add_special_tokens=False)
    if len(prompt) < 2 or len(prompt) > max_context_tokens:
        raise ValueError("The native chat prompt must fit in the selected context")
    steps = min(max_steps, max_context_tokens - len(prompt))
    stop_reason = "context_limit" if steps < max_steps else "max_steps"
    tokens: list[int] = []
    eos_id = None
    block_id = None
    if steps:
        block = await llm.create_block()
        if capture_trace:
            block_id = block.block_id
        try:
            await llm.forward(prompt[:-1], write_to=block, return_logits=False)
            context = AsyncContext(cache_view=[block], next_input_id=prompt[-1])
            async with aclosing(llm.async_generate(context, sampling_params=sampling, max_steps=steps)) as stream:
                async for token in stream:
                    tokens.append(token)
                    if token in eos_token_ids:
                        eos_id = token
                        stop_reason = "eos"
                        break
        finally:
            await llm.free_block(block)
    count = len(tokens)
    answer = extract_boxed_answer(tokenizer.decode(tokens, skip_special_tokens=True))
    result = {
        "boxed_answer": answer,
        "decode_rounds": count,
        "reasoning_generated_tokens": count,
        "total_generated_tokens": count,
        "answer_generated_tokens": 0,
        "generated_tokens": {"Qwen": count},
        "peak_context_tokens": len(prompt) + count,
        "stop_reason": stop_reason,
        "reasoning_stop_reason": stop_reason,
        "used_finisher": False,
        "completed_steps": 0,
        "eos_worker": "Qwen" if eos_id is not None else None,
        "eos_token_id": eos_id,
        "finisher_mode": "none",
        "finisher_eos_token_id": None,
        "finisher_stop_reason": "not_applicable",
        "first_publication_round": None,
        "first_boundary_round": None,
        "first_boundary_prefix_sha256": None,
        "prompt_token_ids_sha256": None,
    }
    if capture_trace:
        result["trace"] = {
            "layout": "native", "workers": ["Qwen"], "prompt_ids": prompt,
            "other_header_ids": [], "own_header_ids": [], "initial_worker_prefix_ids": {"Qwen": []},
            "worker_token_ids": {"Qwen": tokens}, "publications": [],
            "block_ids": {"workers": {"Qwen": block_id}, "history": None, "answer": None},
            "decode_rounds": count, "reasoning_stop_reason": stop_reason,
            "eos_worker": result["eos_worker"], "eos_token_id": eos_id, "boxed_answer": answer,
            "finisher": {
                "used": False, "mode": "none", "worker_marker_ids": [], "suffix_ids": [],
                "sampled_token_ids": [], "stop_reason": "not_applicable", "eos_token_id": None,
            },
        }
    return result
