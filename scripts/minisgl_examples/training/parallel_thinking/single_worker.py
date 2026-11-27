"""One-worker paragraph publication with a chronologically ordered causal cache."""

from collections.abc import Sequence
from contextlib import AsyncExitStack, aclosing
from typing import Any

from minisgl.core import SamplingParams
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext

from minisgl_examples.training.parallel_thinking.rollout import (
    MathFormatting,
    _token_ids_sha256,
    extract_boxed_answer,
)


async def run_single_worker(
    llm: AsyncLLM,
    fmt: MathFormatting,
    *,
    problem: str,
    max_steps: int,
    answer_steps: int,
    sampling: SamplingParams,
    eos_token_ids: Sequence[int],
    max_context_tokens: int,
    publish_paragraphs: bool = True,
    scaffold: bool = False,
    reminder_interval: int = 1024,
    capture_trace: bool = False,
) -> dict[str, Any]:
    """Move completed KV prefixes between two blocks without rearranging their tokens."""
    if len(fmt.workers) != 1:
        raise ValueError("The causal single-worker rollout requires one worker")
    if min(max_steps, answer_steps, max_context_tokens) < 1 or reminder_interval < 0:
        raise ValueError("Budgets must be positive and the reminder interval nonnegative")
    worker = fmt.workers[0]
    tokenizer = llm.tokenizer

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    if scaffold:
        initial = encode(fmt.apply_chat_template(problem))
        others, own = encode(fmt.current_step_header), encode(fmt.current_worker_header)
        greeting = encode(fmt.get_step_prefix(worker, 1) + f"Hi, I'm {worker}.")
    else:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}], tokenize=False, add_generation_prompt=True,
            enable_thinking=True, reasoning_effort="xhigh",
        )
        initial = encode(rendered)
        others, own, greeting = [], [], []
    prompt = initial + others + own + greeting
    context_limit = min(max_context_tokens, getattr(getattr(llm, "engine", None), "max_seq_len", max_context_tokens))
    suffix = fmt.s1_finisher_suffix
    suffix_ids = encode(suffix)
    reasoning_limit = context_limit - answer_steps - len(suffix_ids)
    if len(prompt) < 2 or len(prompt) > reasoning_limit:
        raise ValueError("The prompt and final-answer reserve must fit in the selected context")
    eos = set(eos_token_ids)
    forbidden = [
        tokenizer.vocab[key] for key in ("<|im_start|>", "<|endoftext|>")
        if key in tokenizer.vocab and tokenizer.vocab[key] not in eos
    ] if scaffold else []

    completed, current = [], list(greeting)
    sampled_ids, publication_events = [], []
    rounds = publications = since_reminder = 0
    inserted = len(greeting)
    size = len(prompt)
    first_boundary = first_publication = boundary_hash = eos_id = None
    stop = "max_steps"
    async with AsyncExitStack() as cleanup:
        history = await llm.create_block()
        cleanup.push_async_callback(llm.free_block, history)
        active = await llm.create_block()
        cleanup.push_async_callback(llm.free_block, active)
        # Start on exactly the same single-block path as native generation.
        # The first real publication transfers the prompt and first paragraph
        # together. History then stays at the front of this same live view.
        await llm.forward(prompt[:-1], write_to=active, return_logits=False)
        view = [active]
        ctx = AsyncContext(cache_view=view, next_input_id=prompt[-1])
        budget = min(max_steps, reasoning_limit - size)
        async with aclosing(llm.async_generate(
            ctx, sampling_params=sampling, max_steps=budget, forbid_ids=forbidden,
        )) as stream:
            async for token in stream:
                rounds += 1
                size += 1
                current.append(token)
                if capture_trace:
                    sampled_ids.append(token)
                check_boundary = publish_paragraphs or first_boundary is None
                boundary = check_boundary and fmt.is_end_of_step(current)
                if boundary and first_boundary is None:
                    first_boundary, boundary_hash = rounds, _token_ids_sha256(current)
                if token in eos:
                    eos_id, stop = token, "eos"
                    break
                if boundary and publish_paragraphs:
                    prefix = []
                    if scaffold:
                        label = fmt.get_step_prefix(worker, publications + 2)
                        if reminder_interval and since_reminder > reminder_interval:
                            label += fmt.s1_collab_message
                            since_reminder = 0
                        prefix = encode(label)
                    if size + len(prefix) > reasoning_limit:
                        stop = "context_limit"
                        break
                    # The sampled separator is still pending. Only already
                    # encoded rows move to history. Its next normal decode
                    # (or the forced prefix prefill below) consumes it once.
                    await llm.append_block(history, active)
                    if capture_trace:
                        publication_events.append({
                            "round": rounds, "worker": worker, "step": publications + 1,
                            "committed_token_count": active.num_tokens, "next_prefix_ids": prefix,
                        })
                    await llm.free_block(active)
                    completed.extend(current)
                    current = prefix.copy()
                    publications += 1
                    if first_publication is None:
                        first_publication = rounds
                        view.insert(0, history)
                    if prefix:
                        await llm.forward([token, *prefix[:-1]], cache_view=view, return_logits=False)
                        ctx.next_input_id = prefix[-1]
                        inserted += len(prefix)
                        size += len(prefix)
                    since_reminder += len(prefix)
                since_reminder += 1
                if size >= reasoning_limit and rounds < max_steps:
                    stop = "context_limit"
                    break
        if rounds == 0:
            stop = "context_limit"

        prediction = extract_boxed_answer(tokenizer.decode([*completed, *current]))
        reasoning_stop = stop
        peak = size
        answer_tokens = []
        answer_count = 0
        answer_eos = None
        used_finisher = False
        finisher_stop = "existing_answer"
        if prediction is None:
            # A sampled EOS marks the end of the assistant turn and has no KV
            # yet. Replace it with the finisher suffix. At any other stop, the
            # pending token still belongs to the unfinished turn.
            assert ctx.next_input_id is not None
            inputs = suffix_ids if eos_id is not None else [ctx.next_input_id, *suffix_ids]
            used_finisher, finisher_stop = True, "max_steps"
            for _ in range(answer_steps):
                output = await llm.forward(inputs, cache_view=view, return_logits=True)
                ctx.next_input_id = None
                token = int(output.logits.argmax().item())
                answer_count += 1
                if token in eos:
                    answer_eos, finisher_stop = token, "eos"
                    break
                answer_tokens.append(token)
                prediction = extract_boxed_answer(suffix + tokenizer.decode(answer_tokens))
                if prediction is not None:
                    finisher_stop = "answer"
                    break
                inputs = [token]
            peak = max(peak, size - int(eos_id is not None) + len(suffix_ids) + len(answer_tokens))
    result = {
        "boxed_answer": prediction, "decode_rounds": rounds, "reasoning_generated_tokens": rounds,
        "answer_generated_tokens": answer_count, "total_generated_tokens": rounds + answer_count,
        "generated_tokens": {worker: rounds}, "peak_context_tokens": peak,
        "stop_reason": stop, "reasoning_stop_reason": reasoning_stop,
        "eos_token_id": eos_id, "eos_worker": worker if eos_id is not None else None,
        "used_finisher": used_finisher, "finisher_mode": "cached", "finisher_stop_reason": finisher_stop,
        "finisher_eos_token_id": answer_eos, "completed_steps": publications,
        "first_publication_round": first_publication, "first_boundary_round": first_boundary,
        "first_boundary_prefix_sha256": boundary_hash, "prompt_token_ids_sha256": _token_ids_sha256(prompt),
        "inserted_worker_tokens": inserted, "scaffold": scaffold,
    }
    if capture_trace:
        # The scaffold, when enabled, was encoded into this same initial block.
        # prompt_ids is the complete initial token sequence in both layouts.
        result["trace"] = {
            "layout": "scaffold" if scaffold else "causal", "workers": [worker], "prompt_ids": prompt,
            "other_header_ids": others, "own_header_ids": own,
            "initial_worker_prefix_ids": {worker: greeting}, "worker_token_ids": {worker: sampled_ids},
            "publications": publication_events, "decode_rounds": rounds, "reasoning_stop_reason": reasoning_stop,
            "eos_worker": result["eos_worker"], "eos_token_id": eos_id, "boxed_answer": prediction,
            "block_ids": {
                "workers": {worker: active.block_id}, "history": history.block_id,
                "answer": active.block_id if used_finisher else None,
            },
            "finisher": {
                "used": used_finisher, "mode": "cached", "worker_marker_ids": [],
                "suffix_ids": suffix_ids if used_finisher else [],
                "sampled_token_ids": answer_tokens + ([answer_eos] if answer_eos is not None else []),
                "stop_reason": finisher_stop, "eos_token_id": answer_eos,
            },
        }
    return result
