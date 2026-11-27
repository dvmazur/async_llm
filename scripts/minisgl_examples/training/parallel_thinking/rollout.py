"""Collaborative generation, cache publication, and prompt formatting."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import math
import time
from collections.abc import AsyncIterator, Coroutine, Mapping, Sequence
from contextlib import aclosing, asynccontextmanager
from dataclasses import asdict
from typing import Any, TypeVar

import torch
from minisgl.core import SamplingParams
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext, CacheBlock
from minisgl.training import RecordedAsyncLLM
from transformers import PreTrainedTokenizerBase

from minisgl_examples.training.recording import (
    Samples,
    build_training_example,
    generate_with_targets,
)

T = TypeVar("T")

WORKERS = ("Alice", "Bob")
NATIVE_PUBLICATION_POLICY = "encoded_separator_v1"
TERMINAL_ANSWER_POLICY = "eos_worker_last_box_v1"
PROBLEM = (
    "Three vertices of a cube are $P=(7,12,10)$, $Q=(8,8,1)$, and "
    "$R=(11,3,9)$. What is the surface area of the cube?"
)


async def solve(
    model: str,
    *,
    synchronized: bool = True,
    problem: str = PROBLEM,
    max_steps: int = 384,
    answer_steps: int = 64,
    prompt_ids: list[int] | None = None,
    bob_delay: float = 0.0,
    delay_every: int = 32,
    sampling: SamplingParams | None = None,
    memory_ratio: float = 0.65,
    max_seq_len: int = 16384,
    cuda_graphs: bool = True,
) -> dict[str, Any]:
    if max_steps < 1 or answer_steps < 1 or delay_every < 1 or bob_delay < 0:
        raise ValueError("Token budgets/delay interval must be positive and delay nonnegative")
    started = time.perf_counter()
    llm = AsyncLLM(
        str(model),
        dtype=torch.bfloat16,
        memory_ratio=memory_ratio,
        max_running_req=4,
        max_seq_len_override=max_seq_len,
        cuda_graph_bs=[1, 2] if cuda_graphs else [],
        cuda_graph_max_bs=2 if cuda_graphs else 0,
        shared_cuda_graph_max_depth=6,
        shared_cuda_graph_prefill_rows=[32, 256] if cuda_graphs else [],
        max_prefill_rows=256,
        distributed_addr="tcp://127.0.0.1:0",
    )
    torch.manual_seed(0)
    try:
        samples: Samples = {}
        with RecordedAsyncLLM(llm) as recorder:
            result = await run_agents(
                recorder,
                MathFormatting(llm.tokenizer, WORKERS, enable_thinking=False),
                problem=problem,
                synchronized=synchronized,
                max_steps=max_steps,
                answer_steps=answer_steps,
                prompt_ids=prompt_ids,
                bob_delay=bob_delay,
                delay_every=delay_every,
                sampling=sampling,
                samples=samples,
            )
            result["example"] = build_training_example(recorder, samples)
        result.update(model=str(model), total_seconds=time.perf_counter() - started)
        result["example"].update(
            model=str(model),
            prompt_ids=result["prompt_ids"],
            rollout={key: value for key, value in result.items() if key != "example"},
        )
        return result
    finally:
        await llm.close()
        gc.collect()
        torch.cuda.empty_cache()


async def run_agents(
    llm: AsyncLLM,
    fmt: MathFormatting,
    *,
    problem: str,
    synchronized: bool,
    max_steps: int,
    answer_steps: int,
    prompt_ids: list[int] | None = None,
    bob_delay: float = 0.0,
    delay_every: int = 32,
    sampling: SamplingParams | None = None,
    samples: Samples | None = None,
    total_token_budget: int | None = None,
    max_context_tokens: int | None = None,
    reminder_interval: int = 0,
    finish_if_missing: bool = False,
    eos_token_ids: Sequence[int] | None = None,
    retain_details: bool = True,
    causal_headers: bool = False,
    symbolic_answers: bool = False,
    flat_finisher: bool = False,
    cached_finisher: bool = False,
    publish_paragraphs: bool = True,
    native_workers: bool = False,
    native_answer_cue: bool | None = None,
    capture_trace: bool = False,
) -> dict[str, Any]:
    tokenizer = llm.tokenizer
    workers = tuple(getattr(fmt, "workers", WORKERS))
    count = len(workers)
    if not 1 <= count <= 16 or len(set(workers)) != count:
        raise ValueError("Expected 1 to 16 distinct worker names")
    if max_steps < 1 or answer_steps < 1 or delay_every < 1 or bob_delay < 0:
        raise ValueError("Token budgets/delay interval must be positive and delay nonnegative")
    if any(value is not None and value < 1 for value in (total_token_budget, max_context_tokens)):
        raise ValueError("Token and context budgets must be positive")
    if reminder_interval < 0:
        raise ValueError("Reminder interval must be nonnegative")
    if capture_trace and not synchronized:
        raise ValueError("Capturing an inference trace requires synchronized workers")
    if native_workers:
        if not synchronized:
            raise ValueError("Native workers require synchronized generation")
        cached_finisher, finish_if_missing = True, True
    if native_answer_cue is None:
        native_answer_cue = native_workers
    if native_answer_cue and not cached_finisher:
        raise ValueError("The native answer cue requires a cached finisher")
    if flat_finisher and cached_finisher:
        raise ValueError("Choose either the flat or the cached finisher")
    if flat_finisher and samples is not None:
        raise ValueError("The flat finisher is an inference-only option")
    if cached_finisher and (samples is not None or (isinstance(llm, RecordedAsyncLLM) and not capture_trace)):
        raise ValueError("The cached finisher is an inference-only option")
    if flat_finisher or cached_finisher:
        native_limit = getattr(getattr(llm, "engine", None), "max_seq_len", None)
        if native_limit is not None:
            max_context_tokens = min(native_limit, max_context_tokens or native_limit)
    if not synchronized and (total_token_budget or max_context_tokens or reminder_interval):
        raise ValueError("Evaluation budgets and reminders require synchronized workers")

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    blocks = []
    try:
        for _ in range(count + (2 if native_workers else 4)):
            blocks.append(await llm.create_block())
        common, *middle, answer_block = blocks
        if native_workers:
            worker_blocks, others, own = middle, [], []
            other_header_ids, own_header_ids = [], []
        else:
            others_header, own_header, *worker_blocks = middle
            others, own = [others_header], [own_header]
            other_header_ids, own_header_ids = encode(fmt.current_step_header), encode(fmt.current_worker_header)
        contexts = [
            AsyncContext(cache_view=[
                common, *others,
                *(block for j, block in enumerate(worker_blocks) if j != i),
                *own, worker_blocks[i],
            ])
            for i in range(count)
        ]
        if prompt_ids is not None:
            initial_ids = list(prompt_ids)
        elif native_workers:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}], tokenize=False, add_generation_prompt=True,
                enable_thinking=True, reasoning_effort="xhigh",
            )
            initial_ids = encode(rendered)
        else:
            initial_ids = encode(fmt.apply_chat_template(problem))
        suffix = fmt.s1_finisher_suffix
        finisher_ids = encode(suffix)
        suffix_reserve = len(finisher_ids)
        if native_answer_cue:
            open_think_id, close_think_id = tokenizer.vocab["<think>"], tokenizer.vocab["</think>"]
            answer_cue = "\n\nThe final answer is \\boxed{"
            cues = {False: (answer_cue, encode(answer_cue)),
                    True: ("</think>" + answer_cue, encode("</think>" + answer_cue))}
            suffix_reserve = max(len(ids) for _, ids in cues.values())
        marker = fmt.sep if native_workers else fmt.incomplete_step + fmt.sep
        incomplete_ids = encode(marker) if flat_finisher or cached_finisher else []
        # Both finisher views replace the own-header with one new marker per worker.
        markup_reserve = max(0, count * len(incomplete_ids) - len(own_header_ids))
        reasoning_context_limit = (
            None if max_context_tokens is None
            else max_context_tokens - suffix_reserve - answer_steps - markup_reserve
        )
        current = [[] for _ in workers] if native_workers else [
            encode(
                f"{fmt.get_step_prefix(worker, 1)}Hi, I'm {worker}."
                + (" Here's how we can collaborate" if i == 0 and count > 1 else "")
            )
            for i, worker in enumerate(workers)
        ]
        if capture_trace:
            initial_worker_prefix_ids = {worker: ids.copy() for worker, ids in zip(workers, current)}
            publications = []
        history, events, progress, reminder_events = [], [], [], []
        compare_boundary = synchronized and count == 1
        prompt_token_ids_sha256 = (
            _token_ids_sha256(initial_ids + other_header_ids + own_header_ids + current[0])
            if compare_boundary else None
        )

        def context_tokens() -> int:
            return len(initial_ids) + len(history) + len(other_header_ids) + len(own_header_ids) + sum(map(len, current))

        if reasoning_context_limit is not None and context_tokens() > reasoning_context_limit:
            raise ValueError("Prompt, worker prefixes, and final-answer reserve exceed the context budget")
        prompt_output = await llm.forward(initial_ids, write_to=common, return_logits=native_workers)
        if not native_workers:
            if causal_headers:
                await llm.forward(other_header_ids, cache_view=[common, others_header], return_logits=False)
                await llm.forward(own_header_ids, cache_view=[common, others_header, own_header], return_logits=False)
            else:
                await together(
                    llm.forward(other_header_ids, write_to=others_header, return_logits=False),
                    llm.forward(own_header_ids, write_to=own_header, return_logits=False),
                )
            await together(*(
                llm.forward(ids[:-1], cache_view=[common, others_header, own_header, block], return_logits=False)
                for ids, block in zip(current, worker_blocks)
            ))
            for ctx, ids in zip(contexts, current):
                ctx.next_input_id = ids[-1]
        step_indices, generated, sampled = [1] * count, [0] * count, [[] for _ in workers]
        eos = set(eos_token_ids if eos_token_ids is not None else [tokenizer.eos_token_id])
        forbidden = [] if native_workers else [
            tokenizer.vocab[t]
            for t in ("<|im_start|>", "<|endoftext|>")
            if t in tokenizer.vocab and tokenizer.vocab[t] not in eos
        ]
        gate, stopped = PublicationGate(), False
        stop_reason = "max_steps"
        stopping_eos_id, stopping_worker = None, None
        decode_rounds = 0
        delayed_publications: set[int] = set()
        completed_steps = 0
        first_boundary_round = first_boundary_prefix_sha256 = first_publication_round = None
        max_token_gap = 0
        inserted_worker_tokens = sum(map(len, current))
        tokens_since_reminder = 0
        generation_started = time.perf_counter()

        async def advance(i: int) -> int:
            nonlocal max_token_gap
            if native_workers and decode_rounds == 0:
                # The last prompt token already has KV in common. Each worker
                # takes a separate draw from that one prefill's logits.
                params = sampling or SamplingParams(temperature=0.0, max_tokens=1)
                token = int((await llm.sample(prompt_output.logits, **asdict(params))).item())
                contexts[i].next_input_id = token
            else:
                token = await one_token(llm, contexts[i], forbidden, sampling, samples=samples)
            if cached_finisher and token in eos:
                # EOS ends reasoning. The cached answer continues the open
                # assistant turn, so count the sample without encoding EOS.
                contexts[i].next_input_id = None
            else:
                current[i].append(token)
            generated[i] += 1
            if retain_details or capture_trace or native_workers:
                sampled[i].append(token)
            if retain_details:
                progress.append(tuple(generated))
            max_token_gap = max(max_token_gap, max(generated) - min(generated))
            return token

        async def publish(indices: Sequence[int], advanced: Sequence[int] = ()) -> None:
            nonlocal stopped, stop_reason, inserted_worker_tokens, tokens_since_reminder, completed_steps
            nonlocal first_publication_round
            prefixes = {}
            next_counter = tokens_since_reminder
            reminders = []
            # Follow the reference implementation's stable worker order. The
            # reminder counter excludes initial prefixes and counts each next
            # input token plus any newly injected paragraph prefix.
            for i in (() if native_workers else advanced or indices):
                if i in indices:
                    text = fmt.get_step_prefix(workers[i], step_indices[i] + 1)
                    if reminder_interval and next_counter > reminder_interval:
                        text += fmt.s1_collab_message
                        reminders.append({"worker": workers[i], "generated_tokens": sum(generated)})
                        next_counter = 0
                    prefixes[i] = encode(text)
                next_counter += 1 + len(prefixes.get(i, ()))
            extra = sum(map(len, prefixes.values()))
            if reasoning_context_limit is not None and context_tokens() + extra > reasoning_context_limit:
                stopped, stop_reason = True, "context_limit"
                return
            if not native_workers:
                await together(*(llm.forward(cache_view=contexts[i], return_logits=False) for i in indices))
            for i in indices:
                block = worker_blocks[i]
                # Native publication waits for the next normal decode, so
                # the separator is already encoded. Its newly sampled token
                # stays pending unless it was EOS, which has been discarded.
                encoded_count = block.num_tokens
                paragraph = current[i][:encoded_count] if native_workers else current[i]
                assert block.token_ids == paragraph, "A boundary token was lost or duplicated"
                if native_workers:
                    assert len(current[i]) - encoded_count == int(contexts[i].next_input_id is not None)
                if not paragraph:
                    continue
                await llm.append_block(common, block)
                history.extend(paragraph)
                assert common.token_ids == initial_ids + history
                completed_steps += 1
                if synchronized and first_publication_round is None:
                    first_publication_round = decode_rounds
                if capture_trace:
                    publications.append({
                        "round": decode_rounds, "worker": workers[i], "step": step_indices[i],
                        "committed_token_count": len(paragraph),
                        "next_prefix_ids": prefixes.get(i, []).copy(),
                    })
                if retain_details:
                    events.append({
                        "worker": workers[i],
                        "step": step_indices[i],
                        "tokens": len(paragraph),
                        "generated": tuple(generated),
                        "text": tokenizer.decode(paragraph),
                    })
                await llm.free_block(block)
                step_indices[i] += 1
                current[i] = current[i][encoded_count:] if native_workers else prefixes[i]
            if native_workers:
                return
            tokens_since_reminder = next_counter
            inserted_worker_tokens += extra
            reminder_events.extend(reminders)
            await together(
                *(
                    llm.forward(current[i][:-1], cache_view=contexts[i], return_logits=False)
                    for i in indices
                )
            )
            for i in indices:
                contexts[i].next_input_id = current[i][-1]

        async def generate_agent(i: int) -> None:
            nonlocal stopped, stop_reason, stopping_eos_id, stopping_worker
            for local_step in range(1, max_steps + 1):
                if stopped:
                    break
                if i == 1 and bob_delay and local_step % delay_every == 0:
                    await asyncio.sleep(bob_delay)
                async with gate.forward():
                    token = await advance(i)
                if token in eos:
                    if stopping_eos_id is None:
                        stopping_eos_id, stopping_worker = token, workers[i]
                    stopped, stop_reason = True, "eos"
                    break
                if publish_paragraphs and fmt.is_end_of_step(current[i]):
                    async with gate.publish():
                        await publish([i])

        if synchronized:
            for _ in range(max_steps):
                remaining = count if total_token_budget is None else total_token_budget - sum(generated)
                if remaining <= 0:
                    stop_reason = "token_budget"
                    break
                active = list(range(min(count, remaining)))
                if reasoning_context_limit is not None and context_tokens() + len(active) > reasoning_context_limit:
                    stop_reason = "context_limit"
                    break
                new_tokens = await together(*(advance(i) for i in active))
                decode_rounds += 1
                if native_workers and publish_paragraphs:
                    ready = [i for i in active if i in delayed_publications]
                    await publish(ready)
                    delayed_publications.difference_update(ready)
                if publish_paragraphs or (compare_boundary and first_boundary_round is None):
                    boundaries = [
                        i for i in active
                        if contexts[i].next_input_id is not None and fmt.is_end_of_step(current[i])
                    ]
                    if compare_boundary and boundaries and first_boundary_round is None:
                        first_boundary_round = decode_rounds
                        first_boundary_prefix_sha256 = _token_ids_sha256(current[0])
                    if publish_paragraphs:
                        if native_workers:
                            delayed_publications.update(boundaries)
                        else:
                            await publish(boundaries, active)
                if any(token in eos for token in new_tokens):
                    stopping_worker, stopping_eos_id = next(
                        (workers[i], token) for i, token in zip(active, new_tokens) if token in eos
                    )
                    stop_reason = "eos"
                    break
                if stopped:
                    break
                if total_token_budget is not None and sum(generated) >= total_token_budget:
                    stop_reason = "token_budget"
                    break
        else:
            await together(*(generate_agent(i) for i in range(count)))

        await together(*(
            llm.forward(cache_view=ctx, return_logits=False) for ctx in contexts if ctx.next_input_id is not None
        ))
        for i, ctx in enumerate(contexts):
            assert ctx.output_block.token_ids == current[i]
        parallel_seconds = time.perf_counter() - generation_started
        transcript = ""
        if retain_details or (not flat_finisher and (finish_if_missing or symbolic_answers)):
            if native_workers:
                transcript = tokenizer.decode(history) + fmt.sep.join(tokenizer.decode(ids) for ids in current)
            else:
                transcript = fmt.history_header + fmt.sep + tokenizer.decode(history)
                transcript += fmt.sep + fmt.current_step_header
                transcript += "".join(
                    tokenizer.decode(ids) + fmt.incomplete_step + fmt.sep for ids in current
                )
        if flat_finisher:
            generated_ids = [*history, *other_header_ids]
            for ids in current:
                generated_ids.extend(ids)
                generated_ids.extend(incomplete_ids)
            reasoning_text = tokenizer.decode(generated_ids)
        else:
            reasoning_text = transcript
        answer_selection = None
        if native_workers:
            # Worker order is not completion order. In particular, an unfinished
            # peer's draft box must not override the worker that ended the turn.
            existing_answer, answer_selection = select_terminal_worker_answer(
                tokenizer, dict(zip(workers, sampled)), eos_worker=stopping_worker,
                eos_token_ids=eos, numeric_only=not symbolic_answers,
            )
        else:
            existing_answer = (
                extract_boxed_answer(reasoning_text, numeric_only=not symbolic_answers)
                if finish_if_missing or symbolic_answers else None
            )
        answer_tokens = []
        answer_generated_tokens = 0
        finisher_eos_id = None
        finisher_stop_reason = "existing_answer" if existing_answer is not None else "max_steps"
        reasoning_stop_reason = stop_reason
        reasoning_peak = context_tokens()
        finisher_prefix_length = 0
        used_finisher = existing_answer is None
        if used_finisher and flat_finisher:
            response = tokenizer.decode([*initial_ids, *generated_ids])
            flat_ids = encode(response + suffix)
            if max_context_tokens is not None and len(flat_ids) + answer_steps > max_context_tokens:
                used_finisher = False
                stop_reason = finisher_stop_reason = "context_limit"
            else:
                # The flattened response owns a new causal cache. Release the
                # reasoning pages before allocating another long prompt.
                for block in blocks[:-1]:
                    await llm.free_block(block)
                blocks = [answer_block]
                finisher_prefix_length = len(flat_ids)
                stream = _flat_greedy_tokens(llm, answer_block, flat_ids, answer_steps)
        elif used_finisher:
            if cached_finisher:
                # Move the live worker blocks into answer order. Only the new
                # delimiters, suffix and generated answer enter the engine.
                view = [common, *others]
                for block in worker_blocks:
                    view.append(block)
                    if incomplete_ids:
                        await llm.forward(incomplete_ids, cache_view=view, return_logits=False)
                if native_answer_cue:
                    # Inspect the final view in the same order the model will
                    # read it. Its last thinking tag owns the answer phase.
                    last_tag = next((token for block in reversed(view) for token in reversed(block.token_ids)
                                     if token in (open_think_id, close_think_id)), None)
                    suffix, finisher_ids = cues[last_tag == open_think_id]
            else:
                view = [common, others_header, *worker_blocks[:-1], own_header, worker_blocks[-1]]
            answer_ctx = AsyncContext(cache_view=[*view, answer_block])
            finisher_prefix_length = answer_ctx.num_cached_tokens + len(finisher_ids)
            if len(finisher_ids) > 1:
                await llm.forward(finisher_ids[:-1], cache_view=answer_ctx, return_logits=False)
            answer_ctx.next_input_id = finisher_ids[-1]
            stream = generate_with_targets(
                llm, answer_ctx, samples, max_steps=answer_steps, forbid_ids=forbidden,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=answer_steps),
            )
        if used_finisher:
            async with aclosing(stream):
                async for token in stream:
                    answer_generated_tokens += 1
                    if token in eos:
                        finisher_eos_id = token
                        finisher_stop_reason = "eos"
                        break
                    answer_tokens.append(token)
                    text = suffix + tokenizer.decode(answer_tokens)
                    candidate = extract_boxed_answer(text) if symbolic_answers else fmt.get_final_answer(text)
                    if candidate is not None:
                        finisher_stop_reason = "answer"
                        break
            answer_text = suffix + tokenizer.decode(answer_tokens)
        elif existing_answer is not None:
            answer_text = f"\\boxed{{{existing_answer}}}"
        else:
            answer_text = ""
        answer = extract_boxed_answer(answer_text) if symbolic_answers else fmt.get_final_answer(answer_text)
        if not symbolic_answers and finish_if_missing and answer is not None and not math.isfinite(answer):
            answer = None
        result = {
            "problem": problem,
            "rounds": max(generated),
            "generated_tokens": dict(zip(workers, generated)),
            "reasoning_generated_tokens": sum(generated),
            "answer_generated_tokens": answer_generated_tokens,
            "total_generated_tokens": sum(generated) + answer_generated_tokens,
            "peak_context_tokens": max(reasoning_peak, finisher_prefix_length + len(answer_tokens)),
            "decode_rounds": decode_rounds if synchronized else None,
            "inserted_worker_tokens": inserted_worker_tokens,
            "reminder_events": reminder_events,
            "stop_reason": stop_reason,
            "reasoning_stop_reason": reasoning_stop_reason,
            "eos_token_id": stopping_eos_id,
            "eos_worker": stopping_worker,
            "used_finisher": used_finisher,
            "finisher_mode": "cached" if cached_finisher else "flat" if flat_finisher else "shared",
            "finisher_stop_reason": finisher_stop_reason,
            "finisher_eos_token_id": finisher_eos_id,
            "completed_steps": completed_steps,
            "publication_policy": NATIVE_PUBLICATION_POLICY if native_workers else None,
            "first_publication_round": first_publication_round,
            "prompt_token_ids_sha256": prompt_token_ids_sha256,
            "first_boundary_round": first_boundary_round,
            "first_boundary_prefix_sha256": first_boundary_prefix_sha256,
            "merge_events": events,
            "transcript": transcript,
            "answer_text": answer_text,
            "answer": answer,
            "boxed_answer": extract_boxed_answer(answer_text),
            "parallel_seconds": parallel_seconds,
            "generated_text": {w: tokenizer.decode(ids) for w, ids in zip(workers, sampled)} if retain_details else {},
            "prompt_ids": initial_ids,
            "history_ids": history,
            "current_ids": current,
            "answer_ids": answer_tokens,
            "progress": progress,
            "max_token_gap": max_token_gap,
        }
        if not retain_details:
            for key in ("merge_events", "transcript", "generated_text", "prompt_ids", "history_ids",
                        "current_ids", "answer_ids", "progress", "reminder_events"):
                result.pop(key)
        if answer_selection is not None:
            result["answer_selection"] = answer_selection
        if capture_trace:
            # A worker's nth sample is its token in synchronized round n.
            # Publications are ordered after sampling that round. Native
            # publication includes the previous round's encoded separator.
            result["trace"] = {
                "workers": list(workers), "synchronized": True,
                "layout": "native_workers" if native_workers else "legacy",
                "publication_policy": result["publication_policy"],
                "causal_headers": causal_headers and not native_workers,
                "block_ids": {
                    "history": common.block_id, "others_header": others[0].block_id if others else None,
                    "own_header": own[0].block_id if own else None, "answer": answer_block.block_id,
                    "workers": {w: b.block_id for w, b in zip(workers, worker_blocks)},
                },
                "prompt_ids": initial_ids, "other_header_ids": other_header_ids, "own_header_ids": own_header_ids,
                "initial_worker_prefix_ids": initial_worker_prefix_ids,
                "worker_token_ids": dict(zip(workers, sampled)), "publications": publications,
                "decode_rounds": decode_rounds, "reasoning_stop_reason": reasoning_stop_reason,
                "eos_token_ids": sorted(eos), "eos_token_id": stopping_eos_id, "eos_worker": stopping_worker,
                "boxed_answer": result["boxed_answer"],
                "finisher": {
                    "used": used_finisher, "mode": result["finisher_mode"],
                    "worker_marker_ids": incomplete_ids if used_finisher else [],
                    "suffix_ids": finisher_ids if used_finisher else [],
                    "sampled_token_ids": answer_tokens + ([] if finisher_eos_id is None else [finisher_eos_id]),
                    "stop_reason": finisher_stop_reason, "eos_token_id": finisher_eos_id,
                },
            }
            if flat_finisher and used_finisher:
                result["trace"]["finisher"]["flat_input_ids"] = flat_ids
            if answer_selection is not None:
                result["trace"]["answer_selection"] = answer_selection
        return result
    finally:
        for block in blocks:
            await llm.free_block(block)


async def together(*coroutines: Coroutine[Any, Any, T]) -> list[T]:
    tasks = [asyncio.create_task(coro) for coro in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _flat_greedy_tokens(
    llm: AsyncLLM, block: CacheBlock, prefix: list[int], steps: int,
) -> AsyncIterator[int]:
    """Prefill the complete flattened response, then continue greedily in that block."""
    input_ids = prefix
    for _ in range(steps):
        result = await llm.forward(input_ids, cache_view=[block], return_logits=True)
        token = int(result.logits.argmax().item())
        yield token
        input_ids = [token]


def _token_ids_sha256(ids: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


def extract_boxed_answer(response: str, *, numeric_only: bool = False) -> str | None:
    """Return the last complete boxed expression, preserving nested LaTeX braces."""
    marker = "\\boxed{"
    end = len(response)
    while (start := response.rfind(marker, 0, end)) >= 0:
        depth = 1
        content_start = start + len(marker)
        escaped = False
        for i in range(content_start, len(response)):
            if response[i] == "\\":
                escaped = not escaped
                continue
            if not escaped and response[i] == "{":
                depth += 1
            elif not escaped and response[i] == "}":
                depth -= 1
                if depth == 0:
                    answer = response[content_start:i].strip()
                    if answer:
                        if not numeric_only:
                            return answer
                        try:
                            if math.isfinite(float(answer)):
                                return answer
                        except ValueError:
                            pass
                    break
            escaped = False
        end = start
    return None


def select_terminal_worker_answer(
    tokenizer: PreTrainedTokenizerBase,
    worker_token_ids: Mapping[str, Sequence[int]],
    *,
    eos_worker: str | None,
    eos_token_ids: Sequence[int],
    numeric_only: bool = False,
) -> tuple[str | None, dict[str, Any]]:
    """Select the last complete box from the terminating native worker.

    Full sampled streams retain paragraphs moved into shared history. Selection
    reads those IDs on CPU; it never forwards tokens or changes a cache. A worker
    can inherit a peer's thinking delimiter through shared KV, so its own stream
    does not independently identify the thinking phase. Use the same last-box
    extraction as the baseline, but only on the worker that actually ended the
    turn. Missing EOS or no complete box requires the cached finisher.
    The caller owns the stable worker-order tie-break for simultaneous EOS.
    """
    provenance = {
        "policy": TERMINAL_ANSWER_POLICY, "worker": eos_worker, "reason": "no_eos",
        "answer_token_start": None, "answer_token_end": None, "selected_answer": None,
    }
    if eos_worker is None:
        return None, provenance
    ids = worker_token_ids.get(eos_worker, ())
    if not ids or ids[-1] not in eos_token_ids:
        provenance["reason"] = "invalid_terminal_stream"
        return None, provenance
    start, end = 0, len(ids) - 1
    answer = extract_boxed_answer(tokenizer.decode(ids[start:end]), numeric_only=numeric_only)
    provenance.update(
        reason="terminal_final_box" if answer is not None else "missing_final_box",
        answer_token_start=start, answer_token_end=end, selected_answer=answer,
    )
    return answer, provenance


async def one_token(
    llm: AsyncLLM,
    context: AsyncContext,
    forbidden: Sequence[int],
    sampling: SamplingParams | None = None,
    *,
    samples: Samples | None = None,
) -> int:
    async with aclosing(
        generate_with_targets(
            llm,
            context,
            samples,
            max_steps=1,
            forbid_ids=forbidden,
            sampling_params=sampling or SamplingParams(temperature=0.0, max_tokens=1),
        )
    ) as stream:
        return await anext(stream)


class PublicationGate:
    """Allow concurrent forwards and exclusive cache publication."""

    def __init__(self) -> None:
        self.condition = asyncio.Condition()
        self.readers = 0
        self.writers_waiting = 0
        self.writing = False

    @asynccontextmanager
    async def forward(self) -> AsyncIterator[None]:
        async with self.condition:
            await self.condition.wait_for(lambda: not self.writing and not self.writers_waiting)
            self.readers += 1
        try:
            yield
        finally:
            async with self.condition:
                self.readers -= 1
                self.condition.notify_all()

    @asynccontextmanager
    async def publish(self) -> AsyncIterator[None]:
        async with self.condition:
            self.writers_waiting += 1
            try:
                await self.condition.wait_for(lambda: not self.readers and not self.writing)
                self.writing = True
            finally:
                self.writers_waiting -= 1
                self.condition.notify_all()
        try:
            yield
        finally:
            async with self.condition:
                self.writing = False
                self.condition.notify_all()


class MathFormatting:
    """Math prompts and publication boundaries, preserving the original two-worker prompt."""

    sep = "\n\n"
    history_header = "### Past steps"
    work_in_progress_others = "### Work in progress (others)"
    work_in_progress_self = "### Work in progress (own)"
    incomplete_step = "<...>"
    generation_prefix = f"\n{history_header}{sep}"
    current_step_header = work_in_progress_others + sep
    current_worker_header = incomplete_step + sep + work_in_progress_self + sep
    s1_collab_message = "Quick check: am I doing redundant work? (yes/no): "
    s1_finisher_suffix = (
        "\n\nWait, given the limited time, I have to give an answer right now. "
        "Considering all my previous attempts, I have to conclude that the final answer is \\boxed{"
    )
    end_of_step_chars = [
        ".",
        "?",
        "!",
        "。",
        "۔",
        "؟",
        "।",
        "॥",
        "…",
        "‽",
        "།",
        "᠃",
        "։",
        "჻",
        "¶",
        "❧",
        "！",
        "？",
    ]
    block_borders = ["```", "~~~", "$$"]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        workers: Sequence[str] = ("Alice", "Bob"),
        **template_options: Any,
    ) -> None:
        self.tokenizer, self.workers = tokenizer, tuple(workers)
        if not 1 <= len(self.workers) <= 16 or len(set(self.workers)) != len(self.workers):
            raise ValueError("Expected 1 to 16 distinct worker names")
        self.template_options = template_options
        (separator_id,) = tokenizer.encode(self.sep, add_special_tokens=False)
        separator = {i: token for token, i in tokenizer.vocab.items()}[separator_id]
        self.tokens_containing_sep = {
            i for token, i in tokenizer.vocab.items() if separator in token
        }

    def get_step_prefix(self, worker: str, index: int | str) -> str:
        return f"**{worker} [{index}]:** "

    def is_end_of_step(self, worker_tokens: Sequence[int]) -> bool:
        if worker_tokens[-1] not in self.tokens_containing_sep:
            return False
        text = self.tokenizer.decode(worker_tokens)
        if any(text.count(border) % 2 != 0 for border in self.block_borders):
            return False
        text = text[: text.rindex(self.sep)].strip()
        return any(text.endswith(end) for end in self.end_of_step_chars)

    def apply_chat_template(self, problem: str) -> str:
        system = _make_system_prompt(self)
        return (
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": system + self.sep + problem}],
                tokenize=False,
                add_generation_prompt=True,
                continue_final_message=False,
                **self.template_options,
            )
            + self.generation_prefix
        )

    def get_final_answer(self, response: str) -> float | None:
        while "\\boxed{" in response:
            start = response.rindex("\\boxed{")
            try:
                end = response.index("}", start + 1)
                return float(response[start + len("\\boxed{") : end])
            except ValueError:
                response = response[:start]
        return None


def _make_system_prompt(fmt: MathFormatting) -> str:
    if len(fmt.workers) == 1:
        return (
            f"You are {fmt.workers[0]}. Solve the following problem step by step. "
            "There are no other assistants. The automated system records your completed "
            "steps under '### Past steps' and your current step under '### Work in progress (own)'. "
            "Give the final answer in \\boxed{} notation.\n\n# Solve the following problem"
        )
    return f"""
# Collaborative Reasoning

{_rules(fmt)}

# How to collaborate

{_suggestions(fmt)}{fmt.sep}# Solve the following problem

{f'{", ".join(fmt.workers[:-1])} and {fmt.workers[-1]}'}, you will now solve the next problem together. Keep track of who does what work and communicate to avoid doing the same work twice.
""".strip()


def _rules(fmt: MathFormatting) -> str:
    return f"""
You will collaborate on this problem with another assistant. You will write your thoughts simultaneously with them and collaborate without redundant work. You can collaborate by doing different parts of the problem, double-checking each other's results, trying different approaches, or any other means.

There are {len(fmt.workers)} assistants, including yourself. You will refer to each other as {f'{", ".join(fmt.workers[:-1])} and {fmt.workers[-1]}'}.

You will solve the problem together, writing your thoughts in parallel. You will be able to see each other's past and current thoughts as we write them. You will see each other's previous steps as {fmt.get_step_prefix('AssistantName', 'step')}{fmt.incomplete_step} .

In the '{fmt.history_header}' section, the automated system will gather the thoughts of {f'{", ".join(fmt.workers[:-1])} and {fmt.workers[-1]}'} as you write them.

After the '{fmt.work_in_progress_others}' section, you will see the other assistants' unfinished steps. They will write those steps concurrently with you. You will take into account what they are doing. If another assistant gives you suggestions, you should address them.

You will always see *other* assistants' incomplete thoughts first, and then, after '{fmt.work_in_progress_self}', your own current step. Other assistants will continue writing their thoughts in the background while you will continue writing your own.

Since you and others both write your thoughts in parallel, you will initially see only partial (unfinished) thoughts that others will continue in parallel, while you write yours. Others' thoughts will appear at the end of their unfinished step, near {fmt.incomplete_step}. Other assistants may write new thoughts while you are writing yours.

You will use these partial thoughts to decide how best to collaborate without doing the same work twice. You will periodically check what other assistants are doing and you should adjust your actions based on what they are doing so you collaborate efficiently with them.

If what you are currently doing is the same thing that another assistant has already done or is in process of doing, you will stop (e.g. {fmt.workers[0]} may say 'Wait, I was doing the same as {fmt.workers[1]} ...') and change to a different task right away, so as to avoid doing redundant work.\x20
""".strip()


def _suggestions(fmt: MathFormatting) -> str:
    return f"""
You will take into account what the other assistant is doing and change your actions accordingly. Here is how you can collaborate with them:

- **1. Strategizing:** you should think on how best to divide work between us (e.g. if {fmt.workers[0]} writes: {fmt.workers[1]}, please do this, then {fmt.workers[1]} should take this into account). If assistants disagree about what to do, you should both default to {fmt.workers[0]}'s version.
- **2. Splitting:** you can split the problem into subtasks (simplify one equation or the other equation) and split the tasks between us. Prioritize subtasks that are not redundant (i.e. do not verify minor calculation done by another worker if there is another calculation that wasn't attempted yet).
- **3. Alternatives:** you can each try to solve a problem with different methods (e.g. calculate a mathematical expression with brute force vs mathematical derivations) and see which approach is faster.
- **4. Communicating:** you can look at each other's thoughts, ask each other questions (e.g. '{fmt.workers[0]}, which of these should I do first?'), give each other suggestions or corrections (e.g. 'Hey, {fmt.workers[1]}! You have a mistake in step 3 ...')
- **5. Announcing:** you can announce what you will do next (e.g. 'Let me try x=5 next' or 'I will double-check {fmt.workers[0]}'s result from step 5'). If another assistant says this, you should take it into consideration to avoid redundancy.
- **6. Reacting:** if you notice that another assistant is doing the same thing as you do, you should stop and think what else can you do to avoid redundancy. If you are ahead of the other assistant, you will instead ask them to change task problem (e.g. '{fmt.workers[1]}, please do something else, I am already solving that').
- **7. Pivoting:** if you notice that what you are doing is no longer useful after change in circumstances, you will stop mid-sentence and pivot to another direction (e.g. '... computing p^4 | Wait, {fmt.workers[0]} is already on it, I should switch to adding up the results.')

You can also collaborate in any different way. You can invent new ways that would help you arrive at the correct solution faster together.

To decide how best to collaborate, you will periodically, every few steps or more often, think what you are doing and if you are contributing or doing redundant work. If it is the latter, you will stop and do something else to better contribute to solving the problem together.
""".strip()
