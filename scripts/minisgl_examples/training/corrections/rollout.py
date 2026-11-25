"""Record one reasoning rollout whose second problem shard arrives in flight."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM, Trace

from minisgl_examples.training.corrections.grading import grade_text
from minisgl_examples.training.recording import build_training_example, record_sample

INJECTION_PREFIX = "\n\nADDITIONAL INFORMATION: "
CORRECTION_PROTOCOL = "prompt_only"
PROMPT = (
    "Solve the problem in at most five short reasoning steps. Additional information "
    "may arrive while you work; apply it before finalizing. Do not restate or audit "
    "your instructions. Put the final answer within \\boxed{}.\n\n"
)


def problem_shards(item: Mapping[str, Any]) -> tuple[str, str]:
    """Return the two nonempty shards required by the correction protocol."""
    shards = item.get("problem_shards")
    if (
        not isinstance(shards, (list, tuple))
        or len(shards) != 2
        or not all(isinstance(value, str) and value for value in shards)
    ):
        raise ValueError("Expected exactly two nonempty problem_shards")
    return shards[0], shards[1]


def _encode(tokenizer: Any, text: str) -> torch.Tensor:
    return torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)


async def collect_correction_example(
    llm: AsyncLLM,
    item: Mapping[str, Any],
    *,
    dataset: str,
    dataset_revision: str,
    dataset_split: str,
    dataset_index: int,
    k_steps: int,
    max_tokens: int,
    precision: str,
) -> dict[str, Any]:
    """Collect one format-v2 trace while reusing a caller-owned inference engine."""
    if not 0 <= k_steps < max_tokens:
        raise ValueError("Require 0 <= k_steps < max_tokens")
    first_shard, correction_shard = problem_shards(item)
    tokenizer = llm.tokenizer
    if tokenizer is None:
        raise ValueError("The inference engine has no tokenizer")

    user_prompt = PROMPT + first_shard
    if k_steps == 0:
        user_prompt += correction_shard
    initial = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    correction = correction_shard if k_steps == 0 else INJECTION_PREFIX + correction_shard + "\n\n"
    correction_ids = _encode(tokenizer, correction)
    if correction_ids.numel() == 0:
        raise ValueError("The correction encoded to no tokens")
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if not newline_ids:
        raise ValueError("A newline encoded to no tokens")

    prompt = output = None
    samples: dict[int, tuple[int, float]] = {}
    generated: list[int] = []
    eos_token_positions: list[int] = []
    injection_trace_step = None
    try:
        with RecordedAsyncLLM(llm) as recorded:
            prompt = await recorded.create_block()
            output = await recorded.create_block()
            if k_steps == 0:
                injection_trace_step = recorded.get_trace().steps
            await recorded.forward(
                _encode(tokenizer, initial), write_to=prompt, return_logits=False
            )
            context = AsyncContext([prompt, output], next_input_id=int(newline_ids[-1]))

            for generation_step in range(max_tokens):
                if generation_step == k_steps and k_steps > 0:
                    # Update the prompt before decoding the pending output token.
                    injection_trace_step = recorded.get_trace().steps
                    await recorded.forward(
                        correction_ids,
                        cache_view=[prompt],
                        write_to=prompt,
                        return_logits=False,
                    )
                trace = recorded.get_trace()
                batch_cursor, row_cursor = len(trace.batches), trace.num_tokens
                write_end = output.num_tokens + 1
                result = await recorded.forward(cache_view=context)
                token = int(result.logits.argmax().item())
                record_sample(
                    recorded,
                    samples,
                    block_id=output.block_id,
                    write_end=write_end,
                    batch_cursor=batch_cursor,
                    row_cursor=row_cursor,
                    token=token,
                    logits=result.logits,
                )

                generated.append(token)
                context.next_input_id = token
                if token == tokenizer.eos_token_id:
                    eos_token_positions.append(len(generated) - 1)
                    if injection_trace_step is not None:
                        break

            if injection_trace_step is None:
                raise RuntimeError("The correction was not injected")
            training = build_training_example(recorded, samples)

        pre_correction_tokens = generated[:k_steps]
        output_text = tokenizer.decode(generated, skip_special_tokens=True)
        pre_correction_text = tokenizer.decode(pre_correction_tokens, skip_special_tokens=True)
        reference = str(item.get("answer", ""))
        pre_correction_evaluation = grade_text(pre_correction_text, reference)
        example = {
            "prompt": initial,
            "problem_shards": [first_shard, correction_shard],
            "answer": reference,
            "dataset": dataset,
            "dataset_revision": dataset_revision,
            "dataset_split": dataset_split,
            "dataset_index": dataset_index,
            "k_steps": k_steps,
            "precision": precision,
            "correction_protocol": CORRECTION_PROTOCOL,
            "injection_trace_step": injection_trace_step,
            "correction_tokens": correction_ids.tolist(),
            "pre_correction_tokens": pre_correction_tokens,
            "pre_correction_text": pre_correction_text,
            "output_text": output_text,
            "pre_correction_evaluation": pre_correction_evaluation,
            "answer_evaluation": grade_text(output_text, reference),
            "answer_before_correction": bool(pre_correction_evaluation["answers"]),
            "eos_before_correction": any(position < k_steps for position in eos_token_positions),
            "eos_token_positions": eos_token_positions,
            "stopped_reason": "eos"
            if eos_token_positions and eos_token_positions[-1] >= k_steps
            else "length",
            **training,
            "output_tokens": generated,
        }
        validate_correction_example(example)
        return example
    finally:
        # Free outside the recording scope: resource cleanup is not part of the
        # trajectory that the student must replay.
        if output is not None:
            await llm.free_block(output)
        if prompt is not None:
            await llm.free_block(prompt)


def _validate_prompt_only(trace, targets, correction, k_steps, injection_step):
    if not trace.batches or not trace.batches[0].requests:
        raise ValueError("The trace is missing its prompt prefill")
    initial_step = trace.batches[0].step
    prompt_block = trace.batches[0].requests[0].write_to
    if injection_step is None or not (
        injection_step == initial_step if k_steps == 0 else injection_step > initial_step
    ):
        raise ValueError("The correction step does not match its position in the rollout")

    correction_inputs = []
    decode_steps_before_injection = 0
    saw_prompt_after_injection = False
    prompt_length = 0
    row = 0
    for batch in trace.batches:
        for request in batch.requests:
            end = row + len(request.input_ids)
            if request.write_to == prompt_block:
                if batch.step not in (initial_step, injection_step) or any(
                    value != -100 for value in targets[row:end]
                ):
                    raise ValueError("Prompt writes must be unsupervised and occur during prefill or injection")
                if batch.step == injection_step:
                    correction_inputs.extend(request.input_ids)
                prompt_length = request.write_start + len(request.input_ids)
            else:
                if len(request.input_ids) != 1 or targets[row] == -100 or batch.step == injection_step:
                    raise ValueError("Output writes must be supervised single-token decode steps")
                if batch.step < injection_step:
                    decode_steps_before_injection += 1
                elif not saw_prompt_after_injection:
                    seen_length = dict(request.cache_view).get(prompt_block, 0)
                    if seen_length != prompt_length:
                        raise ValueError("The first post-correction decode must see the updated prompt")
                    saw_prompt_after_injection = True
            row = end
    if not correction_inputs or not saw_prompt_after_injection or decode_steps_before_injection != k_steps:
        raise ValueError("The correction must precede the first post-correction decode")
    if k_steps > 0 and tuple(correction_inputs) != correction:
        raise ValueError("Expected exactly one correction append to the prompt")


def _validate_legacy_correction(trace, targets, correction):
    occurrences: list[tuple[int, Any]] = []
    row = 0
    correction_rows: list[range] = []
    for batch in trace.batches:
        for request in batch.requests:
            end = row + len(request.input_ids)
            if request.input_ids == correction:
                occurrences.append((batch.step, request))
                correction_rows.append(range(row, end))
            row = end
    if len(occurrences) != 2:
        raise ValueError("Expected the correction in exactly two recorded requests")

    (_, prompt_request), (_, output_request) = occurrences
    prompt_block, output_block = prompt_request.write_to, output_request.write_to
    if prompt_block == output_block:
        raise ValueError("Prompt and live reasoning correction writes must use different blocks")
    if [block for block, _ in prompt_request.cache_view] != [prompt_block]:
        raise ValueError("The prompt correction does not extend its prompt block")
    if [block for block, _ in output_request.cache_view] != [prompt_block, output_block]:
        raise ValueError("The live correction does not see the extended prompt and output")
    if any(targets[index] != -100 for index in correction_rows[0]):
        raise ValueError("Prompt correction rows must be unsupervised")
    live_targets = [targets[index] for index in correction_rows[1]]
    if live_targets[-1] == -100 or any(value != -100 for value in live_targets[:-1]):
        raise ValueError("Only the final live correction row may predict a model token")


def validate_correction_example(example: Mapping[str, Any]) -> dict[str, int]:
    """Check prompt-only recordings and earlier two-write traces."""
    raw_trace = example.get("trace")
    trace = raw_trace if isinstance(raw_trace, Trace) else Trace.from_dict(raw_trace)
    targets = list(example.get("targets", ()))
    correction = tuple(example.get("correction_tokens", ()))
    if not correction or len(targets) != trace.num_tokens:
        raise ValueError("Trace targets or correction token metadata are invalid")

    protocol = example.get("correction_protocol")
    if protocol == CORRECTION_PROTOCOL:
        _validate_prompt_only(trace, targets, correction, example["k_steps"], example["injection_trace_step"])
    elif protocol is None:
        _validate_legacy_correction(trace, targets, correction)
    else:
        raise ValueError(f"Unknown correction protocol: {protocol}")
    supervised = sum(value != -100 for value in targets)
    if supervised != len(example.get("output_tokens", ())):
        raise ValueError("Every generated token must have exactly one aligned target")
    return {
        "trace_tokens": trace.num_tokens,
        "supervised_tokens": supervised,
        "trace_steps": trace.steps,
        "correction_tokens": len(correction),
    }
