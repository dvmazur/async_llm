"""Associate sampled targets with actions recorded by RecordedAsyncLLM."""

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from minisgl.core import SamplingParams
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM

Samples = dict[int, tuple[int, float]]


def record_sample(
    recorder: RecordedAsyncLLM,
    samples: Samples,
    *,
    block_id: int,
    write_end: int,
    batch_cursor: int,
    row_cursor: int,
    token: int,
    logits: torch.Tensor,
) -> None:
    """Label the final input row of one completed forward, including its final sample."""
    matches, row = [], row_cursor
    for batch in recorder.get_trace().batches[batch_cursor:]:
        for request in batch.requests:
            row += len(request.input_ids)
            if (
                request.write_to == block_id
                and request.write_start + len(request.input_ids) == write_end
            ):
                matches.append(row - 1)
    if len(matches) != 1 or matches[0] in samples:
        raise RuntimeError("Could not uniquely align a sampled token to its recorded input")
    samples[matches[0]] = (int(token), logits.float().log_softmax(-1)[token].item())


async def generate_with_targets(
    llm: AsyncLLM | RecordedAsyncLLM,
    context: AsyncContext,
    samples: Samples | None = None,
    *,
    forbid_ids: Sequence[int] = (),
    sampling_params: SamplingParams | None = None,
    max_steps: int | None = None,
    first_token_id: int | None = None,
    return_logits: bool = False,
) -> AsyncIterator[int | tuple[int, torch.Tensor]]:
    """Generate normally and optionally retain labels in a caller-owned dictionary."""
    if samples is not None and not isinstance(llm, RecordedAsyncLLM):
        raise TypeError("Sample collection requires RecordedAsyncLLM")
    stream = llm.async_generate(
        context,
        forbid_ids=forbid_ids,
        sampling_params=sampling_params,
        max_steps=max_steps,
        first_token_id=first_token_id,
        return_logits=return_logits or samples is not None,
    )
    try:
        while True:
            if samples is not None:
                trace = llm.get_trace()
                batch_cursor, row_cursor = len(trace.batches), trace.num_tokens
                block = context.output_block
                block_id, write_end = block.block_id, block.num_tokens + 1
            try:
                result = await anext(stream)
            except StopAsyncIteration:
                return
            if samples is not None:
                token, logits = result
                record_sample(
                    llm,
                    samples,
                    block_id=block_id,
                    write_end=write_end,
                    batch_cursor=batch_cursor,
                    row_cursor=row_cursor,
                    token=token,
                    logits=logits,
                )
                result = result if return_logits else token
            yield result
    finally:
        await stream.aclose()


def build_training_example(recorder: RecordedAsyncLLM, samples: Samples) -> dict[str, Any]:
    """Return the trace and row-aligned targets and log probabilities."""
    trace = recorder.get_trace()
    targets, logprobs = [-100] * trace.num_tokens, [None] * trace.num_tokens
    for row, (token, logprob) in samples.items():
        targets[row], logprobs[row] = token, logprob
    return {"trace": trace, "targets": targets, "logprobs": logprobs}


def save_training_example(
    recorder: RecordedAsyncLLM,
    samples: Samples,
    path: str | Path,
    *,
    model: str | Path,
    problem: str,
    thinker_tokens: Sequence[int],
    writer_tokens: Sequence[int],
) -> Path:
    """Write one SFT example without overwriting an existing destination."""
    example = build_training_example(recorder, samples)
    data = {
        "model": str(model),
        "examples": [
            {
                "prompt": problem,
                **example,
                "trace": asdict(example["trace"]),
                "thinker_tokens": list(thinker_tokens),
                "writer_tokens": list(writer_tokens),
            }
        ],
    }
    text = json.dumps(data, ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(text + "\n")
    return path
