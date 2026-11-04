"""Record async thoughts and compare answer, scheduling, and trace structure."""

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from minisgl.llm import AsyncLLM
from minisgl.training import RecordedAsyncLLM, Trace
from transformers import AutoConfig, PreTrainedTokenizerBase

from minisgl_examples.demos.async_thoughts import demo as async_thoughts
from minisgl_examples.training.recording import build_training_example
from minisgl_examples.training.student import TraceSource, load_example

DEFAULT_PROMPT = (
    r"What is 17 * 23? Give one short calculation and put only the final integer in \boxed{...}."
)


async def collect_trace(
    model: str,
    *,
    prompt: str = DEFAULT_PROMPT,
    think_tokens: int = 128,
    answer_tokens: int = 192,
    probe_period: int = 4,
    memory_ratio: float = 0.7,
    max_seq_len: int = 8192,
    cuda_graphs: bool = True,
) -> dict[str, Any]:
    """Return a recorded teacher example without requiring an intermediate file."""
    if torch.cuda.device_count() != 1:
        raise ValueError("Select exactly one GPU")
    if AutoConfig.from_pretrained(model).model_type != "qwen3":
        raise ValueError("This sanity check supports dense text Qwen3")
    if min(think_tokens, answer_tokens, probe_period, max_seq_len) < 1:
        raise ValueError("Token budgets, probe period and sequence limit must be positive")
    if not 0 < memory_ratio < 1:
        raise ValueError("Memory ratio must be between zero and one")
    llm = AsyncLLM(
        model,
        dtype=torch.bfloat16,
        max_running_req=4,
        cuda_graph_bs=[1, 2] if cuda_graphs else [],
        cuda_graph_max_bs=2 if cuda_graphs else 0,
        shared_cuda_graph_max_depth=4,
        shared_cuda_graph_prefill_rows=[32, 256] if cuda_graphs else [],
        max_prefill_rows=256,
        memory_ratio=memory_ratio,
        max_seq_len_override=max_seq_len,
        page_size=1,
        distributed_addr="tcp://127.0.0.1:0",
    )
    config = async_thoughts.DemoConfig(
        model=model,
        problem=prompt,
        max_steps=think_tokens,
        max_answer_tokens=answer_tokens,
        probe_period=probe_period,
    )
    try:
        samples = {}
        with RecordedAsyncLLM(llm) as recorded:
            thinker, writer = await async_thoughts._run_loop(
                config, recorded, llm.tokenizer, async_thoughts.Prompting(prompt), samples
            )
        return {
            "model": model,
            "prompt": prompt,
            **build_training_example(recorded, samples),
            "thinker_tokens": thinker,
            "writer_tokens": writer,
        }
    finally:
        await llm.close()


def check_result(ok: bool, reason: str, **details: Any) -> dict[str, Any]:
    return {"status": "PASS" if ok else "FAIL", "reason": reason, "details": details}


def check_answer(
    example: dict[str, Any], tokenizer: PreTrainedTokenizerBase, reference: list[int]
) -> dict[str, Any]:
    """Check the single boxed integer against an independent reference."""
    if len(reference) != 1:
        raise ValueError("This sanity check expects one final integer")
    text = tokenizer.decode(example["writer_tokens"], skip_special_tokens=True)
    boxes = re.findall(r"\\+boxed\s*\{\s*([+-]?\d+)\s*\}", text)
    actual = [int(value) for value in boxes]
    return check_result(
        actual == reference and len(re.findall(r"\\+boxed", text)) == 1,
        "Expected exactly one numeric boxed answer",
        expected=reference,
        actual=actual,
    )


def check_async_behavior(
    example: dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> dict[str, Any]:
    """Check that writer uses new thoughts while thinker is still producing tokens."""
    from minisgl_examples.demos.async_thoughts.demo import Prompting

    try:
        trace = load_example(example)["trace"]
        trace.to("cpu")
        prompting = Prompting(example["prompt"])
        thinker_prefix = tokenizer.encode(prompting.thinker_output_prefix, add_special_tokens=False)
        writer_prefix = tokenizer.encode(prompting.writer_output_prefix, add_special_tokens=False)
        question = tokenizer.encode(prompting.mode_switching_question, add_special_tokens=False)
        thinker_block = writer_block = None
        probe_steps: list[int] = []
        for batch in trace.batches:
            if batch.kind != "prefill":
                continue
            for request in batch.requests:
                if list(request.input_ids) == thinker_prefix:
                    thinker_block = request.write_to
                elif list(request.input_ids) == writer_prefix:
                    writer_block = request.write_to
                elif list(request.input_ids) == question:
                    probe_steps.append(batch.step)

        if thinker_block is None or writer_block is None:
            raise ValueError("Cannot find thinker/writer prefixes from async_thoughts")
        thinker_steps: list[int] = []
        writer_steps: list[int] = []
        visible_thoughts: list[int] = []
        for token, target in zip(trace.tokens, example["targets"], strict=True):
            if target == -100:
                continue
            if token.block == thinker_block:
                thinker_steps.append(token.step)
            elif token.block == writer_block:
                writer_steps.append(token.step)
                length = sum(n for block, n in token.view if block == thinker_block)
                visible_thoughts.append(length)

        checks = {
            "both_streams": bool(thinker_steps and writer_steps),
            "writer_before_thinker_ends": bool(
                writer_steps and thinker_steps and min(writer_steps) < max(thinker_steps)
            ),
            "probe_before_writer": bool(
                writer_steps and any(s < min(writer_steps) for s in probe_steps)
            ),
            "writer_reads_thoughts": bool(visible_thoughts)
            and min(visible_thoughts) > len(thinker_prefix),
            "writer_receives_new_thoughts": len(set(visible_thoughts)) > 1
            and visible_thoughts == sorted(visible_thoughts),
        }
        return check_result(
            all(checks.values()),
            "Observable async structure, not optimality of probe decisions",
            failed=[name for name, ok in checks.items() if not ok],
            checks=checks,
        )
    except (ValueError, KeyError, TypeError, NotImplementedError) as error:
        return check_result(False, "Invalid or unsupported trace", error=str(error))


def normalize_trace(data: Trace | dict[str, Any]) -> dict[str, Any]:
    """Rename block IDs so different allocations do not count as different behavior."""
    trace = data if isinstance(data, Trace) else Trace.from_dict(data)
    if trace.multimodal:
        raise ValueError("This sanity check uses text only")
    result = asdict(trace)
    block_ids: dict[int, int] = {}

    def rename(block: int) -> int:
        if block not in block_ids:
            block_ids[block] = len(block_ids)
        return block_ids[block]

    for batch in result["batches"]:
        for request in batch["requests"]:
            request["write_to"] = rename(request["write_to"])
            request["cache_view"] = [
                (rename(block), length) for block, length in request["cache_view"]
            ]
    for update in result["updates"]:
        update["block"] = rename(update["block"])
        if update["refs"] is not None:
            update["refs"] = [(rename(block), length) for block, length in update["refs"]]
    return result


def compare_rollouts(
    teacher_trace: TraceSource,
    student_trace: TraceSource,
    tokenizer: PreTrainedTokenizerBase,
    reference: list[int],
    destination: Path,
) -> dict[str, Any]:
    teacher, student = load_example(teacher_trace), load_example(student_trace)
    if teacher["prompt"] != student["prompt"]:
        raise ValueError("Teacher and student prompts differ")
    fields = {
        key: teacher[key] == student[key]
        for key in ("prompt", "thinker_tokens", "writer_tokens", "targets")
    }
    left, right = normalize_trace(teacher["trace"]), normalize_trace(student["trace"])
    fields["execution_trace"] = left == right
    streams = {}
    for role in ("thinker", "writer"):
        teacher_tokens = teacher[f"{role}_tokens"]
        student_tokens = student[f"{role}_tokens"]
        prefix = 0
        for teacher_token, student_token in zip(teacher_tokens, student_tokens):
            if teacher_token != student_token:
                break
            prefix += 1
        streams[role] = {
            "equal": teacher_tokens == student_tokens,
            "common_prefix_tokens": prefix,
            "teacher_length": len(teacher_tokens),
            "student_length": len(student_tokens),
        }
    report = {
        "teacher_answer": check_answer(teacher, tokenizer, reference),
        "student_answer": check_answer(student, tokenizer, reference),
        "teacher_async": check_async_behavior(teacher, tokenizer),
        "student_async": check_async_behavior(student, tokenizer),
        "exact_recorded_rollout": check_result(
            all(fields.values()),
            "Tokens, targets and cache history",
            fields=fields,
            streams=streams,
            first_trace_difference=None if left == right else "Cache execution histories differ",
        ),
    }
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return report
