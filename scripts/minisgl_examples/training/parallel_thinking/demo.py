"""Record Alice/Bob cache sharing, train on its trace, and generate freely."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import time
from contextlib import aclosing
from dataclasses import asdict
from pathlib import Path

import torch
from minisgl.core import SamplingParams
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM, Trace, replay_trace
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from minisgl_examples.training.loss import compute_supervised_loss, evaluate
from minisgl_examples.training.parallel_thinking.async_helpers import (
    PublicationGate,
    isolated,
    one_token,
    together,
)
from minisgl_examples.training.parallel_thinking.hogwild_formatting import MathFormatting
from minisgl_examples.training.recording import (
    Samples,
    build_training_example,
    generate_with_targets,
)

WORKERS = ("Alice", "Bob")
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
) -> dict:
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
        return result
    finally:
        await llm.close()
        gc.collect()
        torch.cuda.empty_cache()


async def run_agents(
    llm: AsyncLLM | RecordedAsyncLLM,
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
) -> dict:
    tokenizer = llm.tokenizer

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    blocks = [await llm.create_block() for _ in range(6)]
    try:
        common, others_header, own_header, alice, bob, answer_block = blocks
        worker_blocks = [alice, bob]
        contexts = [
            AsyncContext(cache_view=[common, others_header, bob, own_header, alice]),
            AsyncContext(cache_view=[common, others_header, alice, own_header, bob]),
        ]
        initial_ids = (
            list(prompt_ids) if prompt_ids is not None else encode(fmt.apply_chat_template(problem))
        )
        await llm.forward(initial_ids, write_to=common, return_logits=False)
        await together(
            llm.forward(
                encode(fmt.current_step_header), write_to=others_header, return_logits=False
            ),
            llm.forward(
                encode(fmt.current_worker_header), write_to=own_header, return_logits=False
            ),
        )
        current = [
            encode(
                f"{fmt.get_step_prefix(WORKERS[0], 1)}Hi, I'm {WORKERS[0]}. Here's how we can collaborate"
            ),
            encode(f"{fmt.get_step_prefix(WORKERS[1], 1)}Hi, I'm {WORKERS[1]}."),
        ]
        await together(
            *(
                llm.forward(
                    ids[:-1],
                    cache_view=[common, others_header, own_header, block],
                    return_logits=False,
                )
                for ids, block in zip(current, worker_blocks)
            )
        )
        for ctx, ids in zip(contexts, current):
            ctx.next_input_id = ids[-1]
        history, events, progress = [], [], []
        step_indices, generated, sampled = [1, 1], [0, 0], [[], []]
        eos = tokenizer.eos_token_id
        forbidden = [
            tokenizer.vocab[t]
            for t in ("<|im_start|>", "<|endoftext|>")
            if t in tokenizer.vocab and tokenizer.vocab[t] != eos
        ]
        gate, stopped = PublicationGate(), False
        generation_started = time.perf_counter()

        async def advance(i):
            token = await one_token(llm, contexts[i], forbidden, sampling, samples=samples)
            current[i].append(token)
            generated[i] += 1
            sampled[i].append(token)
            progress.append(tuple(generated))
            return token

        async def publish(indices):
            await together(
                *(llm.forward(cache_view=contexts[i], return_logits=False) for i in indices)
            )
            for i in indices:
                block = worker_blocks[i]
                assert block.token_ids == current[i], "A boundary token was lost or duplicated"
                await llm.append_block(common, block)
                history.extend(current[i])
                assert common.token_ids == initial_ids + history
                events.append(
                    {
                        "worker": WORKERS[i],
                        "step": step_indices[i],
                        "tokens": len(current[i]),
                        "generated": tuple(generated),
                        "text": tokenizer.decode(current[i]),
                    }
                )
                await llm.free_block(block)
                step_indices[i] += 1
                current[i] = encode(fmt.get_step_prefix(WORKERS[i], step_indices[i]))
            await together(
                *(
                    llm.forward(current[i][:-1], cache_view=contexts[i], return_logits=False)
                    for i in indices
                )
            )
            for i in indices:
                contexts[i].next_input_id = current[i][-1]

        async def generate_agent(i):
            nonlocal stopped
            for local_step in range(1, max_steps + 1):
                if stopped:
                    break
                if i == 1 and bob_delay and local_step % delay_every == 0:
                    await asyncio.sleep(bob_delay)
                async with gate.forward():
                    token = await advance(i)
                if token == eos:
                    stopped = True
                    break
                if fmt.is_end_of_step(current[i]):
                    async with gate.publish():
                        await publish([i])

        if synchronized:
            for _ in range(max_steps):
                new_tokens = await together(advance(0), advance(1))
                await publish([i for i in range(2) if fmt.is_end_of_step(current[i])])
                if eos in new_tokens:
                    break
        else:
            await together(generate_agent(0), generate_agent(1))

        await together(*(llm.forward(cache_view=ctx, return_logits=False) for ctx in contexts))
        for i, ctx in enumerate(contexts):
            assert ctx.output_block.token_ids == current[i]
        parallel_seconds = time.perf_counter() - generation_started
        transcript = fmt.history_header + fmt.sep + tokenizer.decode(history)
        transcript += fmt.sep + fmt.current_step_header
        transcript += "".join(
            tokenizer.decode(ids) + fmt.incomplete_step + fmt.sep for ids in current
        )
        suffix = fmt.s1_finisher_suffix
        answer_ids = encode(suffix)
        answer_ctx = AsyncContext(
            cache_view=[common, others_header, alice, own_header, bob, answer_block]
        )
        await llm.forward(answer_ids[:-1], cache_view=answer_ctx, return_logits=False)
        answer_ctx.next_input_id = answer_ids[-1]
        answer_tokens = []
        async with aclosing(
            generate_with_targets(
                llm,
                answer_ctx,
                samples,
                max_steps=answer_steps,
                forbid_ids=forbidden,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=answer_steps),
            )
        ) as stream:
            async for token in stream:
                if token == eos:
                    break
                answer_tokens.append(token)
                if fmt.get_final_answer(suffix + tokenizer.decode(answer_tokens)) is not None:
                    break
        answer_text = suffix + tokenizer.decode(answer_tokens)
        return {
            "problem": problem,
            "rounds": max(generated),
            "generated_tokens": dict(zip(WORKERS, generated)),
            "completed_steps": len(events),
            "merge_events": events,
            "transcript": transcript,
            "answer_text": answer_text,
            "answer": fmt.get_final_answer(answer_text),
            "parallel_seconds": parallel_seconds,
            "generated_text": {w: tokenizer.decode(ids) for w, ids in zip(WORKERS, sampled)},
            "prompt_ids": initial_ids,
            "history_ids": history,
            "current_ids": current,
            "answer_ids": answer_tokens,
            "progress": progress,
            "max_token_gap": max(abs(a - b) for a, b in progress),
        }
    finally:
        for block in blocks:
            await llm.free_block(block)


def train_student(
    model: str,
    example: dict,
    output: str | Path,
    *,
    steps: int = 80,
    learning_rate: float = 3e-5,
    chunk_size: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    if steps < 1:
        raise ValueError("Training requires at least one optimizer step")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(7)
    trace = example["trace"]
    if isinstance(trace, dict):
        trace = Trace.from_dict(trace)
    trace = trace.to(query_chunk_size=chunk_size, attention_checkpointing=True, cache_plans=True)
    targets = torch.tensor(example["targets"])
    config = AutoConfig.from_pretrained(model)
    multimodal = getattr(config, "vision_config", None) is not None
    model_class = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    processor_class = AutoProcessor if multimodal else AutoTokenizer
    processor = processor_class.from_pretrained(model)
    student = model_class.from_pretrained(model, config=config, dtype=dtype).to(device)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    master = {
        name: torch.nn.Parameter(p.detach().cpu().clone()) for name, p in student.named_parameters()
    }
    optimizer = torch.optim.AdamW(master.values(), lr=learning_rate, foreach=False)
    initial = evaluate(student.eval(), [(trace, targets)], chunk_size=chunk_size)
    losses = []
    for step in range(1, steps + 1):
        student.train()
        student.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        loss = (
            compute_supervised_loss(student, trace, targets, chunk_size=chunk_size)
            / targets.ne(-100).sum()
        )
        loss.backward()
        for name, parameter in student.named_parameters():
            if parameter.grad is not None:
                master[name].grad = parameter.grad.detach().cpu()
                parameter.grad = None
        torch.nn.utils.clip_grad_norm_(master.values(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for name, parameter in student.named_parameters():
                parameter.copy_(master[name])
        losses.append(loss.item())
        if step % 5 == 0:
            metrics = evaluate(student.eval(), [(trace, targets)], chunk_size=chunk_size)
            if metrics["accuracy"] == 1.0 and metrics["loss"] < 0.01:
                break
    final = evaluate(student.eval(), [(trace, targets)], chunk_size=chunk_size)
    rows = targets.ne(-100).nonzero().flatten()[:8]
    with torch.no_grad():
        expected = replay_trace(student, trace, rows=rows, hidden_only=False).cpu()
    student.cpu().save_pretrained(output)
    processor.save_pretrained(output)
    del student, optimizer, master
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reloaded = model_class.from_pretrained(output, dtype=dtype).to(device).eval()
    with torch.no_grad():
        actual = replay_trace(reloaded, trace, rows=rows, hidden_only=False).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return {
        "initial": initial,
        "final": final,
        "losses": losses,
        "checkpoint": str(output),
        "reload_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--student", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--mode", choices=("synchronized", "no-barrier"), default="synchronized")
    parser.add_argument("--problem", default=PROBLEM)
    parser.add_argument("--max-steps", type=int, default=384)
    parser.add_argument("--answer-steps", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--bob-delay", type=float, default=0.0)
    parser.add_argument("--delay-every", type=int, default=32)
    parser.add_argument("--memory-ratio", type=float, default=0.65)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory")
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student)
    if teacher_tokenizer.get_vocab() != student_tokenizer.get_vocab():
        parser.error("Teacher and student must use the same token IDs")
    options = {
        "synchronized": args.mode == "synchronized",
        "problem": args.problem,
        "max_steps": args.max_steps,
        "answer_steps": args.answer_steps,
        "bob_delay": args.bob_delay,
        "delay_every": args.delay_every,
        "memory_ratio": args.memory_ratio,
        "max_seq_len": args.max_seq_len,
        "cuda_graphs": args.cuda_graphs,
    }
    teacher = isolated(
        solve,
        args.teacher,
        **options,
        sampling=SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=1),
    )
    args.output.mkdir(parents=True)
    example = teacher.pop("example")
    data = {"model": args.teacher, "examples": [{**example, "trace": asdict(example["trace"])}]}
    (args.output / "trace.json").write_text(json.dumps(data, allow_nan=False) + "\n")
    training = isolated(
        train_student,
        args.student,
        example,
        args.output / "checkpoint",
        steps=args.train_steps,
        learning_rate=args.learning_rate,
    )
    generated = isolated(solve, training["checkpoint"], **options, prompt_ids=teacher["prompt_ids"])
    generated.pop("example")
    report = {"teacher": teacher, "training": training, "student": generated}
    (args.output / "report.json").write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "trace_rows": example["trace"].num_tokens,
                "targets": sum(t != -100 for t in example["targets"]),
                "initial_loss": training["initial"]["loss"],
                "final_loss": training["final"]["loss"],
                "reload_exact": training["reload_exact"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
