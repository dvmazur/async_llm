"""Alice/Bob generation, cache publication, and prompt formatting."""

from __future__ import annotations

import asyncio
import gc
import time
from collections.abc import AsyncIterator, Coroutine, Sequence
from contextlib import aclosing, asynccontextmanager
from typing import Any, TypeVar

import torch
from minisgl.core import SamplingParams
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM
from transformers import PreTrainedTokenizerBase

from minisgl_examples.training.recording import (
    Samples,
    build_training_example,
    generate_with_targets,
)

T = TypeVar("T")

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
) -> dict[str, Any]:
    tokenizer = llm.tokenizer

    def encode(text: str) -> list[int]:
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

        async def advance(i: int) -> int:
            token = await one_token(llm, contexts[i], forbidden, sampling, samples=samples)
            current[i].append(token)
            generated[i] += 1
            sampled[i].append(token)
            progress.append(tuple(generated))
            return token

        async def publish(indices: Sequence[int]) -> None:
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

        async def generate_agent(i: int) -> None:
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


async def together(*coroutines: Coroutine[Any, Any, T]) -> list[T]:
    tasks = [asyncio.create_task(coro) for coro in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


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
    """The original two-worker math prompt and publication boundaries."""

    sep = "\n\n"
    history_header = "### Past steps"
    work_in_progress_others = "### Work in progress (others)"
    work_in_progress_self = "### Work in progress (own)"
    incomplete_step = "<...>"
    generation_prefix = f"\n{history_header}{sep}"
    current_step_header = work_in_progress_others + sep
    current_worker_header = incomplete_step + sep + work_in_progress_self + sep
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
