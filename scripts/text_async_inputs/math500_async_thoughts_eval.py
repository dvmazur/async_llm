"""Math-500 sharded eval with concurrent thinker/writer AsyncLLM streams.

At ``k_steps`` thinker tokens, route the real shard to the prompt and thinker
blocks and append a short new-input reminder to the writer block. Persistent
cache layouts remain [prompt, thinker] and [prompt, thinker, writer].
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext
from transformers import AutoTokenizer

ASYNC_THOUGHTS = Path(__file__).resolve().parents[1] / "async_thoughts"
sys.path.insert(0, str(ASYNC_THOUGHTS))
from async_thoughts.demo import Prompting, _tokens_with_pending  # noqa: E402
from async_thoughts.engine import (  # noqa: E402
    ModeSwitchProbe,
    encode,
    ends_with_double_newline,
    single_token_id,
    vocab_id_or_none,
)
from math500_async_eval import check_equality, find_last_boxed_answer  # noqa: E402

WRITER_REMINDER = " ... [SYSTEM: additional user input detected]\n"


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--k-steps", type=int, required=True)
    p.add_argument("--path-to-results", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--probe-period", type=int, default=30)
    p.add_argument("--defer-writer-reminder", action="store_true",
                   help="Append the writer reminder at its next paragraph boundary")
    p.add_argument("--memory-ratio", type=float, default=.8)
    p.add_argument("--page-size", type=int, default=1)
    p.add_argument("--distributed-port", type=int, default=2360)
    return p.parse_args()


def forbidden(tokenizer, names):
    return [i for i in (vocab_id_or_none(tokenizer, n) for n in names) if i is not None]


async def append_to_stream(llm, ctx, block, ids, forbid_ids):
    """Commit a pending sampled token, append text in-place, and reseed decode."""
    if ctx.next_input_id is not None:
        await llm.forward(cache_view=ctx)
        ctx.next_input_id = None
    out = await llm.forward(ids, cache_view=ctx.cache_view, write_to=block)
    logits = out.logits
    if forbid_ids:
        logits[forbid_ids] = float("-inf")
    ctx.next_input_id = int(logits.argmax())


async def generate(llm, tokenizer, first, second, k_steps, budget, probe_period,
                   defer_writer_reminder):
    problem = first + second if k_steps == 0 else first
    prompting = Prompting(
        "Please reason step by step, and put your final answer within \\boxed{}.\n\n" + problem
    )
    prompt, thinker, writer = await asyncio.gather(
        llm.create_block(), llm.create_block(), llm.create_block()
    )
    await llm.forward(encode(prompting.input_prompt, tokenizer), write_to=prompt,
                      return_logits=False)
    await llm.forward(encode(prompting.thinker_output_prefix, tokenizer),
                      [prompt, thinker], return_logits=False)
    await llm.forward(encode(prompting.writer_output_prefix, tokenizer),
                      [prompt, thinker, writer], return_logits=False)

    nn = single_token_id("\n\n", tokenizer)
    eos = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1
    thinker_ctx = AsyncContext([prompt, thinker], next_input_id=nn)
    writer_ctx = AsyncContext([prompt, thinker, writer], next_input_id=nn)
    thinker_forbid = forbidden(tokenizer, ["</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"])
    writer_forbid = forbidden(tokenizer, ["</think>", "<|im_start|>", "<|endoftext|>"])
    probe = ModeSwitchProbe(llm, tokenizer, prompting.mode_switching_prompt,
                            prompting.mode_switching_question,
                            prompting.yes_token, prompting.no_token)
    done, thinker_done = asyncio.Event(), asyncio.Event()
    writer_run, probe_due = asyncio.Event(), asyncio.Event()
    injection_done = asyncio.Event(); injection_done.set()
    injected = k_steps <= 0
    writer_reminder_pending = False

    async def inject():
        nonlocal injected, writer_reminder_pending
        injection_done.clear()
        shard = encode(f"\n\nADDITIONAL INFORMATION: {second}\n\n", tokenizer)
        await llm.forward(shard, cache_view=[prompt], write_to=prompt, return_logits=False)
        await append_to_stream(llm, thinker_ctx, thinker, shard, thinker_forbid)
        if defer_writer_reminder:
            writer_reminder_pending = True
        else:
            await append_to_stream(llm, writer_ctx, writer,
                                   encode(WRITER_REMINDER, tokenizer), writer_forbid)
        injected = True
        injection_done.set()

    async def thinker_loop():
        nonlocal injected
        steps = 0
        try:
            while steps < budget and not done.is_set():
                if not injected and k_steps > 0 and steps >= k_steps:
                    await inject()
                out = await llm.forward(cache_view=thinker_ctx)
                logits = out.logits
                if thinker_forbid: logits[thinker_forbid] = float("-inf")
                thinker_ctx.next_input_id = int(logits.argmax())
                steps += 1
                if steps % probe_period == 0 or ends_with_double_newline(
                    _tokens_with_pending(thinker, thinker_ctx), tokenizer
                ): probe_due.set()
        finally:
            thinker_done.set(); probe_due.set(); writer_run.set(); injection_done.set()

    async def writer_loop():
        nonlocal writer_reminder_pending
        steps = 0
        while not done.is_set():
            await writer_run.wait(); await injection_done.wait()
            if done.is_set(): return
            out = await llm.forward(cache_view=writer_ctx)
            logits = out.logits
            if writer_forbid: logits[writer_forbid] = float("-inf")
            token = int(logits.argmax()); writer_ctx.next_input_id = token
            if token == eos: done.set(); return
            steps += 1
            boundary = token == nn or ends_with_double_newline(
                _tokens_with_pending(writer, writer_ctx), tokenizer
            )
            if boundary and writer_reminder_pending:
                await append_to_stream(llm, writer_ctx, writer,
                                       encode(WRITER_REMINDER, tokenizer), writer_forbid)
                writer_reminder_pending = False
            if steps >= budget:
                done.set(); return
            if boundary and not thinker_done.is_set(): writer_run.clear()

    async def probe_loop():
        while not done.is_set() and not thinker_done.is_set():
            await probe_due.wait(); probe_due.clear(); await injection_done.wait()
            if done.is_set() or thinker_done.is_set(): return
            should_write, _, _ = await probe.check_continue_writing(
                _tokens_with_pending(thinker, thinker_ctx),
                _tokens_with_pending(writer, writer_ctx))
            (writer_run.set if should_write else writer_run.clear)()

    try:
        await asyncio.gather(thinker_loop(), writer_loop(), probe_loop())
        thinker_ids = _tokens_with_pending(thinker, thinker_ctx)
        writer_ids = _tokens_with_pending(writer, writer_ctx)
        return (tokenizer.decode(writer_ids, skip_special_tokens=True),
                tokenizer.decode(thinker_ids, skip_special_tokens=True), injected)
    finally:
        await probe.close()
        for block in (prompt, thinker, writer): await llm.free_block(block)


async def run(args):
    data = load_from_disk(args.dataset_path)
    end = args.end if args.end is not None else len(data)
    out_dir = Path(args.path_to_results) / f"k_{args.k_steps}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    llm = AsyncLLM(args.model_name, dtype=torch.bfloat16, max_running_req=4,
                   cuda_graph_bs=[1, 2], cuda_graph_max_bs=2,
                   memory_ratio=args.memory_ratio, page_size=args.page_size,
                   max_seq_len_override=16384,
                   distributed_addr=f"tcp://127.0.0.1:{args.distributed_port}")
    correct = total = 0
    try:
        for idx in range(args.start, end):
            path = out_dir / f"sample_{idx}.json"
            if path.exists():
                old = json.loads(path.read_text()); correct += int(old["is_equal"]); total += 1
                continue
            item = data[idx]
            try:
                response, thoughts, injected = await generate(
                    llm, tokenizer, *item["problem_shards"], args.k_steps,
                    args.budget, args.probe_period, args.defer_writer_reminder)
                predicted = find_last_boxed_answer(response)
                equal = check_equality(predicted, str(item["answer"]))
                result = {"idx": idx, "k_steps": args.k_steps, "is_equal": equal,
                          "predicted_answer": predicted, "correct_answer": str(item["answer"]),
                          "generated_text": response, "thinker_text": thoughts,
                          "shard_injected": injected,
                          "routing": {"prompt": "shard", "thinker": "shard",
                                      "writer": "deferred_reminder" if args.defer_writer_reminder else "reminder"}}
                path.write_text(json.dumps(result, indent=2))
                correct += int(equal); total += 1
                print(f"[{idx}] correct={equal} accuracy={correct/total:.3f}", flush=True)
            except Exception as exc:
                print(f"ERROR sample {idx}: {exc}", flush=True)
                import traceback; traceback.print_exc()
    finally:
        await llm.close()
    summary = {"k_steps": args.k_steps, "accuracy": correct / total if total else 0,
               "correct": correct, "total": total, "model": args.model_name,
               "routing": {"prompt": "shard", "thinker": "shard",
                           "writer": "deferred_reminder" if args.defer_writer_reminder else "reminder"}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(run(arguments()))
