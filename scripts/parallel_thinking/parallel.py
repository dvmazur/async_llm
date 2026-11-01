"""Alice/Bob shared-cache rollout, adapted from the pretrain parallel-thinking notebook.

The agents use concurrent requests with a barrier after every generated token.
The training trace records the actual cache views, batches and block mutations.
"""

import argparse
import asyncio
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import aclosing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts/async_thoughts"))

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ["PATH"] = (
    str(Path(sys.executable).parent) + ":" + os.environ["CUDA_HOME"] + "/bin:" + os.environ["PATH"]
)

# TVM FFI probes physical GPU 0 and ignores CUDA_VISIBLE_DEVICES by default.
# Select the target without initializing CUDA: AsyncLLM owns that initialization.
if "TVM_FFI_CUDA_ARCH_LIST" not in os.environ:
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if "," in gpu:
        raise ValueError("Select one GPU with CUDA_VISIBLE_DEVICES")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=compute_cap", "--format=csv,noheader"],
        text=True,
    ).strip()

import torch
from hogwild_formatting import MathFormatting
from minisgl.core import SamplingParams
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext

WORKERS = ("Alice", "Bob")
PROBLEM = (
    "Three vertices of a cube are $P=(7,12,10)$ , $Q=(8,8,1)$ , and "
    "$R=(11,3,9)$ . What is the surface area of the cube?"
)


def expected_answer():
    from itertools import combinations

    points = [(7, 12, 10), (8, 8, 1), (11, 3, 9)]
    distances = [sum((x - y) ** 2 for x, y in zip(a, b)) for a, b in combinations(points, 2)]
    assert len(set(distances)) == 1
    return 6 * distances[0] / 2


def render_progress(tokenizer, fmt, history, steps, iteration, max_steps):
    pieces = [f"**Problem:** {PROBLEM}\n\n", f"Round {iteration}/{max_steps}\n\n"]
    pieces += [fmt.history_header + fmt.sep, tokenizer.decode(history), fmt.sep]
    pieces.append(fmt.current_step_header)
    pieces.extend(tokenizer.decode(tokens) + fmt.incomplete_step + fmt.sep for tokens in steps)
    return "".join(pieces)


async def together(*coroutines):
    """On an error/cancellation, settle all requests before cache cleanup."""
    tasks = [asyncio.create_task(coro) for coro in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def one_token(llm, context, forbidden):
    # Close each finite stream explicitly; the pending token stays in context.
    async with aclosing(
        llm.async_generate(
            context,
            max_steps=1,
            forbid_ids=forbidden,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
        )
    ) as stream:
        return await anext(stream)


async def solve_parallel(
    model, *, max_steps=1024, memory_ratio=0.65, trace_output=None, prompt_ids=None
):
    MODEL = str(model)
    MAX_STEPS = max_steps
    MEMORY_RATIO = memory_ratio
    FINISHER_MAX_TOKENS = 64
    started = time.perf_counter()
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "Restart the kernel and Run All: minisgl Engine must initialize CUDA itself."
        )
    llm = AsyncLLM(
        MODEL,
        dtype=torch.bfloat16,
        memory_ratio=MEMORY_RATIO,
        max_running_req=4,
        max_seq_len_override=16384,
        cuda_graph_bs=[1, 2],
        cuda_graph_max_bs=2,
        shared_cuda_graph_max_depth=6,
        shared_cuda_graph_prefill_rows=[32, 256],
        max_prefill_rows=256,
        distributed_addr="tcp://127.0.0.1:0",
    )
    if trace_output is not None:
        from async_thoughts.training import DemoRecorder

        llm = DemoRecorder(llm)
    blocks = []
    try:
        tokenizer = llm.tokenizer
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        # Qwen3 thinking template is generated once, then reused verbatim for the student.
        fmt = MathFormatting(tokenizer, WORKERS)
        for _ in range(6):
            blocks.append(await llm.create_block())
        common, others_header, own_header, alice, bob, answer_block = blocks
        worker_blocks = [alice, bob]
        contexts = [
            AsyncContext(cache_view=[common, others_header, bob, own_header, alice]),
            AsyncContext(cache_view=[common, others_header, alice, own_header, bob]),
        ]
        initial_ids = (
            list(prompt_ids)
            if prompt_ids is not None
            else encode(fmt.apply_chat_template(PROBLEM, enable_thinking=True))
        )
        # Graph profiles admit two requests, so initialize the long prompt first.
        await llm.forward(initial_ids, write_to=common, return_logits=False)
        await together(
            llm.forward(
                encode(fmt.current_step_header), write_to=others_header, return_logits=False
            ),
            llm.forward(
                encode(fmt.current_worker_header), write_to=own_header, return_logits=False
            ),
        )

        prefixes = [
            f"{fmt.get_step_prefix(WORKERS[0], 1)}Hi, I'm {WORKERS[0]}. Here's how we can collaborate",
            f"{fmt.get_step_prefix(WORKERS[1], 1)}Hi, I'm {WORKERS[1]}.",
        ]
        current = [encode(prefix) for prefix in prefixes]
        # Leave the last prefix token pending, so round 1 is a joint decode.
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

        history, events = [], []
        step_indices, generated = [1, 1], [0, 0]
        forbidden = [
            tokenizer.vocab[t]
            for t in ("<|im_start|>", "<|endoftext|>", "</think>")
            if t in tokenizer.vocab
        ]
        # EOS remains a stop condition, even if a model uses endoftext as EOS.
        eos = tokenizer.eos_token_id
        forbidden = [token for token in forbidden if token != eos]
        generation_started = time.perf_counter()

        for iteration in range(1, MAX_STEPS + 1):
            new_tokens = await together(*(one_token(llm, ctx, forbidden) for ctx in contexts))
            for i, token in enumerate(new_tokens):
                current[i].append(token)
                generated[i] += 1

            # Both futures are settled: mutations below cannot race a decode.
            finished = [i for i in range(2) if fmt.is_end_of_step(current[i])]
            await together(
                *(llm.forward(cache_view=contexts[i], return_logits=False) for i in finished)
            )
            for i in finished:
                block = worker_blocks[i]
                assert block.token_ids == current[i], "A boundary token was lost or duplicated."
                await llm.append_block(common, block)
                history.extend(current[i])
                assert common.token_ids == initial_ids + history
                events.append(
                    {
                        "round": iteration,
                        "worker": WORKERS[i],
                        "step": step_indices[i],
                        "tokens": len(current[i]),
                    }
                )
                await llm.free_block(block)  # Reset this same block; views keep their references.
                assert block.num_tokens == 0
                step_indices[i] += 1
                start = fmt.get_step_prefix(WORKERS[i], step_indices[i])
                current[i] = encode(start)

            # Prefill prefix[:-1] after all merges; keep the last token pending.
            await together(
                *(
                    llm.forward(current[i][:-1], cache_view=contexts[i], return_logits=False)
                    for i in finished
                )
            )
            for i in finished:
                contexts[i].next_input_id = current[i][-1]

            for i, ctx in enumerate(contexts):
                assert ctx.output_block.token_ids == current[i][:-1]
                assert ctx.next_input_id == current[i][-1]
            if iteration % 128 == 0:
                print(f"round {iteration}: {len(events)} completed steps", flush=True)
            if eos in new_tokens:
                break

        # Flush both final sampled tokens before the answer continuation reads them.
        await together(*(llm.forward(cache_view=ctx, return_logits=False) for ctx in contexts))
        for i, ctx in enumerate(contexts):
            assert ctx.output_block.token_ids == current[i]
        parallel_seconds = time.perf_counter() - generation_started
        transcript = render_progress(tokenizer, fmt, history, current, iteration, MAX_STEPS)

        suffix = fmt.s1_finisher_suffix
        answer_ids = encode(suffix)
        answer_ctx = AsyncContext(
            cache_view=[common, others_header, alice, own_header, bob, answer_block]
        )
        await llm.forward(answer_ids[:-1], cache_view=answer_ctx, return_logits=False)
        answer_ctx.next_input_id = answer_ids[-1]
        answer_tokens = []
        async with aclosing(
            llm.async_generate(
                answer_ctx,
                max_steps=FINISHER_MAX_TOKENS,
                forbid_ids=forbidden,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=FINISHER_MAX_TOKENS),
            )
        ) as stream:
            async for token in stream:
                if token == eos:
                    break
                answer_tokens.append(token)
                if fmt.get_final_answer(suffix + tokenizer.decode(answer_tokens)) is not None:
                    break
        answer_text = suffix + tokenizer.decode(answer_tokens)
        result = {
            "model": MODEL,
            "problem": PROBLEM,
            "rounds": iteration,
            "generated_tokens": dict(zip(WORKERS, generated)),
            "completed_steps": len(events),
            "merge_events": events,
            "transcript": transcript,
            "answer_text": answer_text,
            "answer": fmt.get_final_answer(answer_text),
            "parallel_seconds": parallel_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        result["prompt_ids"] = initial_ids
        result["history_ids"] = history
        result["current_ids"] = current
        result["answer_ids"] = answer_tokens
        result["repo_commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        if trace_output is not None:
            llm.save(
                trace_output, model=MODEL, problem=PROBLEM, thinker_tokens=[], writer_tokens=[]
            )
        return result
    finally:
        # No active streams or queued requests remain here, including on errors.
        try:
            for block in blocks:
                await llm.free_block(block)
        finally:
            await llm.close()
            gc.collect()
            torch.cuda.empty_cache()


def load_training_example(trace_path, model_path):
    from minisgl.training.replay import Replay
    from minisgl.training.trace import Trace
    from transformers import AutoTokenizer

    data = json.loads(Path(trace_path).read_text())
    (example,) = data["examples"]
    trace = Trace.from_dict(example["trace"])
    targets = torch.tensor(example["targets"], dtype=torch.long)
    assert len(targets) == trace.num_tokens and targets.ne(-100).any()
    teacher = AutoTokenizer.from_pretrained(data["model"])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    teacher_vocab = {i: t for t, i in teacher.get_vocab().items()}
    student_vocab = {i: t for t, i in tokenizer.get_vocab().items()}
    used = {t.token_id for t in trace.tokens} | set(targets[targets.ne(-100)].tolist())
    assert all(teacher_vocab[i] == student_vocab.get(i) for i in used), "Token IDs differ"
    replay = Replay.from_trace(
        trace, query_chunk_size=64, attention_checkpointing=True, cache_plans=True
    )
    return replay, targets, tokenizer


def evaluate_student(model, replay, targets):
    from minisgl.training.sft import evaluate

    result = evaluate(model.eval(), [(replay, targets)], chunk_size=64)
    return {k: result[k] for k in ("loss", "accuracy")}


def memorized(metrics):
    return metrics["accuracy"] == 1.0 and metrics["loss"] < 0.01


def train(trace_path, model_path, output, *, steps=80, lr=3e-5):
    from transformers import AutoModelForCausalLM

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    replay, targets, tokenizer = load_training_example(trace_path, model_path)
    from minisgl.training.sft import supervised_loss

    student = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    # CPU FP32 masters keep sub-BF16 updates; only BF16 compute weights live on GPU.
    masters = {
        name: torch.nn.Parameter(p.detach().clone()) for name, p in student.named_parameters()
    }
    del student
    # Load compute weights through HF so RoPE buffers retain their FP32 dtype.
    # model.to(dtype=BF16) would also round inv_freq and change replay positions.
    student = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda()
    assert student.config.model_type == "qwen3", "Only dense Qwen3 is supported"
    assert student.model.rotary_emb.inv_freq.dtype == torch.float32, (
        "RoPE frequencies must stay FP32"
    )
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(masters.values(), lr=lr, foreach=False)
    initial = final = evaluate_student(student, replay, targets)
    history = [{"step": 0, **initial}]
    print(f"initial: {initial}", flush=True)
    started = time.perf_counter()
    first_gradients = None
    step = 0
    for step in range(1, steps + 1):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        student.zero_grad(set_to_none=True)
        loss = supervised_loss(student, replay, targets, chunk_size=64) / targets.ne(-100).sum()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        if first_gradients is None:
            first_gradients = {
                "embedding": float(student.model.embed_tokens.weight.grad.norm()),
                "first_attention": float(
                    student.model.layers[0].self_attn.q_proj.weight.grad.norm()
                ),
                "last_mlp": float(student.model.layers[-1].mlp.down_proj.weight.grad.norm()),
                "lm_head": float(student.lm_head.weight.grad.norm()),
            }
            assert all(v > 0 for v in first_gradients.values())
        for name, parameter in student.named_parameters():
            assert parameter.grad is not None, f"Missing gradient: {name}"
            masters[name].grad = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
            parameter.grad = None
        norm = torch.nn.utils.clip_grad_norm_(masters.values(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for name, parameter in student.named_parameters():
                parameter.copy_(masters[name])
        print(
            f"update {step}: loss={float(loss.detach()):.6f}; grad_norm={float(norm):.4f}",
            flush=True,
        )
        del loss
        if step % 5 == 0 or step == steps:
            final = evaluate_student(student, replay, targets)
            history.append({"step": step, **final})
            (output / "progress.json").write_text(json.dumps(history, indent=2))
            print(f"evaluation {step}: {final}", flush=True)
            if memorized(final):
                break
    checkpoint = output / "checkpoint"
    # Save the full-precision master values, not their rounded BF16 compute copies.
    student = student.cpu().float()
    with torch.no_grad():
        for name, parameter in student.named_parameters():
            parameter.copy_(masters[name])
    student.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    metrics = {
        "initial": initial,
        "final": final,
        "history": history,
        "steps": step,
        "memorized": memorized(final),
        "first_gradient_norms": first_gradients,
        "supervised_tokens": int(targets.ne(-100).sum()),
        "trace_rows": len(targets),
        "training_seconds": time.perf_counter() - started,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
        "lr": lr,
        "seed": 7,
        "model": str(model_path),
        "checkpoint": str(checkpoint),
        "trace_sha256": hashlib.sha256(Path(trace_path).read_bytes()).hexdigest(),
        "repo_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "precision": "CPU FP32 master weights, gradients and AdamW states; GPU BF16 forward/backward",
        "optimizer_device": "cpu",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    assert final["loss"] < initial["loss"], "Training did not decrease loss"
    return metrics


def evaluate_checkpoint(trace_path, checkpoint):
    from transformers import AutoModelForCausalLM

    replay, targets, _ = load_training_example(trace_path, checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16).cuda()
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32, "RoPE frequencies must stay FP32"
    return evaluate_student(model, replay, targets)


def compare_rollouts(teacher, student):
    """Text, token IDs and publication events; not wall-clock/scheduler equivalence."""
    return {
        "answer": student["answer"],
        "correct_answer": student["answer"] == expected_answer(),
        "same_transcript": student["transcript"] == teacher["transcript"],
        "same_merge_events": student["merge_events"] == teacher["merge_events"],
        "same_answer_text": student["answer_text"] == teacher["answer_text"],
        "same_token_ids": all(
            student[k] == teacher[k]
            for k in ("prompt_ids", "history_ids", "current_ids", "answer_ids")
        ),
        "teacher_completed_steps": teacher["completed_steps"],
        "student_completed_steps": student["completed_steps"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rollout = sub.add_parser("rollout")
    rollout.add_argument("--model", required=True)
    rollout.add_argument("--output", type=Path, required=True)
    rollout.add_argument("--trace", type=Path)
    rollout.add_argument("--prompt-from", type=Path)
    rollout.add_argument("--max-steps", type=int, default=1024)
    rollout.add_argument("--memory-ratio", type=float, default=0.65)
    training = sub.add_parser("train")
    training.add_argument("--model", required=True)
    training.add_argument("--trace", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--steps", type=int, default=80)
    training.add_argument("--lr", type=float, default=3e-5)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--trace", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a fresh run directory")
    if args.command == "train":
        train(args.trace, args.model, args.output, steps=args.steps, lr=args.lr)
        return
    if args.command == "rollout":
        prompt = (
            None
            if args.prompt_from is None
            else json.loads(args.prompt_from.read_text())["prompt_ids"]
        )
        result = asyncio.run(
            solve_parallel(
                args.model,
                max_steps=args.max_steps,
                memory_ratio=args.memory_ratio,
                trace_output=args.trace,
                prompt_ids=prompt,
            )
        )
    else:
        result = evaluate_checkpoint(args.trace, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ("answer", "completed_steps", "loss", "accuracy")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
