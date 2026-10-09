"""Mine short AsyncLLM trajectories and overfit their sampled-token targets."""

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .replay import Replay, hidden
from .trace import BlockUpdate, Token, Trace, record


def restore_trace(data):
    tokens = [Token(**{**t, "view": tuple(map(tuple, t["view"]))}) for t in data["tokens"]]
    updates = [
        BlockUpdate(
            **{
                **u,
                "sources": tuple(u.get("sources", ())),
                "refs": None if u.get("refs") is None else tuple(map(tuple, u["refs"])),
            }
        )
        for u in data["updates"]
    ]
    return Trace(tokens, data["steps"], updates)


async def mine(args):
    from minisgl.llm import AsyncLLM
    from minisgl.shared_cache import AsyncContext

    llm = AsyncLLM(
        args.model,
        dtype=torch.float16,
        attention_backend="fi",
        cuda_graph_max_bs=0,
        num_page_override=4096 if args.stress else 1024,
        max_running_req=8,
        max_seq_len_override=1024 if args.stress else 512,
        max_prefill_rows=8 if args.stress else 32,
    )
    session = llm.async_engine.session
    examples = []
    try:
        problems = ["What is 17 + 26?", "What is 9 * 7?", "What is 45 - 18?"]
        for index, problem in enumerate(problems[:2] if args.stress else problems):
            trace = record(llm)
            blocks = [await llm.create_block() for _ in range(7 if args.stress else 3)]
            prompt, *writers = blocks
            a, b = writers[:2]
            targets, logprobs = {}, {}
            operations = Counter()

            async def seed(block, text, context):
                ids = llm.tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > 1:
                    await llm.forward(ids[:-1], cache_view=[*context, block], write_to=block)
                return ids[-1]

            async def sample(block, view, pending, generator):
                cursor = len(trace.tokens)
                context = AsyncContext(cache_view=view, output_block=block, next_input_id=pending)
                output = await llm.forward(cache_view=context)
                rows = [
                    i
                    for i in range(cursor, len(trace.tokens))
                    if trace.tokens[i].block == block.block_id
                ]
                if len(rows) != 1:
                    raise RuntimeError("Sampling requires exactly one writer per block")
                row = rows[0]
                values, ids = output.logits.float().topk(20)
                choice = torch.multinomial(
                    (values / args.temperature).softmax(-1), 1, generator=generator
                )
                token = int(ids[choice].item())
                targets[row] = token
                logprobs[row] = output.logits.float().log_softmax(-1)[token].item()
                return token

            async def generate(block, view, pending, stream):
                generator = torch.Generator(device="cuda").manual_seed(
                    args.seed + 10 * index + stream
                )
                sampled = []
                for _ in range(args.max_tokens):
                    pending = await sample(block, view, pending, generator)
                    sampled.append(pending)
                    if pending == llm.tokenizer.eos_token_id:
                        break
                await llm.forward([pending], cache_view=view, write_to=block, return_logits=False)
                return llm.tokenizer.decode(sampled, skip_special_tokens=False)

            published = list(writers)

            def view(worker, turn, include_self=True):
                width = 2 + (worker + turn) % 5
                context = [prompt, *(published[(worker + j) % len(writers)] for j in range(width))]
                context = list(
                    {
                        block.block_id: block for block in context if block is not writers[worker]
                    }.values()
                )
                offset = turn % len(context)
                context = context[offset:] + context[:offset]
                return context + ([writers[worker]] if include_self else [])

            async def stress_worker(worker, pending):
                block = writers[worker]
                generator = torch.Generator(device="cuda").manual_seed(
                    args.seed + 10 * index + worker
                )
                sampled = []
                for turn in range(args.max_tokens):
                    for _ in range((worker + turn) % 3):
                        await asyncio.sleep(0)
                    if turn and (worker + turn) % 4 == 0:
                        await llm.forward(
                            [pending],
                            cache_view=view(worker, turn),
                            write_to=block,
                            return_logits=False,
                        )
                        merged = await llm.merge_blocks(block, writers[(worker + 1) % len(writers)])
                        copied = await llm.create_block()
                        blocks.extend([merged, copied])
                        await llm.append_block(copied, merged)
                        operations.update(["merge", "copy"])
                        if worker % 2 == 0:
                            await llm.append_block(copied, copied)
                            operations["self_append"] += 1
                        published[worker] = copied
                        while llm.async_engine._block_in_use(block):
                            await asyncio.sleep(0)
                        await llm.free_block(block)
                        await llm.free_block(merged)
                        operations["clear_reuse"] += 1
                        pending = await seed(block, f"\nPhase {turn}: ", view(worker, turn)[:-1])
                    elif turn == worker % 3 + 1:
                        await llm.forward(
                            [pending],
                            cache_view=view(worker, turn),
                            write_to=block,
                            return_logits=False,
                        )
                        pending = await seed(block, "\nNote: ", view(worker, turn)[:-1])
                        operations["prefill_extension"] += 1
                    pending = await sample(
                        block, view(worker, turn, turn % 4 != 2), pending, generator
                    )
                    sampled.append(pending)
                await llm.forward(
                    [pending],
                    cache_view=view(worker, args.max_tokens),
                    write_to=block,
                    return_logits=False,
                )
                return llm.tokenizer.decode(sampled, skip_special_tokens=False)

            prompt_ids = llm.tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "Solve the arithmetic problem. Keep the answer brief.",
                    },
                    {"role": "user", "content": problem},
                ],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            await llm.forward(prompt_ids, write_to=prompt, return_logits=False)
            seeds = await asyncio.gather(
                *(seed(block, f"Worker {i}: ", [prompt]) for i, block in enumerate(writers))
            )
            if args.stress:
                workers = await asyncio.gather(
                    *(stress_worker(i, pending) for i, pending in enumerate(seeds))
                )
                answer = None
            else:
                workers = await asyncio.gather(
                    generate(a, [prompt, b, a], seeds[0], 0),
                    generate(b, [prompt, a, b], seeds[1], 1),
                )
                merged = await llm.merge_blocks(a, b)
                copied = await llm.create_block()
                blocks.extend([merged, copied])
                await llm.append_block(copied, merged)
                for block in (a, b, merged):
                    await llm.free_block(block)
                pending = await seed(a, "\nFinal answer: ", [prompt, copied])
                answer = await generate(a, [prompt, copied, a], pending, 2)
            for block in blocks:
                await llm.free_block(block)
            batches = Counter(t.step for t in trace.tokens if t.kind == "decode")
            coverage = {
                "blocks": len(blocks),
                "operations": dict(operations),
                "ordered_views": len({tuple(b for b, _ in t.view) for t in trace.tokens}),
                "max_view_depth": max(len(t.view) for t in trace.tokens),
                "prefill_steps": len({t.step for t in trace.tokens if t.kind == "prefill"}),
                "decode_batch_sizes": dict(sorted(Counter(batches.values()).items())),
            }
            examples.append(
                {
                    "prompt": problem,
                    "workers": workers,
                    "answer": answer,
                    "trace": asdict(trace),
                    "targets": [targets.get(i, -100) for i in range(len(trace.tokens))],
                    "logprobs": [logprobs.get(i) for i in range(len(trace.tokens))],
                    "coverage": coverage,
                }
            )
            print(
                f"{problem} inputs={len(trace.tokens)} targets={len(targets)} answer={answer!r}",
                flush=True,
            )
            print(f"coverage={coverage}", flush=True)
            llm.async_engine.session = session
    finally:
        await llm.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model": args.model,
                "seed": args.seed,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "stress": args.stress,
                "examples": examples,
            },
            indent=2,
        )
    )


def _supervised(model, replay, targets):
    device = next(model.parameters()).device
    rows = (targets != -100).nonzero().flatten()
    if len(targets) != len(replay.input_ids) or not len(rows):
        raise ValueError("Each trace needs aligned, nonempty supervision")
    return hidden(model, replay.to(device), rows=rows.to(device)), targets[rows].to(device)


def supervised_loss(model, replay, targets, chunk_size=128):
    if chunk_size < 1:
        raise ValueError("Loss chunk size must be positive")
    states, selected = _supervised(model, replay, targets)

    def chunk_loss(x, y):
        return F.cross_entropy(model.lm_head(x).float(), y, reduction="sum")

    losses = []
    for x, y in zip(states.split(chunk_size), selected.split(chunk_size), strict=True):
        losses.append(
            checkpoint(chunk_loss, x, y, use_reentrant=False, preserve_rng_state=False)
            if torch.is_grad_enabled()
            else chunk_loss(x, y)
        )
    return torch.stack(losses).sum()


@torch.no_grad()
def evaluate(model, examples, chunk_size=128):
    if chunk_size < 1:
        raise ValueError("Loss chunk size must be positive")
    results = []
    for replay, targets in examples:
        states, selected = _supervised(model, replay, targets)
        nll, correct = [], 0
        for x, y in zip(states.split(chunk_size), selected.split(chunk_size), strict=True):
            logits = model.lm_head(x).float()
            nll.append(F.cross_entropy(logits, y, reduction="none"))
            correct = correct + (logits.argmax(-1) == y).sum()
            del logits
        nll = torch.cat(nll)
        results.append(
            {
                "tokens": len(selected),
                "loss": nll.mean().item(),
                "accuracy": correct.item() / len(selected),
                "nll": nll.cpu().tolist(),
            }
        )
    count = sum(r["tokens"] for r in results)
    return {
        "loss": sum(r["loss"] * r["tokens"] for r in results) / count,
        "accuracy": sum(r["accuracy"] * r["tokens"] for r in results) / count,
        "examples": results,
    }


def train(args):
    query_chunk_size = getattr(args, "query_chunk_size", 128)
    loss_chunk_size = getattr(args, "loss_chunk_size", 128)
    gdn_segment_size = getattr(args, "gdn_segment_size", 8)
    gdn_read_mode = getattr(args, "gdn_read_mode", "expanded")
    gdn_cache_reused = getattr(args, "gdn_cache_reused", True)
    gdn_read_backend = getattr(args, "gdn_read_backend", "torch")
    if gdn_segment_size < 1:
        raise ValueError("gdn_segment_size must be positive")
    if gdn_read_mode not in {"expanded", "factorized"}:
        raise ValueError("gdn_read_mode must be expanded or factorized")
    if gdn_read_backend not in {"torch", "triton"}:
        raise ValueError("gdn_read_backend must be torch or triton")
    data = json.loads(args.data.read_text())
    examples = []
    for example in data["examples"]:
        trace = restore_trace(example["trace"])
        targets = torch.tensor(example["targets"])
        if len(targets) != len(trace.tokens) or not (targets != -100).any():
            raise ValueError("Each trace needs aligned, nonempty supervision")
        replay = Replay.from_trace(
            trace,
            query_chunk_size=query_chunk_size,
            gdn_segment_size=gdn_segment_size,
            gdn_read_mode=gdn_read_mode,
            gdn_cache_reused=gdn_cache_reused,
            gdn_read_backend=gdn_read_backend,
        )
        examples.append((replay, targets))
    config = AutoConfig.from_pretrained(data["model"]).get_text_config()
    model = AutoModelForCausalLM.from_pretrained(
        data["model"], config=config, dtype=torch.float32
    ).cuda()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    initial = evaluate(model.eval(), examples, loss_chunk_size)
    errors = [
        abs(nll + lp)
        for result, example in zip(initial["examples"], data["examples"])
        for nll, lp in zip(
            result["nll"], (p for p in example["logprobs"] if p is not None), strict=True
        )
    ]
    max_error = max(errors)
    if max_error > 0.1:
        raise RuntimeError(f"Inference/replay target logprob mismatch: {max_error:.6f}")
    print(
        f"initial loss={initial['loss']:.6f} accuracy={initial['accuracy']:.2%} "
        f"max logprob error={max_error:.6f}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    count = sum(int((targets != -100).sum()) for _, targets in examples)
    history = [{"step": 0, "loss": initial["loss"], "accuracy": initial["accuracy"]}]
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for replay, targets in examples:
            loss = supervised_loss(model, replay, targets, loss_chunk_size) / count
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 5 == 0 or step == args.steps:
            final = evaluate(model.eval(), examples, loss_chunk_size)
            history.append({"step": step, "loss": final["loss"], "accuracy": final["accuracy"]})
            print(
                f"step {step}: loss={final['loss']:.6f} accuracy={final['accuracy']:.2%}",
                flush=True,
            )
            if all(r["accuracy"] == 1.0 and r["loss"] < 0.01 for r in final["examples"]):
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    AutoTokenizer.from_pretrained(data["model"]).save_pretrained(args.output)
    metrics = {
        "model": data["model"],
        "data": str(args.data),
        "seed": args.seed,
        "lr": args.lr,
        "steps": step,
        "supervised_tokens": count,
        "gradient_checkpointing": args.gradient_checkpointing,
        "query_chunk_size": query_chunk_size,
        "loss_chunk_size": loss_chunk_size,
        "gdn_segment_size": gdn_segment_size,
        "gdn_read_mode": gdn_read_mode,
        "gdn_cache_reused": gdn_cache_reused,
        "gdn_read_backend": gdn_read_backend,
        "max_logprob_error": max_error,
        "initial": initial,
        "final": final,
        "history": history,
        "torch": torch.__version__,
    }
    (args.output.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved checkpoint to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    mining = commands.add_parser("mine")
    mining.add_argument("--model", default="Qwen/Qwen3-0.6B")
    mining.add_argument("--stress", action="store_true")
    mining.add_argument("--max-tokens", type=int, default=16)
    mining.add_argument("--temperature", type=float, default=0.8)
    mining.add_argument("--seed", type=int, default=7)
    mining.add_argument(
        "--output", type=Path, default=Path("training_runs/qwen3_sft/trajectories.json")
    )
    training = commands.add_parser("train")
    training.add_argument(
        "--data", type=Path, default=Path("training_runs/qwen3_sft/trajectories.json")
    )
    training.add_argument("--output", type=Path, default=Path("training_runs/qwen3_sft/checkpoint"))
    training.add_argument("--steps", type=int, default=80)
    training.add_argument("--lr", type=float, default=3e-5)
    training.add_argument("--seed", type=int, default=7)
    training.add_argument("--gradient-checkpointing", action="store_true")
    training.add_argument("--query-chunk-size", type=int, default=128)
    training.add_argument("--loss-chunk-size", type=int, default=128)
    training.add_argument(
        "--gdn-segment-size",
        type=int,
        default=8,
        help="Maximum compact GDN segment length; 1 disables short-segment compaction",
    )
    training.add_argument(
        "--gdn-read-mode",
        choices=("expanded", "factorized"),
        default="expanded",
        help="Apply compact GDN reads token by token or through segment factors",
    )
    training.add_argument(
        "--gdn-cache-reused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache dense transforms for heavily reused GDN summaries",
    )
    training.add_argument(
        "--gdn-read-backend",
        choices=("torch", "triton"),
        default="torch",
        help="Use Torch or fused Triton kernels for GDN reads",
    )
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        parser.error("Select exactly one GPU with CUDA_VISIBLE_DEVICES")
    if not args.output.resolve().is_relative_to(Path.cwd()):
        parser.error("Outputs must be inside the working directory")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.command == "mine" and (args.max_tokens < 1 or args.temperature <= 0):
        parser.error("--max-tokens and --temperature must be positive")
    if args.command == "train" and (
        args.steps < 1
        or args.lr <= 0
        or args.query_chunk_size < 1
        or args.loss_chunk_size < 1
        or args.gdn_segment_size < 1
    ):
        parser.error(
            "--steps, --lr, --query-chunk-size, --loss-chunk-size and --gdn-segment-size "
            "must be positive"
        )
    torch.manual_seed(args.seed)
    if args.command == "mine":
        asyncio.run(mine(args))
    else:
        train(args)


if __name__ == "__main__":
    main()
