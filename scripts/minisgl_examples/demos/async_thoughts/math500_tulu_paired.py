"""Paired Tülu direct-answer, serial private-thinking, and gated async MATH-500 evaluation."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import gzip
import json
import os
import tempfile
import time
from collections import Counter
from contextlib import aclosing, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

import torch
from minisgl.core import SamplingParams
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM
from transformers import GenerationConfig

from .demo import DemoConfig, LlamaPrompting, _run_loop
from .engine import build_async_llm
from .math500 import (
    BOX,
    DATASET_REVISION,
    DATASET_SHA,
    ORDER,
    digest,
    encode_json,
    last_box,
    ordered_rows,
    save,
)

ARMS = ("direct", "serial-thinking", "evidence-turns")
CHECKPOINT = "allenai/Llama-3.1-Tulu-3-8B"
REVISION = "666943798adbde0b1aff34626007e26986a3c107"
DIRECT = "Answer the user's question directly in the requested format. Give only the final answer."
SERIAL = (
    "Use your first assistant turn for private working notes. Solve the problem and check your reasoning. "
    "You will be asked for the final answer in a subsequent turn."
)
FINAL = "Use your preceding working notes to answer the original question. Give only the final answer in the requested format."
SMOKES = (
    {"unique_id": "smoke/addition", "index": -1, "problem": "What is 2 + 3?", "answer": "5"},
    {
        "unique_id": "smoke/divisors",
        "index": -2,
        "problem": "How many positive whole-number divisors does 196 have?",
        "answer": "9",
    },
)
SOURCES = (
    "math500_tulu_paired.py",
    "math500.py",
    "demo.py",
    "engine.py",
    "probe_log.py",
    "short_retry.py",
)


@dataclass
class Prompt:
    user: str
    system: str | None
    initial: list[int]
    bridge: list[int]
    followup: str | None
    native: LlamaPrompting | None = None


def make_prompt(tok, problem, arm):
    task = problem + "\n\n" + BOX
    if arm == "evidence-turns":
        native = LlamaPrompting(task, tok, protocol="native-chat")
        turns = native.native_turns
        if turns is None:
            raise ValueError("Expected native turn boundary")
        # Extract the exact messages that produced the native template instead of duplicating its wording.
        user = native.input_prompt.removeprefix("<|user|>\n").removesuffix("\n<|assistant|>\n")
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=True, add_generation_prompt=True
        )
        if token_list(rendered) != turns.prompt_ids:
            raise ValueError("Cannot reconstruct the native user message")
        bridge_text = tok.decode(
            turns.writer_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        _, sep, right = bridge_text.partition("<|user|>\n")
        if not sep or not right.endswith("\n<|assistant|>\n"):
            raise ValueError("Unexpected native writer bridge")
        return Prompt(
            user,
            None,
            turns.prompt_ids,
            turns.writer_ids,
            right.removesuffix("\n<|assistant|>\n"),
            native,
        )
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm}")
    system = DIRECT if arm == "direct" else SERIAL
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    prefix = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    initial = encode(prefix)
    if arm == "direct":
        return Prompt(task, system, initial, [], None)
    marker = "MINISGL_PRIVATE_TULU_DRAFT_MARKER"
    full = tok.apply_chat_template(
        messages + [{"role": "assistant", "content": marker}, {"role": "user", "content": FINAL}],
        tokenize=False,
        add_generation_prompt=True,
    )
    before, sep, after = full.rpartition(marker)
    bridge = encode(after)
    if (
        not sep
        or before != prefix
        or encode(prefix + marker) + bridge != encode(full)
        or bridge[0] != tok.eos_token_id
    ):
        raise ValueError("Tülu template does not provide a stable EOS-delimited assistant turn")
    return Prompt(task, system, initial, bridge, FINAL)


def token_list(encoded):
    return list(encoded if isinstance(encoded, list) else encoded["input_ids"])


class Sampler(RecordedAsyncLLM):
    """Bind returned samples to real decode batches, including EOS and simultaneous roles."""

    def __init__(self, llm):
        super().__init__(llm)
        self.phase = "writer"
        self.events = []
        self.first_writer_time = None

    async def async_generate(self, context, **kwargs):
        width = len(context.cache_view)
        role = {1: self.phase, 2: "thinker", 3: "writer"}[width]
        block = context.output_block
        async with aclosing(super().async_generate(context, **kwargs)) as stream:
            while True:
                cursor = len(self.get_trace().batches)
                offset = block.num_tokens
                try:
                    item = await anext(stream)
                except StopAsyncIteration:
                    break
                token = int(item[0] if isinstance(item, tuple) else item)
                matches = [
                    b.step
                    for b in self.get_trace().batches[cursor:]
                    if b.kind == "decode"
                    for request in b.requests
                    if request.write_to == block.block_id and request.write_start == offset
                ]
                if len(matches) != 1:
                    raise ValueError("Sample has no unique recorded decode")
                self.events.append({"role": role, "token": token, "batch_step": matches[0]})
                if (
                    role == "writer"
                    and self.first_writer_time is None
                    and token not in self.eos_token_ids
                ):
                    self.first_writer_time = time.perf_counter()
                yield item


async def decode(rec, ctx, role, seed, limit):
    rec.phase = role
    ids = []
    sampler = SamplingParams(temperature=0.0, max_tokens=limit)
    async with aclosing(
        rec.async_generate(ctx, first_token_id=seed, sampling_params=sampler, max_steps=limit)
    ) as stream:
        async for token in stream:
            if token in rec.eos_token_ids:
                return ids, int(token)
            ids.append(int(token))
    if len(ids) != limit:
        raise ValueError("Serial stream ended before EOS or its sample limit")
    return ids, None


async def run_serial(rec, p, budget, private_cap):
    block = await rec.create_block()
    try:
        ctx = AsyncContext(cache_view=[block])
        await rec.forward(p.initial[:-1], write_to=block, return_logits=False)
        if not p.bridge:
            writer, _ = await decode(rec, ctx, "writer", p.initial[-1], budget)
            return [], writer
        thoughts, stop = await decode(rec, ctx, "thinker", p.initial[-1], private_cap)
        remaining = budget - len(thoughts) - int(stop is not None)
        # EOS is the first token of the template bridge. If the private turn hit its cap,
        # its last sampled content token is pending, and has to be committed exactly once.
        pending = [] if stop is not None else [thoughts[-1]]
        await rec.forward(
            pending + p.bridge[:-1], cache_view=ctx.cache_view, write_to=block, return_logits=False
        )
        writer, _ = await decode(rec, ctx, "writer", p.bridge[-1], remaining)
        return thoughts, writer
    finally:
        await rec.free_block(block)


def visible_at_sample_budget(events, limit, eos_ids):
    """Retain only complete decode batches. A simultaneous pair cannot be split by iterator order."""
    groups = {}
    for event in events:
        groups.setdefault(event["batch_step"], []).append(event)
    left, writer = limit, []
    for step in sorted(groups):
        group = groups[step]
        if len(group) > left:
            break
        left -= len(group)
        writer.extend(
            e["token"] for e in group if e["role"] == "writer" and e["token"] not in eos_ids
        )
    return writer


def validate_sample_log(events, thinker, writer, eos_ids, arm, budget, private_cap):
    stops, counts = {}, {}
    for role, accepted in (("thinker", thinker), ("writer", writer)):
        sampled = [e["token"] for e in events if e["role"] == role]
        tail = sampled[len(accepted) :]
        # When the writer finishes, a concurrently sampled last thought may already be in flight.
        if (
            sampled[: len(accepted)] != accepted
            or len(tail) > 1
            or any(t in eos_ids for t in accepted)
            or (
                tail
                and tail[0] not in eos_ids
                and not (arm == "evidence-turns" and role == "thinker")
            )
        ):
            raise ValueError(f"{arm} {role} samples differ from accepted tokens")
        stops[role] = tail[0] if tail and tail[0] in eos_ids else None
        counts[role] = len(sampled)
    if max(counts.values()) > budget or (arm != "evidence-turns" and sum(counts.values()) > budget):
        raise ValueError("Generation exceeded its declared sample budget")
    if (arm == "direct" and counts["thinker"]) or (
        arm == "serial-thinking" and counts["thinker"] > private_cap
    ):
        raise ValueError("Invalid private allocation")
    return counts, stops


async def measure(llm, args, row, rank, arm, budget, settings_sha, scratch, quiet):
    p = make_prompt(llm.tokenizer, row["problem"], arm)
    if (
        len(p.initial) + len(p.bridge) + budget * (2 if arm == "evidence-turns" else 1)
        > llm.engine.max_seq_len
    ):
        raise ValueError("Insufficient context for the configured streams")
    private_cap = budget // 2
    config = DemoConfig(
        model=args.model,
        problem=row["problem"] + "\n\n" + BOX,
        max_steps=budget,
        max_answer_tokens=budget,
        probe_period=20,
        llama_protocol="native-chat",
    )
    probe_events = []
    with Sampler(llm) as rec:
        torch.cuda.synchronize()
        start = time.perf_counter()
        if arm == "evidence-turns":
            config.llama_gate = "evidence-turns"
            config.probe_include_pending = True
            config.probe_log = scratch / f"{rank:03}.{arm}.json"
            with redirect_stdout(quiet):
                thinker, writer = await _run_loop(config, rec, llm.tokenizer, p.native)
        else:
            thinker, writer = await run_serial(rec, p, budget, private_cap)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        trace = rec.get_trace()
    if arm == "evidence-turns":
        log = json.loads(config.probe_log.read_bytes())
        if (log["llama_gate"], log["probe_include_pending"], log["probe_period"]) != (
            "evidence-turns",
            True,
            20,
        ):
            raise ValueError("Observed async gate differs from the scheduled policy")
        probe_events = log["events"]
    counts, stops = validate_sample_log(
        rec.events, thinker, writer, llm.eos_token_ids, arm, budget, private_cap
    )
    first_batch = trace.batches[0]
    first_decode = next(b for b in trace.batches if b.kind == "decode")
    if (
        list(first_batch.requests[0].input_ids) + list(first_decode.requests[0].input_ids)
        != p.initial
    ):
        raise ValueError("Executed prompt differs from the intended native token IDs")
    prefix = visible_at_sample_budget(rec.events, budget, llm.eos_token_ids)
    text = llm.tokenizer.decode(writer, skip_special_tokens=True)
    box = last_box(text)
    pure = box == row["answer"].strip()
    data = {
        "schema_version": 1,
        "rank": rank,
        "index": row["index"],
        "unique_id": row["unique_id"],
        "arm": arm,
        "settings_sha256": settings_sha,
        "initial_prompt_ids": p.initial,
        "prompt_ids_sha256": digest(encode_json(p.initial)),
        "user_message_sha256": digest(p.user.encode()),
        "bridge_ids": p.bridge,
        "system_message": p.system,
        "user_message": p.user,
        "followup_message": p.followup,
        "thinker_ids": thinker,
        "thinker_stop_id": stops["thinker"],
        "writer_ids": writer,
        "writer_stop_id": stops["writer"],
        "writer_text": text,
        "writer_box": box,
        "gold": row["answer"],
        "pure_exact": pure,
        "thinker_samples": counts["thinker"],
        "writer_samples": counts["writer"],
        "decode_batches": sum(b.kind == "decode" for b in trace.batches),
        "forward_batches": len(trace.batches),
        "joint_decode_batches": sum(
            b.kind == "decode" and {len(r.cache_view) for r in b.requests}.issuperset({2, 3})
            for b in trace.batches
        ),
        "elapsed_seconds": elapsed,
        "first_writer_seconds": rec.first_writer_time - start if rec.first_writer_time else None,
        "shared_prefix_writer_tokens": len(prefix),
        "shared_prefix_exact": last_box(llm.tokenizer.decode(prefix, skip_special_tokens=True))
        == row["answer"].strip(),
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }
    return data, {
        "format_version": 1,
        "trace": asdict(trace),
        "samples": rec.events,
        "probe_events": probe_events,
    }


def verify_saved(output, stem, expected, meta_hash, *, settings=None, tokenizer=None):
    """Load a complete row and independently check its content against the recorded execution."""
    if settings is None:
        settings = json.loads((output / "manifest.json").read_bytes())
    if digest(encode_json(settings)) != meta_hash:
        raise ValueError("Settings hash mismatch")
    row = json.loads((output / "rows" / f"{stem}.json").read_bytes())
    trace = (output / "traces" / f"{stem}.json.gz").read_bytes()
    if (row["unique_id"], row["arm"], row["rank"], row["settings_sha256"]) != (
        *expected,
        meta_hash,
    ):
        raise ValueError(f"{stem}: incompatible row identity or settings")
    if row["trace_sha256"] != digest(trace) or row["pure_exact"] != (
        last_box(row["writer_text"]) == row["gold"].strip()
    ):
        raise ValueError(f"{stem}: inconsistent trace or grade")
    packed = json.loads(gzip.decompress(trace))
    events = packed["samples"]
    counts, stops = validate_sample_log(
        events,
        row["thinker_ids"],
        row["writer_ids"],
        {128001},
        row["arm"],
        settings["budget"],
        settings["serial_draft_cap"],
    )
    batches = packed["trace"]["batches"]
    roles = {}
    for event in events:
        roles.setdefault(event["batch_step"], []).append(event["role"])
    for step, group in roles.items():
        if not 0 <= step < len(batches):
            raise ValueError(f"{stem}: invalid sample step")
        batch = batches[step]
        if (
            batch["step"] != step
            or batch["kind"] != "decode"
            or len(group) > len(batch["requests"])
            or len(set(group)) != len(group)
        ):
            raise ValueError(f"{stem}: samples differ from the recorded batch")
    first_decode = next(batch for batch in batches if batch["kind"] == "decode")
    prompt = batches[0]["requests"][0]["input_ids"] + first_decode["requests"][0]["input_ids"]
    prefix = visible_at_sample_budget(events, settings["budget"], {128001})
    checks = [
        row["initial_prompt_ids"] == prompt,
        row["prompt_ids_sha256"] == digest(encode_json(prompt)),
        row["user_message_sha256"] == digest(row["user_message"].encode()),
        row["thinker_samples"] == counts["thinker"],
        row["writer_samples"] == counts["writer"],
        row["thinker_stop_id"] == stops["thinker"],
        row["writer_stop_id"] == stops["writer"],
        row["decode_batches"] == sum(b["kind"] == "decode" for b in batches),
        row["forward_batches"] == len(batches),
        row["joint_decode_batches"]
        == sum(
            b["kind"] == "decode"
            and {len(q["cache_view"]) for q in b["requests"]}.issuperset({2, 3})
            for b in batches
        ),
        row["shared_prefix_writer_tokens"] == len(prefix),
        row["writer_box"] == last_box(row["writer_text"]),
    ]
    if tokenizer is not None:
        checks.extend(
            [
                row["writer_text"] == tokenizer.decode(row["writer_ids"], skip_special_tokens=True),
                row["shared_prefix_exact"]
                == (
                    last_box(tokenizer.decode(prefix, skip_special_tokens=True))
                    == row["gold"].strip()
                ),
            ]
        )
    if not all(checks):
        raise ValueError(f"{stem}: saved row differs from measured trace")
    return row


def report(output, settings):
    meta_hash = digest(encode_json(settings))
    rows = []
    for rank in range(settings["start_rank"], settings["limit"]):
        for arm in settings["arms"]:
            stem = f"{rank:03}.{arm}"
            if (output / "rows" / f"{stem}.json").exists():
                rows.append(
                    verify_saved(
                        output,
                        stem,
                        (settings["ordered_ids"][rank], arm, rank),
                        meta_hash,
                        settings=settings,
                    )
                )
    by_arm = {}
    for arm in settings["arms"]:
        subset = [r for r in rows if r["arm"] == arm]
        if not subset:
            by_arm[arm] = {"n": 0}
            continue
        stats = {}
        for field in (
            "writer_tokens",
            "thinker_tokens",
            "decode_batches",
            "forward_batches",
            "elapsed_seconds",
        ):
            vals = [
                len(r[field.replace("tokens", "ids")]) if field.endswith("_tokens") else r[field]
                for r in subset
            ]
            stats[field] = {"mean": mean(vals), "median": median(vals), "sum": sum(vals)}
        by_arm[arm] = {
            "n": len(subset),
            "strict_exact": sum(r["pure_exact"] for r in subset),
            "shared_prefix_exact": sum(r["shared_prefix_exact"] for r in subset),
            "stats": stats,
        }
    paired = {}
    for i, a in enumerate(settings["arms"]):
        left = {r["unique_id"]: r for r in rows if r["arm"] == a}
        for b in settings["arms"][i + 1 :]:
            right = {r["unique_id"]: r for r in rows if r["arm"] == b}
            counts = Counter(
                (left[uid]["pure_exact"], right[uid]["pure_exact"])
                for uid in left.keys() & right.keys()
            )
            paired[f"{a}_vs_{b}"] = {
                "both": counts[True, True],
                "left_only": counts[True, False],
                "right_only": counts[False, True],
                "neither": counts[False, False],
            }
    result = {
        "settings_sha256": meta_hash,
        "n_rows": len(rows),
        "expected_rows": (settings["limit"] - settings["start_rank"]) * len(settings["arms"]),
        "arms": by_arm,
        "paired": paired,
    }
    save(output / "summary.json", encode_json(result))
    return result


async def evaluate(args, rows, settings, scratch, quiet):
    gen = GenerationConfig.from_pretrained(args.model, local_files_only=True)
    gen.do_sample, gen.temperature, gen.top_k, gen.top_p = False, 1.0, 1, 1.0
    torch.manual_seed(0)
    llm = build_async_llm(
        args.model, args.memory_ratio, max_context_len=args.context, generation_config=gen
    )
    try:
        if (
            llm.config.model_config.model_type != "llama"
            or not llm.config.get_default_sampling_params().is_greedy
            or llm.eos_token_ids != {128001}
            or llm.engine.max_seq_len < args.context
        ):
            raise ValueError(
                "Expected greedy dense Tülu, pinned EOS 128001, and the requested context"
            )
        meta_hash = digest(encode_json(settings))
        for rank in range(settings["start_rank"], settings["limit"]):
            row = rows[rank]
            arms = args.arms[rank % len(args.arms) :] + args.arms[: rank % len(args.arms)]
            for arm in arms:
                stem = f"{rank:03}.{arm}"
                row_path = args.output / "rows" / f"{stem}.json"
                if row_path.exists():
                    verify_saved(
                        args.output,
                        stem,
                        (row["unique_id"], arm, rank),
                        meta_hash,
                        settings=settings,
                        tokenizer=llm.tokenizer,
                    )
                    continue
                data, trace = await measure(
                    llm, args, row, rank, arm, settings["budget"], meta_hash, scratch, quiet
                )
                packed = gzip.compress(encode_json(trace), compresslevel=1, mtime=0)
                save(args.output / "traces" / f"{stem}.json.gz", packed)
                data["trace_sha256"] = digest(packed)
                save(row_path, encode_json(data))
                print(
                    f"{rank} {arm} {row['unique_id']} {data['pure_exact']} {data['elapsed_seconds']:.2f}s",
                    flush=True,
                )
    finally:
        await llm.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Path to the locally downloaded, pinned Tülu checkpoint")
    parser.add_argument("--model-revision", default=REVISION)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument(
        "--limit", type=int, default=64, help="Exclusive end of the frozen MATH-500 ranks"
    )
    parser.add_argument("--start-rank", type=int, default=0)
    parser.add_argument("--budget", type=int, default=16384)
    parser.add_argument("--context", type=int, default=40960)
    parser.add_argument("--memory-ratio", type=float, default=0.85)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run two arithmetic fixtures with at most 512 samples",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regrade and summarize a complete or partial journal offline",
    )
    args = parser.parse_args()
    output = args.output
    if args.report_only:
        settings = json.loads((output / "manifest.json").read_bytes())
        print(json.dumps(report(output, settings), indent=2))
        return
    if (
        not args.model
        or not args.dataset
        or args.model_revision != REVISION
        or not 0 < args.memory_ratio <= 1
    ):
        parser.error("Provide the pinned local Tülu model and dataset")
    if (
        len(set(args.arms)) != len(args.arms)
        or not 0 <= args.start_rank < args.limit <= 500
        or args.budget < 2
    ):
        parser.error(
            "Use distinct arms, a positive rank range and a budget of at least two samples"
        )
    snapshot = Path(args.model).resolve()
    if (
        snapshot.name != REVISION
        or snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != "models--allenai--Llama-3.1-Tulu-3-8B"
    ):
        parser.error("Use the pinned Hugging Face cache snapshot (a symlink to it is supported)")
    rows = ordered_rows(args.dataset)
    budget = args.budget
    if args.smoke_only:
        rows = list(SMOKES)
        args.start_rank, args.limit = 0, len(rows)
        budget = min(512, budget)
    if budget > 16384 or args.context < 2 or args.context > 40960:
        parser.error("Supported sample cap is 16,384 and maximum context is 40,960")
    here = Path(__file__).resolve().parent
    settings = {
        "schema_version": 1,
        "model": CHECKPOINT,
        "model_revision": REVISION,
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": DATASET_SHA,
        "order": ORDER,
        "ordered_ids": [r["unique_id"] for r in rows[: args.limit]],
        "arms": args.arms,
        "start_rank": args.start_rank,
        "limit": args.limit,
        "budget": budget,
        "serial_draft_cap": budget // 2,
        "context": args.context,
        "memory_ratio": args.memory_ratio,
        "probe_period": 20,
        "llama_gate": "evidence-turns",
        "probe_include_pending": True,
        "dtype": "bfloat16",
        "sampling": "greedy",
        "smoke_only": args.smoke_only,
        "source_sha256": {name: digest((here / name).read_bytes()) for name in SOURCES},
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock, open(os.devnull, "w") as quiet:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_bytes()) != settings:
            raise ValueError("Output directory belongs to a different evaluation")
        save(manifest, encode_json(settings))
        for sub in ("rows", "traces"):
            (output / sub).mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            asyncio.run(evaluate(args, rows, settings, Path(tmp), quiet))
        summary = report(output, settings)
        if summary["n_rows"] != summary["expected_rows"]:
            raise RuntimeError("Incomplete paired schedule")
        if args.smoke_only:
            smoke = [json.loads(p.read_bytes()) for p in sorted((output / "rows").glob("*.json"))]
            if any(not r["pure_exact"] for r in smoke if r["rank"] == 0):
                raise RuntimeError("Addition smoke failed: inspect the retained answers and traces")
            save(
                output / "smoke.json",
                encode_json(
                    [{"arm": r["arm"], "id": r["unique_id"], "box": r["writer_box"]} for r in smoke]
                ),
            )
        save(
            output / "completed.json",
            encode_json(
                {
                    "settings_sha256": summary["settings_sha256"],
                    "planned_rows": summary["expected_rows"],
                    "completed_rows": summary["n_rows"],
                    "smoke_only": args.smoke_only,
                }
            ),
        )


if __name__ == "__main__":
    main()
