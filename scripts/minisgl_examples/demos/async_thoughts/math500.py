"""Paired, resumable Llama AsyncReasoning evaluation on pinned MATH-500."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import gzip
import hashlib
import json
import os
import time
from contextlib import aclosing, redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path

import torch
from minisgl.training import RecordedAsyncLLM
from transformers import GenerationConfig

from .demo import DemoConfig, LlamaPrompting, _run_loop
from .engine import build_async_llm
from .short_retry import generate_native_answer, retry_short_answer

DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
DATASET_SHA = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
ORDER = "tulu-native-concurrent-v1"
BOX = r"Give your final answer in \boxed{}."
ARMS = ("ordinary", "working-notes", "native-chat")


def encode_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def last_box(text):
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    start += len(r"\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i - 1] != "\\":
            depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start:i].strip()
    return None


def ordered_rows(path):
    raw = path.read_bytes()
    if digest(raw) != DATASET_SHA:
        raise ValueError("MATH-500 test.jsonl differs from the pinned dataset")
    rows = [dict(json.loads(line), index=i) for i, line in enumerate(raw.splitlines())]
    if len(rows) != 500 or len({r["unique_id"] for r in rows}) != 500:
        raise ValueError("Expected 500 distinct MATH-500 IDs")
    return sorted(rows, key=lambda row: digest((ORDER + "\0" + row["unique_id"]).encode()))


def schedule(rows, paired, limit):
    for rank, row in enumerate(rows[:limit]):
        # Rotate the first arm to balance timing effects within the paired cohort.
        arms = ARMS[rank % 3 :] + ARMS[: rank % 3] if rank < paired else ("native-chat",)
        for arm in arms:
            yield rank, row, arm


class Observed(RecordedAsyncLLM):
    """Capture the samples the real generator returned, including EOS."""

    def __init__(self, llm):
        super().__init__(llm)
        self.samples = {1: [], 2: [], 3: []}
        self.first_content = {}

    async def async_generate(self, context, **kwargs):
        async with aclosing(super().async_generate(context, **kwargs)) as stream:
            async for item in stream:
                width = len(context.cache_view)
                self.samples[width].append(int(item))
                if width not in self.first_content and int(item) not in self.eos_token_ids:
                    self.first_content[width] = time.perf_counter()
                yield item


def trace_metrics(trace):
    joint, visible = 0, []
    for batch in trace.batches:
        if batch.kind != "decode":
            continue
        got_thinker = any(len(r.cache_view) == 2 for r in batch.requests)
        writers = [r for r in batch.requests if len(r.cache_view) == 3]
        joint += int(got_thinker and bool(writers))
        visible.extend(r.cache_view[1][1] for r in writers)
    return {
        "joint_decode_batches": joint,
        "writer_first_visible_thinker_block_tokens": visible[0] if visible else None,
        "writer_last_visible_thinker_block_tokens": visible[-1] if visible else None,
        "writer_saw_thinker_grow": len(set(visible)) > 1,
    }


async def measure(llm, config, row, arm, quiet):
    tok = llm.tokenizer
    task = row["problem"] + "\n\n" + BOX
    protocol = "working-notes" if arm == "ordinary" else arm
    prompting = LlamaPrompting(task, tok, protocol=protocol)
    prompt_ids = tok.encode(prompting.input_prompt, add_special_tokens=False)
    config = replace(config, problem=task, llama_protocol=protocol)
    torch.cuda.synchronize()
    start = time.perf_counter()
    if arm == "ordinary":
        with Observed(llm) as rec:
            writer, stop = await generate_native_answer(rec, prompt_ids, config.max_answer_tokens)
        thinker = []
    else:
        with Observed(llm) as rec, redirect_stdout(quiet):
            thinker, writer = await _run_loop(config, rec, tok, prompting)
        samples = rec.samples[3]
        tail = samples[len(writer) :]
        if samples[: len(writer)] != writer or (
            tail and (len(tail) != 1 or tail[0] not in llm.eos_token_ids)
        ):
            raise ValueError("Writer token stream and terminal sample disagree")
        stop = tail[0] if tail else None
    if stop is None and len(writer) != config.max_answer_tokens:
        raise ValueError("Writer ended without EOS or its token limit")
    ar_trace = rec.get_trace()
    initial = list(ar_trace.batches[0].requests[0].input_ids)
    if arm in ("ordinary", "native-chat"):
        role_width = 1 if arm == "ordinary" else 2
        first_step = next(
            request
            for batch in ar_trace.batches
            if batch.kind == "decode"
            for request in batch.requests
            if len(request.cache_view) == role_width
        )
        initial.extend(first_step.input_ids)
    if initial != prompt_ids:
        raise ValueError("Executed prompt tokens differ from the pinned ordinary input")
    torch.cuda.synchronize()
    draft_seconds = time.perf_counter() - start
    retry_trace = None
    retry_ids, retry_stop = None, None
    selected, source = writer, arm
    if arm != "ordinary":
        with RecordedAsyncLLM(llm) as retry_rec:
            decision = await retry_short_answer(
                retry_rec,
                prompt_ids=prompt_ids,
                draft_ids=writer,
                draft_stop_id=stop,
                max_answer_tokens=config.max_answer_tokens,
            )
        selected, source = decision.final_ids, decision.source
        retry_ids, retry_stop = decision.retry_ids, decision.retry_stop_id
        if retry_ids is not None:
            retry_trace = retry_rec.get_trace()
    torch.cuda.synchronize()
    pure_text = tok.decode(writer, skip_special_tokens=True)
    selected_text = tok.decode(selected, skip_special_tokens=True)
    result = {
        "index": row["index"],
        "unique_id": row["unique_id"],
        "arm": arm,
        "prompt_ids_sha256": digest(encode_json(prompt_ids)),
        "gold": row["answer"],
        "thinker_ids": thinker,
        "writer_ids": writer,
        "writer_stop_id": stop,
        "writer_text": pure_text,
        "writer_box": last_box(pure_text),
        "pure_exact": last_box(pure_text) == row["answer"].strip(),
        "selected_source": source,
        "retry_ids": retry_ids,
        "retry_stop_id": retry_stop,
        "selected_text": selected_text,
        "selected_box": last_box(selected_text),
        "selected_exact": last_box(selected_text) == row["answer"].strip(),
        "writer_samples_charged": len(writer)
        + int(stop is not None)
        + len(retry_ids or [])
        + int(retry_stop is not None),
        "draft_seconds": draft_seconds,
        "draft_first_content_seconds": (
            rec.first_content[1 if arm == "ordinary" else 3] - start if writer else None
        ),
        "total_seconds": time.perf_counter() - start,
        **trace_metrics(ar_trace),
    }
    if result["writer_samples_charged"] > config.max_answer_tokens:
        raise ValueError("Retry exceeded shared writer budget")
    return result, {
        "primary": asdict(ar_trace),
        "retry": asdict(retry_trace) if retry_trace is not None else None,
    }


async def smoke(llm, config, quiet):
    summary = []
    config = replace(config, max_steps=384, max_answer_tokens=384, probe_period=4)
    for i, (question, gold) in enumerate(
        [
            ("What is 2 + 3?", "5"),
            ("How many positive whole-number divisors does 196 have?", "9"),
        ]
    ):
        row = {"index": i, "unique_id": f"smoke-{i}", "problem": question, "answer": gold}
        normal, _ = await measure(llm, config, row, "ordinary", quiet)
        native, _ = await measure(llm, config, row, "native-chat", quiet)
        common = 0
        for x, y in zip(normal["writer_ids"], native["thinker_ids"]):
            if x != y:
                break
            common += 1
        summary.append(
            {
                "question": question,
                "gold": gold,
                "ordinary_box": normal["writer_box"],
                "native_box": native["writer_box"],
                "matching_native_thinker_prefix": common,
                "same_prompt": normal["prompt_ids_sha256"] == native["prompt_ids_sha256"],
                "joint_decode_batches": native["joint_decode_batches"],
                "writer_saw_thinker_grow": native["writer_saw_thinker_grow"],
            }
        )
    return summary


def verify_smoke(rows):
    if not all(row["same_prompt"] for row in rows) or not any(
        row["matching_native_thinker_prefix"] >= 8 for row in rows
    ):
        raise ValueError(f"Native thinker prompt or initial greedy prefix parity failed: {rows}")
    if rows[0]["ordinary_box"] != "5" or rows[0]["native_box"] != "5":
        raise ValueError(f"Elementary arithmetic smoke failed: {rows}")
    if not any(row["joint_decode_batches"] and row["writer_saw_thinker_grow"] for row in rows):
        raise ValueError(f"Smoke did not demonstrate concurrent reads of growing notes: {rows}")


async def evaluate(args, rows, settings, quiet):
    gen = GenerationConfig.from_pretrained(args.model, local_files_only=True)
    gen.do_sample, gen.temperature, gen.top_k, gen.top_p = False, 1.0, 1, 1.0
    torch.manual_seed(0)
    llm = build_async_llm(
        args.model, args.memory_ratio, max_context_len=args.context, generation_config=gen
    )
    try:
        save(
            args.output / "environment.json",
            encode_json(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "cuda": torch.version.cuda,
                    "effective_context": llm.engine.max_seq_len,
                    "eos_token_ids": sorted(llm.eos_token_ids),
                }
            ),
        )
        if (
            llm.config.model_config.model_type != "llama"
            or not llm.config.get_default_sampling_params().is_greedy
        ):
            raise ValueError("Benchmark requires greedy dense Llama")
        config = DemoConfig(
            model=args.model, max_steps=args.budget, max_answer_tokens=args.budget, probe_period=20
        )
        smokes = await smoke(llm, config, quiet)
        save(args.output / "smoke.json", encode_json(smokes))
        verify_smoke(smokes)
        print("Smoke passed", json.dumps(smokes), flush=True)
        if args.smoke_only:
            return
        for rank, row, arm in schedule(rows, args.paired, args.limit):
            stem = f"{rank:03d}.{arm}"
            result_path = args.output / "rows" / f"{stem}.json"
            trace_path = args.output / "traces" / f"{stem}.json.gz"
            if result_path.exists():
                prior = json.loads(result_path.read_bytes())
                if (
                    prior["arm"] != arm
                    or prior["unique_id"] != row["unique_id"]
                    or prior["trace_sha256"] != digest(trace_path.read_bytes())
                ):
                    raise ValueError(f"Inconsistent completed result at {stem}")
                continue
            data, trace = await measure(llm, config, row, arm, quiet)
            packed = gzip.compress(encode_json(trace), compresslevel=1, mtime=0)
            save(trace_path, packed)
            data.update(rank=rank, trace_sha256=digest(packed))
            save(result_path, encode_json(data))
            print(
                rank,
                arm,
                row["unique_id"],
                data["pure_exact"],
                data["selected_exact"],
                f"{data['total_seconds']:.1f}s",
                flush=True,
            )
        save(
            args.output / "completed.json",
            encode_json(
                {
                    "planned_rows": len(list(schedule(rows, args.paired, args.limit))),
                    "settings_sha256": digest(encode_json(settings)),
                }
            ),
        )
    finally:
        await llm.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Pinned local Hugging Face snapshot")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired", type=int, default=64)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--budget", type=int, default=16384)
    parser.add_argument("--context", type=int, default=40960)
    parser.add_argument("--memory-ratio", type=float, default=0.85)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if (
        not 0 <= args.paired <= args.limit <= 500
        or args.limit < 1
        or args.budget < 1
        or args.context <= 2 * args.budget
    ):
        parser.error("Set a valid cohort size and allow context for both roles plus the prompt")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("Expose exactly one GPU")
    model = Path(args.model).resolve()
    if model.name != args.model_revision or model.parent.name != "snapshots":
        parser.error("--model must point to the named Hub revision under snapshots/")
    if args.output.resolve() == model:
        parser.error("Use a separate evaluation output directory")
    args.model = str(model)
    rows = ordered_rows(args.dataset)
    here = Path(__file__).parent
    settings = {
        "model": model.parent.parent.name,
        "model_revision": args.model_revision,
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": DATASET_SHA,
        "order": ORDER,
        "ordered_ids": [r["unique_id"] for r in rows[: args.limit]],
        "paired": args.paired,
        "limit": args.limit,
        "budget": args.budget,
        "context": args.context,
        "memory_ratio": args.memory_ratio,
        "sampling": "greedy",
        "dtype": "bfloat16",
        "probe_period": 20,
        "smoke_only": args.smoke_only,
        "torch": str(torch.__version__),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_sha256": {
            p: digest((here / p).read_bytes())
            for p in ("math500.py", "demo.py", "engine.py", "short_retry.py")
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock, open(os.devnull, "w") as quiet:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = args.output / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_bytes()) != settings:
            raise ValueError("Output belongs to a different evaluation")
        save(manifest, encode_json(settings))
        for subdir in ("rows", "traces"):
            (args.output / subdir).mkdir(exist_ok=True)
        asyncio.run(evaluate(args, rows, settings, quiet))


if __name__ == "__main__":
    main()
