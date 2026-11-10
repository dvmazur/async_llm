"""Ordinary MATH-500 with either a single stream or the async_thoughts demo."""
import argparse
import asyncio
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import time
import urllib.request

from .grading import grade_one

INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."

DATASET = "HuggingFaceH4/MATH-500"
REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
DATA_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"


def sampling_config(overrides):
    # Standard Gemma IT settings, shared by thinker and writer.
    params = dict(temperature=1.0, top_p=0.95, top_k=64)
    if not isinstance(overrides, dict) or set(overrides) - params.keys():
        raise ValueError("sampling accepts only temperature, top_p and top_k")
    params.update(overrides)
    for name in ("temperature", "top_p"):
        if type(params[name]) not in (int, float) or not math.isfinite(params[name]):
            raise ValueError(f"sampling.{name} must be finite")
    if params["temperature"] < 0 or not 0 < params["top_p"] <= 1:
        raise ValueError("sampling requires temperature >= 0 and 0 < top_p <= 1")
    if type(params["top_k"]) is not int or (params["top_k"] != -1 and params["top_k"] < 1):
        raise ValueError("sampling.top_k must be -1 (disabled) or a positive integer")
    if params["temperature"] == 0 and params["top_p"] != 1:
        raise ValueError("For greedy decoding use temperature=0 and top_p=1")
    return params


def select_cases(cases, *, start=0, limit=None, shard_index=None, num_shards=None):
    if start < 0 or (limit is not None and limit < 1):
        raise ValueError("start must be nonnegative; limit must be positive")
    if (shard_index is None) != (num_shards is None):
        raise ValueError("Pass both --shard-index and --num-shards")
    if num_shards is not None and (num_shards < 1 or not 0 <= shard_index < num_shards):
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    selected = cases[start:None if limit is None else start + limit]
    if num_shards is not None:
        selected = selected[shard_index::num_shards]
    if not selected:
        raise ValueError("Selected no cases: check range and shard options")
    return selected


async def drained(awaitable):
    """Cancellation must not free blocks while their submitted work is running."""
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise


def parse_data(payload):
    cases, seen = [], set()
    for i, line in enumerate(payload.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("problem", "answer")):
            raise ValueError(f"Row {i}: nonempty problem and answer required")
        key = str(row.get("unique_id", row.get("id", i)))
        if key in seen:
            raise ValueError(f"Duplicate question id: {key}")
        seen.add(key)
        cases.append(dict(row, id=key))
    if not cases:
        raise ValueError("Empty dataset")
    return cases


def load_data(path, download=False):
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing dataset: {path}; pass --download-data")
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/test.jsonl"
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read(2 * 1024 * 1024)
        if hashlib.sha256(payload).hexdigest() != DATA_SHA256 or len(parse_data(payload)) != 500:
            raise ValueError("Dataset checksum or 500-row count mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return parse_data(path.read_bytes())


def read_config(path=None, *, model_path=None, backend=None):
    config = json.loads(Path(path).read_text()) if path is not None else {}
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    if model_path is not None:
        config["model_path"] = model_path
    if backend is not None:
        config["backend"] = backend
    if set(config) - {"model_path", "engine", "sampling", "thinking", "max_new_tokens", "seed",
                      "mode", "backend", "max_think_tokens", "max_answer_tokens", "probe_period", "record_training_trace"}:
        raise ValueError("Unknown MATH-500 config fields")
    if not isinstance(config.get("model_path"), str) or not config["model_path"]:
        raise ValueError("Pass --model or set model_path in --config")
    config.setdefault("backend", "minisgl")
    if config["backend"] not in ("minisgl", "sglang"):
        raise ValueError("backend must be minisgl or sglang")
    sglang = config["backend"] == "sglang"
    config.setdefault("mode", "plain" if sglang else "async_thoughts")
    if config["mode"] not in ("plain", "async_thoughts"):
        raise ValueError("mode must be plain or async_thoughts")
    asynchronous = config["mode"] == "async_thoughts"
    if sglang and asynchronous:
        raise ValueError("SGLang baseline supports only mode=plain, not async_thoughts")
    config.setdefault("record_training_trace", asynchronous)
    if type(config["record_training_trace"]) is not bool:
        raise ValueError("record_training_trace must be boolean")
    if config["record_training_trace"] and not asynchronous:
        raise ValueError("record_training_trace currently requires async_thoughts mode")
    if asynchronous:
        for key, default in (("max_think_tokens", 8192), ("max_answer_tokens", 8192), ("probe_period", 20)):
            config.setdefault(key, default)
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        total = config["max_think_tokens"] + config["max_answer_tokens"]
        if "max_new_tokens" in config and config["max_new_tokens"] != total:
            raise ValueError("async_thoughts max_new_tokens must equal thinker + writer budgets")
        config["max_new_tokens"] = total
        if config.get("thinking") is False:
            raise ValueError("async_thoughts requires thinking")
    elif {"max_think_tokens", "max_answer_tokens", "probe_period"} & config.keys():
        raise ValueError("Role budgets/probe_period require mode=async_thoughts")
    config.setdefault("thinking", True)
    config.setdefault("max_new_tokens", 16384)
    config.setdefault("seed", 0)
    if type(config["thinking"]) is not bool:
        raise ValueError("thinking must be boolean")
    if type(config["max_new_tokens"]) is not int or config["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be positive")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    config["sampling"] = sampling_config(config.get("sampling", {}))
    if sglang:
        engine = dict(context_length=32768, max_total_tokens=24576, max_running_requests=1,
                      mem_fraction_static=0.75, cuda_graph_max_bs_decode=1, random_seed=config["seed"],
                      log_level="warning", attention_backend="triton")
        engine.update(config.get("engine", {}))
        if {"model_path", "dtype", "tp_size"} & engine.keys():
            raise ValueError("Model/dtype/TP are not engine overrides; runner is BF16, TP=1")
        for key in ("context_length", "max_total_tokens", "max_running_requests"):
            if type(engine[key]) is not int or engine[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        config["engine"] = engine
        return config
    checkpoint_config = Path(config["model_path"]) / "config.json"
    metadata = json.loads(checkpoint_config.read_text()) if checkpoint_config.is_file() else {}
    # BF16 checkpoints need no quantization. The loader validates the actual FP8
    # scheme; this does not quantize an unquantized checkpoint on the fly.
    quantization = "fp8" if metadata.get("quantization_config") else None
    engine = dict(quantization=quantization, page_size=16, num_page_override=1536,
                  max_running_req=4 if asynchronous else 1, max_prefill_rows=4096, max_seq_len_override=32768,
                  cuda_graph_bs=[], cuda_graph_max_bs=0, use_pynccl=False,
                  distributed_addr="tcp://127.0.0.1:0")
    engine.update(config.get("engine", {}))
    if {"model_path", "dtype", "tp_info"} & engine.keys():
        raise ValueError("Model/dtype/TP are not engine overrides; runner is BF16, TP=1")
    for key in ("page_size", "num_page_override", "max_running_req", "max_seq_len_override"):
        if type(engine[key]) is not int or engine[key] < 1:
            raise ValueError(f"{key} must be a positive integer; this evaluator requires explicit capacities")
    cap = engine["max_prefill_rows"]
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError("max_prefill_rows must be a positive integer or null")
    config["engine"] = engine
    return config


def question_seed(seed, case_id):
    # Single active question: partitioning/resume does not shift its RNG stream.
    payload = json.dumps([seed, case_id]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def final_answer_text(text, *, thinking=False):
    # Do not grade a boxed guess from an unfinished thought channel.
    if (thinking or "<|channel>thought" in text) and "<channel|>" not in text:
        return ""
    return text.rsplit("<channel|>", 1)[-1]


def prompt_ids(tokenizer, problem, thinking, *, demo_prefix=False):
    messages = [{"role": "user", "content": INSTRUCTION + "\n\n" + problem}]
    if demo_prefix and thinking:
        # Match GemmaPrompting + the demo's seeded thinker newline, including
        # separate tokenization at block boundaries; no mini runtime import.
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=False, enable_thinking=True)
        close = "<turn|>\n"
        if not rendered.endswith(close):
            raise ValueError("Unsupported Gemma chat template ending")
        parts = (rendered[:-len(close)].rstrip("\n") + "\n",
                 "<turn|>\n<|turn>model\n<|channel>thought", "\n")
        ids = sum((tokenizer.encode(part, add_special_tokens=False) for part in parts), [])
    else:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=thinking)
    if not isinstance(ids, list) or not ids or not all(type(t) is int for t in ids):
        raise ValueError("Chat template must return a nonempty flat list of token IDs")
    return ids


async def solve(llm, problem, config, sampling_params, stop_ids, *, context_factory=None, on_progress=None):
    if context_factory is None:
        from minisgl.shared_cache import AsyncContext
        context_factory = AsyncContext
    started = time.perf_counter()
    ids = prompt_ids(llm.tokenizer, problem, config["thinking"])
    block, stream, generated = await llm.create_block(), None, []
    stop_reason = "length"
    try:
        if len(ids) > 1:
            await drained(llm.forward(ids[:-1], cache_view=[block], return_logits=False))
        context = context_factory(cache_view=[block], next_input_id=ids[-1])
        stream = llm.async_generate(context, sampling_params=sampling_params,
                                    max_steps=config["max_new_tokens"])
        for _ in range(config["max_new_tokens"]):
            token = await drained(anext(stream))
            generated.append(token)
            if token in stop_ids:
                stop_reason = "eos"
                break
            if on_progress and (len(generated) == 1 or len(generated) % 128 == 0):
                on_progress(len(generated))
        raw = llm.tokenizer.decode(generated, skip_special_tokens=False)
        return dict(completion_text=raw, answer_text=final_answer_text(raw, thinking=config["thinking"]),
                    token_ids=generated, prompt_tokens=len(ids), generated_tokens=len(generated),
                    stop_reason=stop_reason, elapsed_s=time.perf_counter() - started)
    finally:
        if stream is not None:
            await stream.aclose()
        await llm.free_block(block)


def latest(path):
    if not Path(path).exists():
        return {}
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


async def solve_sglang(engine, tokenizer, problem, config, stop_ids, seed):
    """One ordinary request through the public vanilla SGLang API."""
    ids = prompt_ids(tokenizer, problem, config["thinking"], demo_prefix=True)
    started = time.perf_counter()
    out = await engine.async_generate(input_ids=ids, return_logprob=True,
        sampling_params=dict(config["sampling"], max_new_tokens=config["max_new_tokens"],
                             sampling_seed=seed, stop_token_ids=sorted(stop_ids)))
    elapsed = time.perf_counter() - started
    tokens, meta = out["output_ids"], out["meta_info"]
    finish = meta["finish_reason"]["type"]
    if finish not in ("stop", "length"):
        raise RuntimeError(f"SGLang request failed: {meta['finish_reason']}")
    raw = tokenizer.decode(tokens, skip_special_tokens=False)
    return dict(completion_text=raw, answer_text=final_answer_text(raw, thinking=config["thinking"]), token_ids=tokens,
                prompt_tokens=meta["prompt_tokens"], generated_tokens=meta["completion_tokens"],
                stop_reason="length" if finish == "length" else "eos", elapsed_s=elapsed,
                sglang_meta=meta)


def summary(rows, expected):
    done = [r for r in rows.values() if r["status"] == "ok"]
    graded = [r for r in done if isinstance(r.get("correct"), bool)]
    correct = sum(r["correct"] for r in graded)
    seconds = sum(r["elapsed_s"] for r in done)
    tokens = sum(r["generated_tokens"] for r in done)
    return dict(expected=expected, completed=len(done), graded=len(graded), correct=correct,
                errors=sum(r["status"] != "ok" for r in rows.values()),
                accuracy=correct / expected if len(graded) == expected else None,
                accuracy_on_graded=correct / len(graded) if graded else None,
                length_limited=sum(r["stop_reason"] == "length" for r in done),
                generated_tokens=tokens, generation_wall_s=seconds,
                generated_tps=tokens / seconds if seconds else None)


def prepare_run(args, config, cases):
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("python", "eval", "scripts/minisgl_examples"):
        for path in sorted((root / folder).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    asynchronous = config["mode"] == "async_thoughts"
    manifest = dict(protocol="math500-async-thoughts-demo-v1" if asynchronous else "math500-single-stream-v1",
                    config=config, grader="math-verify",
                    case_ids=[c["id"] for c in cases],
                    data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                    source_sha256=digest.hexdigest(),
                    seed_policy="sha256(base_seed, case_id)", concurrency=1,
                    instruction=INSTRUCTION)
    if asynchronous:
        from .async_thoughts import UPSTREAM_COMMIT, UPSTREAM_PATH
        manifest["demo_source"] = dict(repo="dvmazur/async_llm", commit=UPSTREAM_COMMIT, path=UPSTREAM_PATH,
            adaptations=["Gemma chat/thought markers", "explicit sampling", "MATH-500 instruction and telemetry",
                         "eager engine config; optional standard RecordedAsyncLLM trace and sampled targets"])
    output = Path(args.output)
    if args.resume:
        if json.loads((output / "manifest.json").read_text()) != manifest:
            raise ValueError("Resume refused: config, cases, data or source changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def run(args):
    import math_verify  # Fail before constructing the engine if grading is unavailable.
    config = read_config(args.config, model_path=args.model, backend=args.backend)
    cases = select_cases(load_data(args.data, args.download_data), start=args.start, limit=args.limit,
                         shard_index=args.shard_index, num_shards=args.num_shards)
    output = prepare_run(args, config, cases)
    rows = latest(output / "results.jsonl")
    remaining = [c for c in cases if rows.get(c["id"], {}).get("status") != "ok"]
    if not remaining:
        print("All selected questions are complete; no model loaded.")
        return
    import torch
    from transformers import AutoTokenizer, GenerationConfig
    sglang = config["backend"] == "sglang"
    if not sglang:
        from minisgl.core import SamplingParams
        from minisgl.llm import AsyncLLM
        root = Path(__file__).resolve().parents[1]
        if not Path(inspect.getfile(AsyncLLM)).resolve().is_relative_to(root / "python/minisgl"):
            raise ValueError("Imported a different engine checkout")
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    if config["mode"] == "async_thoughts":
        from .async_thoughts import live_token_bound, solve as solve_async
        required = max(live_token_bound(tokenizer, INSTRUCTION + "\n\n" + c["problem"], config) for c in remaining)
    else:
        required = max(len(prompt_ids(tokenizer, c["problem"], config["thinking"], demo_prefix=sglang)) for c in remaining)
        required += config["max_new_tokens"]
    engine = config["engine"]
    capacity = (min(engine["context_length"], engine["max_total_tokens"]) if sglang else
                min(engine["max_seq_len_override"], engine["num_page_override"] * engine["page_size"]))
    if required > capacity:
        raise ValueError(f"Prompt + generation needs {required} token slots; increase engine capacities")
    generation = GenerationConfig.from_pretrained(config["model_path"])
    stop_ids = generation.eos_token_id
    stop_ids = set(stop_ids if isinstance(stop_ids, list) else [stop_ids])
    stop_ids.discard(None)
    params = None if sglang else SamplingParams(**config["sampling"])
    print(f"READY_TO_LOAD questions={len(remaining)} max_live_tokens={required} config={config}", flush=True)
    started = time.perf_counter()
    if sglang:
        from sglang import Engine
        llm = Engine(model_path=config["model_path"], dtype="bfloat16", tp_size=1, **engine)
    else:
        llm = AsyncLLM(config["model_path"], dtype=torch.bfloat16, **engine)
    load_s = time.perf_counter() - started

    async def evaluate():
        try:
            with (output / "results.jsonl").open("a") as out:
                for case in remaining:
                    if shutil.disk_usage(output).free < 256 * 1024 * 1024:
                        raise RuntimeError("Less than 256 MiB free; stopping before the next question")
                    seed = question_seed(config["seed"], case["id"])
                    torch.manual_seed(seed)
                    print(f"START id={case['id']} seed={seed}", flush=True)
                    try:
                        solver = solve_async if config["mode"] == "async_thoughts" else solve
                        problem = INSTRUCTION + "\n\n" + case["problem"] if config["mode"] == "async_thoughts" else case["problem"]
                        options = {}
                        if config["record_training_trace"]:
                            case_key = hashlib.sha256(case["id"].encode()).hexdigest()[:16]
                            options["trace_path"] = output / "traces" / f"{case_key}-{time.time_ns()}.json"
                        if sglang:
                            row = await solve_sglang(llm, tokenizer, problem, config, stop_ids, seed)
                        else:
                            row = await solver(llm, problem, config, params, stop_ids, **options,
                                on_progress=lambda n: print(f"PROGRESS id={case['id']} tokens={n}", flush=True))
                        if row.get("training_trace") is not None:
                            # Artifact bundles must remain usable off the GPU host.
                            row["training_trace"]["path"] = str(options["trace_path"].relative_to(output))
                        row.update(id=case["id"], seed=seed, status="ok", answer=case["answer"])
                    except Exception as exc:
                        row = dict(id=case["id"], status="error", error=f"{type(exc).__name__}: {exc}")
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        rows[case["id"]] = row
                        raise
                    try:
                        row.update(grade_one(dict(writer_text=row["answer_text"], answer=case["answer"])))
                    except Exception as exc:
                        row.update(correct=None, grading_error=f"{type(exc).__name__}: {exc}")
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    rows[case["id"]] = row
                    stats = summary(rows, len(cases))
                    (output / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
                    print(f"DONE id={case['id']} correct={row['correct']} tokens={row['generated_tokens']} "
                          f"seconds={row['elapsed_s']:.2f} completed={stats['completed']}/{len(cases)}", flush=True)
        finally:
            if sglang:
                llm.shutdown()
            else:
                await llm.close()

    try:
        asyncio.run(evaluate())
    finally:
        (output / "summary.json").write_text(json.dumps(summary(rows, len(cases)), indent=2) + "\n")
        (output / "timing.json").write_text(json.dumps(dict(model_load_s=load_s,
            post_load_s=time.perf_counter() - started - load_s), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Fetch pinned MATH-500 (~0.45 MB)")
    download.add_argument("--output", required=True)
    generate = sub.add_parser("run")
    generate.add_argument("--model", help="Local Gemma checkpoint; overrides config model_path")
    generate.add_argument("--backend", choices=("minisgl", "sglang"),
                          help="minisgl async demo (default), or vanilla SGLang single-stream baseline")
    generate.add_argument("--config", help="Optional JSON overrides for budgets, sampling and engine")
    generate.add_argument("--data", default="datasets/math500.jsonl")
    generate.add_argument("--download-data", action="store_true")
    generate.add_argument("--output", required=True)
    generate.add_argument("--start", type=int, default=0)
    generate.add_argument("--limit", type=int)
    generate.add_argument("--shard-index", type=int)
    generate.add_argument("--num-shards", type=int)
    generate.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        print(f"Dataset ready: {len(load_data(args.output, download=True))} questions at {args.output}")
    else:
        if args.model is None and args.config is None:
            parser.error("run requires --model or --config")
        run(args)


if __name__ == "__main__":
    main()
