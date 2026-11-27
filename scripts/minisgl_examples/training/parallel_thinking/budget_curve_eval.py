"""Greedy answer readouts from flattened, truncated saved reasoning token views."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import gzip
import hashlib
import json
import os
import random
import time
from pathlib import Path

from minisgl_examples.training.parallel_thinking import math500, report, trace_io
from minisgl_examples.training.parallel_thinking.budget_views import reconstruct_cutoff
from minisgl_examples.training.parallel_thinking.rollout import extract_boxed_answer

PROTOCOL = "flattened_first_worker_budget_readout_v1"
CUE = "\n\nGiven the current thoughts, the answer is \\boxed{"
CONDITION_NAMES = {f"{mode}-w{workers}": (mode, workers) for mode, workers in report.CONDITIONS}
PINNED_FIELDS = ("model", "model_revision", "dataset", "dataset_revision", "dataset_sha256", "seed")
SOURCE_STOP_MARKERS = ("quality-hold.json", "multiworker-stopped.json", "superseded.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_sha256(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def assert_source_active(source_root: Path) -> None:
    for name in SOURCE_STOP_MARKERS:
        if (source_root / name).exists():
            raise RuntimeError(f"Source run has stop marker: {source_root / name}")


def read_catalog(source_root: Path) -> dict:
    """Read only newline-committed source rows, rejecting duplicate case identities."""
    cases = {}
    for path in sorted(source_root.glob("shards/*/results.jsonl")):
        config = json.loads((path.parent / "config.json").read_text())
        if any(config.get(key) != report.COMMON[key] for key in PINNED_FIELDS):
            raise ValueError(f"Unpinned source model, dataset or seed in {path.parent}")
        for _, row in report.read_results(path, partial=True):
            key = (row["mode"], row["workers"], row["index"])
            if (key[:2] not in report.CONDITIONS or type(key[2]) is not int or not 0 <= key[2] < 500
                    or row["mode"] != config.get("mode") or row["workers"] not in config.get("workers", ())
                    or row["index"] not in config.get("indices", ())):
                raise ValueError(f"Unexpected source case {key} in {path}")
            if key in cases:
                raise ValueError(f"Duplicate source case {key}")
            cases[key] = {"directory": path.parent, "row": row, "config": config}
    return cases


def load_source_trace(case: dict) -> dict:
    """Check immutable compressed bytes before parsing the saved trace."""
    row, config = case["row"], case["config"]
    directory = Path(case["directory"])
    path = directory / row["trace_path"]
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Source trace path escapes its shard")
    if (path.stat().st_size != row["trace_bytes"] or file_sha256(path) != row["trace_sha256"]):
        raise ValueError(f"Source trace SHA-256 or size mismatch: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("format") != trace_io.TRACE_FORMAT:
        raise ValueError("Unsupported source trace format")
    if data.get("case") != {key: row[key] for key in trace_io.IDENTITY}:
        raise ValueError("Source trace case identity mismatch")
    for key in ("model", "model_revision", "dataset", "dataset_revision", "seed"):
        if data.get(key) != config[key]:
            raise ValueError(f"Source trace {key} mismatch")
    if any(row.get(key) != value for key, value in data["metrics"].items()):
        raise ValueError("Source trace metrics mismatch")
    return data


def build_readout_input(rollout: dict, budget: int, tokenizer) -> dict:
    view = reconstruct_cutoff(rollout, budget, include_pending=True)
    ids = view["view_token_ids"]
    opening, closing = tokenizer.vocab["<think>"], tokenizer.vocab["</think>"]
    last_tag = next((token for token in reversed(ids) if token in (opening, closing)), None)
    cue = ("</think>" if last_tag == opening else "") + CUE
    cue_ids = tokenizer.encode(cue, add_special_tokens=False)
    inputs = ids + cue_ids
    return {
        "view_token_ids": ids, "input_ids": inputs, "cue_ids": cue_ids,
        "effective_rounds": view["effective_rounds"], "publication_count": view["publication_count"],
        "reasoning_tokens": len(ids) - len(rollout["prompt_ids"]),
        "prompt_tokens": len(rollout["prompt_ids"]), "context_sha256": token_sha256(inputs),
        "view_sha256": token_sha256(ids),
    }


async def greedy_readout(llm, input_ids: list[int], answer_tokens: int, eos_ids, cue_ids: list[int]) -> dict:
    import torch

    if answer_tokens < 1 or answer_tokens > 8:
        raise ValueError("The answer token cap must be in 1..8")
    block = await llm.create_block()
    sampled, prediction, stop = [], None, "max_tokens"
    try:
        output = await llm.forward(input_ids, write_to=block, return_logits=True)
        for step in range(answer_tokens):
            if not torch.isfinite(output.logits).all().item():
                raise RuntimeError(f"Non-finite readout logits at answer token {step + 1}")
            token = int(output.logits.argmax().item())
            sampled.append(token)
            if token in eos_ids:
                stop = "eos"
                break
            answer_text = llm.tokenizer.decode(cue_ids) + llm.tokenizer.decode(sampled)
            prediction = extract_boxed_answer(answer_text)
            if prediction is not None:
                stop = "answer"
                break
            if step + 1 < answer_tokens:
                output = await llm.forward([token], cache_view=[block], return_logits=True)
    finally:
        await llm.free_block(block)
    return {
        "prediction": prediction, "answer_token_ids": sampled,
        "answer_text": llm.tokenizer.decode(sampled, skip_special_tokens=False), "stop_reason": stop,
    }


def read_existing(output: Path, config: dict) -> dict:
    path = output / "config.json"
    results = output / "results.jsonl"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Existing budget readout configuration does not match this invocation")
    if results.exists() and not path.exists():
        raise ValueError("Existing readout rows have no configuration")
    rows = {}
    if results.exists():
        for _, row in report.read_results(results, partial=True):
            key = (row["mode"], row["workers"], row["index"], row["budget"])
            if key in rows:
                raise ValueError(f"Duplicate committed readout {key}")
            rows[key] = row
        raw = results.read_bytes()
        boundary = raw.rfind(b"\n") + 1
        if boundary != len(raw):
            tail = raw[boundary:]
            recovery = output / "recovery"
            recovery.mkdir(exist_ok=True)
            archive = recovery / f"incomplete-results-{hashlib.sha256(tail).hexdigest()}.bin"
            if archive.exists():
                if archive.read_bytes() != tail:
                    raise ValueError("Incomplete-tail recovery archive was modified")
            else:
                with archive.open("xb") as stream:
                    stream.write(tail)
                    stream.flush()
                    os.fsync(stream.fileno())
            with results.open("r+b") as stream:
                stream.truncate(boundary)
                stream.flush()
                os.fsync(stream.fileno())
    return rows


async def evaluate_case(llm, case, trace, budgets, answer_tokens, eos_ids, existing_rows, *,
                        on_result=None, check_source=None):
    source = case["row"]
    identity = (source["mode"], source["workers"], source["index"])
    cached = {}
    for key, row in existing_rows.items():
        if key[:3] == identity:
            if row["source_trace_sha256"] != source["trace_sha256"]:
                raise ValueError(f"Source trace changed for resumed case {identity}")
            cached.setdefault(row["context_sha256"], row)
    records = []
    for budget in budgets:
        key = (*identity, budget)
        if key in existing_rows:
            continue
        if check_source is not None:
            check_source()
        started = time.perf_counter()
        view = build_readout_input(trace["rollout"], budget, llm.tokenizer)
        cap = getattr(getattr(llm, "engine", None), "max_seq_len", None)
        if cap is not None and len(view["input_ids"]) + answer_tokens > cap:
            raise ValueError(f"Readout context exceeds model capacity for {identity}, budget {budget}")
        previous = cached.get(view["context_sha256"])
        if previous is None:
            answer = await greedy_readout(llm, view["input_ids"], answer_tokens, eos_ids, view["cue_ids"])
        else:
            answer = {name: previous[name] for name in
                      ("prediction", "answer_token_ids", "answer_text", "stop_reason")}
        grade = math500.grade_math500(source["gold"], answer["prediction"])
        count = len(answer["answer_token_ids"])
        row = {
            "mode": identity[0], "workers": identity[1], "index": identity[2], "budget": budget,
            "unique_id": source["unique_id"], "gold": source["gold"], **answer,
            "correct": grade == "correct", "grading_status": grade,
            "generated_tokens": view["reasoning_tokens"] + count, "answer_tokens": count,
            "reasoning_tokens": view["reasoning_tokens"], "effective_rounds": view["effective_rounds"],
            "prompt_tokens": view["prompt_tokens"], "context_tokens": len(view["input_ids"]),
            "context_sha256": view["context_sha256"], "view_sha256": view["view_sha256"],
            "cue_token_ids": view["cue_ids"], "publication_count": view["publication_count"],
            "source_trace_sha256": source["trace_sha256"], "source_trace_path": str(
                Path(case["directory"]) / source["trace_path"]),
            "wall_seconds": time.perf_counter() - started, "reused_readout": previous is not None,
            "reused_from_budget": previous["budget"] if previous is not None else None,
        }
        if on_result is not None:
            on_result(row)
        records.append(row)
        cached.setdefault(view["context_sha256"], row)
    return records


def configuration(args, context: int) -> dict:
    files = (Path(__file__), Path(__file__).with_name("budget_views.py"),
             Path(math500.__file__), Path(__file__).with_name("rollout.py"))
    return {
        "protocol": PROTOCOL, "source_root": str(args.source_root.resolve()),
        **{key: report.COMMON[key] for key in PINNED_FIELDS if key != "seed"},
        "seed": args.seed, "budgets": args.budgets, "answer_tokens": args.answer_tokens,
        "conditions": args.conditions, "indices": args.indices, "index_modulo": args.index_modulo,
        "index_remainder": args.index_remainder, "grading": math500.GRADING_POLICY,
        "readout_context": "flat_token_id_prefill_first_worker_last_including_pending_without_eos",
        "budget_unit": "sampled_reasoning_rounds", "always_force_readout": True,
        "answer_sampling": {"temperature": 0.0}, "answer_cue": CUE,
        "close_thinking": "append_close_tag_if_last_thinking_tag_is_open",
        "context_cap": context, "memory_ratio": args.memory_ratio,
        "max_prefill_rows": args.max_prefill_rows, "max_running_req": 4,
        "cuda_graph_bs": [1], "shared_cuda_graph_max_depth": 1,
        "source_sha256": {path.name: file_sha256(path) for path in files},
        "metric_definitions": {
            "generated_tokens": "Visible saved reasoning token count plus new sampled answer tokens",
            "answer_tokens": "New sampled answer tokens, including sampled EOS",
            "wall_seconds": "Current row reconstruction, readout or deduplication, and grading time",
        },
    }


async def evaluate(args) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from minisgl.llm import AsyncLLM

    from minisgl_examples.training.parallel_thinking.eval import gpu_preflight

    assert_source_active(args.source_root)
    model_path = Path(snapshot_download(report.COMMON["model"], revision=report.COMMON["model_revision"],
                                        local_files_only=True))
    model = json.loads((model_path / "config.json").read_text())
    native_cap = model.get("text_config", model)["max_position_embeddings"]
    maximum_workers = max(CONDITION_NAMES[name][1] for name in args.conditions)
    context = min(native_cap, args.max_context or max(args.budgets) * maximum_workers + 8192)
    config = configuration(args, context)
    existing = read_existing(args.output, config)
    dataset_path, dataset = math500.load_math500()
    if file_sha256(dataset_path) != report.COMMON["dataset_sha256"]:
        raise ValueError("Pinned MATH-500 dataset digest mismatch")
    chosen_indices = args.indices if args.indices is not None else range(500)
    chosen_indices = {i for i in chosen_indices if i % args.index_modulo == args.index_remainder}
    wanted = {(*CONDITION_NAMES[name], i) for name in args.conditions for i in chosen_indices}
    if any(key[:3] not in wanted or key[3] not in args.budgets for key in existing):
        raise ValueError("Existing readout rows fall outside this lane configuration")
    if not (args.output / "config.json").exists():
        temp = args.output / "config.json.tmp"
        with temp.open("w") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(args.output / "config.json")
    llm = None

    def commit(row):
        with (args.output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        existing[(row["mode"], row["workers"], row["index"], row["budget"])] = row
        print(f"DONE {row['mode']}-w{row['workers']} i{row['index']} budget={row['budget']} "
              f"correct={row['correct']} answer_tokens={row['answer_tokens']} reused={row['reused_readout']}", flush=True)

    try:
        while True:
            assert_source_active(args.source_root)
            catalog = read_catalog(args.source_root)
            for key in sorted(wanted & catalog.keys(), key=lambda k: (k[2], k[0], k[1])):
                case = catalog[key]
                for budget in args.budgets:
                    prior = existing.get((*key, budget))
                    if prior is not None and prior["source_trace_sha256"] != case["row"]["trace_sha256"]:
                        raise ValueError(f"Source trace changed for resumed case {key}")
                if all((*key, budget) in existing for budget in args.budgets):
                    continue
                gold = dataset[key[2]]
                if (case["row"]["unique_id"], case["row"]["gold"]) != (gold["unique_id"], gold["answer"]):
                    raise ValueError(f"Source case differs from pinned MATH-500 problem {key}")
                trace = load_source_trace(case)
                if llm is None:
                    gpu = gpu_preflight()
                    print(f"GPU_PREFLIGHT {json.dumps(gpu)}", flush=True)
                    random.seed(args.seed)
                    torch.manual_seed(args.seed)
                    torch.cuda.manual_seed_all(args.seed)
                    llm = AsyncLLM(
                        str(model_path), dtype=torch.bfloat16, max_running_req=4,
                        memory_ratio=args.memory_ratio, max_seq_len_override=context,
                        cuda_graph_bs=[1], cuda_graph_max_bs=1, shared_cuda_graph_max_depth=1,
                        shared_cuda_graph_prefill_rows=sorted({32, args.max_prefill_rows}),
                        max_prefill_rows=args.max_prefill_rows, distributed_addr="tcp://127.0.0.1:0",
                    )
                print(f"START {key[0]}-w{key[1]} i{key[2]}", flush=True)
                await evaluate_case(llm, case, trace, args.budgets, args.answer_tokens,
                                    report.COMMON["eos_token_ids"], existing, on_result=commit,
                                    check_source=lambda: assert_source_active(args.source_root))
                del trace
            if all((*key, budget) in existing for key in wanted for budget in args.budgets):
                break
            if not args.wait_for_source:
                break
            print(f"WAIT_FOR_SOURCE completed={len(existing)} target={len(wanted) * len(args.budgets)}", flush=True)
            await asyncio.sleep(30)
    finally:
        if llm is not None:
            await llm.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--conditions", choices=CONDITION_NAMES, nargs="+", default=list(CONDITION_NAMES))
    parser.add_argument("--index-modulo", type=int, default=1)
    parser.add_argument("--index-remainder", type=int, default=0)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(range(1000, 8001, 1000)))
    parser.add_argument("--answer-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-context", type=int, default=0)
    parser.add_argument("--memory-ratio", type=float, default=0.75)
    parser.add_argument("--max-prefill-rows", type=int, default=256)
    parser.add_argument("--wait-for-source", action="store_true")
    args = parser.parse_args(argv)
    if args.index_modulo < 1 or not 0 <= args.index_remainder < args.index_modulo:
        parser.error("Require positive index-modulo and 0 <= index-remainder < index-modulo")
    if not args.budgets or min(args.budgets) < 1 or len(set(args.budgets)) != len(args.budgets):
        parser.error("Use unique positive budgets")
    args.budgets.sort()
    if args.indices is not None and (len(set(args.indices)) != len(args.indices)
                                     or any(not 0 <= i < 500 for i in args.indices)):
        parser.error("Use unique indices in 0..499")
    if (not 1 <= args.answer_tokens <= 8 or args.max_context < 0 or args.max_prefill_rows < 1
            or not 0 < args.memory_ratio < 1 or len(set(args.conditions)) != len(args.conditions)):
        parser.error("Invalid answer cap, context, prefill size, memory ratio or repeated condition")
    if not args.source_root.is_dir():
        parser.error("Source run directory does not exist")
    if args.output.resolve().is_relative_to(args.source_root.resolve()):
        parser.error("Readout output must be outside the immutable source run")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lane.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another readout worker holds this output lane")
        asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
