"""Evaluate shared-cache Hogwild inference on MATH-500 (default) or LIMO.

The budget counts synchronized decode rounds, not aggregate worker tokens.
Paragraph publication uses additional forwards in this implementation. The
greedy finisher is outside that budget and runs only if no boxed answer was
found. A native-context stop is reported rather than extending/truncating RoPE.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from minisgl.core import SamplingParams
from minisgl.llm import AsyncLLM
from minisgl.training import RecordedAsyncLLM

from minisgl_examples.training.parallel_thinking import math500, trace_io
from minisgl_examples.training.parallel_thinking.native import run_native
from minisgl_examples.training.parallel_thinking.rollout import MathFormatting, run_agents
from minisgl_examples.training.parallel_thinking.single_worker import run_single_worker

MODEL = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "8c0b3d30f3ae96f5c7f774de704bf0ada83e422a"
LIMO_DATASET = "GAIR/LIMO"
LIMO_REVISION = "b60f4462da9d927930b9c9bd43399cf875564416"
WORKER_COUNTS = (1, 2, 4, 6, 8, 16)
NAMES = ("Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
         "Ivan", "Judy", "Karl", "Laura", "Mallory", "Niaj", "Olivia", "Peggy")


def numeric_answer(value: object) -> Decimal | None:
    """Strict LIMO numeric grading; never turn arbitrary text into digits."""
    text = str(value).strip().strip("$").strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return None
    try:
        answer = Decimal(text)
        return answer if answer.is_finite() else None
    except InvalidOperation:
        return None


def select_rows(rows: list[dict], indices: list[int] | None, start: int, limit: int, dataset: str = "math500"):
    selected = indices if indices is not None else list(range(start, min(start + limit, len(rows))))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select at least one unique problem index")
    if any(index < 0 or index >= len(rows) for index in selected):
        raise ValueError(f"Problem indices must be in 0..{len(rows) - 1}")
    result = []
    for index in selected:
        row = rows[index]
        if dataset == "math500":
            if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("problem", "answer", "unique_id")):
                raise ValueError(f"Invalid MATH-500 row {index}")
        elif not isinstance(row.get("question"), str) or numeric_answer(row.get("answer")) is None:
            raise ValueError(f"Invalid question/numeric answer in LIMO row {index}")
        result.append((index, row))
    return result


def sampling_defaults(config: dict) -> SamplingParams:
    for name, neutral in (("min_p", 0), ("presence_penalty", 0), ("repetition_penalty", 1)):
        value = config.get(name, neutral)
        if value is not None and value != neutral:
            raise ValueError(f"Unsupported non-neutral sampling parameter: {name}={value}")
    return SamplingParams(
        temperature=config.get("temperature", 1.0) if config.get("do_sample", True) else 0.0,
        top_p=config.get("top_p", 1.0), top_k=config.get("top_k", -1), max_tokens=1,
    )


def gpu_preflight() -> dict:
    """Read fresh physical GPU state after downloads, immediately before load."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-") or "," in visible:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU UUID")
    subprocess.run(["nvidia-smi"], check=True)
    query = subprocess.check_output([
        "nvidia-smi", "-i", visible,
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(", ")
    processes = subprocess.check_output([
        "nvidia-smi", "-i", visible, "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True).strip()
    selected = os.environ.get("HOGWILD_GPU", "7")
    if not selected.isdecimal() or int(query[0]) != int(selected) or query[1] != visible:
        raise RuntimeError(f"Expected physical GPU {selected} with UUID {visible}, got {query[:2]}")
    # Some idle devices retain a few MiB of driver memory without a compute process.
    if processes or not 0 <= int(query[3]) <= 8 or int(query[5]) != 0:
        raise RuntimeError(f"GPU {selected} is occupied: memory={query[3]} MiB, processes={processes!r}")
    return dict(zip(("index", "uuid", "name", "memory_used_mib", "memory_total_mib", "utilization"), query))


def summarize(records: list[dict]) -> dict:
    result = {}
    for count in sorted({record["workers"] for record in records}):
        group = [record for record in records if record["workers"] == count]
        result[str(count)] = {
            "problems": len(group), "correct": sum(record["correct"] for record in group),
            "accuracy": sum(record["correct"] for record in group) / len(group),
            "context_limited": sum(record["stop_reason"] == "context_limit" for record in group),
            "generated_tokens": sum(record["generated_tokens"] for record in group),
            "decoding_steps": sum(record["decoding_steps"] for record in group),
            "total_decoding_steps": sum(record["total_decoding_steps"] for record in group),
            "used_finisher": sum(record["used_finisher"] for record in group),
            "finisher_tokens": sum(record["finisher_tokens"] for record in group),
            "total_seconds": sum(record["wall_seconds"] for record in group),
        }
    return result


async def evaluate(args: argparse.Namespace) -> None:
    native = args.mode == "native"
    solo_layout = args.single_worker_layout if not native and 1 in args.workers else None
    solo_only = args.workers == [1] and solo_layout in {"causal", "scaffold"}
    multiple = not native and max(args.workers) > 1
    native_workers = multiple and args.multi_worker_layout == "native"
    model_path = Path(snapshot_download(MODEL, revision=MODEL_REVISION))
    symbolic = args.dataset == "math500"
    if symbolic:
        dataset, revision = math500.DATASET, math500.DATASET_REVISION
        dataset_path, rows = math500.load_math500()
    else:
        dataset, revision = LIMO_DATASET, LIMO_REVISION
        dataset_path = Path(hf_hub_download(dataset, "limo.jsonl", repo_type="dataset", revision=revision))
        rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    selected = select_rows(rows, args.indices, args.start, args.limit, args.dataset)
    answer_steps = 0 if native else args.finisher_tokens or (128 if symbolic else 16)
    config = json.loads((model_path / "config.json").read_text())
    generation = json.loads((model_path / "generation_config.json").read_text())
    sampling = sampling_defaults(generation)
    eos_ids = generation["eos_token_id"]
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
    native_context = config.get("text_config", config)["max_position_embeddings"]
    requested_context = args.max_context or args.budget * max(args.workers) + 8192
    context = min(native_context, requested_context)
    output = args.output or Path("training_runs") / datetime.now(timezone.utc).strftime("hogwild-eval-%Y%m%dT%H%M%S-%fZ")
    repo = Path(__file__).resolve().parents[4]
    sources = [Path(__file__), Path(__file__).with_name("rollout.py"), Path(__file__).with_name("native.py"),
               Path(__file__).with_name("single_worker.py"), Path(math500.__file__), repo / "run_hogwild.sh"]
    if args.save_traces:
        sources.append(Path(trace_io.__file__))
    manifest = {
        "model": MODEL, "model_revision": MODEL_REVISION, "dataset": dataset,
        "dataset_revision": revision, "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "indices": [index for index, _ in selected], "workers": args.workers, "seed": args.seed,
        "mode": args.mode, "budget": args.budget,
        "budget_unit": "generated_tokens" if native else "synchronized_decode_rounds_per_worker",
        "paragraph_publication": not native and args.paragraph_publication,
        "publication_forwards_are_additional": not native, "finisher_tokens_outside_budget": answer_steps,
        "single_worker_layout": solo_layout,
        "multi_worker_layout": args.multi_worker_layout if multiple else None,
        "multi_worker_answer_cue": args.multi_worker_answer_cue if multiple else None,
        "answer_selection_policy": "eos_worker_last_box_v1" if native_workers else None,
        "publication_policy": "encoded_separator_v1" if native_workers else None,
        "pending_eos_in_finisher": False,
        "header_context": ("none" if native or native_workers or (solo_only and solo_layout == "causal") else
                           "causal" if args.causal_headers or solo_only else "independent"),
        "finisher_mode": "none" if native else "cached",
        "reminder_interval": 0 if native or native_workers or (solo_only and solo_layout == "causal") else 1024,
        "enable_thinking": True, "reasoning_effort": "xhigh",
        "sampling": asdict(sampling), "eos_token_ids": eos_ids,
        "native_context": native_context, "context_cap": context, "cuda_graphs": args.cuda_graphs,
        "grading": math500.GRADING_POLICY if symbolic else "strict numeric equality of final boxed answer",
        "metric_definitions": {
            "generated_tokens": "All sampled reasoning and finisher tokens, excluding prompt/inserted prefixes",
            "decoding_steps": "Synchronized reasoning rounds; publication/prefill forwards are additional",
            "total_decoding_steps": "Reasoning rounds plus sampled finisher tokens",
        },
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources if path.exists()},
    }
    manifest["gpu"] = gpu_preflight()
    if args.save_traces:
        manifest["trace_format"] = trace_io.TRACE_FORMAT
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"OUTPUT={output.resolve()}", flush=True)
    maximum = max(args.workers)
    # Even a one-worker rollout prefills the two shared headers concurrently.
    graph_sizes = ([1] if native else sorted({1, 2, *args.workers})) if args.cuda_graphs else []
    started = time.perf_counter()
    llm = AsyncLLM(
        str(model_path), dtype=torch.bfloat16, max_running_req=max(4, maximum),
        memory_ratio=args.memory_ratio, max_seq_len_override=context,
        cuda_graph_bs=graph_sizes, cuda_graph_max_bs=max(graph_sizes, default=0),
        shared_cuda_graph_max_depth=1 if native else maximum + 4,
        shared_cuda_graph_prefill_rows=[32, 256] if args.cuda_graphs else [],
        max_prefill_rows=256, distributed_addr="tcp://127.0.0.1:0",
    )
    print(f"Model initialized in {time.perf_counter() - started:.1f}s", flush=True)
    records = []
    try:
        for count in args.workers:
            formatter = None if native else MathFormatting(
                llm.tokenizer, NAMES[:count], enable_thinking=True, reasoning_effort="xhigh",
            )
            for index, row in selected:
                random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
                print(f"START workers={count} index={index} budget={args.budget}", flush=True)
                begin = time.perf_counter()
                problem = "Please reason step by step, and put your final answer within \\boxed{}.\n\n" + row[
                    "problem" if symbolic else "question"
                ]
                scope = RecordedAsyncLLM(llm) if args.save_traces else nullcontext(llm)
                with scope as case_llm:
                    if native:
                        result = await run_native(
                            case_llm, problem=problem, max_steps=args.budget, sampling=sampling,
                            eos_token_ids=eos_ids, max_context_tokens=context, capture_trace=args.save_traces,
                        )
                    elif count == 1 and solo_layout != "legacy":
                        result = await run_single_worker(
                            case_llm, formatter, problem=problem, max_steps=args.budget, answer_steps=answer_steps,
                            sampling=sampling, eos_token_ids=eos_ids, max_context_tokens=context,
                            publish_paragraphs=args.paragraph_publication, scaffold=solo_layout == "scaffold",
                            capture_trace=args.save_traces,
                        )
                    else:
                        result = await run_agents(
                            case_llm, formatter, problem=problem, synchronized=True, max_steps=args.budget,
                            answer_steps=answer_steps, bob_delay=0, delay_every=32, sampling=sampling, samples=None,
                            max_context_tokens=context, reminder_interval=1024, finish_if_missing=True, eos_token_ids=eos_ids,
                            retain_details=False, symbolic_answers=symbolic, cached_finisher=True,
                            causal_headers=args.causal_headers, publish_paragraphs=args.paragraph_publication,
                            native_workers=count > 1 and native_workers, capture_trace=args.save_traces,
                            native_answer_cue=count > 1 and args.multi_worker_answer_cue == "qwen",
                        )
                predicted = result.get("boxed_answer")
                if symbolic:
                    grading = math500.grade_math500(row["answer"], predicted)
                    if grading == "unparsed_gold":
                        raise ValueError(f"Grader could not parse published MATH-500 answer for {row['unique_id']}")
                else:
                    value = numeric_answer(predicted)
                    grading = ("unparsed_prediction" if value is None else
                               "correct" if value == numeric_answer(row["answer"]) else "incorrect")
                record = {
                    "mode": args.mode, "workers": count, "index": index, "gold": str(row["answer"]),
                    "single_worker_layout": solo_layout if count == 1 else None,
                    "multi_worker_layout": args.multi_worker_layout if count > 1 else None,
                    "multi_worker_answer_cue": args.multi_worker_answer_cue if count > 1 else None,
                    **{key: row[key] for key in ("unique_id", "subject", "level") if key in row},
                    "prediction": predicted, "grading_status": grading, "correct": grading == "correct",
                    "wall_seconds": time.perf_counter() - begin,
                    "decoding_steps": result["decode_rounds"],
                    "total_decoding_steps": result["decode_rounds"] + result["answer_generated_tokens"],
                    "generated_tokens": result["total_generated_tokens"],
                    "reasoning_tokens": result["reasoning_generated_tokens"],
                    "finisher_tokens": result["answer_generated_tokens"],
                    "per_worker_tokens": result["generated_tokens"],
                    "stop_reason": result["stop_reason"],
                    "peak_context_tokens": result["peak_context_tokens"],
                    "used_finisher": result["used_finisher"],
                    "finisher_mode": result["finisher_mode"],
                    "published_paragraphs": result["completed_steps"],
                    **{key: result[key] for key in (
                        "first_publication_round", "first_boundary_round", "first_boundary_prefix_sha256",
                        "prompt_token_ids_sha256", "eos_worker", "eos_token_id", "finisher_stop_reason",
                        "finisher_eos_token_id",
                    ) if key in result},
                }
                if args.save_traces:
                    record.setdefault("unique_id", f"{args.dataset}/{index}")
                    trace_io.write_trace(output, record, manifest, problem, llm.tokenizer,
                                         result.pop("trace"), case_llm.get_trace())
                # Release the case recorder before starting the next problem.
                del scope, case_llm
                with (output / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                records.append(record)
                summary = summarize(records)
                (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(f"DONE workers={count} index={index} correct={record['correct']} "
                      f"answer={record['prediction']} stop={result['stop_reason']} "
                      f"rounds={result['decode_rounds']} seconds={record['wall_seconds']:.1f}", flush=True)
    finally:
        await llm.close()
    (output / "completed.json").write_text(json.dumps({"cases": len(records), "summary": summarize(records)}, indent=2) + "\n")
    print(json.dumps(summarize(records), indent=2), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["math500", "limo"], default="math500")
    parser.add_argument("--mode", choices=["hogwild", "native"], default="hogwild")
    parser.add_argument("--workers", nargs="+", type=int, choices=WORKER_COUNTS)
    parser.add_argument("--budget", type=int, default=16384, help="Decode rounds per worker; finisher excluded")
    parser.add_argument("--limit", type=int, default=1, help="Default: one fixed problem. Use 500 for all of MATH-500.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--indices", nargs="+", type=int, help="Explicit zero-based dataset rows, instead of start/limit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="New directory; existing results are never overwritten")
    parser.add_argument("--save-traces", action="store_true",
                        help="Save one compressed token and engine-cache trace per scored case")
    parser.add_argument("--max-context", type=int, default=0, help="Optional cap, always bounded by native context")
    parser.add_argument("--memory-ratio", type=float, default=0.65)
    parser.add_argument("--finisher-tokens", type=int, help="Extra greedy answer steps (MATH-500: 128; LIMO: 16)")
    parser.add_argument("--causal-headers", action="store_true", help="Diagnostic: condition shared headers on the prompt")
    parser.add_argument("--single-worker-layout", choices=["causal", "scaffold", "legacy"], default="causal",
                        help="One worker: causal publication by default; scaffold retains step labels; legacy reorders KV")
    parser.add_argument("--multi-worker-layout", choices=["collaboration", "native"], default="native",
                        help="Multiple workers: collaboration scaffold or native Qwen prompt with shared peer KV")
    parser.add_argument("--multi-worker-answer-cue", choices=["paper", "qwen"], default="qwen",
                        help="Cached finisher cue for multiple workers; qwen closes an open thinking span")
    parser.add_argument("--paragraph-publication", action=argparse.BooleanOptionalAction, default=True,
                        help="Publish completed paragraphs to shared history and start a new worker step (default: on)")
    parser.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.workers is None:
        args.workers = [1] if args.mode == "native" else list(WORKER_COUNTS)
    if args.mode == "native" and (args.workers != [1] or args.causal_headers or args.finisher_tokens is not None):
        parser.error("Ordinary Qwen uses one worker, a native chat prompt and no finisher")
    if max(args.workers) > 1 and args.causal_headers and args.multi_worker_layout == "native":
        parser.error("The native multi-worker layout has no role headers; use --multi-worker-layout collaboration")
    if args.budget < 1 or args.limit < 1 or args.start < 0 or args.max_context < 0:
        parser.error("Budget/limit must be positive; start/max-context must be nonnegative")
    if not 0 < args.memory_ratio < 1 or len(set(args.workers)) != len(args.workers):
        parser.error("Use a memory ratio in (0,1) and unique worker counts")
    if args.finisher_tokens is not None and args.finisher_tokens < 1:
        parser.error("Finisher token limit must be positive")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
