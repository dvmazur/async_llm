"""Report the seven-condition MATH-500 sweep from its local result files.

Run `python -m minisgl_examples.training.parallel_thinking.report RUN_ROOT`.
Add `--partial` to report completed cases during a sweep. The command writes
summary.json, summary.csv and tasks.csv in RUN_ROOT/report/.
Use `--only-one-worker` for an isolated corrected causal one-worker run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

N_PROBLEMS = 500
COUNTS = (1, 2, 4, 6, 8, 16)
CONDITIONS = (("native", 1), *(("hogwild", n) for n in COUNTS))
COMMON = {
    "model": "Qwen/Qwen3.8-27B", "model_revision": "8c0b3d30f3ae96f5c7f774de704bf0ada83e422a",
    "dataset": "HuggingFaceH4/MATH-500", "dataset_revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "dataset_sha256": "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
    "seed": 42, "budget": 16384, "enable_thinking": True, "reasoning_effort": "xhigh",
    "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "max_tokens": 1, "ignore_eos": False},
    "eos_token_ids": [248046, 248044],
}
PROTOCOL = {
    "native": {"budget_unit": "generated_tokens", "paragraph_publication": False,
               "header_context": "none", "reminder_interval": 0, "finisher_mode": "none",
               "finisher_tokens_outside_budget": 0, "publication_forwards_are_additional": False},
    "hogwild": {"budget_unit": "synchronized_decode_rounds_per_worker", "paragraph_publication": True,
                "header_context": "independent", "reminder_interval": 1024, "finisher_mode": "flat",
                "finisher_tokens_outside_budget": 128, "publication_forwards_are_additional": True},
}
# The original native driver for this comparison was audited against the
# corrected driver. Revisit the pairing if the native call or prompt changes.
REFERENCE_COMMIT = "5b77f2b04e752a4d523b6998dbab9af973c8494b"
REFERENCE_SOURCES = {
    "eval.py": "e76c96c9b6da9c86488f2bb2167a14b3a404df1b4512e131eeb53b6a29f02d85",
    "rollout.py": "9ae054bcae72845e17393cc893491916b708f39183a0aa8621ad61adf1d89417",
    "native.py": "8b408624b4a0d6c625c881768eb499a7c2cdaa9ad2d9db8968eae3895b87f1f6",
    "math500.py": "aded81fcec2a2a9a7f607979f74a51f1ed66fb53a04020eecca10efc4f34e9d4",
    "run_hogwild.sh": "63207febbfb8aa6b947e82368b956dc987fcdf76b2aed8cb929d03d75279cc52",
}
V2_EVAL_SHA256 = "04a74d9e824250e06c5b3f3c270eb5be753e2aa16191b72754eb94b75015371e"
TARGET_EVAL_SHA256 = "122c137abf666f8f313269ae533f9ed8a234d27ecd25a27d129c95177fbfdbab"
# With capture disabled, this pair uses the same native forward, sampler and
# extraction path as the earlier runner. Traced comparisons execute all cases.
OPTIONAL_TRACE_NATIVE = ("eb796ba3056802e865ec557261c6f81ddb45fcd688f4c9dc0ea866124fe3bef1",
                         "01d9392692e920d720128fb0050b7321abe13097b125868cb6e81ea415079b4f")
CATALOG_SHA256 = "2a9de08992d234023bfd3f1028831bef14e8ef5707fe7aaff4c4f07d7fb1d51b"
NATIVE_SNAPSHOT = "reused-native"
TASK_FIELDS = (
    "condition", "workers", "single_worker_layout", "index", "unique_id", "correct", "gold", "prediction", "grading_status",
    "generated_tokens", "reasoning_tokens", "decoding_steps", "total_decoding_steps", "used_finisher",
    "finisher_tokens", "stop_reason", "eos_token_id", "finisher_stop_reason", "finisher_eos_token_id",
)
MULTI_FIELDS = ("multi_worker_layout", "multi_worker_answer_cue")
TRACE_FIELDS = ("trace_path", "trace_sha256", "trace_bytes")
# These exact evaluator builds differ only in the GPU occupancy preflight:
# the latter permits up to 8 MiB of idle driver memory with no compute process.
GPU_PREFLIGHT_SOURCE = "scripts/minisgl_examples/training/parallel_thinking/eval.py"
GPU_PREFLIGHT_ORIGINAL_SHA256 = "eb796ba3056802e865ec557261c6f81ddb45fcd688f4c9dc0ea866124fe3bef1"
GPU_PREFLIGHT_RESIDUAL_SHA256 = "c1af092270325acef6c5060cb9c0031b84c91f6143be6595177152db3e3f7348"


def canonical_sources(sources: dict) -> dict:
    """Compare the two audited occupancy checks while retaining all other hashes."""
    canonical = sources.copy()
    if canonical.get(GPU_PREFLIGHT_SOURCE) == GPU_PREFLIGHT_RESIDUAL_SHA256:
        canonical[GPU_PREFLIGHT_SOURCE] = GPU_PREFLIGHT_ORIGINAL_SHA256
    return canonical


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def worker_layout(config: dict, count: int) -> str | None:
    if config["mode"] == "hogwild" and count == 1:
        return config.get("single_worker_layout", "legacy")
    return None


def manifest_version(config: dict) -> int:
    if "trace_format" in config:
        return 4
    if "multi_worker_layout" in config:
        return 3
    return 2 if "single_worker_layout" in config else 1


def condition_protocol(config: dict, count: int) -> dict:
    protocol = PROTOCOL[config["mode"]].copy()
    if config["mode"] == "hogwild" and "single_worker_layout" in config:
        protocol["finisher_mode"] = "cached"
    layout = worker_layout(config, count)
    if layout == "causal":
        protocol.update(header_context="none", reminder_interval=0)
    elif layout == "scaffold":
        protocol.update(header_context="causal", reminder_interval=1024)
    elif layout == "legacy" and "single_worker_layout" in config and config["header_context"] == "causal":
        protocol["header_context"] = "causal"  # Explicit legacy-header diagnostic.
    if count > 1 and config.get("multi_worker_layout") == "native":
        protocol.update(header_context="none", reminder_interval=0)
    return protocol


def check_config(config: dict) -> None:
    mode = config["mode"]
    require(mode in PROTOCOL, f"Unknown evaluation mode {mode!r}")
    workers = config["workers"]
    require(isinstance(workers, list) and workers and len(set(workers)) == len(workers)
            and all(type(n) is int and (mode, n) in CONDITIONS for n in workers),
            f"Unexpected worker counts {workers!r}")
    explicit = "single_worker_layout" in config
    if explicit:
        layout = config["single_worker_layout"]
        if mode == "hogwild" and 1 in workers:
            require(layout in ("causal", "scaffold", "legacy"), f"Unknown one-worker layout {layout!r}")
        else:
            require(layout is None, "A one-worker layout was recorded for another condition")
    version = manifest_version(config)
    policy = config.get("answer_selection_policy")
    publication = config.get("publication_policy")
    if publication is not None:
        require(publication == "encoded_separator_v1" and mode == "hogwild" and max(workers) > 1
                and config.get("multi_worker_layout") == "native", "Invalid publication policy")
    if policy is not None:
        require(policy == "eos_worker_last_box_v1" and mode == "hogwild" and max(workers) > 1
                and config.get("multi_worker_layout") == "native", "Invalid answer-selection policy")
    if version >= 3:
        require("multi_worker_layout" in config, "Missing multi-worker layout")
        require(explicit and config.get("pending_eos_in_finisher") is False,
                "v3 requires an explicit one-worker layout and pending_eos_in_finisher=false")
        require("multi_worker_answer_cue" in config, "Missing multi-worker answer cue")
        layout, cue = (config[key] for key in MULTI_FIELDS)
        if mode == "hogwild" and max(workers) > 1:
            require(layout in ("collaboration", "native") and cue in ("paper", "qwen"),
                    f"Unknown multi-worker layout or answer cue: {layout!r}, {cue!r}")
        else:
            require(layout is None and cue is None, "A multi-worker layout or answer cue was recorded for another condition")
    else:
        require("pending_eos_in_finisher" not in config and "multi_worker_answer_cue" not in config,
                "Incomplete v3 manifest")
    # In a mixed-width manifest the shared fields describe the multiworker
    # path; the explicit layout describes the one-worker override.
    protocol = condition_protocol(config, next((n for n in workers if n != 1), workers[0]))
    for key, expected in {**COMMON, **protocol}.items():
        require(config[key] == expected, f"Expected {key}={expected!r}, got {config[key]!r}")
    indices = config["indices"]
    require(isinstance(indices, list) and indices and len(set(indices)) == len(indices)
            and all(type(i) is int and 0 <= i < N_PROBLEMS for i in indices), "Invalid shard indices")
    sources = config["source_sha256"]
    filenames = {Path(path).name for path in sources}
    if version == 4:
        from .trace_io import TRACE_FORMAT

        require(config["trace_format"] == TRACE_FORMAT and "trace_io.py" in filenames,
                "The traced benchmark requires saved inference traces and their source hash")
    require(("single_worker.py" in filenames) == explicit, "One-worker layout and source version disagree")
    require({"eval.py", "rollout.py", "native.py", "math500.py", "run_hogwild.sh"} <= filenames
            and all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in sources.values()),
            "Missing or malformed source hashes")


def check_task(row: dict, config: dict) -> dict:
    mode, workers = config["mode"], row["workers"]
    require(row["mode"] == mode and workers in config["workers"], "Condition outside shard manifest")
    layout = worker_layout(config, workers)
    if "single_worker_layout" in config:
        require("single_worker_layout" in row and row["single_worker_layout"] == layout,
                "Result one-worker layout differs from its shard manifest")
    else:
        require("single_worker_layout" not in row, "Explicit one-worker result in a historical manifest")
    multi = {}
    version = manifest_version(config)
    if version >= 3:
        multi = {key: config[key] if workers > 1 else None for key in MULTI_FIELDS}
        for key, value in multi.items():
            require(key in row and row[key] == value, f"Result {key} differs from its shard manifest")
        require(row.get("finisher_mode") == config["finisher_mode"], "Result finisher mode differs from its shard manifest")
    else:
        require(all(key not in row for key in MULTI_FIELDS), "Explicit multi-worker result in an older manifest")
    require(type(row["index"]) is int and row["index"] in config["indices"], "Problem outside shard manifest")
    trace = {}
    if version == 4:
        name = f"traces/{mode}-w{workers}-i{row['index']:03d}.json.gz"
        require(row.get("trace_path") == name and re.fullmatch(r"[0-9a-f]{64}", row.get("trace_sha256") or ""),
                "Missing or invalid trace path or SHA-256 digest")
        require(type(row.get("trace_bytes")) is int and row["trace_bytes"] > 0, "Invalid trace file size")
        trace = {key: row[key] for key in TRACE_FIELDS}
    else:
        require(all(key not in row for key in TRACE_FIELDS), "Unexpected trace fields on an older manifest")
    require(isinstance(row["unique_id"], str) and row["unique_id"].strip(), "Missing MATH-500 ID")
    require(type(row["correct"]) is bool and row["grading_status"] in {"correct", "incorrect", "unparsed_prediction"}
            and row["correct"] == (row["grading_status"] == "correct"), "Inconsistent grade")
    for key in ("generated_tokens", "reasoning_tokens", "finisher_tokens", "decoding_steps", "total_decoding_steps"):
        require(type(row[key]) is int and row[key] >= 0, f"Invalid {key}")
    rounds, extra, used = row["decoding_steps"], row["finisher_tokens"], row["used_finisher"]
    require(rounds <= COMMON["budget"] and row["total_decoding_steps"] == rounds + extra,
            "Inconsistent decoding step counts")
    require(row["reasoning_tokens"] == rounds * workers
            and row["generated_tokens"] == row["reasoning_tokens"] + extra, "Inconsistent sampled token counts")
    require(type(used) is bool and used == (extra > 0) and extra <= config["finisher_tokens_outside_budget"],
            "Inconsistent finisher token count or usage")
    require(isinstance(row["stop_reason"], str) and bool(row["stop_reason"]), "Missing stop reason")
    eos = row["eos_token_id"]
    require(eos is None or eos in COMMON["eos_token_ids"], "Unknown reasoning EOS token")
    require(row["stop_reason"] != "eos" or eos is not None, "Missing reasoning EOS token")
    end = row["finisher_stop_reason"]
    final_eos = row["finisher_eos_token_id"]
    require((end == "eos" and final_eos in COMMON["eos_token_ids"]) or (end != "eos" and final_eos is None),
            "Inconsistent finisher stop reason and EOS token")
    if mode == "native":
        require(not used and end == "not_applicable", "Native baseline unexpectedly used a finisher")
    else:
        valid = {"answer", "eos", "max_steps"} if used else {"existing_answer", "context_limit"}
        require(end in valid, f"Unexpected finisher stop reason {end!r} for used_finisher={used}")
    return {"condition": mode, "single_worker_layout": layout, **multi, **trace,
            **{key: row[key] for key in TASK_FIELDS if key not in ("condition", "single_worker_layout")}}


def read_results(path: Path, partial: bool):
    # A writer appends and flushes one newline-terminated case at a time.
    with path.open("rb") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.endswith(b"\n"):
                if partial:
                    break
                raise ValueError(f"{path}:{number}: unterminated result row")
            try:
                yield number, json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{number}: invalid JSONL") from exc


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_target_sources(hashes: dict) -> None:
    names = {Path(path).name: value for path, value in hashes.items()}
    require(len(names) == len(hashes) and set(REFERENCE_SOURCES) | {"single_worker.py"} <= names.keys()
            and all(re.fullmatch(r"[a-f0-9]{64}", value) for value in names.values()),
            "Expected corrected evaluator sources")
    audited_pairs = ((V2_EVAL_SHA256, REFERENCE_SOURCES["native.py"]),
                     (TARGET_EVAL_SHA256, REFERENCE_SOURCES["native.py"]), OPTIONAL_TRACE_NATIVE)
    require((names["eval.py"], names["native.py"]) in audited_pairs
            and names["math500.py"] == REFERENCE_SOURCES["math500.py"] and "trace_io.py" not in names,
            "Unaudited native driver, runner, or grader")


def load_native_catalog() -> dict[int, tuple[str, str]]:
    from .math500 import load_math500

    path, rows = load_math500()
    require(digest(path.read_bytes()) == COMMON["dataset_sha256"], "Incorrect MATH-500 dataset bytes")
    return {i: (row["unique_id"], row["answer"]) for i, row in enumerate(rows)}


def verify_traced_case(directory: Path, row: dict, config: dict, catalog: dict) -> dict:
    from .math500 import GRADING_POLICY, grade_math500
    from .trace_io import verify_trace

    require(config["grading"] == GRADING_POLICY and catalog.get(row["index"]) == (row["unique_id"], row["gold"]),
            "Traced case does not match the pinned MATH-500 catalog or grading policy")
    require(grade_math500(row["gold"], row["prediction"]) == row["grading_status"], "Traced case regrade differs")
    data = verify_trace(directory, row, config)
    if "regrade_origin" in row:
        from .regrade import verify_regraded_origin

        verify_regraded_origin(directory, row, config)
    if "baseline_origin" in row:
        from .baseline_reuse import verify_baseline_origin

        verify_baseline_origin(directory, row, config)
    return data


def verify_native_reuse(root: Path, hashes: dict | None = None):
    """Audit a portable, byte-preserving snapshot of native v1 results."""
    base = root / NATIVE_SNAPSHOT
    if not base.exists():
        return set(), None
    from .math500 import GRADING_POLICY, grade_math500

    meta = json.loads((base / "provenance.json").read_text())
    require(meta["snapshot_format"] == 1 and meta["source_commit"] == REFERENCE_COMMIT,
            "Unexpected native import provenance")
    require(Path(meta["source_run"]).is_absolute(), "Missing original native run path")
    validate_target_sources(meta["target_source_sha256"])
    if hashes is not None:
        require(meta["target_source_sha256"] == hashes, "Native import belongs to a different corrected build")
    catalog_data = (base / "catalog.json").read_bytes()
    require(digest(catalog_data) == CATALOG_SHA256 == meta["catalog_sha256"], "Invalid native-import ID/gold catalog")
    catalog = json.loads(catalog_data)
    manifests, seen = set(), set()
    for shard in meta["shards"]:
        rel = Path(shard["source_manifest"])
        require(len(rel.parts) == 3 and rel.parts[0] == "shards" and rel.name == "config.json"
                and rel.parts[1] != ".." and rel not in manifests, f"Invalid original native manifest path {rel}")
        manifests.add(rel)
        path = base / rel
        raw = path.read_bytes()
        require(digest(raw) == shard["manifest_sha256"], f"Changed original native manifest {rel}")
        cfg = json.loads(raw)
        check_config(cfg)
        names = {Path(p).name: v for p, v in cfg["source_sha256"].items()}
        require(cfg["mode"] == "native" and cfg["workers"] == [1] and "single_worker_layout" not in cfg
                and len(names) == len(cfg["source_sha256"]) and names == REFERENCE_SOURCES,
                f"Expected known v1 native sources at {rel}")
        require(cfg["branch"] == "hogwild-multi-workers" and cfg["commit"] == REFERENCE_COMMIT
                and cfg["native_context"] == 262144 and cfg["context_cap"] == 24576
                and cfg["cuda_graphs"] is True and cfg["grading"] == GRADING_POLICY,
                f"Incompatible native runtime or grader at {rel}")
        data = path.with_name("results.jsonl").read_bytes()
        require(data.endswith(b"\n") and digest(data) == shard["results_sha256"], f"Changed native results at {rel}")
        lines = data.splitlines()
        require(len(lines) == shard["cases"] > 0, f"Invalid native-import case count at {rel}")
        for number, line in enumerate(lines, 1):
            row = json.loads(line)
            task = check_task(row, cfg)
            index = task["index"]
            key = ("native", 1, index)
            require(key not in seen, f"Duplicate imported native case {index}")
            require([task["unique_id"], task["gold"]] == catalog[index], f"Wrong MATH-500 row at {rel}:{number}")
            require(task["grading_status"] == grade_math500(task["gold"], task["prediction"]),
                    f"Native regrade differs at {rel}:{number}")
            require(row["published_paragraphs"] == 0 and row["per_worker_tokens"] == {"Qwen": task["reasoning_tokens"]}
                    and math.isfinite(row["wall_seconds"]) and row["wall_seconds"] >= 0,
                    f"Invalid native metrics at {rel}:{number}")
            require(task["stop_reason"] in ("max_steps", "context_limit", "eos")
                    and (task["stop_reason"] == "eos") == (task["eos_token_id"] is not None)
                    and (task["stop_reason"] != "max_steps" or task["decoding_steps"] == COMMON["budget"]),
                    f"Invalid native stop at {rel}:{number}")
            seen.add(key)
        completed = path.with_name("completed.json")
        if shard.get("completed_sha256") is not None:
            raw = completed.read_bytes()
            require(digest(raw) == shard["completed_sha256"] and json.loads(raw)["cases"] == len(lines)
                    and len(lines) == len(cfg["indices"]), f"Changed/incomplete native completion marker at {rel}")
        else:
            require(not completed.exists(), f"Unexpected completion marker at {rel}")
    require(len(seen) == meta["cases"] > 0 and manifests == {p.relative_to(base) for p in base.rglob("config.json")},
            "Native snapshot is empty or contains unregistered manifests")
    return seen, meta


def collect(root: Path, partial: bool, only_one_worker: bool = False, native_reuse=None):
    tasks = {}
    ids_by_index, indices_by_id = {}, {}
    identity, settings = None, {}
    source_variants = []
    catalog = None
    multi_settings = None
    if native_reuse is None and (root / NATIVE_SNAPSHOT).exists():
        _, native_reuse = verify_native_reuse(root)
    imported = {root / NATIVE_SNAPSHOT / shard["source_manifest"]: shard
                for shard in (native_reuse or {}).get("shards", [])}
    manifests = sorted(root.rglob("config.json"))
    for result_path in root.rglob("results.jsonl"):
        require(result_path.with_name("config.json").is_file(), f"Missing manifest beside {result_path}")
    for manifest in manifests:
        path = manifest.with_name("results.jsonl")
        # The sweep may contain a just-created manifest and no results yet.
        if partial and not path.exists() and not manifest.read_bytes().strip():
            continue
        try:
            config = json.loads(manifest.read_text(encoding="utf-8"))
            check_config(config)
            if only_one_worker:
                require(config["mode"] == "hogwild" and config["workers"] == [1]
                        and config.get("single_worker_layout") == "causal",
                        "This report expects only corrected causal one-worker shards")
            shared = {key: config[key] for key in ("branch", "commit", "source_sha256", "native_context", "grading")}
            shared["manifest_version"] = manifest_version(config)
            if manifest not in imported:
                if identity is None:
                    identity = shared
                comparison = {**shared, "source_sha256": canonical_sources(shared["source_sha256"])}
                reference = {**identity, "source_sha256": canonical_sources(identity["source_sha256"])}
                require(comparison == reference, "Mixed code versions or grading protocols across shards")
                if shared["source_sha256"] not in source_variants:
                    source_variants.append(shared["source_sha256"])
                if native_reuse:
                    require(shared["manifest_version"] in (2, 3) and shared["commit"] == REFERENCE_COMMIT
                            and shared["source_sha256"] == native_reuse["target_source_sha256"],
                            "Native reuse and executed shard have different target builds")
            for n in config["workers"]:
                key = (config["mode"], n)
                runtime = {field: config[field] for field in ("context_cap", "cuda_graphs")}
                runtime["single_worker_layout"] = worker_layout(config, n)
                runtime["answer_selection_policy"] = config.get("answer_selection_policy") if n > 1 else None
                runtime["publication_policy"] = config.get("publication_policy") if n > 1 else None
                runtime.update({k: v for k, v in condition_protocol(config, n).items()
                                if k in ("header_context", "reminder_interval", "finisher_mode")})
                if shared["manifest_version"] >= 3 and config["mode"] == "hogwild":
                    runtime.update({k: config[k] if n > 1 else None for k in MULTI_FIELDS},
                                   pending_eos_in_finisher=False)
                    if n > 1:
                        choices = tuple(runtime[k] for k in MULTI_FIELDS)
                        require(multi_settings is None or multi_settings == choices,
                                "Mixed multi-worker layouts or answer cues across shards")
                        multi_settings = choices
                require(settings.setdefault(key, runtime) == runtime,
                        f"Inconsistent context, graph, or one-worker layout settings for {key}")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{manifest}: {exc}") from exc
        if not path.exists():
            continue
        for line, row in read_results(path, partial=not manifest.with_name("completed.json").exists()):
            try:
                task = check_task(row, config)
                if manifest_version(config) == 4:
                    if catalog is None:
                        catalog = load_native_catalog()
                    verify_traced_case(manifest.parent, row, config, catalog)
                    task["trace_path"] = str(manifest.parent.relative_to(root) / task["trace_path"])
                if "regrade_origin" in row:
                    task["result_origin"] = row["regrade_origin"]["classification"]
                    task["source_manifest"] = row["regrade_origin"]["source_config"]
                if "baseline_origin" in row:
                    task["result_origin"] = row["baseline_origin"]["classification"]
                    task["source_manifest"] = row["baseline_origin"]["source_config"]
                if native_reuse:
                    source = imported.get(manifest)
                    task["result_origin"] = "reused_native" if source else "executed"
                    task["source_manifest"] = source["source_manifest"] if source else str(manifest.relative_to(root))
                index, uid = task["index"], task["unique_id"]
                require(ids_by_index.setdefault(index, uid) == uid and indices_by_id.setdefault(uid, index) == index,
                        f"Inconsistent mapping between index {index} and problem {uid!r}")
                key = (config["mode"], task["workers"], index)
                require(key not in tasks, f"Duplicate case {key}; first seen in {tasks.get(key, (None,))[0]}")
                tasks[key] = (f"{path}:{line}", task)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
    if identity is not None:
        identity = {**identity, "source_variants": source_variants}
    return [task for _, task in tasks.values()], identity, settings


def summarize(mode: str, workers: int, tasks: list[dict]) -> dict:
    count = len(tasks)
    correct = sum(task["correct"] for task in tasks)
    finishers = sum(task["used_finisher"] for task in tasks)
    summary = {
        "condition": mode, "workers": workers, "completed_cases": count, "expected_cases": N_PROBLEMS,
        "correct": correct, "accuracy": correct / count if count == N_PROBLEMS else None,
        "observed_accuracy": correct / count if count else None,
        "finisher_uses": finishers, "finisher_rate": finishers / count if count else None,
    }
    for field in ("generated_tokens", "decoding_steps", "total_decoding_steps", "finisher_tokens"):
        total = sum(task[field] for task in tasks)
        summary[f"sum_{field}"] = total
        summary[f"mean_{field}"] = total / count if count else None
    return summary


def csv_text(rows: list[dict], fields) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def atomic_write(path: Path, text: str) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
        try:
            tmp.write(text)
            tmp.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def report(root: Path, partial: bool = False, only_one_worker: bool = False) -> dict:
    require(root.is_dir(), f"Run directory does not exist: {root}")
    _, native_reuse = verify_native_reuse(root)
    tasks, identity, settings = collect(root, partial, only_one_worker, native_reuse)
    conditions = (("hogwild", 1),) if only_one_worker else CONDITIONS
    summaries = []
    v3 = identity is not None and identity["manifest_version"] >= 3
    traced = identity is not None and identity["manifest_version"] == 4
    if v3 and ("native", 1) in settings:
        settings[("native", 1)].update(multi_worker_layout=None, multi_worker_answer_cue=None,
                                       pending_eos_in_finisher=False)
    for mode, workers in conditions:
        group = [task for task in tasks if task["condition"] == mode and task["workers"] == workers]
        summary = summarize(mode, workers, group)
        summary["single_worker_layout"] = settings.get((mode, workers), {}).get("single_worker_layout")
        if v3:
            summary.update({k: settings.get((mode, workers), {}).get(k) for k in MULTI_FIELDS})
        if not partial:
            require(len(group) == N_PROBLEMS, f"{mode} {workers} workers: {len(group)}/{N_PROBLEMS} cases")
        summaries.append(summary)
    tasks.sort(key=lambda task: (CONDITIONS.index((task["condition"], task["workers"])), task["index"]))
    payload = {
        "complete": len(tasks) == N_PROBLEMS * len(conditions),
        "completed_cases": len(tasks), "expected_cases": N_PROBLEMS * len(conditions),
        "scope": "corrected_one_worker" if only_one_worker else "seven_conditions",
        "protocol": COMMON, "source": identity, "native_reuse": native_reuse,
        "mode_protocol": PROTOCOL if identity and identity["manifest_version"] == 1 else {
            "native": PROTOCOL["native"],
            "hogwild": {**{key: value for key, value in PROTOCOL["hogwild"].items()
                           if key not in ("header_context", "reminder_interval")}, "finisher_mode": "cached"},
        },
        "condition_protocol": {f"{mode}-{n}": {**PROTOCOL[mode], **settings[(mode, n)]}
                               for mode, n in conditions if (mode, n) in settings},
        "metric_definitions": {
            "accuracy": "Correct cases divided by 500; null until the condition is complete.",
            "observed_accuracy": "Correct cases divided by the number completed in this snapshot.",
            "generated_tokens": "All sampled tokens, including finisher tokens.",
            "decoding_steps": "Reasoning decode rounds. Prompt and publication forwards are excluded.",
            "total_decoding_steps": "Reasoning decode rounds plus sampled finisher tokens.",
            "finisher_rate": "Cases using the finisher divided by the number completed.",
        },
        "runtime_per_condition": {f"{mode}-{n}": value for (mode, n), value in sorted(settings.items())},
        "conditions": summaries,
        "stop_reasons": {f"{mode}-{n}": dict(Counter(task["stop_reason"] for task in tasks
                         if (task["condition"], task["workers"]) == (mode, n))) for mode, n in conditions},
    }
    output = root / "report"
    output.mkdir(exist_ok=True)
    fields = TASK_FIELDS + (MULTI_FIELDS if v3 else ()) + (("result_origin", "source_manifest") if native_reuse else ())
    if traced:
        payload["verified_traces"] = len(tasks)
        fields += TRACE_FIELDS
    if any("result_origin" in task for task in tasks) and not native_reuse:
        fields += ("result_origin", "source_manifest")
        for task in tasks:
            task.setdefault("result_origin", "executed")
            task.setdefault("source_manifest", "")
        payload["result_origins"] = dict(Counter(task["result_origin"] for task in tasks))
    atomic_write(output / "tasks.csv", csv_text(tasks, fields))
    atomic_write(output / "summary.csv", csv_text(summaries, summaries[0].keys()))
    atomic_write(output / "summary.json", json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--partial", action="store_true", help="Allow fewer than 500 cases per condition")
    parser.add_argument("--only-one-worker", action="store_true", help="Verify an isolated 500-case causal one-worker run")
    args = parser.parse_args(argv)
    try:
        payload = report(args.run_root, args.partial, args.only_one_worker)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Report failed: {exc}\n")
    print(f"{payload['completed_cases']}/{payload['expected_cases']} cases; {args.run_root / 'report'}")


if __name__ == "__main__":
    main()
