"""Import auditable saved traces into a fresh sweep after fixing answer selection."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

from . import report, trace_io
from .math500 import grade_math500
from .rollout import TERMINAL_ANSWER_POLICY, select_terminal_worker_answer

SOURCE_HASHES = {
    "scripts/minisgl_examples/training/parallel_thinking/eval.py":
        "eb796ba3056802e865ec557261c6f81ddb45fcd688f4c9dc0ea866124fe3bef1",
    "scripts/minisgl_examples/training/parallel_thinking/rollout.py":
        "9a71ffd47048e8a1c21d2d1dc5632293ced6fb7d21c11fbaf4271ab5c19b04bf",
    "scripts/minisgl_examples/training/parallel_thinking/native.py":
        "01d9392692e920d720128fb0050b7321abe13097b125868cb6e81ea415079b4f",
    "scripts/minisgl_examples/training/parallel_thinking/single_worker.py":
        "587c717a08bfb37c27b5d6d7b77c5ff6f0b5db7ae55777b55369ee54b7b8cd6f",
    "scripts/minisgl_examples/training/parallel_thinking/math500.py":
        "aded81fcec2a2a9a7f607979f74a51f1ed66fb53a04020eecca10efc4f34e9d4",
    "scripts/minisgl_examples/training/parallel_thinking/trace_io.py":
        "be2a111c320ef02cb3d1ae6081ec3af12b28038f8ba621b58164529f573a67ba",
    "run_hogwild.sh": "671d2836accc595b1beb149c4707cc52ef432680a1c1f6c7c5f01912ecacc727",
}
RESIDUAL_PREFLIGHT_SHA256 = "c1af092270325acef6c5060cb9c0031b84c91f6143be6595177152db3e3f7348"
SUMMARY = "regrade-summary.json"


def check_source(config: dict) -> None:
    report.check_config(config)
    hashes = config["source_sha256"].copy()
    evaluation = "scripts/minisgl_examples/training/parallel_thinking/eval.py"
    if hashes.get(evaluation) == RESIDUAL_PREFLIGHT_SHA256:
        hashes[evaluation] = SOURCE_HASHES[evaluation]
    report.require(hashes == SOURCE_HASHES, "Unexpected source build for selector regrading")
    report.require(config["branch"] == "hogwild-multi-workers"
                   and config["commit"] == report.REFERENCE_COMMIT
                   and report.manifest_version(config) == 4
                   and config.get("answer_selection_policy") is None,
                   "Unexpected source identity for selector regrading")
    report.require(len(config["workers"]) == 1, "Expected a single source condition per shard")
    workers = config["workers"][0]
    single = config["mode"] == "hogwild" and workers == 1
    multiple = config["mode"] == "hogwild" and workers > 1
    report.require(config["native_context"] == 262144
                   and config["context_cap"] == min(262144, report.COMMON["budget"] * workers + 8192)
                   and config["cuda_graphs"] is True
                   and config["single_worker_layout"] == ("causal" if single else None)
                   and config["multi_worker_layout"] == ("native" if multiple else None)
                   and config["multi_worker_answer_cue"] == ("qwen" if multiple else None),
                   "Unexpected source runtime protocol for selector regrading")


@lru_cache(maxsize=1)
def load_tokenizer():
    return trace_io.selection_tokenizer(report.COMMON["model"], report.COMMON["model_revision"])


@lru_cache(maxsize=1)
def load_catalog():
    return report.load_native_catalog()


def classify_case(row: dict, trace: dict, tokenizer) -> tuple[str, str | None, dict | None]:
    """Choose reuse or rerun from recorded execution before consulting the gold answer."""
    if row["mode"] == "native" or row["workers"] == 1:
        return "unchanged", row["prediction"], None
    rollout = trace["rollout"]
    answer, selection = select_terminal_worker_answer(
        tokenizer, rollout["worker_token_ids"], eos_worker=rollout["eos_worker"],
        eos_token_ids=rollout["eos_token_ids"],
    )
    if answer is not None:
        return ("finisher_no_longer_needed" if row["used_finisher"] else "selected_terminal_answer"), answer, selection
    return ("existing_finisher" if row["used_finisher"] else "missing_cached_finisher"), row["prediction"], selection


def project_case(row: dict, trace: dict, tokenizer) -> tuple[dict, dict] | None:
    classification, prediction, selection = classify_case(row, trace, tokenizer)
    if classification in ("missing_cached_finisher", "finisher_no_longer_needed"):
        return None
    projected_row = row.copy()
    if classification == "unchanged":
        return projected_row, trace
    grading = grade_math500(row["gold"], prediction)
    report.require(grading in ("correct", "incorrect", "unparsed_prediction"), "Cannot grade derived prediction")
    projected_row.update(prediction=prediction, grading_status=grading, correct=grading == "correct")
    projected_trace = {
        **trace,
        "metrics": {**trace["metrics"], **{key: projected_row[key] for key in ("prediction", "grading_status", "correct")}},
        "rollout": {**trace["rollout"], "boxed_answer": prediction, "answer_selection": selection},
    }
    return projected_row, projected_trace


def _relative_file(root: Path, name: str) -> Path:
    relative = Path(name)
    report.require(not relative.is_absolute() and ".." not in relative.parts, "Invalid regrade provenance path")
    path = root / relative
    report.require(path.is_file() and path.resolve().is_relative_to(root.resolve()), "Missing regrade provenance file")
    return path


def _verified_bytes(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    report.require(report.digest(data) == expected, f"Changed regrade provenance: {path}")
    return data


def _source_row(root: Path, row: dict) -> tuple[dict, dict, Path]:
    origin = row["regrade_origin"]
    config_path = _relative_file(root, origin["source_config"])
    results_path = _relative_file(root, origin["source_results"])
    report.require(config_path.name == "source-config.json"
                   and results_path == config_path.with_name("source-results.jsonl"), "Mismatched source provenance paths")
    config = json.loads(_verified_bytes(config_path, origin["source_config_sha256"]))
    check_source(config)
    raw = _verified_bytes(results_path, origin["source_results_sha256"])
    matches = [json.loads(line) for line in raw.splitlines(keepends=True)
               if line.endswith(b"\n") and json.loads(line)["index"] == row["index"]]
    report.require(len(matches) == 1, "Missing or duplicate source result")
    source_row = matches[0]
    report.require("regrade_origin" not in source_row, "Cannot recursively regrade a derived result")
    report.require(all(source_row[key] == row[key] for key in trace_io.IDENTITY), "Source case identity differs")
    trace_path = _relative_file(root, origin["source_trace"])
    report.require(trace_path == config_path.parent / source_row["trace_path"]
                   and origin["source_trace_sha256"] == source_row["trace_sha256"],
                   "Source trace provenance differs")
    report.check_task(source_row, config)
    report.verify_traced_case(config_path.parent, source_row, config, load_catalog())
    return source_row, config, trace_path


def _source_metadata(config: dict, config_relative: str, config_hash: str,
                     results_relative: str, results_hash: str, completed_hash: str | None) -> dict:
    return {
        "source_config": config_relative, "source_config_sha256": config_hash,
        "source_results": results_relative, "source_results_sha256": results_hash,
        "original_source_sha256": config["source_sha256"], "original_commit": config["commit"],
        "original_completed_sha256": completed_hash,
    }


def _target_config(source: dict, indices: list[int], hashes: dict, provenance: dict) -> dict:
    return {
        **source, "indices": indices, "source_sha256": hashes, "regrade_source": provenance,
        "answer_selection_policy": TERMINAL_ANSWER_POLICY
        if source["mode"] == "hogwild" and source["workers"][0] > 1 else None,
    }


def verify_regraded_origin(attempt: Path, row: dict, manifest: dict) -> None:
    """Rebuild a derived result from its hash-bound original execution on CPU."""
    root = attempt.parent.parent
    source_row, config, trace_path = _source_row(root, row)
    origin = row["regrade_origin"]
    completed = trace_path.parent.parent / "source-completed.json"
    provenance = _source_metadata(
        config, origin["source_config"], origin["source_config_sha256"],
        origin["source_results"], origin["source_results_sha256"],
        report.digest(completed.read_bytes()) if completed.exists() else None,
    )
    report.require(manifest == _target_config(config, manifest["indices"], manifest["source_sha256"], provenance),
                   "Derived manifest changed original runtime settings or provenance")
    report.require(set(manifest["indices"]) <= set(config["indices"]), "Derived indices outside source attempt")
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        source_trace = json.load(stream)
    tokenizer = load_tokenizer() if row["mode"] == "hogwild" and row["workers"] > 1 else None
    classification, _, _ = classify_case(source_row, source_trace, tokenizer)
    report.require(origin["classification"] == classification, "Regrade classification changed")
    projected = project_case(source_row, source_trace, tokenizer)
    report.require(projected is not None, "A case requiring new cache computation was imported")
    expected_row, expected_trace = projected
    actual_row = {key: value for key, value in row.items() if key not in ("regrade_origin", "trace_sha256", "trace_bytes")}
    expected_row = {key: value for key, value in expected_row.items() if key not in ("trace_sha256", "trace_bytes")}
    report.require(actual_row == expected_row, "Derived row differs from deterministic source projection")
    derived = attempt / row["trace_path"]
    report.require(derived.stat().st_size == row["trace_bytes"]
                   and trace_io._digest(derived) == row["trace_sha256"], "Derived trace bytes changed")
    with gzip.open(derived, "rt", encoding="utf-8") as stream:
        report.require(json.load(stream) == expected_trace, "Derived trace changed recorded execution")
    if classification == "unchanged":
        report.require(row["trace_sha256"] == source_row["trace_sha256"], "Unchanged trace bytes differ")


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_trace(path: Path, trace: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as stream:
        # Preserve the exact sampled streams, cache operations and finisher payload.
        stream.write(json.dumps(trace, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode())


def _source_inventory(root: Path) -> list[dict]:
    inventory = []
    for config_path in sorted((root / "shards").glob("*/config.json")):
        results = config_path.with_name("results.jsonl")
        if results.exists():
            inventory.append({"shard": config_path.parent.name, "config_sha256": trace_io._digest(config_path),
                              "results_sha256": trace_io._digest(results)})
    return inventory


def import_regraded(old_root: Path, new_root: Path) -> dict:
    """Publish a complete new root atomically, preserving every original artifact."""
    from .sweep import ROOT, source_hashes

    old, new = old_root.resolve(), new_root.resolve()
    report.require(old.is_dir() and old != new and not old.is_relative_to(new) and not new.is_relative_to(old),
                   "Source and destination must be separate run directories")
    # Validate engine content so a branch containing the published sweep can
    # still verify historical imports against the audited engine revision.
    subprocess.run([
        "git", "-c", "core.filemode=false", "diff", "--quiet", report.REFERENCE_COMMIT, "--",
        "python/minisgl", "pyproject.toml", "uv.lock", "scripts/pyproject.toml",
    ], cwd=ROOT, check=True)
    hashes = source_hashes(True)
    inventory = _source_inventory(old)
    report.require(bool(inventory), "No committed source attempts")
    inventory_hash = report.digest(json.dumps(inventory, sort_keys=True).encode())
    if new.exists():
        saved = json.loads((new / SUMMARY).read_text())
        report.require(saved["source_root"] == str(old) and saved["source_inventory_sha256"] == inventory_hash
                       and saved["target_source_sha256"] == hashes and saved["policy"] == TERMINAL_ANSWER_POLICY,
                       "Existing destination belongs to another selector migration")
        return saved
    new.parent.mkdir(parents=True, exist_ok=True)
    tokenizer, catalog = load_tokenizer(), load_catalog()
    seen, rerun, flips = set(), [], []
    counts, conditions = Counter(), {}
    with TemporaryDirectory(prefix=f".{new.name}.regrade-", dir=new.parent) as temporary:
        stage = Path(temporary) / "run"
        (stage / "shards").mkdir(parents=True)
        (stage / "provenance").mkdir()
        for entry in inventory:
            source = old / "shards" / entry["shard"]
            config = json.loads((source / "config.json").read_text())
            check_source(config)
            origin_dir = stage / "provenance" / entry["shard"]
            _copy_file(source / "config.json", origin_dir / "source-config.json")
            _copy_file(source / "results.jsonl", origin_dir / "source-results.jsonl")
            _verified_bytes(origin_dir / "source-config.json", entry["config_sha256"])
            _verified_bytes(origin_dir / "source-results.jsonl", entry["results_sha256"])
            completed_hash = None
            if (source / "completed.json").exists():
                _copy_file(source / "completed.json", origin_dir / "source-completed.json")
                completed_hash = trace_io._digest(source / "completed.json")
            provenance = _source_metadata(
                config, str((origin_dir / "source-config.json").relative_to(stage)), entry["config_sha256"],
                str((origin_dir / "source-results.jsonl").relative_to(stage)), entry["results_sha256"], completed_hash,
            )
            derived = stage / "shards" / entry["shard"]
            rows, source_count = [], 0
            for _, row in report.read_results(source / "results.jsonl", partial=completed_hash is None):
                key = (row["mode"], row["workers"], row["index"])
                report.require(key not in seen, f"Duplicate source case: {key}")
                seen.add(key)
                source_count += 1
                report.check_task(row, config)
                report.verify_traced_case(source, row, config, catalog)
                raw_trace = source / row["trace_path"]
                _copy_file(raw_trace, origin_dir / row["trace_path"])
                snapshot_trace = origin_dir / row["trace_path"]
                report.require(trace_io._digest(snapshot_trace) == row["trace_sha256"], "Source trace changed during copy")
                with gzip.open(snapshot_trace, "rt", encoding="utf-8") as stream:
                    trace = json.load(stream)
                classification, _, _ = classify_case(row, trace, tokenizer)
                counts[classification] += 1
                condition = conditions.setdefault(f"{key[0]}-{key[1]}", {
                    "source_cases": 0, "kept_cases": 0, "rerun_cases": 0, "original_kept_correct": 0,
                    "regraded_correct": 0, "generated_tokens": 0, "reasoning_steps": 0,
                    "total_steps": 0, "finisher_uses": 0,
                    "original_correct": 0, "original_generated_tokens": 0, "original_reasoning_steps": 0,
                    "original_total_steps": 0, "original_finisher_uses": 0,
                })
                condition["source_cases"] += 1
                condition["original_correct"] += int(row["correct"])
                condition["original_generated_tokens"] += row["generated_tokens"]
                condition["original_reasoning_steps"] += row["decoding_steps"]
                condition["original_total_steps"] += row["total_decoding_steps"]
                condition["original_finisher_uses"] += int(row["used_finisher"])
                projected = project_case(row, trace, tokenizer)
                if projected is None:
                    condition["rerun_cases"] += 1
                    rerun.append({"mode": key[0], "workers": key[1], "index": key[2], "unique_id": row["unique_id"],
                                  "reason": classification, "source_trace_sha256": row["trace_sha256"],
                                  "source_trace": str((origin_dir / row["trace_path"]).relative_to(stage))})
                    continue
                derived_row, derived_trace = projected
                if classification == "unchanged":
                    _copy_file(snapshot_trace, derived / row["trace_path"])
                else:
                    _write_trace(derived / row["trace_path"], derived_trace)
                derived_row.update(
                    trace_sha256=trace_io._digest(derived / row["trace_path"]),
                    trace_bytes=(derived / row["trace_path"]).stat().st_size,
                    regrade_origin={
                        **{key: provenance[key] for key in ("source_config", "source_config_sha256", "source_results", "source_results_sha256")},
                        "source_trace": str((origin_dir / row["trace_path"]).relative_to(stage)),
                        "source_trace_sha256": row["trace_sha256"], "classification": classification,
                    },
                )
                rows.append(derived_row)
                condition["kept_cases"] += 1
                condition["original_kept_correct"] += int(row["correct"])
                condition["regraded_correct"] += int(derived_row["correct"])
                condition["generated_tokens"] += row["generated_tokens"]
                condition["reasoning_steps"] += row["decoding_steps"]
                condition["total_steps"] += row["total_decoding_steps"]
                condition["finisher_uses"] += int(row["used_finisher"])
                if row["correct"] != derived_row["correct"]:
                    flips.append({"mode": key[0], "workers": key[1], "index": key[2], "was_correct": row["correct"],
                                  "correct": derived_row["correct"], "old_prediction": row["prediction"],
                                  "prediction": derived_row["prediction"]})
            if completed_hash is not None:
                report.require(json.loads((source / "completed.json").read_text())["cases"] == source_count
                               and source_count == len(config["indices"]), "Incomplete source marked completed")
            if rows:
                manifest = _target_config(config, [row["index"] for row in rows], hashes, provenance)
                _write_json(derived / "config.json", manifest)
                with (derived / "results.jsonl").open("w", encoding="utf-8") as stream:
                    for row in rows:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                _write_json(derived / "completed.json", {"cases": len(rows), "origin": "saved_trace_regrade"})
                for row in rows:
                    report.verify_traced_case(derived, row, manifest, catalog)
            print(f"Regraded {entry['shard']}: kept {len(rows)}/{source_count}", flush=True)
        if (old / "quality-hold.json").exists():
            _copy_file(old / "quality-hold.json", stage / "provenance" / "source-quality-hold.json")
        summary = {
            "format": 1, "policy": TERMINAL_ANSWER_POLICY, "source_root": str(old),
            "source_inventory_sha256": inventory_hash, "source_inventory": inventory,
            "regrade_code_sha256": trace_io._digest(Path(__file__)),
            "target_source_sha256": hashes, "source_cases": len(seen), "kept_cases": len(seen) - len(rerun),
            "rerun_cases": len(rerun), "classifications": dict(counts), "conditions": conditions,
            "accuracy_flips": flips, "rerun": rerun,
        }
        report.require(_source_inventory(old) == inventory, "Source attempts changed during regrading")
        _write_json(stage / SUMMARY, summary)
        os.rename(stage, new)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    result = import_regraded(args.source, args.destination)
    print(json.dumps({key: result[key] for key in ("source_cases", "kept_cases", "rerun_cases", "classifications")}, indent=2))


if __name__ == "__main__":
    main()
