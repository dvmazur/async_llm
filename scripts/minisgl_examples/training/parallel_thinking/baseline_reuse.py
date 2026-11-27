"""Reuse verified native and causal one-worker executions across a multi-worker fix."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from . import regrade, report, trace_io

SUMMARY = "baseline-reuse-summary.json"
PREFIX = "scripts/minisgl_examples/training/parallel_thinking/"
SELECTOR_EVAL_SHA256 = "152dfef57421829ef9ce7ddc4c85273eac3785330aea9811e5c7689532edb107"
SELECTOR_SOURCES = {
    **regrade.SOURCE_HASHES,
    PREFIX + "eval.py": SELECTOR_EVAL_SHA256,
    PREFIX + "rollout.py": "938c09a9b1d2b9eb9d4502ce8b2f3d7c969359fefea47e4315aca7442440063e",
    PREFIX + "trace_io.py": "ee0290cf7ad44ed6e68da1860bfefe0791ce2a5e2d94dfe5ccc835abe6cad2cf",
}
HELPER_HASHES = {
    "_token_ids_sha256": "f259b5412bf82f87753cc3d62d999d59464b3685b5001b0116a270bc9e137420",
    "extract_boxed_answer": "d88eb77230ce7b271bb008eacfc16def6f2c0ee2faa17622bfccd539b30daaac",
    "MathFormatting": "adc354d74a9677ddf782b894952dbf6d0198451040cf03c2b7f9c930ddb5098e",
    "_make_system_prompt": "eff8ede0d6dacb599d8b5dbc6727e30895b3202d23186a7a74fea979a74970dc",
    "_rules": "3f3712259be67482301391e25e2781f5eedab67175a6e31d1b2aaa91a9811d62",
    "_suggestions": "edf730a061007cff64f18065e82564a63406e624602c803beffb824b50607a3b",
}


def audit_target() -> dict:
    from .sweep import ROOT, source_hashes
    subprocess.run([
        "git", "-c", "core.filemode=false", "diff", "--quiet", report.REFERENCE_COMMIT, "--",
        "python/minisgl", "pyproject.toml", "uv.lock", "scripts/pyproject.toml",
    ], cwd=ROOT, check=True)
    hashes = source_hashes(True)
    for name in ("native.py", "single_worker.py", "math500.py"):
        path = PREFIX + name
        report.require(hashes[path] == regrade.SOURCE_HASHES[path], f"Baseline implementation changed: {name}")
    report.require(hashes["run_hogwild.sh"] == regrade.SOURCE_HASHES["run_hogwild.sh"], "Original launcher changed")
    evaluation = (ROOT / PREFIX / "eval.py").read_text()
    publication_field = '        "publication_policy": "encoded_separator_v1" if native_workers else None,\n'
    report.require(evaluation.count(publication_field) == 1
                   and report.digest(evaluation.replace(publication_field, "").encode()) == SELECTOR_EVAL_SHA256,
                   "Evaluator changes extend beyond the publication-policy manifest field")
    rollout = (ROOT / PREFIX / "rollout.py").read_text()
    helpers = {node.name: report.digest(ast.get_source_segment(rollout, node).encode())
               for node in ast.parse(rollout).body if getattr(node, "name", None) in HELPER_HASHES}
    report.require(helpers == HELPER_HASHES, "A rollout helper used by the baselines changed")
    return hashes


def check_source(config: dict) -> None:
    report.check_config(config)
    report.require(config["workers"] == [1] and config["mode"] in ("native", "hogwild"),
                   "Only native and one-worker baseline cases may be reused")
    report.require(config["single_worker_layout"] == ("causal" if config["mode"] == "hogwild" else None),
                   "Only the causal one-worker baseline may be reused")
    report.require(config["branch"] == "hogwild-multi-workers" and config["commit"] == report.REFERENCE_COMMIT
                   and report.manifest_version(config) == 4 and config["native_context"] == 262144
                   and config["context_cap"] == 24576 and config["cuda_graphs"] is True
                   and config.get("answer_selection_policy") is None and config.get("publication_policy") is None,
                   "Baseline source runtime protocol differs")
    hashes = config["source_sha256"]
    original = {**regrade.SOURCE_HASHES, PREFIX + "eval.py": regrade.RESIDUAL_PREFLIGHT_SHA256}
    report.require(hashes in (regrade.SOURCE_HASHES, original, SELECTOR_SOURCES), "Unaudited baseline source build")


def _inventory(root: Path) -> list[dict]:
    entries = []
    for config_path in sorted((root / "shards").glob("*/config.json")):
        config = json.loads(config_path.read_text())
        if config.get("workers") != [1]:
            continue
        check_source(config)
        results = config_path.with_name("results.jsonl")
        if results.exists():
            entries.append({"shard": config_path.parent.name, "config_sha256": trace_io._digest(config_path),
                            "results_sha256": trace_io._digest(results)})
    return entries


def _original_execution(root: Path, attempt: Path, row: dict, config: dict):
    if "regrade_origin" in row:
        origin = row["regrade_origin"]
        report.require(origin["classification"] == "unchanged", "Baseline regrade changed its execution")
        source_row, source_config, trace = regrade._source_row(root, row)
        report.require({key: value for key, value in row.items() if key != "regrade_origin"} == source_row,
                       "Regraded baseline row differs from its original execution")
        report.require(row["trace_sha256"] == origin["source_trace_sha256"], "Regraded baseline trace changed")
        config_path = regrade._relative_file(root, origin["source_config"])
        results_path = regrade._relative_file(root, origin["source_results"])
        completed = config_path.with_name("source-completed.json")
    else:
        report.require("baseline_origin" not in row, "Expected a direct execution or unchanged selector import")
        source_row, source_config = row, config
        config_path, results_path = attempt / "config.json", attempt / "results.jsonl"
        trace, completed = attempt / row["trace_path"], attempt / "completed.json"
    check_source(source_config)
    return source_row, source_config, config_path, results_path, trace, completed


def _metadata(config: dict, relative: Path, config_hash: str, results_hash: str, completed_hash: str | None) -> dict:
    return {
        "source_config": str(relative / "source-config.json"), "source_config_sha256": config_hash,
        "source_results": str(relative / "source-results.jsonl"), "source_results_sha256": results_hash,
        "original_source_sha256": config["source_sha256"], "original_commit": config["commit"],
        "original_completed_sha256": completed_hash,
    }


def _target_config(source: dict, indices: list[int], hashes: dict, provenance: dict) -> dict:
    return {
        **source, "indices": indices, "source_sha256": hashes, "baseline_source": provenance,
        "answer_selection_policy": None, "publication_policy": None,
    }


def verify_baseline_origin(attempt: Path, row: dict, manifest: dict) -> None:
    """Verify that an imported baseline is byte-identical to its original execution."""
    root, origin = attempt.parent.parent, row["baseline_origin"]
    config_path = regrade._relative_file(root, origin["source_config"])
    results_path = regrade._relative_file(root, origin["source_results"])
    report.require(config_path.name == "source-config.json"
                   and results_path == config_path.with_name("source-results.jsonl"), "Mismatched baseline provenance paths")
    config = json.loads(regrade._verified_bytes(config_path, origin["source_config_sha256"]))
    check_source(config)
    raw_results = regrade._verified_bytes(results_path, origin["source_results_sha256"])
    matches = [json.loads(line) for line in raw_results.splitlines(keepends=True)
               if line.endswith(b"\n") and json.loads(line)["index"] == row["index"]]
    report.require(len(matches) == 1, "Missing or duplicate original baseline result")
    original = matches[0]
    report.require("regrade_origin" not in original and "baseline_origin" not in original,
                   "Baseline provenance must point to its original execution")
    report.require({key: value for key, value in row.items() if key != "baseline_origin"} == original,
                   "Imported baseline row changed")
    expected_classification = "unchanged_native" if config["mode"] == "native" else "unchanged_causal_w1"
    report.require(origin["classification"] == expected_classification, "Baseline classification changed")
    source_trace = regrade._relative_file(root, origin["source_trace"])
    report.require(source_trace == config_path.parent / original["trace_path"]
                   and origin["source_trace_sha256"] == original["trace_sha256"], "Baseline trace provenance differs")
    completed = config_path.with_name("source-completed.json")
    provenance = _metadata(
        config, config_path.parent.relative_to(root), origin["source_config_sha256"], origin["source_results_sha256"],
        trace_io._digest(completed) if completed.exists() else None,
    )
    report.require(manifest == _target_config(config, manifest["indices"], manifest["source_sha256"], provenance)
                   and set(manifest["indices"]) <= set(config["indices"]), "Baseline manifest changed its runtime or provenance")
    report.check_task(original, config)
    report.verify_traced_case(config_path.parent, original, config, regrade.load_catalog())
    derived = attempt / row["trace_path"]
    report.require(derived.stat().st_size == original["trace_bytes"]
                   and trace_io._digest(derived) == original["trace_sha256"], "Imported baseline trace bytes changed")


def import_baselines(old_root: Path, new_root: Path) -> dict:
    """Publish a new run containing only audited native and causal one-worker cases."""
    old, new = old_root.resolve(), new_root.resolve()
    report.require(old.is_dir() and old != new and not old.is_relative_to(new) and not new.is_relative_to(old),
                   "Baseline source and destination must be separate directories")
    hashes = audit_target()
    inventory = _inventory(old)
    report.require(bool(inventory), "No eligible baseline attempts")
    inventory_hash = report.digest(json.dumps(inventory, sort_keys=True).encode())
    if new.exists():
        summary = json.loads((new / SUMMARY).read_text())
        report.require(summary["source_root"] == str(old) and summary["source_inventory_sha256"] == inventory_hash
                       and summary["target_source_sha256"] == hashes, "Existing root belongs to another baseline import")
        return summary
    new.parent.mkdir(parents=True, exist_ok=True)
    catalog = regrade.load_catalog()
    seen, groups = set(), {}
    counts = {"native": 0, "hogwild-1": 0}
    original_regrades = 0
    with TemporaryDirectory(prefix=f".{new.name}.baselines-", dir=new.parent) as temporary:
        stage = Path(temporary) / "run"
        (stage / "shards").mkdir(parents=True)
        (stage / "baseline-provenance").mkdir()
        for entry in inventory:
            attempt = old / "shards" / entry["shard"]
            config = json.loads(regrade._verified_bytes(attempt / "config.json", entry["config_sha256"]))
            regrade._verified_bytes(attempt / "results.jsonl", entry["results_sha256"])
            for _, row in report.read_results(attempt / "results.jsonl", partial=not (attempt / "completed.json").exists()):
                report.check_task(row, config)
                report.verify_traced_case(attempt, row, config, catalog)
                key = (row["mode"], row["workers"], row["index"])
                report.require(key not in seen, f"Duplicate source baseline: {key}")
                seen.add(key)
                original, source_config, source_manifest, source_results, source_trace, source_completed = \
                    _original_execution(old, attempt, row, config)
                original_regrades += int("regrade_origin" in row)
                config_hash, results_hash = trace_io._digest(source_manifest), trace_io._digest(source_results)
                name = f"{row['mode']}-w1-{report.digest((config_hash + results_hash).encode())[:16]}"
                archived, derived = stage / "baseline-provenance" / name, stage / "shards" / name
                if name not in groups:
                    regrade._copy_file(source_manifest, archived / "source-config.json")
                    regrade._copy_file(source_results, archived / "source-results.jsonl")
                    regrade._verified_bytes(archived / "source-config.json", config_hash)
                    regrade._verified_bytes(archived / "source-results.jsonl", results_hash)
                    completed_hash = None
                    if source_completed.exists():
                        regrade._copy_file(source_completed, archived / "source-completed.json")
                        completed_hash = trace_io._digest(source_completed)
                    metadata = _metadata(source_config, archived.relative_to(stage), config_hash, results_hash, completed_hash)
                    groups[name] = {"source": source_config, "metadata": metadata, "rows": []}
                regrade._copy_file(source_trace, archived / original["trace_path"])
                report.require(trace_io._digest(archived / original["trace_path"]) == original["trace_sha256"],
                               "Original baseline trace changed during copy")
                regrade._copy_file(archived / original["trace_path"], derived / original["trace_path"])
                metadata = groups[name]["metadata"]
                kept = {**original, "baseline_origin": {
                    **{field: metadata[field] for field in ("source_config", "source_config_sha256", "source_results", "source_results_sha256")},
                    "source_trace": str((archived / original["trace_path"]).relative_to(stage)),
                    "source_trace_sha256": original["trace_sha256"],
                    "classification": "unchanged_native" if row["mode"] == "native" else "unchanged_causal_w1",
                }}
                groups[name]["rows"].append(kept)
                counts["native" if row["mode"] == "native" else "hogwild-1"] += 1
            print(f"Verified baseline source {entry['shard']}", flush=True)
        for name, group in groups.items():
            derived, rows = stage / "shards" / name, group["rows"]
            config = _target_config(group["source"], [row["index"] for row in rows], hashes, group["metadata"])
            regrade._write_json(derived / "config.json", config)
            (derived / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            regrade._write_json(derived / "completed.json", {"cases": len(rows), "origin": "unchanged_baseline"})
            for row in rows:
                report.verify_traced_case(derived, row, config, catalog)
        report.require(_inventory(old) == inventory, "Source baselines changed during import")
        summary = {
            "format": 1, "source_root": str(old), "source_inventory": inventory,
            "source_inventory_sha256": inventory_hash, "target_source_sha256": hashes,
            "helper_sha256": HELPER_HASHES, "baseline_reuse_code_sha256": trace_io._digest(Path(__file__)),
            "cases": len(seen), "conditions": counts, "unwrapped_regraded_cases": original_regrades,
            "multiworker_cases": 0,
        }
        regrade._write_json(stage / SUMMARY, summary)
        os.rename(stage, new)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    summary = import_baselines(args.source, args.destination)
    print(json.dumps({field: summary[field] for field in ("cases", "conditions", "multiworker_cases")}, indent=2))


if __name__ == "__main__":
    main()
