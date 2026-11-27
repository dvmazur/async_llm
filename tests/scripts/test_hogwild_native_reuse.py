"""Offline checks for importing native results into a corrected Hogwild sweep."""

import csv
import hashlib
import json
from pathlib import Path

import pytest
from minisgl_examples.training.parallel_thinking import math500, report, sweep

CATALOG = {0: ("test/algebra/0.json", "1"), 1: ("test/algebra/1.json", "2")}
MODULES = Path("scripts/minisgl_examples/training/parallel_thinking")


def source_path(name):
    return str(MODULES / name) if name.endswith(".py") else name


@pytest.fixture
def runs(tmp_path, monkeypatch):
    old = tmp_path / "historical"
    new = tmp_path / "corrected"
    (old / "shards").mkdir(parents=True)
    (new / "shards").mkdir(parents=True)
    original = {source_path(name): digest for name, digest in report.REFERENCE_SOURCES.items()}
    target = {**original, source_path("eval.py"): "e" * 64, source_path("rollout.py"): "b" * 64,
              source_path("single_worker.py"): "a" * 64}
    monkeypatch.setattr(report, "TARGET_EVAL_SHA256", target[source_path("eval.py")])
    monkeypatch.setattr(report, "load_native_catalog", lambda: CATALOG)
    catalog = json.dumps(list(CATALOG.values()), ensure_ascii=False, separators=(",", ":")).encode()
    monkeypatch.setattr(report, "CATALOG_SHA256", hashlib.sha256(catalog).hexdigest())
    monkeypatch.setattr(sweep, "current_commit", lambda: report.REFERENCE_COMMIT)
    return old, new, original, target


def native_manifest(hashes, indices, *, current=False):
    cfg = {
        **report.COMMON, **report.PROTOCOL["native"], "mode": "native", "workers": [1],
        "indices": list(indices), "context_cap": 24576, "native_context": 262144,
        "cuda_graphs": True, "branch": "hogwild-multi-workers", "commit": report.REFERENCE_COMMIT,
        "grading": math500.GRADING_POLICY, "source_sha256": hashes,
    }
    if current:
        cfg["single_worker_layout"] = None
    return cfg


def native_row(index, *, current=False):
    uid, gold = CATALOG[index]
    row = {
        "mode": "native", "workers": 1, "index": index, "unique_id": uid, "gold": gold,
        "prediction": gold, "correct": True, "grading_status": "correct", "generated_tokens": 3,
        "reasoning_tokens": 3, "finisher_tokens": 0, "decoding_steps": 3, "total_decoding_steps": 3,
        "used_finisher": False, "stop_reason": "eos", "eos_token_id": 248046,
        "finisher_stop_reason": "not_applicable", "finisher_eos_token_id": None,
        "published_paragraphs": 0, "per_worker_tokens": {"Qwen": 3}, "wall_seconds": 0.1,
    }
    if current:
        row["single_worker_layout"] = None
    return row


def write_attempt(root, name, manifest, rows):
    path = root / "shards" / name
    path.mkdir()
    (path / "config.json").write_text(json.dumps(manifest) + "\n")
    (path / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_import_freezes_native_only_is_idempotent_and_survives_source_removal(runs):
    old, new, original, target = runs
    native = write_attempt(old, "native-partial", native_manifest(original, [0, 1]), [native_row(0)])
    hogwild = {**native_manifest(original, [1]), **report.PROTOCOL["hogwild"], "mode": "hogwild"}
    write_attempt(old, "legacy-one-worker", hogwild, [{**native_row(1), "mode": "hogwild"}])

    sweep.import_native(old, new, target)
    reused = new / "reused-native"
    keys, provenance = report.verify_native_reuse(new, target)
    assert keys == {("native", 1, 0)}
    assert provenance["source_run"] == str(old.resolve())
    assert provenance["target_source_sha256"] == target
    assert sum(shard["cases"] for shard in provenance["shards"]) == 1
    for filename in ("config.json", "results.jsonl"):
        assert (reused / native.relative_to(old) / filename).read_bytes() == (native / filename).read_bytes()
    assert not list(reused.rglob("*legacy-one-worker*"))
    with pytest.raises(ValueError):
        report.report(new, partial=True, only_one_worker=True)
    snapshot = (reused / "provenance.json").read_bytes()
    sweep.import_native(old, new, target)
    assert (reused / "provenance.json").read_bytes() == snapshot

    old.rename(old.with_name("source-moved"))
    assert report.verify_native_reuse(new, target) == (keys, provenance)


@pytest.mark.parametrize("problem", ["source_hash", "commit", "unique_id", "gold", "grade", "hogwild_row"])
def test_import_rejects_incompatible_or_unverifiable_results(runs, problem):
    old, new, original, target = runs
    cfg, row = native_manifest(original, [0]), native_row(0)
    if problem == "source_hash":
        cfg["source_sha256"] = {**original, source_path("rollout.py"): "0" * 64}
    elif problem == "commit":
        cfg["commit"] = "0" * 40
    elif problem == "unique_id":
        row["unique_id"] = CATALOG[1][0]
    elif problem == "gold":
        row["gold"] = row["prediction"] = "9"
    elif problem == "grade":
        row["correct"], row["grading_status"] = False, "incorrect"
    else:
        row["mode"] = "hogwild"
    write_attempt(old, "native-one", cfg, [row])
    with pytest.raises(ValueError):
        sweep.import_native(old, new, target)
    assert not (new / "reused-native").exists()


@pytest.mark.parametrize("filename", ["config.json", "results.jsonl"])
def test_verification_detects_post_import_byte_changes(runs, filename):
    old, new, original, target = runs
    native = write_attempt(old, "native", native_manifest(original, [0]), [native_row(0)])
    sweep.import_native(old, new, target)
    path = new / "reused-native" / native.relative_to(old) / filename
    # Keep the JSON semantically identical to exercise file fingerprinting.
    path.write_bytes(b" " + path.read_bytes())
    with pytest.raises(ValueError):
        report.verify_native_reuse(new, target)


def test_overlapping_historical_attempts_fail_atomically(runs):
    old, new, original, target = runs
    for attempt in ["native-original", "native-retry"]:
        write_attempt(old, attempt, native_manifest(original, [0]), [native_row(0)])
    with pytest.raises(ValueError):
        sweep.import_native(old, new, target)
    assert not (new / "reused-native").exists()


@pytest.mark.parametrize("version", [2, 3])
def test_partial_report_counts_verified_reuse_and_new_native_once(runs, version):
    old, new, original, target = runs
    write_attempt(old, "native-original", native_manifest(original, [0]), [native_row(0)])
    sweep.import_native(old, new, target)
    config, row = native_manifest(target, [1], current=True), native_row(1, current=True)
    if version == 3:
        options = {"multi_worker_layout": None, "multi_worker_answer_cue": None}
        config.update(options, pending_eos_in_finisher=False)
        row.update(options, finisher_mode="none")
    write_attempt(new, "native-new", config, [row])
    payload = report.report(new, partial=True)
    assert payload["native_reuse"]["source_run"] == str(old.resolve())
    assert payload["completed_cases"] == 2
    assert payload["source"]["manifest_version"] == version
    native = next(group for group in payload["conditions"] if group["condition"] == "native")
    assert (native["completed_cases"], native["correct"], native["sum_generated_tokens"]) == (2, 2, 6)
    with (new / "report" / "tasks.csv").open() as stream:
        tasks = sorted(csv.DictReader(stream), key=lambda task: int(task["index"]))
    assert [task["unique_id"] for task in tasks] == [CATALOG[0][0], CATALOG[1][0]]
    assert [task["result_origin"] for task in tasks] == ["reused_native", "executed"]
    assert "native-original" in tasks[0]["source_manifest"]

    write_attempt(new, "native-overlap", {**config, "indices": [0]}, [{**row, **native_row(0, current=True)}])
    with pytest.raises(ValueError):
        report.report(new, partial=True)


def test_verification_accepts_frozen_v2_and_audited_current_native_driver(runs):
    _, _, _, target = runs
    for digest in (report.V2_EVAL_SHA256, report.TARGET_EVAL_SHA256):
        report.validate_target_sources({**target, source_path("eval.py"): digest})
    with pytest.raises(ValueError, match="Unaudited native"):
        report.validate_target_sources({**target, source_path("eval.py"): "0" * 64})
    new_eval, new_native = report.OPTIONAL_TRACE_NATIVE
    report.validate_target_sources({**target, source_path("eval.py"): new_eval, source_path("native.py"): new_native})
    with pytest.raises(ValueError, match="Unaudited native"):
        report.validate_target_sources({**target, source_path("eval.py"): new_eval})


def test_without_reuse_reports_no_imported_keys(runs):
    _, new, _, target = runs
    assert report.verify_native_reuse(new, target) == (set(), None)


def test_reuse_rejects_changed_catalog_and_different_target_build(runs):
    old, new, original, target = runs
    write_attempt(old, "native", native_manifest(original, [0]), [native_row(0)])
    sweep.import_native(old, new, target)
    with pytest.raises(ValueError):
        report.verify_native_reuse(new, {**target, "run_hogwild.sh": "c" * 64})
    catalog = new / "reused-native" / "catalog.json"
    catalog.write_bytes(catalog.read_bytes() + b" ")
    with pytest.raises(ValueError):
        report.verify_native_reuse(new, target)
