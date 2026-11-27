"""Saved-case migration preserves execution and reruns changed finisher branches."""

import gzip
import json

import pytest

from minisgl_examples.training.parallel_thinking import regrade, report, sweep, trace_io
from minisgl_examples.training.parallel_thinking.math500 import GRADING_POLICY
from tests.scripts.test_hogwild_sweep_protocol import manifest, traced, v3


class TextTokenizer:
    def decode(self, ids, **kwargs):
        return "".join(chr(token) for token in ids if token != 248046)


def record(index, *, mode="hogwild", workers=2, terminal=True, boxed=True, finisher=False):
    text = "reasoning " + (r"\boxed{7}" if boxed else "unfinished")
    stream = list(map(ord, text)) + ([248046] if terminal else [])
    names = ["Alice", "Bob"][:workers]
    streams = {name: list(stream) for name in names}
    rounds = len(stream)
    prediction = "999" if index == 0 else "7"
    row = {
        "mode": mode, "workers": workers, "index": index, "unique_id": f"test/{index}",
        "gold": "7", "prediction": prediction, "grading_status": "correct" if prediction == "7" else "incorrect",
        "correct": prediction == "7", "generated_tokens": rounds * workers + 2 * finisher,
        "reasoning_tokens": rounds * workers, "finisher_tokens": 2 * finisher,
        "decoding_steps": rounds, "total_decoding_steps": rounds + 2 * finisher,
        "used_finisher": finisher, "stop_reason": "eos" if terminal else "max_steps",
        "eos_token_id": 248046 if terminal else None, "eos_worker": names[0] if terminal else None,
        "finisher_stop_reason": "answer" if finisher else "existing_answer", "finisher_eos_token_id": None,
        "single_worker_layout": "causal" if workers == 1 and mode == "hogwild" else None,
        "multi_worker_layout": "native" if workers > 1 else None,
        "multi_worker_answer_cue": "qwen" if workers > 1 else None,
        "finisher_mode": "cached" if mode == "hogwild" else "none", "published_paragraphs": 0,
        "per_worker_tokens": {name: rounds for name in names}, "wall_seconds": 1.0,
    }
    if mode == "native":
        row["finisher_stop_reason"] = "not_applicable"
    trace = {
        "metrics": {key: row[key] for key in trace_io.METRICS},
        "rollout": {
            "workers": names, "worker_token_ids": streams, "eos_worker": row["eos_worker"],
            "eos_token_ids": [248046, 248044], "boxed_answer": prediction,
            "finisher": {"used": finisher, "sampled_token_ids": [ord("7"), ord("}")] if finisher else []},
        },
        "engine": {"requests": [{"cache_view": [[1, 20], [2, rounds]], "input_ids": [4]}]},
    }
    return row, trace


def write_source(root, index, **options):
    row, trace = record(index, **options)
    workers, mode = row["workers"], row["mode"]
    cfg = traced(v3(manifest("causal" if workers == 1 else None, indices=[index], workers=(workers,)),
                    layout="native" if workers > 1 else None, cue="qwen" if workers > 1 else None))
    cfg.update(source_sha256=regrade.SOURCE_HASHES.copy(), commit=report.REFERENCE_COMMIT, grading=GRADING_POLICY,
               context_cap=min(262144, 16384 * workers + 8192))
    cfg.pop("answer_selection_policy", None)
    if mode == "native":
        cfg.update(report.PROTOCOL["native"], mode="native", single_worker_layout=None)
    directory = root / "shards" / f"case-{index}"
    path = directory / "traces" / f"{mode}-w{workers}-i{index:03d}.json.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(json.dumps(trace).encode()))
    row.update(trace_path=str(path.relative_to(directory)), trace_sha256=trace_io._digest(path), trace_bytes=path.stat().st_size)
    (directory / "config.json").write_text(json.dumps(cfg))
    (directory / "results.jsonl").write_text(json.dumps(row) + "\n")
    (directory / "completed.json").write_text(json.dumps({"cases": 1}))
    return directory, row, trace


@pytest.fixture
def migration_environment(monkeypatch):
    monkeypatch.setattr(regrade, "load_tokenizer", lambda: TextTokenizer())
    monkeypatch.setattr(regrade, "load_catalog", lambda: {})
    monkeypatch.setattr(sweep, "source_hashes", lambda save_traces: {**regrade.SOURCE_HASHES, "target": "a" * 64})

    def verify_source(directory, row, config, catalog):
        path = directory / row["trace_path"]
        report.require(trace_io._digest(path) == row["trace_sha256"], "Trace checksum mismatch")
        with gzip.open(path, "rt") as stream:
            report.require(json.load(stream)["metrics"] == {key: row[key] for key in trace_io.METRICS},
                           "Trace metrics mismatch")

    monkeypatch.setattr(report, "verify_traced_case", verify_source)


@pytest.mark.parametrize("terminal,boxed,finisher,classification", [
    (True, True, False, "selected_terminal_answer"),
    (False, True, False, "missing_cached_finisher"),
    (True, False, False, "missing_cached_finisher"),
    (True, False, True, "existing_finisher"),
    (False, True, True, "existing_finisher"),
    (True, True, True, "finisher_no_longer_needed"),
])
def test_classification_precedes_grading(terminal, boxed, finisher, classification):
    row, trace = record(0, terminal=terminal, boxed=boxed, finisher=finisher)
    del row["gold"]
    assert regrade.classify_case(row, trace, TextTokenizer())[0] == classification


def test_projection_changes_only_answer_and_grade():
    row, trace = record(0)
    projected_row, projected_trace = regrade.project_case(row, trace, TextTokenizer())
    assert projected_row["prediction"] == "7" and projected_row["correct"] is True
    assert trace["rollout"]["boxed_answer"] == "999" and "answer_selection" not in trace["rollout"]
    assert projected_trace["engine"] == trace["engine"]
    assert projected_trace["rollout"]["worker_token_ids"] == trace["rollout"]["worker_token_ids"]
    assert projected_trace["rollout"]["finisher"] == trace["rollout"]["finisher"]
    for key in row:
        if key not in ("prediction", "correct", "grading_status"):
            assert projected_row[key] == row[key]


def test_atomic_migration_retry_and_provenance(tmp_path, migration_environment):
    old, new = tmp_path / "held", tmp_path / "corrected"
    write_source(old, 0)
    write_source(old, 1, terminal=False)
    write_source(old, 2, boxed=False, finisher=True)
    write_source(old, 3, finisher=True)
    write_source(old, 4, mode="native", workers=1)
    write_source(old, 5, workers=1)
    (old / "quality-hold.json").write_text('{"reason":"selector"}')
    before = {str(path.relative_to(old)): path.read_bytes() for path in old.rglob("*") if path.is_file()}

    summary = regrade.import_regraded(old, new)
    assert (summary["source_cases"], summary["kept_cases"], summary["rerun_cases"]) == (6, 4, 2)
    assert {row["index"] for row in summary["rerun"]} == {1, 3}
    assert len(summary["accuracy_flips"]) == 1 and summary["accuracy_flips"][0]["index"] == 0
    assert not (new / "quality-hold.json").exists()
    assert (new / "provenance" / "source-quality-hold.json").is_file()
    assert regrade.import_regraded(old, new) == summary
    for path, raw in before.items():
        assert (old / path).read_bytes() == raw
    assert len(list(new.rglob("config.json"))) == len(list(new.rglob("results.jsonl"))) == 4
    for attempt in (new / "shards").iterdir():
        row = json.loads((attempt / "results.jsonl").read_text())
        cfg = json.loads((attempt / "config.json").read_text())
        regrade.verify_regraded_origin(attempt, row, cfg)
        origin = row["regrade_origin"]
        assert (new / origin["source_trace"]).is_file()
        if row["workers"] == 1:
            assert row["trace_sha256"] == origin["source_trace_sha256"]


def test_failure_does_not_publish_partial_migration(tmp_path, migration_environment):
    old, new = tmp_path / "held", tmp_path / "corrected"
    write_source(old, 0)
    directory, _, _ = write_source(old, 1)
    cfg = json.loads((directory / "config.json").read_text())
    cfg["source_sha256"]["scripts/minisgl_examples/training/parallel_thinking/rollout.py"] = "f" * 64
    (directory / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="Unexpected source build"):
        regrade.import_regraded(old, new)
    assert not new.exists()
    assert not list(tmp_path.glob(".corrected.regrade-*"))


def test_invalid_derived_trace_aborts_before_publishing(tmp_path, migration_environment, monkeypatch):
    old, new = tmp_path / "held", tmp_path / "corrected"
    write_source(old, 0)
    write_trace = regrade._write_trace

    def corrupt_derived_metrics(path, trace):
        broken = {**trace, "metrics": {**trace["metrics"], "prediction": "corrupted"}}
        write_trace(path, broken)

    monkeypatch.setattr(regrade, "_write_trace", corrupt_derived_metrics)
    with pytest.raises(ValueError, match="Trace metrics mismatch"):
        regrade.import_regraded(old, new)
    assert not new.exists()
    assert not list(tmp_path.glob(".corrected.regrade-*"))


@pytest.mark.parametrize("filename", ["config.json", "hogwild-w2-i000.json.gz"])
def test_source_mutation_during_copy_aborts_import(tmp_path, migration_environment, monkeypatch, filename):
    old, new = tmp_path / "held", tmp_path / "corrected"
    write_source(old, 0)
    copy_file = regrade._copy_file

    def mutate_before_copy(source, destination):
        if source.name == filename:
            source.write_bytes(source.read_bytes() + b" ")
        copy_file(source, destination)

    monkeypatch.setattr(regrade, "_copy_file", mutate_before_copy)
    with pytest.raises(ValueError, match="[Cc]hanged"):
        regrade.import_regraded(old, new)
    assert not new.exists()


@pytest.mark.parametrize("mutation", ["row_metric", "engine", "source_config", "classification"])
def test_origin_verification_rejects_tampering(tmp_path, migration_environment, mutation):
    old, new = tmp_path / "held", tmp_path / "corrected"
    write_source(old, 0)
    regrade.import_regraded(old, new)
    attempt = new / "shards" / "case-0"
    row = json.loads((attempt / "results.jsonl").read_text())
    cfg = json.loads((attempt / "config.json").read_text())
    if mutation == "row_metric":
        row["generated_tokens"] += 1
    elif mutation == "engine":
        path = attempt / row["trace_path"]
        with gzip.open(path, "rt") as stream:
            trace = json.load(stream)
        trace["engine"]["requests"][0]["input_ids"] = [99]
        path.write_bytes(gzip.compress(json.dumps(trace).encode()))
        row.update(trace_sha256=trace_io._digest(path), trace_bytes=path.stat().st_size)
    elif mutation == "source_config":
        (new / row["regrade_origin"]["source_config"]).write_text("{}")
    else:
        row["regrade_origin"]["classification"] = "unchanged"
    with pytest.raises(ValueError):
        regrade.verify_regraded_origin(attempt, row, cfg)
