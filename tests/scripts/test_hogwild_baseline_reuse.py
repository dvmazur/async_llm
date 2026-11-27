"""Baseline reuse preserves original execution and excludes every multi-worker case."""

import json

import pytest

from minisgl_examples.training.parallel_thinking import baseline_reuse, regrade, report, sweep, trace_io
from tests.scripts.test_hogwild_regrade import migration_environment, write_source


@pytest.fixture
def baseline_environment(migration_environment, monkeypatch):
    hashes = {**baseline_reuse.SELECTOR_SOURCES, "target": "a" * 64}
    monkeypatch.setattr(baseline_reuse, "audit_target", lambda: hashes)
    original_verify = report.verify_traced_case

    def verify(directory, row, config, catalog):
        original_verify(directory, row, config, catalog)
        if "baseline_origin" in row:
            baseline_reuse.verify_baseline_origin(directory, row, config)
        elif "regrade_origin" in row:
            regrade.verify_regraded_origin(directory, row, config)

    monkeypatch.setattr(report, "verify_traced_case", verify)
    return hashes


def test_baseline_import_is_exact_atomic_and_idempotent(tmp_path, baseline_environment):
    old, new = tmp_path / "held", tmp_path / "publication-fixed"
    _, native, _ = write_source(old, 0, mode="native", workers=1)
    _, solo, _ = write_source(old, 1, workers=1)
    write_source(old, 2, workers=2)
    before = {str(path.relative_to(old)): path.read_bytes() for path in old.rglob("*") if path.is_file()}

    summary = baseline_reuse.import_baselines(old, new)
    assert summary["cases"] == 2 and summary["conditions"] == {"native": 1, "hogwild-1": 1}
    assert summary["multiworker_cases"] == 0
    assert baseline_reuse.import_baselines(old, new) == summary
    imported = []
    for path in new.glob("shards/*/results.jsonl"):
        row = json.loads(path.read_text())
        original = native if row["mode"] == "native" else solo
        assert {key: value for key, value in row.items() if key != "baseline_origin"} == original
        origin = row["baseline_origin"]
        assert (path.parent / row["trace_path"]).read_bytes() == (new / origin["source_trace"]).read_bytes()
        config = json.loads(path.with_name("config.json").read_text())
        assert config["publication_policy"] is None and config["answer_selection_policy"] is None
        baseline_reuse.verify_baseline_origin(path.parent, row, config)
        imported.append(row)
    assert len(imported) == 2
    assert len(list(new.rglob("config.json"))) == 2
    assert len(list(new.rglob("results.jsonl"))) == 2
    assert {str(path.relative_to(old)): path.read_bytes() for path in old.rglob("*") if path.is_file()} == before


def test_regraded_baseline_is_unwrapped_to_original_execution(tmp_path, baseline_environment, monkeypatch):
    original, selector, fixed = tmp_path / "original", tmp_path / "selector", tmp_path / "fixed"
    _, row, _ = write_source(original, 1, workers=1)
    monkeypatch.setattr(sweep, "source_hashes", lambda save_traces: baseline_reuse.SELECTOR_SOURCES.copy())
    regrade.import_regraded(original, selector)
    summary = baseline_reuse.import_baselines(selector, fixed)
    assert summary["unwrapped_regraded_cases"] == 1
    path = next(fixed.glob("shards/*/results.jsonl"))
    copied = json.loads(path.read_text())
    assert "regrade_origin" not in copied
    assert {key: value for key, value in copied.items() if key != "baseline_origin"} == row
    archived = fixed / copied["baseline_origin"]["source_results"]
    assert json.loads(archived.read_text()) == row
    baseline_reuse.verify_baseline_origin(path.parent, copied, json.loads(path.with_name("config.json").read_text()))


@pytest.mark.parametrize("mutation", ["scaffold", "source_hash"])
def test_unaudited_baseline_is_rejected_without_publishing(tmp_path, baseline_environment, mutation):
    old, new = tmp_path / "held", tmp_path / "fixed"
    directory, _, _ = write_source(old, 1, workers=1)
    config = json.loads((directory / "config.json").read_text())
    if mutation == "scaffold":
        config.update(single_worker_layout="scaffold", header_context="causal", reminder_interval=1024)
    else:
        config["source_sha256"][baseline_reuse.PREFIX + "native.py"] = "f" * 64
    (directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        baseline_reuse.import_baselines(old, new)
    assert not new.exists()


@pytest.mark.parametrize("mutation", ["row", "trace", "source", "classification"])
def test_baseline_origin_rejects_changed_execution(tmp_path, baseline_environment, mutation):
    old, new = tmp_path / "held", tmp_path / "fixed"
    write_source(old, 1, workers=1)
    baseline_reuse.import_baselines(old, new)
    path = next(new.glob("shards/*/results.jsonl"))
    row, config = json.loads(path.read_text()), json.loads(path.with_name("config.json").read_text())
    if mutation == "row":
        row["generated_tokens"] += 1
    elif mutation == "trace":
        (path.parent / row["trace_path"]).write_bytes(b"changed")
    elif mutation == "source":
        (new / row["baseline_origin"]["source_results"]).write_text("{}\n")
    else:
        row["baseline_origin"]["classification"] = "unchanged_native"
    with pytest.raises(ValueError):
        baseline_reuse.verify_baseline_origin(path.parent, row, config)


def test_invalid_copy_is_detected_before_publication(tmp_path, baseline_environment, monkeypatch):
    old, new = tmp_path / "held", tmp_path / "fixed"
    write_source(old, 1, workers=1)
    original_copy = regrade._copy_file

    def corrupt_copy(source, destination):
        original_copy(source, destination)
        if destination.suffix == ".gz" and "shards" in destination.parts:
            destination.write_bytes(b"invalid derived copy")

    monkeypatch.setattr(regrade, "_copy_file", corrupt_copy)
    with pytest.raises(ValueError):
        baseline_reuse.import_baselines(old, new)
    assert not new.exists()
    assert not list(tmp_path.glob(".fixed.baselines-*"))


@pytest.mark.parametrize("mutation", ["none", "native", "helper", "driver"])
def test_target_audit_permits_only_multiplayer_publication_change(tmp_path, monkeypatch, mutation):
    actual_root = sweep.ROOT
    directory = tmp_path / baseline_reuse.PREFIX
    directory.mkdir(parents=True)
    for name in ("eval.py", "rollout.py"):
        (directory / name).write_bytes((actual_root / baseline_reuse.PREFIX / name).read_bytes())
    hashes = baseline_reuse.SELECTOR_SOURCES.copy()
    if mutation == "native":
        hashes[baseline_reuse.PREFIX + "native.py"] = "f" * 64
    elif mutation == "helper":
        p = directory / "rollout.py"
        p.write_text(p.read_text().replace("class MathFormatting:", "class MathFormatting:\n    changed = True"))
    elif mutation == "driver":
        p = directory / "eval.py"
        p.write_text(p.read_text() + "\nraise RuntimeError('unexpected driver change')\n")
    monkeypatch.setattr(sweep, "ROOT", tmp_path)
    monkeypatch.setattr(sweep, "current_commit", lambda: report.REFERENCE_COMMIT)
    monkeypatch.setattr(sweep, "source_hashes", lambda save_traces: hashes)
    monkeypatch.setattr(baseline_reuse.subprocess, "check_output", lambda *args, **kwargs: "hogwild-multi-workers\n")
    monkeypatch.setattr(baseline_reuse.subprocess, "run", lambda *args, **kwargs: None)
    if mutation == "none":
        assert baseline_reuse.audit_target() == hashes
    else:
        with pytest.raises(ValueError):
            baseline_reuse.audit_target()
