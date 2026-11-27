import csv
import json

import pytest
from minisgl_examples.training.parallel_thinking import budget_short_report as report


def config():
    result = dict.fromkeys(report.IDENTITY_FIELDS, "same")
    for key in ("model", "model_revision", "dataset", "dataset_revision", "dataset_sha256", "seed"):
        result[key] = report.COMMON[key]
    return {**result, "answer_tokens": 8, "context_cap": 136192, "answer_sampling": {"temperature": 0.0}}


def record(mode, workers, index, budget):
    return {"mode": mode, "workers": workers, "index": index, "budget": budget, "correct": True,
            "generated_tokens": 305, "answer_tokens": 5, "reasoning_tokens": 300, "wall_seconds": 0.5,
            "effective_rounds": min(budget, 200), "source_trace_sha256": f"{index * 20 + workers:064x}",
            "context_sha256": f"{index + 15:064x}", "answer_token_ids": [11, 12, 13, 14, 15],
            "prediction": "12", "stop_reason": "answer", "grading_status": "correct"}


def create(root, budgets, indices=(0, 1)):
    manifest = root / "lanes/lane-000/config.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({**config(), "budgets": list(budgets)}))
    rows = [record(mode, workers, index, budget) for mode, workers in report.CONDITIONS
            for index in indices for budget in budgets]
    write(root, rows)
    return rows


def write(root, rows):
    (root / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def studies(tmp_path):
    old, new = tmp_path / "previous", tmp_path / "short"
    create(old, report.frozen.BUDGETS)
    create(new, report.SHORT_BUDGETS)
    return old, new


def build(studies, **options):
    old, new = studies
    return report.build_report(old, new, new / "report", **options)


def test_exact_budget_join_global_cohort_and_counts(studies):
    old, new = studies
    selected = [json.loads(line) for line in (new / "results.jsonl").read_text().splitlines()]
    selected.pop()  # Only one of 49 cells is missing index 1.
    selected[0].update(correct=False, grading_status="unparsed_prediction", stop_reason="max_tokens",
                       prediction=None, answer_tokens=8, answer_token_ids=list(range(8)), generated_tokens=308)
    write(new, selected)
    result = build(studies)
    assert result["matched_indices"] == [0] and result["status"] == "matched_partial"
    assert result["completed_evaluations"] == 97 and result["expected_evaluations"] == 24500
    assert result["budgets"] == [256, 512, 1000, 1024, 2000, 3000, 4000] and len(result["cells"]) == 49
    first = result["cells"][0]
    assert first["accuracy"] == 0 and first["matched_tasks"] == first["answer_cap_reached_count"] == 1
    assert first["unparsed_prediction_count"] == 1 and first["total_generated_tokens"] == 308
    with (new / "report/tasks.csv").open() as stream:
        tasks = list(csv.DictReader(stream))
    assert len(tasks) == 97 and sum(row["in_matched_cohort"] == "True" for row in tasks) == 49
    assert {int(row["budget"]) for row in tasks} == set(report.BUDGETS)
    assert {row["input_run"] for row in tasks if int(row["budget"]) == 1000} == {"previous"}
    assert {row["input_run"] for row in tasks if int(row["budget"]) == 1024} == {"short"}


def test_equal_context_mismatch_visible_but_raw_observations_preserved(studies):
    _, new = studies
    rows = [json.loads(line) for line in (new / "results.jsonl").read_text().splitlines()]
    rows[0].update(prediction="99", correct=False, grading_status="incorrect", reused_readout=False, wall_seconds=9)
    rows[1].update(reused_readout=True, wall_seconds=0.01)
    write(new, rows)
    result = build(studies)
    audit = result["identical_context_audit"]
    assert audit["identical_contexts"] == 14 and audit["short_rows_with_identical_context"] == 42
    assert audit["mismatched_short_rows"] == audit["mismatched_contexts"] == 1
    assert audit["mismatch_examples"][0]["short_budget"] == 256
    assert result["cells"][0]["accuracy"] == 0.5


@pytest.mark.parametrize("field,new_value", [("model_revision", "other"), ("context_cap", 24576),
                                             ("grading", "different"), ("source_sha256", {"runtime": "other"})])
def test_incompatible_run_manifest_rejected(studies, field, new_value):
    _, new = studies
    path = new / "lanes/lane-000/config.json"
    data = json.loads(path.read_text())
    data[field] = new_value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=field):
        build(studies)


@pytest.mark.parametrize("field", ["source_trace_sha256", "unique_id", "gold"])
def test_conflicting_source_task_identity_rejected(studies, field):
    old, new = studies
    for root, value in ((old, "a" * 64), (new, "b" * 64)):
        rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
        for row in rows:
            if row["index"] == 0:
                row[field] = value
        write(root, rows)
    with pytest.raises(ValueError, match="Conflicting"):
        build(studies)


@pytest.mark.parametrize("change", [{"answer_token_ids": [11]}, {"generated_tokens": 306}, {"effective_rounds": 257},
                                   {"grading_status": "incorrect"}, {"stop_reason": "max_tokens"},
                                   {"source_trace_sha256": "bad"}, {"context_sha256": "bad"},
                                   {"wall_seconds": -1}, {"correct": 1}, {"budget": 1000}])
def test_corrupted_short_rows_rejected(studies, change):
    _, new = studies
    row = record("native", 1, 0, 256)
    write(new, [{**row, **change}])
    with pytest.raises(ValueError):
        report.load_short_rows(new / "results.jsonl")


def test_short_adapter_supports_core_schema_and_partial_tail_but_rejects_committed_duplicate(studies):
    _, new = studies
    row = {key: value for key, value in record("native", 1, 1, 512).items() if key in report.frozen.REQUIRED}
    write(new, [row])
    path = new / "results.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"mode":')
    rows, source = report.load_short_rows(path)
    assert rows == {("native", 1, 1, 512): row} and source["ignored_uncommitted_trailing_bytes"] == 8
    write(new, [row, row])
    with pytest.raises(ValueError, match="Duplicate result"):
        report.load_short_rows(path)


def test_precase_progress_without_manifests_and_empty_overlap(studies):
    _, new = studies
    (new / "results.jsonl").unlink()
    (new / "lanes/lane-000/config.json").unlink()
    result = build(studies)
    assert result["status"] == "progress_only" and result["matched_tasks"] == 0
    assert result["sources"]["protocol_checked_across_runs"] is False
    assert all(c["accuracy"] is None for c in result["cells"])


def test_nonempty_rows_require_protocol_and_cannot_overwrite_old_report(studies):
    old, new = studies
    (new / "lanes/lane-000/config.json").unlink()
    with pytest.raises(ValueError, match="no lane manifest"):
        build(studies)
    with pytest.raises(ValueError, match="outside the frozen"):
        report.build_report(old, new, old / "report")


def test_plan_runtime_subset_and_source_manifests_are_identical(studies):
    old, new = studies
    previous = {"source_manifest_sha256": "original", "active_source_manifest_sha256": "active",
                "code_sha256": {"engine.py": "engine", "budget_views.py": "views"}}
    extended = {**previous, "code_sha256": {**previous["code_sha256"], "budget_short_report.py": "extension"}}
    for root, settings in ((old, previous), (new, extended)):
        (root / "plan.json").write_text(json.dumps({"config": settings}))
    assert report.compare_plans(old, new)["shared_runtime_files"] == 2
    extended["code_sha256"]["engine.py"] = "changed"
    (new / "plan.json").write_text(json.dumps({"config": extended}))
    with pytest.raises(ValueError, match="runtime or engine"):
        build(studies)
