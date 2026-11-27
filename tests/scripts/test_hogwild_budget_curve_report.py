import csv
import hashlib
import json

import pytest
from minisgl_examples.training.parallel_thinking import budget_curve_report as curve


def row(mode="native", workers=1, index=0, budget=1000, **extra):
    return {"mode": mode, "workers": workers, "index": index, "budget": budget,
            "correct": True, "generated_tokens": budget * workers + 4, "answer_tokens": 4,
            "source_trace_sha256": f"{index:064x}", "wall_seconds": 1.5, **extra}


def cohort(indices=(0,)):
    return [row(mode, workers, index, budget) for mode, workers in curve.CONDITIONS
            for budget in curve.BUDGETS for index in indices]


def write_rows(tmp_path, rows, name="results.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(value) + "\n" for value in rows))
    return path


def test_completion_certificate_binds_plan_keys_and_result_bytes(tmp_path):
    data = [row()]
    results = write_rows(tmp_path, data)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"config": {"indices": [0], "conditions": ["native-w1"], "budgets": [1000]}}))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    complete = {"completed": 1, "target": 1, "plan_sha256": digest(plan), "results_sha256": digest(results)}
    (tmp_path / "complete.json").write_text(json.dumps(complete))
    rows, evidence = curve.load_rows(results)
    assert curve.verify_completion(tmp_path, rows, evidence)["indices"] == [0]
    results.write_bytes(results.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Completed result bytes changed"):
        curve.verify_completion(tmp_path, *curve.load_rows(results))
    plan.write_text(json.dumps({"config": {"indices": [0, 1], "conditions": ["native-w1"], "budgets": [1000]}}))
    with pytest.raises(ValueError, match="Incomplete or unexpected"):
        curve.verify_completion(tmp_path, *curve.load_rows(results))


def test_one_cohort_across_all_conditions_and_budgets_not_per_point(tmp_path):
    data = cohort((0, 1, 2))
    data = [r for r in data if (r["mode"], r["workers"], r["index"], r["budget"]) != ("hogwild", 16, 2, 8000)]
    data = [r for r in data if (r["mode"], r["workers"], r["index"], r["budget"]) != ("native", 1, 1, 1000)]
    result = curve.build_report(write_rows(tmp_path, data), tmp_path / "report")
    assert result["matched_indices"] == [0]
    assert len(result["cells"]) == 56 and all(r["matched_tasks"] == 1 for r in result["cells"])
    assert result["completed_evaluations"] == 166 and result["status"] == "matched_partial"
    tasks = list(csv.DictReader((tmp_path / "report/tasks.csv").open()))
    assert len(tasks) == 166 and sum(r["in_matched_cohort"] == "True" for r in tasks) == 56


def test_false_unparsed_answers_stay_in_denominator_and_arithmetic_is_exact(tmp_path):
    data = cohort((0, 1))
    for r in data:
        if r["index"] == 1:
            r.update(correct=False, grading_status="unparsed_prediction", prediction=None,
                     answer_tokens=8, generated_tokens=r["budget"] * r["workers"] + 8, wall_seconds=2.5)
    result = curve.build_report(write_rows(tmp_path, data), tmp_path / "report")
    for cell in result["cells"]:
        assert cell["matched_tasks"] == 2 and cell["correct"] == 1 and cell["accuracy"] == 0.5
        assert cell["total_generated_tokens"] == 2 * cell["budget"] * cell["workers"] + 12
        assert cell["mean_answer_tokens"] == 6 and cell["total_wall_seconds"] == 4
        assert cell["mean_wall_seconds"] == 2
    tasks = list(csv.DictReader((tmp_path / "report/tasks.csv").open()))
    assert sum(r["correct"] == "False" for r in tasks) == 56
    assert all(r["grading_status"] == "unparsed_prediction" for r in tasks if r["correct"] == "False")


def test_cap_and_unparsed_counts_use_matched_rows_and_expose_optional_coverage(tmp_path):
    data = cohort((0, 1, 2))
    for record in data:
        if record["index"] == 0:
            record.update(stop_reason="max_tokens", grading_status="unparsed_prediction", correct=False)
        elif record["index"] == 1:
            record.update(stop_reason="boxed_answer", grading_status="correct")
    result = curve.build_report(write_rows(tmp_path, data), tmp_path / "report")
    for cell in result["cells"]:
        assert cell["matched_tasks"] == 3
        assert cell["stop_reason_tasks"] == cell["grading_status_tasks"] == 2
        assert cell["answer_cap_reached_count"] == cell["unparsed_prediction_count"] == 1
        assert cell["answer_cap_reached_rate"] == cell["unparsed_prediction_rate"] == 0.5
        assert cell["accuracy"] == 2 / 3


def test_missing_optional_statuses_are_unknown_and_observed_no_caps_are_zero(tmp_path):
    data = cohort()
    data[0].update(stop_reason="boxed_answer", grading_status="correct")
    result = curve.build_report(write_rows(tmp_path, data), tmp_path / "report")
    first, missing = result["cells"][0], result["cells"][1]
    assert first["stop_reason_tasks"] == first["grading_status_tasks"] == 1
    assert first["answer_cap_reached_count"] == first["unparsed_prediction_count"] == 0
    assert first["answer_cap_reached_rate"] == first["unparsed_prediction_rate"] == 0
    assert missing["stop_reason_tasks"] == missing["grading_status_tasks"] == 0
    assert missing["answer_cap_reached_count"] is None and missing["answer_cap_reached_rate"] is None
    assert missing["unparsed_prediction_count"] is None and missing["unparsed_prediction_rate"] is None


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_results_rejected_even_if_identical(tmp_path, conflicting):
    first = row()
    second = {**first, "correct": not first["correct"]} if conflicting else first.copy()
    with pytest.raises(ValueError, match="Duplicate result"):
        curve.build_report(write_rows(tmp_path, [first, second]), tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_source_trace_cannot_change_between_budgets(tmp_path):
    with pytest.raises(ValueError, match="Different source traces"):
        curve.load_rows(write_rows(tmp_path, [row(), row(budget=2000, source_trace_sha256="a" * 64)]))


@pytest.mark.parametrize("updates", [
    {"mode": "native", "workers": 2}, {"workers": True}, {"index": 500}, {"budget": 1500},
    {"correct": 1}, {"answer_tokens": 9}, {"answer_tokens": -1}, {"generated_tokens": 2},
    {"wall_seconds": float("nan")}, {"wall_seconds": -1}, {"source_trace_sha256": "bad"},
])
def test_invalid_metric_or_identity_rejected(tmp_path, updates):
    with pytest.raises(ValueError):
        curve.load_rows(write_rows(tmp_path, [row(**updates)]))


def test_empty_intersection_is_progress_only(tmp_path):
    output = tmp_path / "report"
    output.mkdir()
    result = curve.build_report(write_rows(tmp_path, [row()]), output)
    assert result["status"] == "progress_only" and result["matched_tasks"] == 0
    assert result["completed_evaluations"] == 1
    assert all(r["accuracy"] is None and r["correct"] is None for r in result["cells"])
    assert sum(r["completed_tasks"] for r in result["cells"]) == 1


def test_empty_file_and_uncommitted_tail_are_pending_not_incorrect(tmp_path):
    path = write_rows(tmp_path, cohort())
    with path.open("ab") as file:
        file.write(b'{"mode":"native","workers":1')
    result = curve.build_report(path, tmp_path / "report")
    assert result["matched_tasks"] == 1 and result["completed_evaluations"] == 56
    assert result["source"]["ignored_uncommitted_trailing_bytes"] > 0
    empty = curve.build_report(write_rows(tmp_path, [], "empty.jsonl"), tmp_path / "empty")
    assert empty["completed_evaluations"] == 0 and empty["matched_tasks"] == 0


def test_committed_malformed_row_fails_instead_of_disappearing(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"bad":\n')
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        curve.load_rows(path)


def test_numeric_values_and_optional_fields_survive_csv(tmp_path):
    data = cohort()
    for r in data:
        r["correct"] = r["budget"] % 2000 == 1000
        r["diagnostic"] = {"early_eos": True}
    result = curve.build_report(write_rows(tmp_path, data), tmp_path / "report")
    for condition in curve.CONDITIONS:
        cells = [c for c in result["cells"] if (c["mode"], c["workers"]) == condition]
        assert [c["accuracy"] for c in cells] == [1, 0, 1, 0, 1, 0, 1, 0]
    tasks = list(csv.DictReader((tmp_path / "report/tasks.csv").open()))
    assert json.loads(tasks[0]["diagnostic"]) == {"early_eos": True}


def test_native_reference_is_matched_and_explicitly_reference(tmp_path):
    path = write_rows(tmp_path, cohort((0, 1)))
    reference = write_rows(tmp_path, [
        {"mode": "native", "workers": 1, "index": 0, "correct": True},
        {"mode": "native", "workers": 1, "index": 1, "correct": False},
        {"mode": "native", "workers": 1, "index": 2, "correct": True},
    ], "reference.jsonl")
    result = curve.build_report(path, tmp_path / "report", native_reference=reference)
    ref = result["native_16k_reference"]
    assert ref["accuracy"] == 0.5 and ref["tasks"] == 2 and ref["paired"]
    assert "reference" in ref["label"] and "upper" not in ref["label"]


def test_summary_reference_is_labeled_separate_cohort_and_missing_paired_reference_rejected(tmp_path):
    path = write_rows(tmp_path, cohort((0, 1)))
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"accuracy": 0.98, "tasks": 500, "budget": 16384}))
    result = curve.build_report(path, tmp_path / "report", native_reference=reference)
    assert result["native_16k_reference"]["paired"] is False
    assert "separate cohort" in result["native_16k_reference"]["label"]
    reference.write_text(json.dumps({"accuracy": 0.98, "tasks": 500, "mode": "hogwild", "workers": 1}))
    with pytest.raises(ValueError, match="ordinary native"):
        curve.build_report(path, tmp_path / "wrong-reference", native_reference=reference)
    short = write_rows(tmp_path, [{"mode": "native", "workers": 1, "index": 0, "correct": True}], "short.jsonl")
    with pytest.raises(ValueError, match="missing tasks"):
        curve.build_report(path, tmp_path / "bad-reference", native_reference=short)






def test_cli_emits_progress_and_rejects_bad_input(tmp_path, capsys):
    path = write_rows(tmp_path, [])
    curve.main([str(path), "--output", str(tmp_path / "report")])
    assert "progress_only" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        curve.main([str(tmp_path / "missing"), "--output", str(tmp_path / "missing-output")])
    assert error.value.code == 1
