import csv
import hashlib
import json

import pytest
from minisgl_examples.training.parallel_thinking import budget_long_report as report


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def studies(tmp_path):
    roots = [tmp_path / "low", tmp_path / "high"]
    for root, budgets, cap in zip(roots, [report.LOW, report.HIGH], [report.LOW_CAP, report.HIGH_CAP]):
        lane = root / "lanes/lane-000"
        lane.mkdir(parents=True)
        config = dict.fromkeys(report.checks.IDENTITY_FIELDS, "identical")
        config.update({k: report.checks.COMMON[k] for k in ("model", "model_revision", "dataset", "dataset_revision", "dataset_sha256", "seed")})
        config.update(context_cap=cap, answer_tokens=8, answer_sampling={"temperature": 0.0}, budgets=budgets)
        (lane / "config.json").write_text(json.dumps(config))
        plan = {"config": {"source_manifest_sha256": "a" * 64, "active_source_manifest_sha256": "b" * 64,
                           "code_sha256": {"eval.py": "c" * 64}, "budgets": list(budgets), "indices": [0, 1],
                           "conditions": [f"{mode}-w{workers}" for mode, workers in report.CONDITIONS]}}
        (root / "plan.json").write_text(json.dumps(plan))
        rows = []
        for mode, workers in report.CONDITIONS:
            for index in (0, 1):
                for budget in budgets:
                    rows.append({"index": index, "mode": mode, "workers": workers, "budget": budget, "unique_id": str(index),
                                 "gold": "17", "answer_tokens": 3, "answer_token_ids": [1, 2, 3], "correct": True,
                                 "prediction": "17", "grading_status": "correct", "stop_reason": "answer", "cue_token_ids": [20],
                                 "reused_readout": budget != budgets[0], "prompt_tokens": 22, "publication_count": 0,
                                 "reasoning_tokens": 2, "generated_tokens": 5, "effective_rounds": 2, "context_tokens": 25,
                                 "context_sha256": f"{index+workers:064x}", "source_trace_sha256": "d" * 64,
                                 "view_sha256": "e" * 64, "wall_seconds": 0.02})
        write_rows(root / "results.jsonl", rows)
        if root == roots[0]:
            (root / "complete.json").write_text(json.dumps({"completed": len(rows), "target": len(rows),
                "results_sha256": digest(root / "results.jsonl"), "plan_sha256": digest(root / "plan.json")}))
    return roots


def run(studies):
    return report.build_report(*studies, studies[1] / "report-1k-to-16k")


def test_global_cohort_cap_and_graded_rows(studies):
    old, high = studies
    path = high / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    missing = rows.pop()
    assert missing["index"] == 1 and missing["budget"] == 16000
    rows[0].update(prediction=None, answer_token_ids=list(range(8)), answer_tokens=8, correct=False,
                   grading_status="unparsed_prediction", stop_reason="max_tokens", generated_tokens=10)
    write_rows(path, rows)
    result = run(studies)
    assert result["matched_indices"] == [0]
    assert result["budgets"] == list(range(1000, 16001, 1000))
    assert len(result["cells"]) == 112 and result["completed_evaluations"] == 223
    assert result["expected_evaluations"] == 56000
    cell = next(r for r in result["cells"] if r["mode"] == "native" and r["budget"] == 9000)
    assert cell["accuracy"] == 0 and cell["answer_cap_reached_count"] == cell["unparsed_prediction_count"] == 1
    assert result["identical_context_audit"]["mismatched_extended_rows"] == 1
    assert result["identical_context_audit"]["mismatch_examples"][0]["extended_budget"] == 9000
    assert result["protocols"]["previous"]["context_cap"] == 136192
    assert result["protocols"]["extended"]["context_cap"] == 262144
    with (high / "report-1k-to-16k/tasks.csv").open() as f:
        all_rows = list(csv.DictReader(f))
    assert sum(r["in_matched_cohort"] == "True" for r in all_rows) == 112
    assert {r["input_run"] for r in all_rows} == {"previous", "extended"}


@pytest.mark.parametrize("where,field,value", [
    ("manifest", "context_cap", 250000), ("manifest", "memory_ratio", 0.99),
    ("manifest", "source_sha256", "other"), ("plan", "code_sha256", {"eval.py": "changed"}),
    ("plan", "budgets", [16000]), ("row", "gold", "18"), ("row", "source_trace_sha256", "0"*64),
    ("row", "answer_token_ids", [9]), ("row", "effective_rounds", 9001), ("row", "context_tokens", 262145)])
def test_integrity(studies, where, field, value):
    _, high = studies
    if where == "row":
        path = high / "results.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        rows[0][field] = value
        write_rows(path, rows)
    else:
        path = high / ("lanes/lane-000/config.json" if where == "manifest" else "plan.json")
        data = json.loads(path.read_text())
        (data if where == "manifest" else data["config"])[field] = value
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        run(studies)


def test_prior_results_pinned_and_no_write_into_prior_root(studies):
    low, high = studies
    with pytest.raises(ValueError, match="dedicated"):
        report.build_report(low, high, low / "report")
    path = low / "results.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Completed result bytes changed"):
        run(studies)


def test_partial_start_and_torn_tail(studies):
    _, high = studies
    path = high / "results.jsonl"
    path.write_bytes(path.read_bytes() + b'{"mode":')
    assert run(studies)["sources"]["extended"]["ignored_uncommitted_trailing_bytes"] == 8
    path.unlink()
    assert run(studies)["status"] == "progress_only"
