import hashlib
import json
from argparse import Namespace

import pytest

from minisgl_examples.training.parallel_thinking import budget_long_sweep as sweep
from tests.scripts.test_hogwild_budget_short_sweep import case, catalog_for, serialized


def options(source, output):
    return Namespace(source_root=source, output=output, gpus=[0, 2], indices=list(range(500)),
                     conditions=list(sweep.lanes.CONDITIONS), budgets=list(sweep.BUDGETS),
                     wait_for_source=False, max_prefill_rows=256, memory_ratio=0.75)


def test_high_context_command_and_unchanged_runtime(tmp_path, monkeypatch):
    opts = options(tmp_path / "source", tmp_path / "out")
    tasks = opts.source_root / "report/tasks.csv"
    tasks.parent.mkdir(parents=True)
    tasks.write_text("case")
    monkeypatch.setattr(sweep.lanes, "identity", lambda _: dict(code_sha256={},
                        source_manifest_sha256='a' * 64, active_source_manifest_sha256='b' * 64,
                        source_tasks_sha256=sweep.short.sha256(tasks)))
    config = sweep.configuration(opts)
    assert config["max_context"] == sweep.CONTEXT_CAP == 262144
    assert config["format"] == sweep.FORMAT and len(config["code_sha256"]) == 4
    assert config["gpu_acquisition"] == "immediate_only_no_reacquisition"
    cmd = sweep.worker_command(opts, dict(modulo=2, remainder=1))
    assert cmd[:2] == [str(sweep.lanes.LAUNCHER), "--worker"]
    assert cmd[-2:] == ["--max-context", "262144"]
    assert cmd[cmd.index("--budgets")+1:cmd.index("--conditions")] == list(map(str, range(9000, 16001, 1000)))
    tasks.write_text("changed")
    assert sweep.configuration(opts) != config


def test_context_limit_and_exact_within_case_reuse(tmp_path):
    row = case(("hogwild", 16, 0, 9000))
    row.update(prompt_tokens=10, reasoning_tokens=144000, generated_tokens=144001, context_tokens=144012,
               effective_rounds=9000)
    reused = dict(row, budget=10000, reused_readout=True, reused_from_budget=9000)
    path = tmp_path/"rows.jsonl"
    catalog = catalog_for([row])
    path.write_text(serialized([row,reused])+'{"uncommitted":')
    assert len(sweep.read_rows(path, catalog)[0]) == 2
    path.write_text(serialized([dict(row, reasoning_tokens=262134, generated_tokens=262135, context_tokens=262146)]))
    with pytest.raises(ValueError):
        sweep.read_rows(path,catalog)
    path.write_text(serialized([row, dict(reused, answer_token_ids=[2])]))
    with pytest.raises(ValueError, match="reuse"):
        sweep.read_rows(path, catalog)


def test_reject_unrelated_root_or_budgets(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    other = tmp_path / "previous"; other.mkdir()
    (other / "plan.json").write_text(json.dumps({"config": {"format": "older-run"}}))
    for path in [other, other / "child", source / "child"]:
        with pytest.raises(SystemExit):
            sweep.main(["--source-root", str(source), "--output", str(path), "--gpus", "0"])
    runner = sweep.coordinator({})
    assert runner.report.BUDGETS == tuple(range(9000, 16001, 1000))
    assert sweep.lanes.report.BUDGETS == tuple(range(1000, 8001, 1000))
    opts = options(source, tmp_path / "long")
    opts.budgets = [9000, 16384]
    with pytest.raises(ValueError, match="9k to 16k"):
        sweep.configuration(opts)


def test_completed_run_28000_rows(tmp_path):
    keys = [(mode, workers, index, budget) for mode, workers in sweep.lanes.CONDITIONS.values()
            for index in range(500) for budget in sweep.BUDGETS]
    path = tmp_path / "results.jsonl"
    with path.open("w") as stream:
        for key in keys:
            stream.write(serialized([case(key)]))
    catalog = catalog_for(case(key) for key in keys)
    sweep.finish_report(path, tmp_path / "report", catalog)
    report = json.loads((tmp_path / "report/summary.json").read_text())
    assert report["completed_evaluations"] == report["expected_evaluations"] == 28000
    assert report["matched_tasks"] == 500 and len(report["cells"]) == 56
    assert all(cell["correct"] == cell["tasks"] == 500 for cell in report["cells"])
