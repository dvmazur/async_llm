import gzip
import json
from contextlib import nullcontext

import pytest
from minisgl_examples.training.parallel_thinking import report, sweep


def manifest(layout, indices=(0,), workers=(1,), historical=False):
    files = ["eval.py", "rollout.py", "native.py", "math500.py", "run_hogwild.sh"]
    cfg = dict(report.COMMON, **report.PROTOCOL["hogwild"], mode="hogwild", workers=list(workers),
               indices=list(indices), context_cap=24576, native_context=262144, cuda_graphs=True,
               branch="hogwild-multi-workers", commit="fixture", grading="symbolic")
    if not historical:
        files.append("single_worker.py")
        cfg["finisher_mode"] = "cached"
        cfg["single_worker_layout"] = layout
        if workers == (1,) and layout == "causal":
            cfg.update(header_context="none", reminder_interval=0)
        elif workers == (1,) and layout == "scaffold":
            cfg["header_context"] = "causal"
    cfg["source_sha256"] = dict.fromkeys(files, "a" * 64)
    return cfg


def v3(cfg, layout=None, cue=None):
    cfg = {**cfg, "multi_worker_layout": layout, "multi_worker_answer_cue": cue, "pending_eos_in_finisher": False}
    if layout == "native":
        cfg.update(header_context="none", reminder_interval=0)
        cfg["answer_selection_policy"] = "eos_worker_last_box_v1"
        cfg["publication_policy"] = "encoded_separator_v1"
    return cfg


def traced(cfg):
    return {**cfg, "trace_format": "hogwild-inference-v1",
            "source_sha256": {**cfg["source_sha256"], "trace_io.py": "a" * 64}}


@pytest.mark.parametrize("memory,utilization,processes,expected", [
    (0, 0, "", True), (5, 0, "", True), (8, 0, "", True),
    (-1, 0, "", False), (9, 0, "", False),
    (5, 1, "", False), (5, 0, "72142\n", False),
])
def test_sweep_gpu_occupancy_with_idle_driver_memory(monkeypatch, memory, utilization, processes, expected):
    def check_output(args, *, text):
        if "--query-compute-apps=pid" in args:
            return processes
        return f"1, GPU-fixture, {memory}, {utilization}\n"

    monkeypatch.setattr(sweep.subprocess, "check_output", check_output)
    assert sweep.gpu_available(1) is expected


def test_audited_evaluator_versions_only_change_gpu_memory_preflight():
    source = (sweep.ROOT / report.GPU_PREFLIGHT_SOURCE).read_bytes()
    source = source.replace(b'        "answer_selection_policy": "eos_worker_last_box_v1" if native_workers else None,\n', b'')
    source = source.replace(b'        "publication_policy": "encoded_separator_v1" if native_workers else None,\n', b'')
    assert report.digest(source) == report.GPU_PREFLIGHT_RESIDUAL_SHA256
    original = source.replace(
        b"    # Some idle devices retain a few MiB of driver memory without a compute process.\n"
        b"    if processes or not 0 <= int(query[3]) <= 8 or int(query[5]) != 0:",
        b"    if processes or int(query[3]) != 0 or int(query[5]) != 0:",
    )
    assert report.digest(original) == report.GPU_PREFLIGHT_ORIGINAL_SHA256


def gpu_preflight_manifest(index, evaluator_hash):
    cfg = traced(v3(manifest("causal", indices=[index])))
    del cfg["source_sha256"]["eval.py"]
    cfg["source_sha256"][report.GPU_PREFLIGHT_SOURCE] = evaluator_hash
    return cfg


def test_resume_and_report_preserve_both_audited_source_versions(tmp_path, trace_fixture):
    original = gpu_preflight_manifest(0, report.GPU_PREFLIGHT_ORIGINAL_SHA256)
    residual = gpu_preflight_manifest(1, report.GPU_PREFLIGHT_RESIDUAL_SHA256)
    write_shard(tmp_path / "shards" / "a", original)
    write_shard(tmp_path / "shards" / "b", residual)
    before = (tmp_path / "shards" / "a" / "config.json").read_bytes()

    assert sweep.read_existing(tmp_path / "shards", residual["source_sha256"]) == {
        ("hogwild", 1, 0), ("hogwild", 1, 1),
    }
    payload = report.report(tmp_path, partial=True)
    assert payload["source"]["source_variants"] == [original["source_sha256"], residual["source_sha256"]]
    assert (tmp_path / "shards" / "a" / "config.json").read_bytes() == before


@pytest.mark.parametrize("changed_path", [report.GPU_PREFLIGHT_SOURCE, "rollout.py", "run_hogwild.sh", "extra.py"])
def test_resume_and_report_reject_unaudited_source_changes(tmp_path, trace_fixture, changed_path):
    original = gpu_preflight_manifest(0, report.GPU_PREFLIGHT_ORIGINAL_SHA256)
    changed = gpu_preflight_manifest(1, report.GPU_PREFLIGHT_RESIDUAL_SHA256)
    changed["source_sha256"][changed_path] = "f" * 64
    write_shard(tmp_path / "shards" / "a", original)
    write_shard(tmp_path / "shards" / "b", changed)
    with pytest.raises(sweep.IncompatibleRun, match="source_sha256"):
        sweep.read_existing(tmp_path / "shards", original["source_sha256"])
    with pytest.raises(ValueError, match="Mixed code"):
        report.report(tmp_path, partial=True)


def verify_fixture(directory, row, config, catalog):
    # The recorder tests cover the MiniSGL schema; this exercises on-disk hash binding.
    data = (directory / row["trace_path"]).read_bytes()
    report.require(report.digest(data) == row["trace_sha256"] and len(data) == row["trace_bytes"],
                   "Trace fixture digest mismatch")
    report.require(json.loads(gzip.decompress(data)) == [row["mode"], row["workers"], row["index"]],
                   "Trace fixture identity mismatch")
    return {"rollout": {"worker_token_ids": {"fixture": [1]}}}


def verify_graded_fixture(directory, row, config, catalog):
    from minisgl_examples.training.parallel_thinking.math500 import grade_math500

    data = verify_fixture(directory, row, config, catalog)
    report.require(grade_math500(row["gold"], row["prediction"]) == row["grading_status"], "Fixture regrade differs")
    return data


def verify_monitored_graded_fixture(directory, row, config, catalog):
    data = verify_graded_fixture(directory, row, config, catalog)
    return sweep.repeated_sampled_tokens(data["rollout"]["worker_token_ids"])


@pytest.fixture
def trace_fixture(monkeypatch):
    monkeypatch.setattr(report, "verify_traced_case", verify_fixture)
    monkeypatch.setattr(report, "load_native_catalog", lambda: {})


def write_shard(path, cfg):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(cfg))
    with (path / "results.jsonl").open("w") as stream:
        for n in cfg["workers"]:
            for index in cfg["indices"]:
                row = {
                    "mode": "hogwild", "workers": n, "index": index, "unique_id": f"test/{index}",
                    "gold": "1", "prediction": "1", "grading_status": "correct", "correct": True,
                    "generated_tokens": n, "reasoning_tokens": n, "finisher_tokens": 0, "decoding_steps": 1,
                    "total_decoding_steps": 1, "used_finisher": False, "stop_reason": "eos", "eos_token_id": 248046,
                    "finisher_stop_reason": "existing_answer", "finisher_eos_token_id": None,
                }
                if "single_worker_layout" in cfg:
                    row["single_worker_layout"] = cfg["single_worker_layout"] if n == 1 else None
                if "multi_worker_layout" in cfg:
                    row.update({key: cfg[key] if n > 1 else None for key in report.MULTI_FIELDS})
                    row["finisher_mode"] = cfg["finisher_mode"]
                if cfg.get("trace_format"):
                    row.update(mode=cfg["mode"], published_paragraphs=0)
                    if cfg["mode"] == "native":
                        row["finisher_stop_reason"] = "not_applicable"
                    name = f"traces/{cfg['mode']}-w{n}-i{index:03d}.json.gz"
                    data = gzip.compress(json.dumps([cfg["mode"], n, index]).encode())
                    target = path / name
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(data)
                    row.update(trace_path=name, trace_sha256=report.digest(data), trace_bytes=len(data))
                stream.write(json.dumps(row) + "\n")


def test_historical_reports_identify_legacy_protocol(tmp_path):
    write_shard(tmp_path / "old", manifest("legacy", historical=True))
    payload = report.report(tmp_path, partial=True)
    assert payload["source"]["manifest_version"] == 1
    assert payload["conditions"][1]["single_worker_layout"] == "legacy"
    assert payload["runtime_per_condition"]["hogwild-1"]["header_context"] == "independent"
    assert payload["condition_protocol"]["hogwild-1"]["finisher_mode"] == "flat"


@pytest.mark.parametrize("layout", ["causal", "scaffold", "legacy"])
def test_explicit_layout_and_source_must_agree(layout):
    cfg = manifest(layout)
    report.check_config(cfg)
    with pytest.raises(ValueError, match="finisher_mode"):
        report.check_config({**cfg, "finisher_mode": "flat"})
    del cfg["source_sha256"]["single_worker.py"]
    with pytest.raises(ValueError, match="source version"):
        report.check_config(cfg)


def test_report_rejects_mixed_one_worker_layouts(tmp_path):
    write_shard(tmp_path / "a", manifest("causal", indices=[0]))
    write_shard(tmp_path / "b", manifest("scaffold", indices=[1]))
    with pytest.raises(ValueError, match="layout"):
        report.report(tmp_path, partial=True)


def test_isolated_report_rejects_missing_indices(tmp_path):
    write_shard(tmp_path / "causal", manifest("causal"))
    with pytest.raises(ValueError, match="1/500"):
        report.report(tmp_path, only_one_worker=True)
    payload = report.report(tmp_path, only_one_worker=True, partial=True)
    assert not payload["complete"] and payload["expected_cases"] == 500


def test_mixed_width_manifest_records_effective_condition_protocol(tmp_path):
    write_shard(tmp_path / "a", manifest("causal", workers=(1, 2)))
    payload = report.report(tmp_path, partial=True)
    assert payload["condition_protocol"]["hogwild-1"]["header_context"] == "none"
    assert payload["condition_protocol"]["hogwild-2"]["header_context"] == "independent"
    assert payload["condition_protocol"]["hogwild-2"]["single_worker_layout"] is None
    assert payload["condition_protocol"]["hogwild-2"]["finisher_mode"] == "cached"


def test_isolated_sweep_resumes_and_verifies_all_500(tmp_path, monkeypatch, trace_fixture):
    cfg = traced(v3(manifest("causal", indices=range(500))))
    write_shard(tmp_path / "shards" / "causal", cfg)
    monkeypatch.setattr(sweep, "source_hashes", lambda save_traces=False: cfg["source_sha256"])

    def finish(command, **kwargs):
        assert command[-2:] == [str(tmp_path), "--only-one-worker"]
        payload = report.report(tmp_path, only_one_worker=True)
        assert payload["complete"] and payload["completed_cases"] == payload["expected_cases"] == 500
        assert len(payload["conditions"]) == 1 and payload["conditions"][0]["single_worker_layout"] == "causal"
        assert payload["verified_traces"] == 500

    monkeypatch.setattr(sweep.subprocess, "run", finish)
    sweep.run_sweep(tmp_path, 50, 0, only_one_worker=True, save_traces=True)


def test_new_sweep_rejects_scaffold_rows(tmp_path):
    cfg = manifest("scaffold")
    write_shard(tmp_path / "shards" / "scaffold", cfg)
    with pytest.raises(ValueError):
        sweep.read_existing(tmp_path / "shards", cfg["source_sha256"])
    with pytest.raises(ValueError, match="corrected causal"):
        report.report(tmp_path, partial=True, only_one_worker=True)


@pytest.mark.parametrize("layout", ["collaboration", "native"])
@pytest.mark.parametrize("cue", ["paper", "qwen"])
def test_v3_effective_protocol_for_mixed_widths(tmp_path, layout, cue):
    cfg = v3(manifest("causal", workers=(1, 2)), layout, cue)
    write_shard(tmp_path / "shards" / "attempt", cfg)
    payload = report.report(tmp_path, partial=True)
    assert payload["source"]["manifest_version"] == 3
    solo, multi = (payload["condition_protocol"][f"hogwild-{n}"] for n in (1, 2))
    assert solo["multi_worker_layout"] is solo["multi_worker_answer_cue"] is None
    assert solo["pending_eos_in_finisher"] is multi["pending_eos_in_finisher"] is False
    assert multi["multi_worker_layout"] == layout and multi["multi_worker_answer_cue"] == cue
    assert (multi["header_context"], multi["reminder_interval"]) == (
        ("none", 0) if layout == "native" else ("independent", 1024))
    assert payload["conditions"][2]["multi_worker_answer_cue"] == cue
    assert "multi_worker_answer_cue" in (tmp_path / "report" / "tasks.csv").read_text().splitlines()[0]


@pytest.mark.parametrize("key,value", [("pending_eos_in_finisher", True), ("multi_worker_answer_cue", None),
                                     ("multi_worker_layout", "unknown"), ("header_context", "independent")])
def test_v3_rejects_inconsistent_manifest(key, value):
    cfg = v3(manifest(None, workers=(2,)), "native", "qwen")
    with pytest.raises(ValueError):
        report.check_config({**cfg, key: value})


@pytest.mark.parametrize("second", [("native", "qwen"), ("collaboration", "paper")])
def test_v3_cross_width_protocols_must_match(tmp_path, second):
    write_shard(tmp_path / "a", v3(manifest(None, workers=(2,)), "native", "paper"))
    write_shard(tmp_path / "b", v3(manifest(None, workers=(4,)), *second))
    with pytest.raises(ValueError, match="Mixed multi-worker"):
        report.report(tmp_path, partial=True)


def test_report_rejects_mixed_v2_v3_and_result_cues(tmp_path):
    write_shard(tmp_path / "old", manifest(None, workers=(2,)))
    attempt = tmp_path / "new"
    cfg = v3(manifest(None, workers=(4,)), "collaboration", "paper")
    write_shard(attempt, cfg)
    with pytest.raises(ValueError, match="Mixed code versions"):
        report.report(tmp_path, partial=True)
    results = attempt / "results.jsonl"
    row = {**json.loads(results.read_text()), "multi_worker_answer_cue": "qwen"}
    with pytest.raises(ValueError, match="multi_worker_answer_cue"):
        report.check_task(row, cfg)


def test_sweep_sticks_to_requested_layout_and_cue(tmp_path, trace_fixture):
    cfg = traced(v3(manifest(None, workers=(2,)), "native", "qwen"))
    cfg["context_cap"] = 16384 * 2 + 8192
    attempt = tmp_path / "shards" / "current"
    write_shard(attempt, cfg)
    expected = {("hogwild", 2, 0)}
    assert sweep.read_existing(attempt.parent, cfg["source_sha256"]) == expected
    with pytest.raises(ValueError, match="multi_worker_layout"):
        sweep.read_existing(attempt.parent, cfg["source_sha256"],
                            multi_worker=sweep.MultiWorkerProtocol("collaboration", "qwen"))
    with pytest.raises(ValueError, match="multi_worker_answer_cue"):
        sweep.read_existing(attempt.parent, cfg["source_sha256"], multi_worker=sweep.MultiWorkerProtocol("native", "paper"))


@pytest.mark.parametrize("explicit_none", [False, True])
def test_sweep_requires_corrected_multiworker_publication_policy(tmp_path, explicit_none):
    cfg = traced(v3(manifest(None, workers=(2,)), "native", "qwen"))
    cfg["context_cap"] = sweep.BUDGET * 2 + 8192
    if explicit_none:
        cfg["publication_policy"] = None
    else:
        del cfg["publication_policy"]
    with pytest.raises(sweep.IncompatibleRun, match="publication_policy"):
        sweep.check_attempt(tmp_path, cfg, cfg["source_sha256"], sweep.CONDITIONS, sweep.MultiWorkerProtocol())


@pytest.mark.parametrize("marker", ["multiworker-stopped.json", "superseded.json", "quality-hold.json"])
def test_stopped_run_never_creates_an_attempt_or_dispatches(tmp_path, marker):
    (tmp_path / marker).write_text("{}")
    with pytest.raises(ValueError, match="closed to new attempts"):
        sweep.main([str(tmp_path), "--gpus", "0"])
    assert [p.name for p in tmp_path.iterdir()] == [marker]


def test_traced_sweep_rejects_old_cases_and_preserves_incomplete_attempt_tail(tmp_path, trace_fixture):
    cfg = traced(v3(manifest("causal")))
    attempt = tmp_path / "shards" / "one"
    write_shard(attempt, cfg)
    with (attempt / "results.jsonl").open("ab") as stream:
        stream.write(b'{"incomplete":')
    assert sweep.attempt_rows(attempt, cfg["source_sha256"]) == {("hogwild", 1, 0)}
    (attempt / "completed.json").write_text('{"cases":1}')
    with pytest.raises(ValueError, match="unterminated"):
        sweep.attempt_rows(attempt, cfg["source_sha256"])
    (attempt / "completed.json").unlink()
    (attempt / "traces" / "hogwild-w1-i000.json.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        report.report(tmp_path, partial=True)
    old = v3(manifest("causal"))
    write_shard(tmp_path / "untraced", old)
    with pytest.raises(ValueError, match="different benchmark protocol"):
        sweep.attempt_rows(tmp_path / "untraced", cfg["source_sha256"])
    with pytest.raises(ValueError, match="fresh execution"):
        sweep.run_sweep(tmp_path, 50, 0, reuse_native_from=tmp_path.parent, save_traces=True)


def observation(index, workers, *, mode="hogwild", correct=False, early=False):
    return {"mode": mode, "workers": workers, "index": index, "correct": correct,
            "grading_status": "correct" if correct else "incorrect", "decoding_steps": 80 if early else 800,
            "used_finisher": early, "stop_reason": "eos", "finisher_stop_reason": "eos" if early else "existing_answer",
            "finisher_tokens": int(early), "published_paragraphs": 3, "trace_sha256": "a" * 64,
            "trace_path": f"traces/{mode}-{workers}-{index}.json.gz"}


@pytest.mark.parametrize("token", [4558, 0, 27])
def test_sampled_repetition_gate_counts_consecutive_tokens_and_exact_boundaries(token):
    limit = sweep.REPEATED_TOKEN_LIMIT
    assert limit == 128
    assert sweep.repeated_sampled_tokens({
        "short": [token] * (limit - 1),
        "separated": [token] * (limit - 1) + [token + 1] + [token] * (limit - 1),
        "alternating": [token, token + 1] * limit,
        "empty": [],
    }) == []
    assert sweep.repeated_sampled_tokens({"Alice": [token + 1] * 3 + [token] * limit + [token + 1]}) == [{
        "kind": "identical_token_run", "worker": "Alice", "token_id": token, "run_length": limit,
        "start_index": 3, "end_index_exclusive": 3 + limit,
        "threshold": limit, "stream": "worker_token_ids",
    }]


def test_repeated_phrase_heuristic_uses_bounded_raw_tail_and_avoids_short_or_diverse_controls():
    tokens = list(range(43)) * 47
    assert sweep.repeated_sampled_tokens({"Alice": tokens}) == []
    finding, = sweep.repeated_sampled_phrases({"Alice": tokens})
    assert finding["kind"] == "low_diversity_8gram_tail" and finding["heuristic"] is True
    assert finding["start_index"] == len(tokens) - 2000 and finding["end_index_exclusive"] == len(tokens)
    assert finding["window_count"] == 1993 and finding["unique_windows"] == 43
    assert finding["pattern_occurrences"] >= 46 and len(finding["pattern_token_ids"]) == 8
    assert sweep.repeated_sampled_phrases({
        "short": [1, 2] * 500, "diverse": list(range(2000)),
        "rare": list(range(200)) * 6,
    }) == []
    assert sweep.repeated_sampled_phrases({"Alice": tokens}, exclude_workers={"Alice"}) == []


def test_verifier_reuses_loaded_raw_sampled_streams_and_returns_compact_evidence(monkeypatch):
    # Shared text/history duplication is irrelevant; only actual sampled IDs count.
    data = {"rollout": {"worker_token_ids": {"Alice": [1, 2] * 100, "Bob": [4558] * 137},
                        "worker_text": {"Alice": "duplicate" * 1000}, "history": [4558] * 200},
            "engine": {"large_history": "not returned"}}
    calls = []

    def verify(*args):
        calls.append(args)
        return data

    monkeypatch.setattr(report, "verify_traced_case", verify)
    evidence = sweep.verify_monitored_case("attempt", {"correct": True}, "manifest", "catalog")
    assert calls == [("attempt", {"correct": True}, "manifest", "catalog")]
    assert len(evidence) == 1 and evidence[0]["worker"] == "Bob" and evidence[0]["run_length"] == 137
    assert "engine" not in evidence[0]


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("policy", ["hold", "record"])
def test_sampled_repetition_saved_and_live_correct_finisher_policy(tmp_path, monkeypatch, live, policy):
    from threading import Event
    from types import SimpleNamespace

    cfg = traced(v3(manifest(None, workers=(6,)), "native", "qwen"))
    cfg["context_cap"] = sweep.BUDGET * 6 + 8192
    attempt = tmp_path / "shards" / "attempt"
    write_shard(attempt, cfg)
    path = attempt / "results.jsonl"
    row = json.loads(path.read_text())
    row.update(decoding_steps=137, total_decoding_steps=139, reasoning_tokens=822, generated_tokens=824,
               used_finisher=True, finisher_tokens=2, stop_reason="max_steps", eos_token_id=None,
               finisher_stop_reason="answer")
    path.write_text(json.dumps(row) + "\n")

    def verify(*args):
        verify_fixture(*args)
        return {"rollout": {"worker_token_ids": {"Alice": [4558] * 137}}}

    monkeypatch.setattr(report, "verify_traced_case", verify)
    monitor = sweep.QualityMonitor(tmp_path, {}, policy)
    with (pytest.raises(RuntimeError, match="Suspicious sampled repetition") if policy == "hold" else nullcontext()):
        if live:
            log = tmp_path / "evaluator.log"
            log.touch()
            with sweep.ThreadPoolExecutor(max_workers=1) as verifier:
                sweep.watch_attempt(SimpleNamespace(poll=lambda: 0), attempt, cfg["source_sha256"],
                                    ("hogwild", 6), [0], sweep.MultiWorkerProtocol(), monitor,
                                    Event(), "pilot", log, verifier)
        else:
            sweep.read_existing(attempt.parent, cfg["source_sha256"], monitor=monitor)
    if policy == "hold":
        marker = json.loads((tmp_path / sweep.HOLD).read_text())
        case, = marker["cases"]
        assert not monitor.rows  # It was never accepted as a healthy result.
    else:
        marker = json.loads((tmp_path / "suspicious-samples/hogwild-w6-i000.json").read_text())
        case = marker["case"]
        assert marker["cause_status"] == "unresolved; recorded cache/token integrity validated"
        assert monitor.rows[("hogwild", 6)][0]["correct"] and not (tmp_path / sweep.HOLD).exists()
    assert (case["mode"], case["workers"], case["index"]) == ("hogwild", 6, 0)
    assert case["correct"] and case["used_finisher"]
    assert case["trace"] == "shards/attempt/traces/hogwild-w6-i000.json.gz"
    assert case["trace_sha256"] == row["trace_sha256"]
    repeated, = case["repeated_sampled_tokens"]
    assert (repeated["worker"], repeated["token_id"], repeated["run_length"]) == ("Alice", 4558, 137)


@pytest.mark.parametrize("kind", ["token", "phrase"])
def test_recorded_repetition_is_persistent_idempotent_and_conflict_holds(tmp_path, kind):
    row = observation(156, 8, correct=True)
    repetitions = (sweep.repeated_sampled_tokens({"Alice": [4558] * 128}) if kind == "token" else
                   sweep.repeated_sampled_phrases({"Alice": list(range(43)) * 47}))
    monitor = sweep.QualityMonitor(tmp_path, {}, "record")
    directory = tmp_path / "attempt"
    monitor.observe(directory, row, repetitions)
    path = tmp_path / "suspicious-samples/hogwild-w8-i156.json"
    original = path.read_bytes()
    monitor.observe(directory, row, repetitions)
    restarted = sweep.QualityMonitor(tmp_path, {}, "record")
    restarted.observe(directory, row, repetitions)
    assert path.read_bytes() == original and len(list(path.parent.iterdir())) == 1
    restarted.status("benchmark")
    status = json.loads((tmp_path / "progress.json").read_text())
    assert status["repetition_policy"] == "record" and status["suspicious_cases"] == 1
    conflicting = sweep.QualityMonitor(tmp_path, {}, "record")
    with pytest.raises(RuntimeError, match="Conflicting suspicious-sample record"):
        conflicting.observe(directory, {**row, "trace_sha256": "b" * 64}, repetitions)
    assert path.read_bytes() == original and (tmp_path / sweep.HOLD).exists()


def test_phrase_heuristic_holds_by_default_and_single_token_findings_are_not_duplicated(tmp_path, monkeypatch):
    row = observation(156, 8, correct=True)
    phrases = sweep.repeated_sampled_phrases({"Alice": list(range(43)) * 47})
    with pytest.raises(RuntimeError, match="Suspicious sampled repetition"):
        sweep.QualityMonitor(tmp_path, {}).observe(tmp_path / "attempt", row, phrases)
    monkeypatch.setattr(report, "verify_traced_case", lambda *args: {
        "rollout": {"worker_token_ids": {"Alice": [4558] * 2000}},
    })
    findings = sweep.verify_monitored_case(None, None, None, None)
    assert len(findings) == 1 and findings[0]["kind"] == "identical_token_run"


def test_cli_repetition_policy_is_explicit_and_requires_traces(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sweep, "run_sweep", lambda *args, **kwargs: calls.append(kwargs))
    (tmp_path / "report").mkdir()
    (tmp_path / "report/tasks.csv").write_text("dispatch fixture")
    sweep.main([str(tmp_path), "--gpus", "0", "--save-traces", "--repetition-policy", "record"])
    assert calls[-1]["repetition_policy"] == "record"
    sweep.main([str(tmp_path), "--gpus", "0", "--save-traces"])
    assert calls[-1]["repetition_policy"] == "hold"
    with pytest.raises(SystemExit):
        sweep.main([str(tmp_path), "--gpus", "0", "--repetition-policy", "record"])


def test_pilot_quality_hold_records_stable_case_evidence(tmp_path):
    mon = sweep.QualityMonitor(tmp_path, {})
    with pytest.raises(RuntimeError, match="matched pilot"):
        for i in sweep.WIDE_PILOT:
            mon.observe(tmp_path / "attempt", observation(i, 16, correct=i in sweep.WIDE_PILOT[:3]))
    marker = json.loads((tmp_path / sweep.HOLD).read_text())
    assert [r["index"] for r in marker["cases"]] == list(sweep.WIDE_PILOT)
    with pytest.raises(RuntimeError):
        mon.hold("second simultaneous fault")
    assert json.loads((tmp_path / sweep.HOLD).read_text()) == marker
    with pytest.raises(ValueError, match="closed"):
        sweep.check_run_open(tmp_path)


@pytest.mark.parametrize("policy", ["hold", "record"])
def test_wider_unmatched_pilot_requires_corroborating_signals(tmp_path, policy):
    mon = sweep.QualityMonitor(tmp_path, {}, policy)
    for i in sweep.WIDE_PILOT:
        mon.observe(tmp_path / "good", observation(i, 8))
    assert not (tmp_path / sweep.HOLD).exists()
    with pytest.raises(RuntimeError, match="Repeated unparsed"):
        for i in sweep.WIDE_PILOT:
            mon.observe(tmp_path / "suspicious", observation(i, 6, early=True))


def test_paired_h1_gap_detected_when_wrong_answers_parse(tmp_path):
    mon = sweep.QualityMonitor(tmp_path, {})
    # An ordinary 6/6 vs H1 4/6 warm-up is allowed. On a fresh complete
    # 16-case window, a six-case paired deficit triggers the separate rule.
    for i in sweep.WIDE_PILOT:
        mon.observe(tmp_path / "native", observation(i, 1, mode="native", correct=True))
        mon.observe(tmp_path / "hogwild", observation(i, 1, correct=i not in sweep.WIDE_PILOT[:2]))
    with pytest.raises(RuntimeError, match="Paired native accuracy"):
        for i in range(384, 400):
            mon.observe(tmp_path / "native", observation(i, 1, mode="native", correct=True))
            mon.observe(tmp_path / "hogwild", observation(i, 1, correct=i >= 390))


def test_low_disk_space_holds_and_never_dispatches(tmp_path, monkeypatch, trace_fixture):
    from types import SimpleNamespace

    monkeypatch.setattr(sweep.shutil, "disk_usage", lambda *args: SimpleNamespace(free=sweep.DISK_RESERVE - 1))
    with pytest.raises(RuntimeError, match="bytes free"):
        sweep.run_sweep(tmp_path, 50, 0, save_traces=True)
    assert not list((tmp_path / "shards").iterdir()) and (tmp_path / sweep.HOLD).exists()


@pytest.mark.parametrize("no_wait", [False, True])
def test_two_thread_synthetic_traced_sweep_gates_all_widths_before_bulk(tmp_path, monkeypatch, trace_fixture, no_wait):
    from pathlib import Path
    from threading import Barrier
    from types import SimpleNamespace

    monkeypatch.setattr(sweep, "TOTAL", 2)
    monkeypatch.setattr(report, "N_PROBLEMS", 2)
    monkeypatch.setattr(sweep, "pilot_indices", lambda condition: (0,))
    availability_checks = []

    def available(gpu):
        availability_checks.append(gpu)
        # It becomes free on a second check, but no-wait must never check it
        # again, including after the pilot phase finishes on the other GPU.
        return not (no_wait and gpu == 1 and availability_checks.count(1) == 1)

    monkeypatch.setattr(sweep, "gpu_available", available)
    monkeypatch.setattr(report, "verify_traced_case", verify_graded_fixture)
    # The spawn worker imports module globals anew; submit the standalone
    # synthetic verifier so its fixture does not depend on inherited patches.
    monkeypatch.setattr(sweep, "verify_monitored_case", verify_monitored_graded_fixture)
    hashes = traced(v3(manifest("causal")))["source_sha256"]
    monkeypatch.setattr(sweep, "source_hashes", lambda save_traces=False: hashes)
    starts = []
    devices = set()
    first_pair = Barrier(1 if no_wait else 2)

    def evaluator(command, **options):
        assert "--save-traces" in command and options["start_new_session"]
        gpu = options["env"]["HOGWILD_GPU"]
        assert gpu in {"1", "4"}
        if gpu not in devices:
            devices.add(gpu)
            first_pair.wait(timeout=10)
        mode = command[command.index("--mode") + 1]
        n = int(command[command.index("--workers") + 1])
        i = int(command[command.index("--indices") + 1])
        starts.append((mode, n, i))
        cfg = traced(v3(manifest("causal" if n == 1 else None, indices=(i,), workers=(n,)),
                        "native" if n > 1 else None, "qwen" if n > 1 else None))
        if mode == "native":
            cfg.update(report.PROTOCOL["native"], mode=mode, single_worker_layout=None)
        cfg["context_cap"] = min(cfg["native_context"], sweep.BUDGET * n + 8192)
        dest = Path(command[command.index("--output") + 1])
        write_shard(dest, cfg)
        result = dest / "results.jsonl"
        row = json.loads(result.read_text())
        row.update(gold=r"\frac{1}{2}", prediction="0.5")
        result.write_text(json.dumps(row) + "\n")
        (dest / "completed.json").write_text('{"cases":1}')
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(sweep.subprocess, "Popen", evaluator)
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **kw: report.report(tmp_path))
    sweep.main([str(tmp_path), "--save-traces", "--gpus", "1", "4", *(["--no-wait-gpus"] if no_wait else [])])
    assert devices == ({"4"} if no_wait else {"1", "4"})
    if no_wait:
        assert availability_checks.count(1) == 1
    assert [i for _, _, i in starts] == [0] * 7 + [1] * 7
    payload = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert payload["complete"] and payload["verified_traces"] == 14
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["phase"] == "complete" and progress["paired_pilots"]["2"]["indices"] == [0]


def test_no_wait_all_occupied_exits_unavailable_without_quality_hold_and_preserves_rows(tmp_path, monkeypatch, trace_fixture):
    sweep.snapshot_sources(tmp_path)
    cfg = traced(v3(manifest("causal")))
    attempt = tmp_path / "shards" / "existing"
    write_shard(attempt, cfg)
    before = (attempt / "results.jsonl").read_bytes()
    monkeypatch.setattr(sweep, "source_hashes", lambda save_traces=False: cfg["source_sha256"])
    checks = []
    monkeypatch.setattr(sweep, "gpu_available", lambda gpu: checks.append(gpu) or False)
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *a, **k: pytest.fail("Occupied GPU launched"))
    with pytest.raises(SystemExit) as exc:
        sweep.main([str(tmp_path), "--save-traces", "--gpus", "1", "6", "--no-wait-gpus"])
    assert exc.value.code == 3 and sorted(checks) == [1, 6]
    assert not (tmp_path / sweep.HOLD).exists() and (attempt / "results.jsonl").read_bytes() == before
    status = json.loads((tmp_path / "gpu-unavailable.json").read_text())
    assert status["dropped_gpus"] == [1, 6] and status["pending_shards"] == 7 and status["no_wait_gpus"]
    assert json.loads((tmp_path / "progress.json").read_text())["phase"] == "gpu_unavailable"


@pytest.mark.parametrize("failure,expected_checks", [("GPU 6 is occupied", 1), ("ReadTimeout", 2)])
def test_no_wait_preflight_or_retry_race_drops_gpu_without_relaunch(tmp_path, monkeypatch, trace_fixture,
                                                               failure, expected_checks):
    from types import SimpleNamespace

    checks, launches = [], []
    monkeypatch.setattr(sweep, "gpu_available", lambda gpu: checks.append(gpu) or len(checks) == 1)

    def failed_launch(command, **options):
        launches.append(command)
        options["stdout"].write(failure + "\n")
        options["stdout"].flush()
        return SimpleNamespace(poll=lambda: 1)

    monkeypatch.setattr(sweep.subprocess, "Popen", failed_launch)
    with pytest.raises(SystemExit) as exc:
        sweep.main([str(tmp_path), "--save-traces", "--gpus", "6", "--no-wait-gpus"])
    assert exc.value.code == 3 and len(launches) == 1 and checks == [6] * expected_checks
    assert not (tmp_path / sweep.HOLD).exists()
    status = json.loads((tmp_path / "gpu-unavailable.json").read_text())
    assert status["dropped_gpus"] == [6] and status["pending_shards"] == 7


@pytest.mark.parametrize("values", [["1", "1"], ["1", "-1"]])
def test_cli_rejects_invalid_gpu_ids_before_creating_run(tmp_path, values):
    root = tmp_path / "benchmark"
    with pytest.raises(SystemExit) as exc:
        sweep.main([str(root), "--gpus", *values])
    assert exc.value.code == 2 and not root.exists()


def test_concurrent_live_regrade_reports_actual_symbolic_mismatch(tmp_path):
    from multiprocessing import get_context
    from threading import Event
    from types import SimpleNamespace

    from minisgl_examples.training.parallel_thinking.math500 import GRADING_POLICY

    cfg = traced(v3(manifest("causal")))
    cfg["grading"] = GRADING_POLICY
    attempt = tmp_path / "attempt"
    write_shard(attempt, cfg)
    path = attempt / "results.jsonl"
    row = json.loads(path.read_text())
    row.update(gold=r"\frac{1}{2}", prediction="0.75")  # Row incorrectly claims this is correct.
    path.write_text(json.dumps(row) + "\n")
    log = tmp_path / "evaluator.log"
    log.touch()
    monitor = sweep.QualityMonitor(tmp_path, {0: (row["unique_id"], row["gold"])})
    with sweep.ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as verifier:
        with sweep.ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(sweep.watch_attempt, SimpleNamespace(poll=lambda: 0), attempt,
                                   cfg["source_sha256"], ("hogwild", 1), [0], sweep.MultiWorkerProtocol(),
                                   monitor, Event(), "pilot", log, verifier)
            with pytest.raises(RuntimeError, match="Traced case regrade differs"):
                future.result()
    assert json.loads((tmp_path / sweep.HOLD).read_text())["reason"] == "Traced case regrade differs"


@pytest.mark.parametrize("policy", ["hold", "record"])
def test_live_error_stops_only_the_owned_process_group(tmp_path, policy):
    import subprocess
    import sys
    import threading

    mon = sweep.QualityMonitor(tmp_path, {}, policy)
    log = tmp_path / "synthetic.log"
    log.write_text("RuntimeError: non-finite logits\n")
    args = [sys.executable, "-c", "import time; time.sleep(30)"]
    with subprocess.Popen(args, start_new_session=True) as owned, subprocess.Popen(args, start_new_session=True) as unrelated:
        try:
            with pytest.raises(RuntimeError, match="non-finite logits"):
                sweep.watch_attempt(owned, tmp_path, {}, ("hogwild", 2), [0], sweep.MultiWorkerProtocol(),
                                    mon, threading.Event(), "pilot", log, verifier=None)
            assert owned.wait(timeout=3) < 0 and unrelated.poll() is None
            assert "non-finite logits" in json.loads((tmp_path / sweep.HOLD).read_text())["reason"]
        finally:
            if owned.poll() is None:
                owned.terminate()
            unrelated.terminate()


@pytest.mark.parametrize("policy", ["hold", "record"])
def test_bulk_paired_collapse_holds_even_with_parsed_boxes_and_no_finisher(tmp_path, policy):
    mon = sweep.QualityMonitor(tmp_path, {}, policy)
    for index in range(384, 400):
        mon.observe(tmp_path / "hogwild", observation(index, 8, correct=index < 388))
    with pytest.raises(RuntimeError, match="Matched 16-case collapse"):
        for index in range(384, 400):
            mon.observe(tmp_path / "native", observation(index, 1, mode="native", correct=index < 396))
    assert len(json.loads((tmp_path / sweep.HOLD).read_text())["cases"]) == 32


def test_untraced_isolated_sweep_can_resume_without_artifacts(tmp_path, monkeypatch):
    cfg = v3(manifest("causal", indices=range(500)))
    write_shard(tmp_path / "shards" / "old", cfg)
    monkeypatch.setattr(sweep, "source_hashes", lambda save_traces=False: cfg["source_sha256"])
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **kw: report.report(tmp_path, only_one_worker=True))
    sweep.run_sweep(tmp_path, 50, 0, only_one_worker=True)
    assert not (tmp_path / "progress.json").exists()
    assert json.loads((tmp_path / "report" / "summary.json").read_text())["complete"]
