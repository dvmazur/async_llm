"""CPU unit tests of fresh-run orchestration and failure handling; no real models."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tests.e2e.fp8.common import ROOT, save_json, source_hashes, validate_current_mini
from tests.e2e.fp8.comparison import compare
from tests.e2e.fp8.suite import RunConfig, PROFILES, collect_profile, run_command


def _fake_worker(command, log_path, env, timeout):
    """Produce small deterministic artifacts, in the exact model-worker schema."""
    def option(name):
        return command[command.index(name) + 1]

    out = Path(option("--output"))
    out.mkdir()
    tokens = int(option("--tokens"))
    model, engine = option("--model"), option("--engine")
    if engine == "fixtures":
        cases = [dict(name=f"{kind}_{i}", prompt_ids=[1, 2, i + 3],
                      teacher_tokens=[3] * tokens, cold_steps=[0, 7, tokens - 1])
                 for i in range(8) for kind in ("text", "image")]
        save_json(out / "fixtures.json", dict(model=model, cases=cases))
        return
    fixture_path = Path(option("--fixtures"))
    fixture = json.loads(fixture_path.read_text())
    names = command[command.index("--cases") + 1:]
    names = names[:next((i for i, name in enumerate(names) if name.startswith("--")), len(names))]
    args = dict(model=model, engine=engine, tokens=tokens)
    if engine == "mini":
        mode = option("--mini-scheduling")
        args.update(mini_scheduling=mode, quantization="fp8")
        save_json(out / "storage.json", dict(fp8_tensors=1, source=str(ROOT)))
        if mode != "shared-cache":
            forwards = [dict(label="decode", mixed=False, rows=[dict(phase="decode")])]
            if mode == "mixed":
                forwards.append(dict(label="decode", mixed=True,
                                     rows=[dict(phase="prefill"), dict(phase="decode")]))
            save_json(out / "schedule.json", dict(mode=mode, overlap_scheduling=True, forwards=forwards))
    for case in fixture["cases"]:
        if case["name"] not in names:
            continue
        logits = torch.arange(tokens * 7).float().reshape(tokens, 7) % 11
        torch.save(logits, out / f"{case['name']}_decode.pt")
        for step in case["cold_steps"]:
            torch.save(logits[step], out / f"{case['name']}_cold{step}.pt")
    save_json(out / "complete.json", dict(arguments=args, cases=names,
        fixtures_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        source_hashes=source_hashes() if engine == "mini" else {}))


def _collect(tmp_path, calls, profile="serving", **kwargs):
    def worker(command, log, env, timeout):
        calls.append((command, env))
        _fake_worker(command, log, env, timeout)
    config = RunConfig(tmp_path / "run", model="fake-Qwen-FP8", **kwargs)
    return collect_profile(config, profile, command_runner=worker)


def test_new_collections_always_run_all_engines(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_PTXAS_PATH", "/example/user-toolchain/ptxas")
    calls = []
    artifacts = _collect(tmp_path, calls, sglang_python="/example/sglang/python",
                    transformers_python="/example/transformers/python")
    assert len(calls) == 5  # fixtures, mini mixed, mini sequential, SG, TF
    assert [cmd[cmd.index("--engine") + 1] for cmd, _ in calls] == [
        "fixtures", "mini", "mini", "sglang", "transformers"]
    assert calls[3][0][0] == "/example/sglang/python"
    assert calls[4][0][0] == "/example/transformers/python"
    for command, env in calls:
        assert command[1:3] == ["-m", "tests.e2e.fp8.worker"]
        assert env["TRITON_PTXAS_PATH"] == "/example/user-toolchain/ptxas"
        assert str(ROOT) in env["PYTHONPATH"].split(os.pathsep)
    for mode in artifacts.profile.modes:
        validate_current_mini(artifacts.mini(mode), mode)
        assert compare(artifacts.mini(mode), artifacts.root / "sglang", artifacts.root / "transformers")["passed"]
    _collect(tmp_path / "another-invocation", calls)
    assert len(calls) == 10  # The next invocation collects everything afresh.


def test_shared_cache_profile_keeps_original_image_and_token_coverage(tmp_path):
    calls = []
    artifacts = _collect(tmp_path, calls, "shared-cache")
    assert len(calls) == 4
    assert artifacts.profile.tokens == 32 and len(artifacts.profile.cases) == 16
    assert sum(name.startswith("image_") for name in artifacts.profile.cases) == 8
    result = compare(artifacts.mini("shared-cache"), artifacts.root / "sglang", artifacts.root / "transformers")
    assert result["metrics"]["mini"]["decode"]["positions"] == 496
    assert result["metrics"]["mini"]["prefill"]["positions"] == 48
    assert result["passed"]


def test_completed_results_cannot_be_reused_or_overwritten(tmp_path):
    calls = []
    artifacts = _collect(tmp_path, calls)
    before = {p: p.read_bytes() for p in artifacts.root.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        _collect(tmp_path, calls)
    assert len(calls) == 5
    assert before == {p: p.read_bytes() for p in artifacts.root.rglob('*') if p.is_file()}


def test_missing_artifacts_and_stale_sources_are_rejected(tmp_path):
    artifacts = _collect(tmp_path, [])
    path = artifacts.mini("mixed") / "complete.json"
    manifest = json.loads(path.read_text())
    key = next(iter(manifest["source_hashes"]))
    manifest["source_hashes"][key] = "not-the-current-code"
    save_json(path, manifest)
    with pytest.raises(AssertionError, match="CURRENT"):
        validate_current_mini(artifacts.mini("mixed"), "mixed")
    # Delete only the test's own synthetic logits, to simulate an interrupted run.
    (artifacts.mini("sequential") / "text_0_decode.pt").unlink()
    with pytest.raises(AssertionError, match="missing/extra"):
        artifacts.validate()


def test_no_overwrite_and_no_automatic_retry_after_worker_failure(tmp_path):
    calls = []
    def fail(*args):
        calls.append(args)
        raise RuntimeError("worker failed")
    config = RunConfig(tmp_path / "run", model="fake-model")
    with pytest.raises(RuntimeError, match="worker failed"):
        collect_profile(config, "serving", command_runner=fail)
    with pytest.raises(FileExistsError):
        collect_profile(config, "serving", command_runner=fail)
    assert len(calls) == 1


def test_worker_failure_log_and_timeout(tmp_path):
    log = tmp_path / "failed.log"
    with pytest.raises(RuntimeError, match="intentional-error"):
        run_command([sys.executable, "-c", "print('intentional-error'); raise SystemExit(4)"],
                    log, dict(os.environ), timeout=10)
    assert "intentional-error" in log.read_text()
    with pytest.raises(subprocess.TimeoutExpired):
        run_command([sys.executable, "-c", "import time; time.sleep(10)"],
                    tmp_path / "timeout.log", dict(os.environ), timeout=.1)


@pytest.mark.parametrize("args", [
    ["--fp8-model", "a", "--fp8-reuse", "b"],
    ["--fp8-results", "unused"],
    ["--fp8-timeout", "0"],
])
def test_invalid_cli_fails_before_model_start(args):
    proc = subprocess.run([sys.executable, "-m", "pytest", "-o", "addopts=",
                           "tests/e2e/fp8", "--collect-only", *args], cwd=ROOT,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 4, proc.stdout + proc.stderr


def test_profiles_remain_explicit():
    assert PROFILES["serving"].modes == ("mixed", "sequential")
    assert PROFILES["shared-cache"].modes == ("shared-cache",)


def test_accuracy_thresholds_are_relative_to_transformers(tmp_path):
    artifacts = _collect(tmp_path, [])
    external = compare(artifacts.mini("mixed"), artifacts.root / "sglang", artifacts.root / "transformers")
    assert external["relative_tolerance_percent"] == 5.0
    assert "additive_margins" not in external
    assert all(value == 0 for phase in external["relative_error_delta_percent"].values()
               for value in phase.values())
    path = artifacts.mini("mixed") / "text_0_decode.pt"
    logits = torch.load(path, weights_only=True)
    logits[1:] = logits[1:].flip(-1)
    torch.save(logits, path)
    # Worse than Transformers -> fail; equal deviation -> pass. This is not
    # an absolute-logit-invariance test between two mini schedules.
    assert not compare(artifacts.mini("mixed"), artifacts.root / "sglang",
                       artifacts.root / "transformers")["passed"]
    torch.save(logits, artifacts.root / "transformers" / path.name)
    assert compare(artifacts.mini("mixed"), artifacts.root / "sglang",
                   artifacts.root / "transformers")["passed"]
