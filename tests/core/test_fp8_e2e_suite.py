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
        if "--fixture-set" in command and option("--fixture-set") == "chains":
            cases = [dict(name=f"chain_{i}", prompt_ids=[1]*32+[i+3]*12,
                          teacher_tokens=[3]*tokens, cold_steps=[0, 3, tokens-1]) for i in range(3)]
        save_json(out / "fixtures.json", dict(model=model, cases=cases))
        return
    fixture_path = Path(option("--fixtures"))
    fixture = json.loads(fixture_path.read_text())
    names = command[command.index("--cases") + 1:]
    names = names[:next((i for i, name in enumerate(names) if name.startswith("--")), len(names))]
    args = dict(model=model, engine=engine, tokens=tokens)
    if engine == "mini":
        mode = option("--mini-scheduling")
        if mode == "shared-chains":
            from tests.e2e.fp8.chains import LAYOUTS
            for layout in LAYOUTS:
                variant = list(command)
                variant[variant.index("--mini-scheduling")+1] = layout
                variant[variant.index("--output")+1] = str(out.parent/layout)
                _fake_worker(variant, log_path, env, timeout)
            return
        quantization = option("--quantization") if "--quantization" in command else None
        args.update(mini_scheduling=mode, quantization=quantization)
        save_json(out / "storage.json", dict(fp8_tensors=int(quantization == "fp8"), bf16_tensors=2, source=str(ROOT)))
        if mode in ("mixed", "sequential"):
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
    engines = {name: [cmd for cmd, _ in calls if cmd[cmd.index("--engine") + 1] == name]
               for name in ("fixtures", "mini", "sglang", "transformers")}
    assert all(engines.values())
    assert {cmd[cmd.index("--mini-scheduling") + 1] for cmd in engines["mini"]} == set(artifacts.profile.modes)
    assert all(cmd[0] == "/example/sglang/python" for cmd in engines["sglang"])
    assert all(cmd[0] == "/example/transformers/python" for cmd in engines["transformers"])
    for command, env in calls:
        assert command[1:3] == ["-m", "tests.e2e.fp8.worker"]
        assert env["TRITON_PTXAS_PATH"] == "/example/user-toolchain/ptxas"
        assert str(ROOT) in env["PYTHONPATH"].split(os.pathsep)
    for mode in artifacts.profile.modes:
        validate_current_mini(artifacts.mini(mode), mode)
        assert compare(artifacts.mini(mode), artifacts.root / "sglang", artifacts.root / "transformers")["passed"]
    previous_count = len(calls)
    _collect(tmp_path / "another-invocation", calls)
    assert len(calls) == 2 * previous_count  # The next invocation collects everything afresh.


def test_shared_cache_profile_keeps_original_image_and_token_coverage(tmp_path):
    calls = []
    artifacts = _collect(tmp_path, calls, "shared-cache")
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
    previous_count = len(calls)
    with pytest.raises(FileExistsError):
        _collect(tmp_path, calls)
    assert len(calls) == previous_count
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
    ["--fp8-reference-memory-fraction", "1"],
    ["--fp8-reference-memory-fraction", "-1"],
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
    assert PROFILES["shared-chains"].modes == ("chain-flat", "chain-split", "chain-shared")
    assert all(p.cases and p.tokens > 1 and p.modes for p in PROFILES.values())


@pytest.mark.parametrize("flag", ["--audit-linear", "--audit-moe", "--reference-quant-outputs",
                                  "--capture-prefill", "--sglang-native-mrope"])
def test_acceptance_worker_rejects_diagnostic_substitution_flags(flag):
    from tests.e2e.fp8.worker import make_parser
    with pytest.raises(SystemExit) as error:
        make_parser().parse_args(["--engine", "mini", "--model", "unused",
                                  "--output", "unused", flag])
    assert error.value.code == 2


def test_batched_shared_collection_keeps_full_numerical_coverage(tmp_path):
    calls = []
    artifacts = _collect(tmp_path, calls, "shared-batched")
    assert len(calls) == 4
    mini_command = calls[1][0]
    assert mini_command[mini_command.index('--mini-scheduling')+1] == 'shared-batched'
    result = compare(artifacts.mini('shared-batched'), artifacts.root/'sglang', artifacts.root/'transformers')
    assert result['metrics']['mini']['prefill']['positions'] == 48
    assert result['metrics']['mini']['decode']['positions'] == 496
    assert result['passed']


@pytest.mark.parametrize('broken', [None, 'eager_decode', 'old_prefill', 'single_prefill', 'extra_graph'])
def test_batched_coverage_rejects_silent_fallback(broken):
    from tests.e2e.fp8.shared_batch import check_coverage
    coverage = dict(forwards=[dict(phase='prefill', workers=4), dict(phase='decode', workers=4)],
                    linear_layers=18, prefill_layer_calls=18, decode_replays=1, capture_profiles=[1, 2, 4])
    if broken == 'eager_decode':
        coverage['decode_replays'] = 0
    elif broken == 'old_prefill':
        coverage['prefill_layer_calls'] = 0
    elif broken == 'single_prefill':
        coverage['forwards'][0]['workers'] = 1
    elif broken == 'extra_graph':
        coverage['capture_profiles'].append(8)
    if broken:
        with pytest.raises(AssertionError):
            check_coverage(coverage)
    else:
        check_coverage(coverage)


@pytest.mark.parametrize('broken', [None, 'prefill_eager', 'model_eager', 'layer_eager', 'extra_graph'])
def test_full_graph_coverage_requires_both_replays_without_host_model_calls(broken):
    from tests.e2e.fp8.shared_batch import FULL_PREFILL_ROWS, check_coverage
    coverage = dict(forwards=[dict(phase='prefill', workers=4), dict(phase='decode', workers=4)],
        full_prefill_graph=True, prefill_layer_calls=0, model_forward_calls=0,
        prefill_replays=1, decode_replays=1, capture_profiles=[1, 2, 4],
        prefill_capture_profiles=list(FULL_PREFILL_ROWS))
    if broken == 'prefill_eager':
        coverage['prefill_replays'] = 0
    elif broken == 'model_eager':
        coverage['model_forward_calls'] = 1
    elif broken == 'layer_eager':
        coverage['prefill_layer_calls'] = 18
    elif broken == 'extra_graph':
        coverage['prefill_capture_profiles'].append(2048)
    if broken:
        with pytest.raises(AssertionError):
            check_coverage(coverage)
    else:
        check_coverage(coverage)


def test_chains_share_one_load_per_engine_but_never_past_predictions(tmp_path):
    calls = []
    artifacts = _collect(tmp_path, calls, "shared-chains")
    assert len(calls) == 4
    assert [c[c.index("--engine")+1] for c,_ in calls] == ["fixtures", "mini", "sglang", "transformers"]
    assert "--fixture-set" in calls[0][0]
    for mode in artifacts.profile.modes:
        validate_current_mini(artifacts.mini(mode), mode)
        result = compare(artifacts.mini(mode), artifacts.root/"sglang", artifacts.root/"transformers")
        assert result["metrics"]["mini"]["decode"]["positions"] == 21
        assert result["metrics"]["mini"]["prefill"]["positions"] == 9
        assert result["passed"] and result["relative_tolerance_percent"] == 100
    _collect(tmp_path/"next", calls, "shared-chains")
    assert len(calls) == 8


def test_bf16_control_cannot_masquerade_as_fp8(tmp_path):
    calls = []
    artifacts = _collect(tmp_path, calls, "shared-chains", quantization=None)
    assert "--bf16-control" in calls[0][0]
    assert "--quantization" not in calls[1][0]
    for mode in artifacts.profile.modes:
        validate_current_mini(artifacts.mini(mode), mode, quantization=None)
        result = compare(artifacts.mini(mode), artifacts.root/"sglang", artifacts.root/"transformers")
        assert result["bf16_control"] and result["acceptance_policy_fragile"]
        assert result["error_factors"]["prefill"]["p95_tv"] == 2.2
        assert result["error_factors"]["decode"]["p95_tv"] == 2
        with pytest.raises(AssertionError, match="FP8 weights"):
            validate_current_mini(artifacts.mini(mode), mode)
    with pytest.raises(AssertionError, match="chain-only"):
        collect_profile(RunConfig(tmp_path/"wrong", "bf16", quantization=None), "serving")
    path = artifacts.mini("chain-shared")/"storage.json"
    storage = json.loads(path.read_text()); storage["fp8_tensors"] = 1; save_json(path, storage)
    with pytest.raises(AssertionError, match="accidentally used FP8"):
        validate_current_mini(artifacts.mini("chain-shared"), "chain-shared", quantization=None)


def test_moe_collection_keeps_its_own_checkpoint_and_explicit_memory_budget(tmp_path):
    calls = []
    artifacts = _collect(tmp_path, calls, "shared-chains", require_moe=True, reference_memory_fraction=.6)
    assert len(calls) == 4
    for command,_ in calls:
        assert "--require-moe" in command
        assert command[command.index("--model")+1] == "fake-Qwen-FP8"
    sg = calls[2][0]
    assert sg[sg.index("--mem-fraction-static")+1] == "0.6"
    assert all(compare(artifacts.mini(m), artifacts.root/"sglang", artifacts.root/"transformers")["passed"]
               for m in artifacts.profile.modes)


def test_accuracy_thresholds_are_relative_to_transformers(tmp_path):
    artifacts = _collect(tmp_path, [])
    external = compare(artifacts.mini("mixed"), artifacts.root / "sglang", artifacts.root / "transformers")
    assert external["relative_tolerance_percent"] == 100
    assert external["acceptance_policy_fragile"] and not external["bf16_control"]
    assert all(v == 2 for phase in external["error_factors"].values() for v in phase.values())
    assert "additive_margins" not in external
    assert external["mini_vs_transformers"]["decode"]["positions"] == 88
    assert external["mini_vs_transformers"]["decode"]["mean_tv"] == 0
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


def test_positive_error_cannot_pass_against_an_exact_reference(tmp_path):
    artifacts = _collect(tmp_path, [])
    path = artifacts.mini("mixed")/"text_0_decode.pt"
    logits = torch.load(path, weights_only=True)
    logits[1:, 0] += .001
    torch.save(logits, path)
    result = compare(artifacts.mini("mixed"), artifacts.root/"sglang", artifacts.root/"transformers")
    assert not result["passed"]
    assert result["mini_vs_transformers"]["decode"]["mean_tv"] > 0


def test_optional_moe_flag_is_accepted_without_primary_model_at_collection():
    proc = subprocess.run([sys.executable, "-m", "pytest", "-o", "addopts=", "tests/e2e/fp8",
                           "--collect-only", "--fp8-moe-model", "/not-a-real-checkpoint", "-k", "moe_chain"],
                          cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for mode in PROFILES["shared-chains"].modes:
        assert f"test_moe_chain_external_parity[{mode}]" in proc.stdout


def test_moe_serving_still_forms_two_explicit_modes(tmp_path):
    calls = []
    artifacts = _collect(tmp_path, calls, "serving", require_moe=True, reference_memory_fraction=.6)
    assert len(calls) == 5
    assert all("--require-moe" in c for c,_ in calls)
    assert artifacts.profile.modes == ("mixed", "sequential")
