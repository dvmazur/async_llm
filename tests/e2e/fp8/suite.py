"""Stateless collection helpers for fresh model runs, called by pytest fixtures."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

from .common import ROOT, save_json


@dataclass(frozen=True)
class Profile:
    name: str
    cases: tuple[str, ...]
    tokens: int
    modes: tuple[str, ...]


PROFILES = {
    "serving": Profile("serving", tuple(f"text_{i}" for i in range(8)), 12, ("mixed", "sequential")),
    "shared-cache": Profile("shared-cache", tuple(f"{kind}_{i}" for i in range(8)
                                                 for kind in ("text", "image")), 32, ("shared-cache",)),
}


def mini_directory(mode):
    return "mini" if mode == "shared-cache" else mode


def _unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _stop_worker_group(process):
    # Only the process group created by this test; includes SGLang's workers.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def run_command(command, log_path, env, timeout):
    save_json(log_path.with_suffix(".command.json"), dict(command=command, cwd=str(ROOT)))
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            _stop_worker_group(process)
            raise
        if code:
            _stop_worker_group(process)
    if code:
        tail = log_path.read_text(errors="replace")[-8000:]
        raise RuntimeError(f"FP8 worker exited {code}; full log: {log_path}\n{tail}")


@dataclass(frozen=True)
class Artifacts:
    profile: Profile
    root: Path

    def mini(self, mode):
        assert mode in self.profile.modes
        return self.root / mini_directory(mode)

    def report(self, name, result):
        path = self.root / "reports" / name
        save_json(path, result)
        return path

    def validate(self):
        """Verify that the fresh workers produced the entire requested workload."""
        fixture_path = self.root / "fixtures/fixtures.json"
        fixture_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
        fixtures = json.loads(fixture_path.read_text())
        cases = {case["name"]: case for case in fixtures["cases"]}
        expected = set()
        for name in self.profile.cases:
            case = cases[name]
            assert len(case["teacher_tokens"]) >= self.profile.tokens
            expected.add(f"{name}_decode.pt")
            for step in {min(s, self.profile.tokens - 1) for s in case["cold_steps"]}:
                expected.add(f"{name}_cold{step}.pt")
        roots = [self.mini(mode) for mode in self.profile.modes]
        roots += [self.root / name for name in ("sglang", "transformers")]
        models = set()
        for root in roots:
            manifest = json.loads((root / "complete.json").read_text())
            assert manifest["fixtures_sha256"] == fixture_sha, f"wrong fixtures: {root}"
            assert manifest["cases"] == list(self.profile.cases), f"wrong cases: {root}"
            assert manifest["arguments"]["tokens"] == self.profile.tokens, f"wrong token budget: {root}"
            assert {p.name for p in root.glob("*.pt")} == expected, f"missing/extra logits: {root}"
            models.add(manifest["arguments"]["model"])
        assert len(models) == 1, "different checkpoints across references"


@dataclass(frozen=True)
class RunConfig:
    root: Path
    model: str
    sglang_python: str = sys.executable
    transformers_python: str = sys.executable
    timeout: float = 1800


def collect_profile(config, name, *, command_runner=run_command):
    """Always run fresh. Pytest's session fixtures alone own sharing/lifetime."""
    profile = PROFILES[name]
    artifacts = Artifacts(profile, config.root / name)
    # No overwrite, resume or reuse, including after an interrupted collection.
    artifacts.root.mkdir(parents=True, exist_ok=False)
    stages = [("fixtures", "fixtures", None)]
    stages += [("mini", mini_directory(mode), mode) for mode in profile.modes]
    stages += [(engine, engine, None) for engine in ("sglang", "transformers")]
    interpreters = {"mini": sys.executable, "fixtures": sys.executable,
                    "sglang": config.sglang_python, "transformers": config.transformers_python}
    for engine, label, mode in stages:
        command = [interpreters[engine], "-m", "tests.e2e.fp8.worker",
                   "--engine", engine, "--model", config.model, "--tokens", str(profile.tokens),
                   "--output", str(artifacts.root / label)]
        if engine != "fixtures":
            command += ["--fixtures", str(artifacts.root / "fixtures/fixtures.json"),
                        "--cases", *profile.cases]
        if engine == "mini":
            command += ["--mini-repo", str(ROOT), "--quantization", "fp8",
                        "--mini-scheduling", mode, "--port", str(_unused_port())]
        # Inherit the user's CUDA/toolchain/HF environment, never rewrite a venv
        # or hard-code a machine's /workspace and CUDA installation paths.
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "python"), str(ROOT), env.get("PYTHONPATH"))))
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "0"
        log = artifacts.root / f"{label}.log"
        print(f"FP8 {profile.name}: {label}; log: {log}", flush=True)
        command_runner(command, log, env, config.timeout)
    artifacts.validate()
    return artifacts
