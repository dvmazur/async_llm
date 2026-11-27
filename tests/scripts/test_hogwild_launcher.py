"""CPU coverage for the portable launcher's isolated checkpoint transfer adapter."""

from pathlib import Path
from types import SimpleNamespace
import json
import os
import shutil
import subprocess
import sys

import pytest
import safetensors
import torch

from minisgl.models import weight
from tests.training.models import tiny_hybrid, tiny_model


@pytest.fixture
def install_reader():
    launcher = Path(__file__).resolve().parents[2] / "run_hogwild.sh"
    source = launcher.read_text().split("<<'PYTHON'\n", 1)[1].rsplit("\nPYTHON", 1)[0]
    namespace = {"__name__": "launcher_adapter_test"}
    exec(compile(source, str(launcher), "exec"), namespace)
    return namespace["_install_cpu_weight_reader"]


@pytest.mark.parametrize("fail", [False, True])
def test_adapter_delegates_cleanup_and_only_changes_transfer_source(install_reader, fail):
    events = []

    class Tensor:
        def to(self, **kwargs):
            events.append(("to", kwargs))
            if fail:
                raise RuntimeError("transfer failed")
            return "copied tensor"

    class Context:
        def __enter__(self):
            events.append(("enter",))
            return self

        def __exit__(self, *exception):
            events.append(("exit", exception))

        def keys(self):
            return ["lm_head.weight"]

        def get_tensor(self, name):
            events.append(("get_tensor", name))
            return Tensor()

    def safe_open(filename, **kwargs):
        events.append(("open", filename, kwargs))
        return Context()

    original = SimpleNamespace(safe_open=safe_open)
    consumer = SimpleNamespace(safetensors=original)
    install_reader(consumer)
    assert original.safe_open is safe_open and consumer.safetensors is not original

    def read():
        with consumer.safetensors.safe_open("weights.safetensors", framework="pt", device="cuda:0") as reader:
            assert reader.keys() == ["lm_head.weight"]
            return reader.get_tensor("lm_head.weight")

    if fail:
        with pytest.raises(RuntimeError, match="transfer failed"):
            read()
    else:
        assert read() == "copied tensor"
    assert events[:4] == [
        ("open", "weights.safetensors", {"framework": "pt", "device": "cpu"}),
        ("enter",), ("get_tensor", "lm_head.weight"),
        ("to", {"device": "cuda:0", "non_blocking": False}),
    ]
    assert events[4][0] == "exit"
    assert events[4][1][0] is (RuntimeError if fail else None)


@pytest.mark.parametrize("make_model", [tiny_model, tiny_hybrid])
def test_existing_loader_preserves_every_tensor_with_cpu_adapter(tmp_path, monkeypatch, install_reader, make_model):
    with torch.device("cpu"):
        model = make_model().to(dtype=torch.bfloat16)
    model.save_pretrained(tmp_path)
    monkeypatch.setattr(weight, "get_tp_info", lambda: SimpleNamespace(rank=0, size=1, is_primary=lambda: False))
    original_loader, original_safe_open = weight.load_weight, safetensors.safe_open
    expected = dict(original_loader(str(tmp_path), torch.device("cpu")))
    # Ensure monkeypatch restores this consumer after the test.
    monkeypatch.setattr(weight, "safetensors", safetensors)
    install_reader(weight)
    actual = dict(weight.load_weight(str(tmp_path), torch.device("cpu")))
    assert weight.load_weight is original_loader and safetensors.safe_open is original_safe_open
    assert actual.keys() == expected.keys()
    for name, reference in expected.items():
        assert actual[name].dtype == reference.dtype
        assert actual[name].shape == reference.shape
        assert torch.equal(actual[name].view(torch.uint8), reference.view(torch.uint8)), name


@pytest.mark.parametrize("budget,worker", [(False, False), (True, False), (False, True), (True, True)])
def test_launchers_dispatch_with_offline_cache_and_one_gpu(tmp_path, budget, worker):
    project = Path(__file__).resolve().parents[2]
    for name in ("run_hogwild.sh", "run_hogwild_budget.sh"):
        shutil.copy2(project / name, tmp_path / name)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    interpreter = tmp_path / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    recorder = binaries / "uv"
    recorder.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "keys = ('CUDA_VISIBLE_DEVICES', 'HF_HOME', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE', "
        "'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF', 'HOGWILD_EVAL_MODULE')\n"
        "print(json.dumps({'argv': sys.argv[1:], 'env': {k: os.environ.get(k) for k in keys}}))\n"
    )
    recorder.chmod(0o755)
    query = binaries / "nvidia-smi"
    query.write_text("#!/bin/sh\n[ \"$2\" = 3 ] || exit 1\nprintf '%s\\n' GPU-test-3\n")
    query.chmod(0o755)
    script = tmp_path / ("run_hogwild_budget.sh" if budget else "run_hogwild.sh")
    args = ["--worker"] if budget and worker else []
    if not budget and not worker:
        args.append("--full-math500")
    environment = {
        **os.environ, "PATH": f"{binaries}:{os.environ['PATH']}", "HOGWILD_GPU": "3",
        "HF_HOME": str(tmp_path / "model-cache"), "HF_HUB_CACHE": str(tmp_path / "model-cache/hub"),
        "PYTORCH_ALLOC_CONF": "should_be_unset", "PYTORCH_CUDA_ALLOC_CONF": "should_be_unset",
    }
    tail = ["--output", "fixture-output"] if worker else ["--help"]
    result = subprocess.run([str(script), *args, *tail], env=environment, text=True, capture_output=True, check=True)
    data = json.loads(result.stdout)
    suffix = "budget_curve_eval" if budget and worker else "budget_curve_sweep" if budget else "eval" if worker else "sweep"
    if worker:
        assert data["env"]["HOGWILD_EVAL_MODULE"] == f"minisgl_examples.training.parallel_thinking.{suffix}"
        assert data["argv"][-2:] == tail
    else:
        assert data["argv"][-2:] == [f"minisgl_examples.training.parallel_thinking.{suffix}", "--help"]
    assert data["env"]["HF_HOME"] == environment["HF_HOME"] and data["env"]["HF_HUB_OFFLINE"] == "1"
    assert data["env"]["PYTORCH_ALLOC_CONF"] is None and data["env"]["PYTORCH_CUDA_ALLOC_CONF"] is None
    assert data["env"]["CUDA_VISIBLE_DEVICES"] == ("GPU-test-3" if worker else "")


def test_evaluator_requires_explicit_physical_gpu(tmp_path):
    project = Path(__file__).resolve().parents[2]
    shutil.copy2(project / "run_hogwild.sh", tmp_path / "run_hogwild.sh")
    interpreter = tmp_path / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    environment = {k: v for k, v in os.environ.items() if k != "HOGWILD_GPU"}
    result = subprocess.run([str(tmp_path / "run_hogwild.sh")], env=environment, text=True, capture_output=True)
    assert result.returncode == 2 and "HOGWILD_GPU must select" in result.stderr
