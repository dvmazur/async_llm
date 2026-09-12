from pathlib import Path

import pytest

from .suite import RunConfig, collect_profile


@pytest.fixture(scope="session")
def fp8_config(pytestconfig, tmp_path_factory):
    model = pytestconfig.getoption("--fp8-model")
    if not model:
        pytest.skip("Opt in with --fp8-model CHECKPOINT for a fresh GPU run")
    import torch

    if not torch.cuda.is_available():
        pytest.fail("--fp8-model requests a GPU run, but CUDA is unavailable")
    output = Path(pytestconfig.getoption("--fp8-results") or tmp_path_factory.mktemp("fp8-e2e")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        pytest.fail(f"Refusing to overwrite {output}; choose a new --fp8-results directory")
    return RunConfig(output, model=model,
                    sglang_python=pytestconfig.getoption("--fp8-sglang-python"),
                    transformers_python=pytestconfig.getoption("--fp8-transformers-python"),
                    timeout=pytestconfig.getoption("--fp8-timeout"))


@pytest.fixture(scope="session")
def serving_artifacts(fp8_config):
    return collect_profile(fp8_config, "serving")


@pytest.fixture(scope="session")
def shared_cache_artifacts(fp8_config):
    return collect_profile(fp8_config, "shared-cache")


@pytest.fixture(scope="session")
def shared_batched_artifacts(fp8_config):
    return collect_profile(fp8_config, "shared-batched")
