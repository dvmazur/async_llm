from pathlib import Path

import pytest

from .suite import RunConfig, collect_profile


@pytest.fixture(scope="session")
def fp8_output_root(pytestconfig, tmp_path_factory):
    output = Path(pytestconfig.getoption("--fp8-results") or tmp_path_factory.mktemp("fp8-e2e")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        pytest.fail(f"Refusing to overwrite {output}; choose a new --fp8-results directory")
    return output


def _config(pytestconfig, request, flag, *, subdir=None, quantization="fp8", require_moe=False):
    model = pytestconfig.getoption(flag)
    if not model:
        pytest.skip(f"Opt in with {flag} CHECKPOINT for a fresh GPU run")
    import torch
    if not torch.cuda.is_available():
        pytest.fail(f"{flag} requests a GPU run, but CUDA is unavailable")
    output = request.getfixturevalue("fp8_output_root")
    fraction = pytestconfig.getoption("--fp8-reference-memory-fraction")
    return RunConfig(output / subdir if subdir else output, model=model,
                    quantization=quantization, require_moe=require_moe,
                    reference_memory_fraction=fraction if fraction is not None else (.6 if require_moe else .15),
                    sglang_python=pytestconfig.getoption("--fp8-sglang-python"),
                    transformers_python=pytestconfig.getoption("--fp8-transformers-python"),
                    timeout=pytestconfig.getoption("--fp8-timeout"))


@pytest.fixture(scope="session")
def fp8_config(pytestconfig, request):
    return _config(pytestconfig, request, "--fp8-model")


@pytest.fixture(scope="session")
def chain_artifacts(fp8_config):
    return collect_profile(fp8_config, "shared-chains")


@pytest.fixture(scope="session")
def chain_bf16_artifacts(pytestconfig, request):
    config = _config(pytestconfig, request, "--chain-bf16-model", subdir="bf16", quantization=None)
    return collect_profile(config, "shared-chains")


@pytest.fixture(scope="session")
def moe_config(pytestconfig, request):
    return _config(pytestconfig, request, "--fp8-moe-model", subdir="moe", require_moe=True)


@pytest.fixture(scope="session")
def chain_moe_artifacts(moe_config):
    return collect_profile(moe_config, "shared-chains")


@pytest.fixture(scope="session")
def moe_serving_artifacts(moe_config):
    return collect_profile(moe_config, "serving")


@pytest.fixture(scope="session")
def serving_artifacts(fp8_config):
    return collect_profile(fp8_config, "serving")


@pytest.fixture(scope="session")
def shared_cache_artifacts(fp8_config):
    return collect_profile(fp8_config, "shared-cache")
