"""Public pytest entry points; all full-model tests consume fresh session fixtures."""
import pytest
import json

from .common import validate_current_mini
from .comparison import compare
from .serving import check_schedule

pytestmark = pytest.mark.fp8_e2e


def _external_result(artifacts, mode, *, quantization="fp8"):
    mini = artifacts.mini(mode)
    validate_current_mini(mini, mode, quantization=quantization)
    result = compare(mini, artifacts.root / "sglang", artifacts.root / "transformers")
    artifacts.report(f"{mode}_vs_sglang.json", result)
    assert not result["quant_output_ablation"], "Reference-output ablation cannot validate production"
    assert not result["sglang_native_mrope_override"], "Parity requires the default SGLang reference"
    return result


def _check_chain(artifacts, mode, *, quantization="fp8", require_moe=False):
    from .chains import check_coverage, check_moe_execution
    fixtures = json.loads((artifacts.root / "fixtures/fixtures.json").read_text())
    directory = artifacts.mini(mode)
    check_coverage(json.loads((directory / "chain_coverage.json").read_text()), fixtures["cases"], mode)
    storage = json.loads((directory / "storage.json").read_text())
    assert storage["execution"] == "eager"
    if require_moe:
        check_moe_execution(storage)
    result = _external_result(artifacts, mode, quantization=quantization)
    assert result["metrics"]["mini"]["decode"]["positions"] == 21
    assert result["metrics"]["mini"]["prefill"]["positions"] == 9
    assert result["passed"], result["metrics"]


@pytest.mark.parametrize("mode", ["chain-flat", "chain-split", "chain-shared"])
def test_chain_external_parity(chain_artifacts, mode):
    _check_chain(chain_artifacts, mode)


@pytest.mark.parametrize("mode", ["chain-flat", "chain-split", "chain-shared"])
def test_bf16_chain_external_parity(chain_bf16_artifacts, mode):
    _check_chain(chain_bf16_artifacts, mode, quantization=None)


@pytest.mark.parametrize("mode", ["chain-flat", "chain-split", "chain-shared"])
def test_moe_chain_external_parity(chain_moe_artifacts, mode):
    _check_chain(chain_moe_artifacts, mode, require_moe=True)


def test_shared_cache_external_parity(shared_cache_artifacts):
    result = _external_result(shared_cache_artifacts, "shared-cache")
    # Preserve the full panel; the single temporary, fragile 2x TF-error gate
    # is user-approved in comparison.py (not a task-quality guarantee).
    assert result["metrics"]["mini"]["decode"]["positions"] >= 400
    assert result["metrics"]["mini"]["prefill"]["positions"] >= 32
    assert result["passed"], result["metrics"]


@pytest.mark.parametrize("mode", ["mixed", "sequential"])
def test_serving_external_parity(serving_artifacts, mode):
    _check_serving(serving_artifacts, mode)


@pytest.mark.parametrize("mode", ["mixed", "sequential"])
def test_moe_serving_external_parity(moe_serving_artifacts, mode):
    _check_serving(moe_serving_artifacts, mode, require_moe=True)


def _check_serving(serving_artifacts, mode, *, require_moe=False):
    if require_moe:
        from .chains import check_moe_execution
        storage = json.loads((serving_artifacts.mini(mode)/"storage.json").read_text())
        check_moe_execution(storage, require_mixed=mode == "mixed")
    result = _external_result(serving_artifacts, mode)
    assert result["mini_scheduling"] == mode
    schedule = result["mixed_schedule"]
    check_schedule(schedule["forwards"], mode)
    assert schedule["overlap_scheduling"] is True
    assert result["metrics"]["mini"]["decode"]["positions"] >= 32
    assert result["metrics"]["mini"]["prefill"]["positions"] >= 8
    assert result["passed"], result["metrics"]
