"""Public pytest entry points; all full-model tests consume fresh session fixtures."""
import pytest

from .common import validate_current_mini
from .comparison import compare
from .serving import check_schedule

pytestmark = pytest.mark.fp8_e2e


def _external_result(artifacts, mode):
    mini = artifacts.mini(mode)
    validate_current_mini(mini, mode)
    result = compare(mini, artifacts.root / "sglang", artifacts.root / "transformers")
    artifacts.report(f"{mode}_vs_sglang.json", result)
    assert not result["quant_output_ablation"], "Reference-output ablation cannot validate production"
    if mode != "shared-cache":
        assert not result["sglang_native_mrope_override"], "Serving parity requires the default SGLang reference"
    return result


def test_shared_cache_external_parity(shared_cache_artifacts):
    result = _external_result(shared_cache_artifacts, "shared-cache")
    # Original full-parity coverage and numerical thresholds, unchanged.
    assert result["metrics"]["mini"]["decode"]["positions"] >= 400
    assert result["metrics"]["mini"]["prefill"]["positions"] >= 32
    assert result["passed"], result["metrics"]


@pytest.mark.parametrize("mode", ["mixed", "sequential"])
def test_serving_external_parity(serving_artifacts, mode):
    result = _external_result(serving_artifacts, mode)
    assert result["mini_scheduling"] == mode
    schedule = result["mixed_schedule"]
    check_schedule(schedule["forwards"], mode)
    assert schedule["overlap_scheduling"] is True
    assert result["metrics"]["mini"]["decode"]["positions"] >= 32
    assert result["metrics"]["mini"]["prefill"]["positions"] >= 8
    assert result["passed"], result["metrics"]
