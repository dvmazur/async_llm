"""Test CLI options. Model execution is opt-in and owned by E2E fixtures."""
import sys

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("FP8 end-to-end correctness")
    group.addoption("--fp8-model", help="Serialized Qwen FP8 checkpoint; enables fresh GPU E2E runs")
    group.addoption("--chain-bf16-model", help="Unquantized Qwen for the separate, short chain control")
    group.addoption("--fp8-moe-model", help="Separate MoE FP8 checkpoint; enables teacher-forced MoE chain parity")
    group.addoption("--fp8-reference-memory-fraction", type=float, default=None,
                    help="SGLang static memory fraction (default: .15; .6 for separate MoE)")
    group.addoption("--fp8-results", help="New artifact directory (default: pytest temporary directory)")
    group.addoption("--fp8-sglang-python", default=sys.executable,
                    help="Already-installed SGLang interpreter (default: current Python)")
    group.addoption("--fp8-transformers-python", default=sys.executable,
                    help="Already-installed Transformers interpreter (default: current Python)")
    group.addoption("--fp8-timeout", type=float, default=1800,
                    help="Timeout per isolated model worker, seconds (default: 1800)")


def pytest_configure(config):
    if config.getoption("--fp8-results") and not any(config.getoption(flag) for flag in
            ("--fp8-model", "--chain-bf16-model", "--fp8-moe-model")):
        raise pytest.UsageError("--fp8-results requires a model option")
    if config.getoption("--fp8-timeout") <= 0:
        raise pytest.UsageError("--fp8-timeout must be positive")
    fraction = config.getoption("--fp8-reference-memory-fraction")
    if fraction is not None and not 0 < fraction < 1:
        raise pytest.UsageError("--fp8-reference-memory-fraction must be between 0 and 1")
