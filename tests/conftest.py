"""Test CLI options. Model execution is opt-in and owned by E2E fixtures."""
import sys

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("FP8 end-to-end correctness")
    group.addoption("--fp8-model", help="Serialized Qwen FP8 checkpoint; enables fresh GPU E2E runs")
    group.addoption("--fp8-results", help="New artifact directory (default: pytest temporary directory)")
    group.addoption("--fp8-sglang-python", default=sys.executable,
                    help="Already-installed SGLang interpreter (default: current Python)")
    group.addoption("--fp8-transformers-python", default=sys.executable,
                    help="Already-installed Transformers interpreter (default: current Python)")
    group.addoption("--fp8-timeout", type=float, default=1800,
                    help="Timeout per isolated model worker, seconds (default: 1800)")


def pytest_configure(config):
    if config.getoption("--fp8-results") and not config.getoption("--fp8-model"):
        raise pytest.UsageError("--fp8-results requires --fp8-model")
    if config.getoption("--fp8-timeout") <= 0:
        raise pytest.UsageError("--fp8-timeout must be positive")
