"""Original ordinary-serving greedy invariant, run in isolated model processes.

Set MINISGL_E2E_MODEL to Gemma FP8. Unlike the teacher-forced external panel,
this checks actual generated sequences with chunking, radix reuse and mixing.
"""
import os

import pytest

from test_mixed_batch import TestMixedEquivalence as _Mixed


@pytest.fixture(autouse=True)
def require_gemma():
    path = os.environ.get("MINISGL_E2E_MODEL", "")
    if not path:
        pytest.skip("Set MINISGL_E2E_MODEL to the Gemma checkpoint")
    from transformers import AutoConfig
    assert AutoConfig.from_pretrained(path).model_type in ("gemma4", "gemma4_text")


class TestGemmaServing(_Mixed):
    pass
