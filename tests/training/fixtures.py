import inspect
from functools import partial

import pytest
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5

from tests.training.references import Float64AttentionMath


@pytest.fixture
def fp64_attention(monkeypatch):
    from minisgl.training import attention as attention_module

    monkeypatch.setattr(
        attention_module,
        "checkpoint",
        partial(
            attention_module.checkpoint,
            context_fn=lambda: (Float64AttentionMath(), Float64AttentionMath()),
        ),
    )


@pytest.fixture(autouse=True)
def native_torch_kernels(monkeypatch):
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    ):
        monkeypatch.setattr(modeling_qwen3_5, name, inspect.unwrap(getattr(modeling_qwen3_5, name)))


@pytest.fixture
def cuda_delta_math():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.fixture(scope="module", autouse=True)
def torch_reference_kernels():
    """Keep FP32 reference arithmetic independent of optional inference kernels."""
    with pytest.MonkeyPatch.context() as patch:
        for name in (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
            "FusedRMSNormGated",
        ):
            # Legacy optional hooks; current HF dispatch is unwrapped below.
            patch.setattr(modeling_qwen3_5, name, None, raising=False)
        for name in (
            "torch_causal_conv1d_update",
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
        ):
            patch.setattr(modeling_qwen3_5, name, inspect.unwrap(getattr(modeling_qwen3_5, name)))
        patch.setattr(torch.backends.cudnn, "allow_tf32", False)
        yield
