"""CPU-only dispatch/contract checks; numerical CUDA parity is gated separately."""

import os
from unittest.mock import patch
import torch
from minisgl.kernel.fp8 import use_sm90_backend, block_fp8_linear


def test_default_and_emulation_never_enter_hopper_backend():
    x = torch.ones(2, 256, dtype=torch.bfloat16)
    with (
        patch.dict(os.environ, {"MINISGL_FP8_BACKEND": "triton"}),
        patch("torch.cuda.get_device_capability") as cap,
    ):
        assert not use_sm90_backend(x)
        cap.assert_not_called()
    with (
        patch.dict(
            os.environ, {"MINISGL_FP8_BACKEND": "flashinfer_sm90", "MINISGL_FP8_EMULATE": "1"}
        ),
        patch("torch.cuda.get_device_capability") as cap,
    ):
        assert not use_sm90_backend(x)
        cap.assert_not_called()


def test_hopper_requires_sm90_and_supported_output_dtype():
    for capability, expected in [((9, 0), True), ((8, 9), False), ((8, 6), False)]:
        with (
            patch.dict(
                os.environ, {"MINISGL_FP8_BACKEND": "flashinfer_sm90", "MINISGL_FP8_EMULATE": "0"}
            ),
            patch("torch.cuda.get_device_capability", return_value=capability),
        ):
            assert use_sm90_backend(torch.empty(1, 256, dtype=torch.bfloat16)) == expected
    with patch.dict(os.environ, {"MINISGL_FP8_BACKEND": "flashinfer_sm90"}):
        assert not use_sm90_backend(torch.empty(1, 256, dtype=torch.float32))


def test_hopper_preserves_block_scales_shape_and_bias():
    x = torch.ones(2, 3, 256, dtype=torch.bfloat16).transpose(0, 1)
    w = torch.zeros(128, 256, dtype=torch.float8_e4m3fn)
    s = torch.ones(1, 2)
    bias = torch.ones(128, dtype=x.dtype)
    q = torch.zeros(6, 256, dtype=torch.float8_e4m3fn)
    qs = torch.ones(6, 2)

    def fill(a, b, sa, sb, out):
        assert a is q and b is w and sa is qs and sb is s
        out.fill_(2)

    with (
        patch("minisgl.kernel.fp8.quantize_fp8_groups", return_value=(q, qs)),
        patch("minisgl.kernel.fp8.use_sm90_backend", return_value=True),
        patch("minisgl.kernel.fp8.sm90_linear", side_effect=fill),
    ):
        result = block_fp8_linear(x, w, s, bias)
    assert result.shape == (3, 2, 128)
    torch.testing.assert_close(result, torch.full_like(result, 3), rtol=0, atol=0)
