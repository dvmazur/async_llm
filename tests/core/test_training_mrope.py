"""The compile-friendly mRoPE rewrite preserves the recorded axis convention."""

import pytest
import torch
from minisgl.training.mrope import rotate


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("section", [(11, 11, 10), (16, 0, 16), (1, 1, 1)])
def test_functional_mrope_matches_frequency_by_frequency_reference(dtype, section):
    generator = torch.Generator().manual_seed(211)
    x = torch.randn(5, 4, 256, generator=generator).to(dtype).requires_grad_()
    reference = x.detach().clone().requires_grad_()
    positions = torch.tensor([[3, 11, 29], [4, 12, 31], [5, 15, 38], [6, 19, 41], [7, 23, 47]])
    inv_freq = 10000000 ** (-torch.arange(0, 64, 2).float() / 64)
    angles = []
    for frequency in range(len(inv_freq)):
        axis = frequency % 3
        if axis and frequency >= section[axis] * 3:
            axis = 0
        angles.append(positions[:, axis].float() * inv_freq[frequency])
    angles = torch.stack(angles, dim=-1)[:, None]
    left, right = reference[..., :64].float().chunk(2, dim=-1)
    rotated = torch.cat(
        (left * angles.cos() - right * angles.sin(), right * angles.cos() + left * angles.sin()),
        dim=-1,
    )
    expected = torch.cat((rotated.to(dtype), reference[..., 64:]), dim=-1)
    actual = rotate(x, positions, inv_freq, section)
    upstream = torch.randn(actual.shape, generator=generator).to(dtype)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, upstream)[0],
        torch.autograd.grad(expected, reference, upstream)[0],
        atol=0,
        rtol=0,
    )
