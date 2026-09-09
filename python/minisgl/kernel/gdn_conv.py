"""Causal GDN convolution with the stock SGLang rounding contract.

Products round to the input dtype; their sum and SiLU are FP32. State stores
unmodified projection inputs, not convolved values. There is no cache policy
here: callers own gathering/storing the returned raw window.
"""
import torch
import torch.nn.functional as F


def causal_conv1d_silu(x, weight, prior=None):
    """[B,T,C] + [C,1,K] + optional [B,C,K] -> output and new raw window.

    Does not modify or alias inputs. Arbitrary tensor strides are supported.
    The eager CPU path also serves the optional no-FLA implementation.
    """
    if x.ndim != 3 or weight.ndim != 3 or weight.shape[1] != 1:
        raise ValueError('Expected x[B,T,C] and depthwise weight[C,1,K]')
    batch, length, channels = x.shape
    kernel = weight.shape[-1]
    if kernel < 1 or channels != weight.shape[0]:
        raise ValueError('Invalid causal-convolution dimensions')
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('Expected FP16/BF16/FP32 projection inputs')
    if x.dtype != weight.dtype or x.device != weight.device:
        raise ValueError('Projection inputs and convolution weights must match dtype/device')
    if prior is not None and (prior.shape != (batch, channels, kernel)
            or prior.dtype != x.dtype or prior.device != x.device):
        raise ValueError('Expected matching raw prior window[B,C,K]')
    if not x.is_cuda or not batch or not length or not channels:
        history = x.new_zeros(batch, channels, kernel) if prior is None else prior
        joined = torch.cat((history, x.transpose(1, 2)), dim=-1)
        acc = torch.zeros(batch, channels, length, dtype=torch.float32, device=x.device)
        for tap in range(kernel):
            # Keep this rounding explicit: FP32 products followed by a single
            # cast after the sum implement a DIFFERENT inference arithmetic.
            product = joined[..., tap + 1:tap + 1 + length] * weight[:, 0, tap][None, :, None]
            acc = acc + product.float()
        return F.silu(acc).to(x.dtype).transpose(1, 2).contiguous(), joined[..., -kernel:].clone()

    from .triton.gdn_conv import launch_causal_conv

    return launch_causal_conv(x, weight, prior)
