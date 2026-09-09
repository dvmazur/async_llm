"""Qwen GDN RMSNorm then SiLU gate, with a portable eager fallback."""
import torch
import torch.nn.functional as F


def gated_rmsnorm_eager(x,weight,gate,eps):
    xf=x.float()
    normalized=xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+eps)
    return (normalized*weight.float()*F.silu(gate.float())).to(x.dtype)


def gated_rmsnorm(x,weight,gate,eps):
    if (x.ndim<1 or not x.shape[-1] or weight.shape!=(x.shape[-1],) or gate.shape!=x.shape
        or x.device!=gate.device or x.device!=weight.device):
        raise ValueError('Expected matching x/gate[...,D] and weight[D]')
    if x.dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise ValueError('GDN norm requires floating-point model activations')
    if x.is_cuda and x.numel() and x.shape[-1]<=65536//x.element_size():
        from .triton.gdn_norm import rmsnorm_sglang
        return rmsnorm_sglang(x,weight.contiguous(),gate,eps)
    return gated_rmsnorm_eager(x,weight,gate,eps)
