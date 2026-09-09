"""Qwen gate/RoPE rounding boundaries, independent mathematical references."""
from types import SimpleNamespace

import pytest
import torch

from minisgl.models.qwen3_5_attn import Qwen3_5Attention, _apply_output_gate
from minisgl.models.qwen3_5_delta import _gdn_gates_eager, _compiled_gdn_gates
from minisgl.shared_cache.attention import SharedCacheAttention


def test_output_gate_keeps_sigmoid_in_fp32():
    torch.manual_seed(724)
    x=torch.randn(7,4,32,dtype=torch.bfloat16)
    gate=torch.randn_like(x)
    actual=_apply_output_gate(x,gate)
    expected=(x.float()*gate.float().sigmoid()).to(x.dtype)
    old=x*gate.sigmoid()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert (old!=expected).any(), 'fixture must expose premature sigmoid rounding'


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_decode_beta_fp32_and_prefill_contract(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.manual_seed(412)
    a=torch.randn(13,16,device=device,dtype=torch.bfloat16)
    b=torch.randn_like(a)
    log=torch.randn(16,device=device,dtype=torch.bfloat16)
    bias=torch.randn_like(log)
    fn=_compiled_gdn_gates if device=='cuda' else _gdn_gates_eager
    pre,g_pre=fn(a,b,log,bias)
    dec,g_dec=fn(a,b,log,bias,beta_fp32=True)
    assert pre.dtype==torch.bfloat16 and dec.dtype==torch.float32
    torch.testing.assert_close(pre,b.sigmoid(),rtol=0,atol=0)
    torch.testing.assert_close(dec,b.float().sigmoid(),rtol=3e-6,atol=3e-7)
    torch.testing.assert_close(g_dec,g_pre,rtol=3e-6,atol=3e-7)
    assert (dec!=pre.float()).any()


def rotate_reference(x,freq):
    half=freq.shape[-1]
    result=x.clone()
    # scalar frequency columns make the head/half ordering explicit.
    for column in range(half):
        c=freq[:,column].cos()[:,None]
        s=freq[:,column].sin()[:,None]
        left=x[...,column].float();right=x[...,column+half].float()
        result[...,column]=(left*c-right*s).to(x.dtype)
        result[...,column+half]=(right*c+left*s).to(x.dtype)
    return result


def test_rope_rotation_keeps_cosines_and_products_fp32():
    torch.manual_seed(66)
    x=torch.randn(7,2,96,dtype=torch.bfloat16)
    freqs=torch.randn(7,32)
    before=x.clone()
    out=Qwen3_5Attention._apply_from_freqs(SimpleNamespace(rotary_dim=64),x,freqs)
    torch.testing.assert_close(out,rotate_reference(x,freqs),rtol=0,atol=0)
    torch.testing.assert_close(out[...,64:],x[...,64:],rtol=0,atol=0)
    torch.testing.assert_close(x,before,rtol=0,atol=0)


@pytest.mark.parametrize('three_axes',[False,True])
def test_shared_mrope_matches_explicit_axis_reference(three_axes):
    torch.manual_seed(606)
    x=torch.randn(7,2,96,dtype=torch.bfloat16)
    inv=1/(10000000**(torch.arange(0,64,2).float()/64))
    positions=torch.arange(7) if not three_axes else torch.arange(21).reshape(3,7)
    owner=SimpleNamespace(device=torch.device('cpu'),rotary_dim=64,
        _mrope_inv_freq=inv,mrope_section=(11,11,10))
    out=SharedCacheAttention._rope_mrope(owner,x,positions)
    pos3=positions.expand(3,-1) if not three_axes else positions
    freq=torch.empty(7,32)
    for col in range(32):
        axis=1 if col%3==1 and col<33 else 2 if col%3==2 and col<30 else 0
        freq[:,col]=pos3[axis].float()*inv[col]
    torch.testing.assert_close(out,rotate_reference(x,freq),rtol=0,atol=0)
