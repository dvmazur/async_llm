"""Causal windows, streaming, and low-product/FP32-accumulator precision."""
import pytest
import torch
import torch.nn.functional as F

from minisgl.kernel.gdn_conv import causal_conv1d_silu


def reference(x, weight, prior):
    b, _, c = x.shape
    k = weight.shape[-1]
    prior = x.new_zeros(b, c, k) if prior is None else prior
    raw = torch.cat((prior, x.transpose(1,2)), -1)
    windows = raw.unfold(-1, k, 1)[...,1:,:]
    products = windows * weight[:,0,:][None,:,None,:]
    output = F.silu(products.float().sum(-1)).to(x.dtype).transpose(1,2)
    return output, raw[...,-k:]


@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('length,kernel,with_prior', [(1,4,False),(1,4,True),(2,4,True),
                                                    (7,4,False),(11,3,True),(5,1,False)])
def test_conv_precision_and_windows(device, dtype, length, kernel, with_prior):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA required')
    torch.manual_seed(420)
    # Non-contiguous input, weight and state with odd channel count.
    x = torch.randn(3,length,74,device=device,dtype=dtype)[...,::2]
    weight = torch.randn(37,1,kernel*2,device=device,dtype=dtype)[...,::2]
    prior = torch.randn(3,74,kernel,device=device,dtype=dtype)[:,::2] if with_prior else None
    snapshots = [t.clone() for t in (x,weight) + ((prior,) if prior is not None else ())]
    output, window = causal_conv1d_silu(x,weight,prior)
    expected, expected_window = reference(x,weight,prior)
    torch.testing.assert_close(output,expected,rtol=.008 if dtype==torch.bfloat16 else .002,atol=1e-5)
    relative = (output.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-9)
    assert relative < 3e-4
    torch.testing.assert_close(window,expected_window,rtol=0,atol=0)
    for t,before in zip((x,weight)+((prior,) if prior is not None else ()),snapshots):
        torch.testing.assert_close(t,before,rtol=0,atol=0)
    assert output.is_contiguous() and window.is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_streaming_and_batch_equivalence():
    torch.manual_seed(991)
    x = torch.randn(3,13,128,device='cuda',dtype=torch.bfloat16)
    weight = torch.randn(128,1,4,device='cuda',dtype=x.dtype)
    whole, final = causal_conv1d_silu(x,weight)
    pieces=[];state=None
    for start,end in [(0,1),(1,3),(3,9),(9,13)]:
        out,state=causal_conv1d_silu(x[:,start:end],weight,state);pieces.append(out)
    torch.testing.assert_close(torch.cat(pieces,1),whole,rtol=0,atol=0)
    torch.testing.assert_close(state,final,rtol=0,atol=0)
    for i in range(3):
        row,row_state=causal_conv1d_silu(x[i:i+1],weight)
        torch.testing.assert_close(row,whole[i:i+1],rtol=0,atol=0)
        torch.testing.assert_close(row_state,final[i:i+1],rtol=0,atol=0)
    changed=x.clone();changed[:,-1]*=10
    out,_=causal_conv1d_silu(changed,weight)
    torch.testing.assert_close(out[:,:-1],whole[:,:-1],rtol=0,atol=0)


def test_empty_and_invalid():
    x=torch.empty(2,0,3,dtype=torch.bfloat16)
    w=torch.ones(3,1,4,dtype=x.dtype)
    old=torch.randn(2,3,4,dtype=x.dtype)
    out,state=causal_conv1d_silu(x,w,old)
    assert out.shape==x.shape and state.data_ptr()!=old.data_ptr()
    torch.testing.assert_close(state,old,rtol=0,atol=0)
    with pytest.raises(ValueError): causal_conv1d_silu(x,w.float())
    with pytest.raises(ValueError): causal_conv1d_silu(x,w,old[:,:,:3])
