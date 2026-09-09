"""BF16 vision arithmetic contracts independently of loading a checkpoint."""
import pytest
import torch
from minisgl.models.qwen3_5_vision import _vision_rope, _interpolate_position_embedding


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16,torch.float32])
def test_vision_rotary_matches_fp32_reference(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.manual_seed(918)
    x=torch.randn(19,3,48,device=device,dtype=dtype)
    angle=torch.randn(19,24,device=device).repeat(1,2)
    cos,sin=angle.cos(),angle.sin()
    actual=_vision_rope(x,cos,sin)
    left,right=x.float().chunk(2,-1)
    c,s=cos[:,:24,None].transpose(1,2),sin[:,:24,None].transpose(1,2)
    expected=torch.cat((left*c-right*s,right*c+left*s),-1).to(dtype)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    if dtype!=torch.float32:
        old=x*cos.to(dtype)[:,None]+torch.cat((-x[...,24:],x[...,:24]),-1)*sin.to(dtype)[:,None]
        assert (old!=actual).any()


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
def test_position_embedding_rounding_matches_reference(dtype):
    torch.manual_seed(817)
    table=torch.randn(36,12,dtype=dtype)
    idx=torch.randint(36,(4,19))
    fraction=torch.rand(4,19);fraction/=fraction.sum(0,keepdim=True)
    actual=_interpolate_position_embedding(table,idx,fraction)
    terms=[(table[idx[i]]*fraction[i,:,None].to(dtype)).float() for i in range(4)]
    expected=torch.stack(terms).sum(0).to(dtype)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
