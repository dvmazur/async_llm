"""Ragged image attention: numerical oracle, boundaries, strides, CPU fallback."""
import pytest
import torch
from minisgl.kernel.vision_attention import prepare_vision_attention,vision_attention


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('dim,lengths',[(32,(1,7)),(64,(16,97,3)),(80,(13,129)),(128,(67,))])
def test_attention_against_float64_oracle(device,dtype,dim,lengths):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.manual_seed(905)
    # Non-contiguous row/head layout exercises real QKV projection views.
    q,k,v=[torch.randn(sum(lengths),3,dim*2,device=device,dtype=dtype)[...,::2] for _ in range(3)]
    plan=prepare_vision_attention(lengths,q.device)
    before=[x.clone() for x in (q,k,v)]
    actual=vision_attention(q,k,v,plan)
    expected=[];offset=0
    for n in lengths:
        qq,kk,vv=[x[offset:offset+n].double().transpose(0,1) for x in (q,k,v)]
        expected.append(((qq@kk.transpose(-1,-2)/(dim**.5)).softmax(-1)@vv).transpose(0,1))
        offset+=n
    expected=torch.cat(expected)
    relative=(actual.double()-expected).norm()/expected.norm()
    assert relative < {torch.float32:2e-5,torch.float16:1e-3,torch.bfloat16:8e-3}[dtype]
    assert actual.dtype==dtype and actual.shape==q.shape
    for x,original in zip((q,k,v),before):torch.testing.assert_close(x,original,rtol=0,atol=0)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_images_do_not_attend_to_each_other(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.manual_seed(919)
    q,k,v=[torch.randn(29,2,64,device=device,dtype=torch.bfloat16) for _ in range(3)]
    plan=prepare_vision_attention((13,16),q.device)
    first=vision_attention(q,k,v,plan)
    v[13:]+=100
    second=vision_attention(q,k,v,plan)
    torch.testing.assert_close(first[:13],second[:13],rtol=0,atol=0)
    assert (first[13:]!=second[13:]).any()


def test_reject_invalid_segments_and_plan():
    for lengths in [(),(2,0),(3,-1)]:
        with pytest.raises(ValueError):prepare_vision_attention(lengths,'cpu')
    x=torch.zeros(7,2,32)
    with pytest.raises(ValueError):vision_attention(x,x,x,prepare_vision_attention((8,),'cpu'))
