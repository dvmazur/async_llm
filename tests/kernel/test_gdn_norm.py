"""Independent eager/FP64 oracles for SGLang-compatible fused Qwen GDN norm."""
import pytest
import torch
from minisgl.kernel.gdn_norm import gated_rmsnorm,gated_rmsnorm_eager


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
@pytest.mark.parametrize('shape',[(1,128),(3,4,80),(37,16,128),(0,8,128)])
def test_gated_norm(device,dtype,shape):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.manual_seed(816)
    x=torch.randn(*shape[:-1],shape[-1]*2,device=device,dtype=dtype)[...,::2]
    gate=torch.randn_like(x);w=torch.randn(shape[-1],device=device,dtype=dtype)
    before=x.clone();actual=gated_rmsnorm(x,w,gate,1e-6)
    ref=gated_rmsnorm_eager(x,w,gate,1e-6)
    atol=2e-2 if dtype==torch.bfloat16 else 2e-3 if dtype==torch.float16 else 3e-6
    torch.testing.assert_close(actual,ref,atol=atol,rtol=atol)
    torch.testing.assert_close(x,before,rtol=0,atol=0)
    if x.numel():
        xf=x.double();gf=gate.double()
        oracle=xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-6)*w.double()*(gf*gf.sigmoid())
        assert float((actual.double()-oracle).norm()/oracle.norm()) < (0.004 if dtype==torch.bfloat16 else 0.001 if dtype==torch.float16 else 1e-6)


def test_reject_mismatched_gate():
    with pytest.raises(ValueError):gated_rmsnorm(torch.zeros(3,128),torch.ones(128),torch.zeros(3,64),1e-6)
