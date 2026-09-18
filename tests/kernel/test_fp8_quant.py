"""Independent arithmetic oracles; no model, cache or scheduler optimizations."""
import pytest
import torch
import torch.nn.functional as F

from minisgl.kernel.fp8 import block_fp8_linear, quantize_fp8_groups, validate_weight


def quantize_reference(x):
    groups = x.float().reshape(x.shape[0], -1, 128)
    maximum = groups.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scales = maximum / 448.
    q = (groups * (448. / maximum)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape_as(x), scales.squeeze(-1)


def dequant(weight, scales):
    return weight.float() * scales.repeat_interleave(128, -2).repeat_interleave(128, -1)


def random_weight(shape):
    return ((torch.randn(shape, device='cuda') * 20).to(torch.float8_e4m3fn),
            torch.rand((*shape[:-2], shape[-2]//128, shape[-1]//128), device='cuda')*.002+.001)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('rows', [0, 1, 2, 5, 16, 33, 64, 128, 137])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
def test_quantizer_and_linear(rows, dtype):
    torch.manual_seed(107)
    x = torch.randn(rows, 384, device='cuda', dtype=dtype)
    q, scales = quantize_fp8_groups(x)
    if not rows:
        w, ws = random_weight((256,384))
        assert block_fp8_linear(x,w,ws).shape == (0,256)
        return
    qr, sr = quantize_reference(x)
    torch.testing.assert_close(scales, sr, rtol=2e-7, atol=0)
    # CUDA div.approx and torch division can choose opposite neighbors at
    # an E4M3 halfway point. Require every difference to be such a tie;
    # non-boundary quantization errors must still fail this independent oracle.
    different=q.float()!=qr.float()
    if different.any():
        groups=x.double().reshape(rows,-1,128)
        scaled=(groups*448/groups.abs().amax(-1,keepdim=True).clamp_min(1e-10)).reshape_as(x)
        midpoint=(q.float()+qr.float())/2
        torch.testing.assert_close(scaled[different],midpoint[different].double(),rtol=1e-6,atol=1e-6)
        assert ((q.float()-qr.float()).abs()[different] <= qr.float().abs()[different]*.125).all()
    w, ws = random_weight((256,384))
    actual = block_fp8_linear(x,w,ws)
    reference = F.linear(q.float()*scales.repeat_interleave(128,-1), dequant(w,ws)).to(dtype)
    relative = (actual.float()-reference.float()).norm()/reference.float().norm()
    assert relative < .003
    torch.testing.assert_close(actual, reference, rtol=.02, atol=.006)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_bias_noncontiguous_and_zero():
    torch.manual_seed(42)
    w,ws = random_weight((128,256))
    x = torch.randn(2,3,256,device='cuda',dtype=torch.bfloat16).transpose(0,1)
    bias = torch.randn(128,device='cuda',dtype=x.dtype)
    expected = block_fp8_linear(x.contiguous(),w,ws)+bias
    torch.testing.assert_close(block_fp8_linear(x,w,ws,bias),expected,rtol=0,atol=0)
    q,s = quantize_fp8_groups(torch.zeros(3,256,device='cuda',dtype=x.dtype))
    assert not q.float().count_nonzero() and torch.isfinite(s).all() and (s>0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_scale_product_rounding_boundary():
    x = torch.zeros(1,256,device='cuda',dtype=torch.bfloat16)
    x[0,0],x[0,128]=1.5,.625
    w = torch.zeros(128,256,device='cuda').to(torch.float8_e4m3fn)
    w[:,0],w[:,128]=1,1
    scales = torch.tensor([[.4874778985977173,.436303049325943]],device='cuda')
    actual = block_fp8_linear(x,w,scales)
    torch.testing.assert_close(actual,torch.ones_like(actual),rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('rows',[1,5,64,128,137])
def test_exact_sglang_quantizer(rows):
    if torch.cuda.get_device_capability() < (8,9):
        pytest.skip('SGLang native FP8 reference requires Ada or newer')
    reference=pytest.importorskip('sglang.kernels.ops.quantization.fp8_kernel')
    torch.manual_seed(107)
    x=torch.randn(rows,384,device='cuda',dtype=torch.bfloat16)
    q,s=quantize_fp8_groups(x)
    qr,sr=reference.sglang_per_token_group_quant_fp8(x,128)
    torch.testing.assert_close(q.float(),qr.float(),rtol=0,atol=0)
    torch.testing.assert_close(s,sr,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('rows,n,k',[(1,1024,3584),(2,1024,2048),(5,3584,1024),
                                    (64,1024,2048),(128,1024,3584),(137,2048,1024)])
def test_linear_matches_sglang_cutlass(rows,n,k):
    if torch.cuda.get_device_capability() < (8,9):
        pytest.skip('SGLang native FP8 reference requires Ada or newer')
    reference=pytest.importorskip('sglang.srt.layers.quantization.fp8_utils')
    torch.manual_seed(412)
    x=torch.randn(rows,k,device='cuda',dtype=torch.bfloat16)
    w,s=random_weight((n,k))
    actual=block_fp8_linear(x,w,s)
    expected=reference.cutlass_w8a8_block_fp8_linear_with_fallback(x,w,[128,128],s)
    relative=(actual.float()-expected.float()).norm()/expected.float().norm()
    assert relative<2e-4
    assert (actual==expected).float().mean()>.999


@pytest.mark.parametrize('bad', ['missing','dtype','shape','unaligned'])
def test_bad_weight_metadata(bad):
    w = torch.empty((128,256),dtype=torch.float8_e4m3fn)
    s = torch.ones(1,2)
    if bad=='missing':s=None
    elif bad=='dtype':s=s.bfloat16()
    elif bad=='shape':s=torch.ones(2,2)
    else:w=torch.empty((127,256),dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError):validate_weight(w,s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('rows,on_input', [(1,False),(5,False),(16,False),(33,False),(5,True)])
def test_moe_fp8(rows,on_input):
    from minisgl.moe.fused import fused_experts_impl
    torch.manual_seed(9)
    e,h,d,topk=4,256,128,2
    x=torch.randn(rows,h,device='cuda',dtype=torch.bfloat16)
    w1,s1=random_weight((e,2*d,h));w2,s2=random_weight((e,h,d))
    ids=torch.stack([torch.randperm(e,device='cuda')[:topk] for _ in range(rows)]).int()
    scores=torch.softmax(torch.randn(rows,topk,device='cuda'),-1)
    qx,xs=quantize_reference(x)
    expected=torch.zeros_like(x,dtype=torch.float32)
    for row in range(rows):
        for slot in range(topk):
            expert=int(ids[row,slot])
            hidden=F.linear(qx[row].float()*xs[row].repeat_interleave(128),dequant(w1[expert],s1[expert]))
            if on_input:hidden*=scores[row,slot]
            gate,up=hidden.to(x.dtype).float().chunk(2)
            hidden=(F.silu(gate)*up).to(x.dtype)[None]
            qh,hs=quantize_reference(hidden)
            value=F.linear(qh.float()*hs.repeat_interleave(128,-1),dequant(w2[expert],s2[expert]))[0]
            if not on_input:value*=scores[row,slot]
            expected[row]+=value.to(x.dtype).float()
    actual=fused_experts_impl(x.clone(),w1,w2,scores,ids,
        apply_router_weight_on_input=on_input,w1_scale=s1,w2_scale=s2)
    assert (actual.float()-expected).norm()/expected.norm()<.02
    torch.testing.assert_close(actual,expected.to(x.dtype),rtol=.08,atol=.008)
