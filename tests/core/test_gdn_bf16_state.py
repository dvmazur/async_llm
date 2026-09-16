"""Opt-in BF16 storage, FP32 accumulation and dtype-safe pointer lifetimes."""
import pytest
import torch

from minisgl.shared_cache.gdn import SharedCacheGDN
from minisgl.shared_cache.gdn_decode import GDNDecodeBuffers
from minisgl.shared_cache.gdn_affine import compose_gdn_affines, update_affine_summary
from minisgl.shared_cache.gdn_state import prepare_state
from minisgl.shared_cache.shared_block import CacheBlock


def test_state_addresses_include_storage_dtype_and_own_conversions():
    block=CacheBlock(torch.device('cpu'))
    a=torch.randn(1,2,8,8);b=torch.randn(1,2,6,8)
    block.linear_affine[0]=(a,b)
    sig=(block.device,1,2,8,6,(7,4),torch.bfloat16)
    fp=prepare_state(block,sig)
    bf=prepare_state(block,(*sig,torch.bfloat16))
    assert bf.owners[0].dtype==bf.owners[1].dtype==torch.bfloat16
    assert fp.owners[0] is a
    assert bf.pointers[0,0]==bf.owners[0].data_ptr()!=a.data_ptr()
    old=bf.owners[0].clone()
    a.add_(1)
    # Converted external/debug tensors aren't cached: changed bytes are seen.
    newer=prepare_state(block,(*sig,torch.bfloat16))
    assert torch.equal(newer.owners[0],a.bfloat16())
    assert torch.equal(bf.owners[0],old)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_bf16_merge_rounds_after_fp32_add(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA')
    torch.manual_seed(947)
    a,c=[torch.randn(2,3,32,32,device=device,dtype=torch.bfloat16)*.02 for _ in range(2)]
    b,d=[torch.randn(2,3,24,32,device=device,dtype=torch.bfloat16)*.02 for _ in range(2)]
    out=compose_gdn_affines(A_first=a,B_first=b,A_second=c,B_second=d)
    expected=((a.float()@c.float()).bfloat16(),(b.float()@c.float()+d.float()).bfloat16())
    for x,y in zip(out,expected):
        assert x.dtype==torch.bfloat16
        torch.testing.assert_close(x,y,atol=1e-5,rtol=.008)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA BF16 pointer kernels')
@pytest.mark.parametrize('dk,dv',[(32,24),(128,128)])
@pytest.mark.parametrize('captured',[False,True])
@torch.inference_mode()
def test_bf16_compose_capture_reordered_chains_and_owned_storage(dk,dv,captured):
    from test_gdn_compose import capture
    torch.manual_seed(948)
    ar=SharedCacheGDN(num_heads=2,head_k_dim=dk,head_v_dim=dv,conv_dim=7,
                      conv_kernel=4,device=torch.device('cuda'),state_dtype=torch.bfloat16)
    buf=GDNDecodeBuffers(ar,2,6,5,torch.bfloat16)
    fp=GDNDecodeBuffers(SharedCacheGDN(num_heads=2,head_k_dim=dk,head_v_dim=dv,
        conv_dim=7,conv_kernel=4,device=ar.device),2,6,5,torch.bfloat16)
    for name in ('b','frontier','initial'):
        small,large=getattr(buf,name),getattr(fp,name)
        assert small.dtype==torch.bfloat16
        assert small.nbytes*2==large.nbytes
    blocks=[CacheBlock(ar.device) for _ in range(4)]
    for i,b in enumerate(blocks):
        for layer in range(2):
            if (i+layer)%3:
                a=(torch.eye(dk,device='cuda').repeat(1,2,1,1)+torch.randn(1,2,dk,dk,device='cuda')*.01).bfloat16()
                bb=(torch.randn(1,2,dv,dk,device='cuda')*.03).bfloat16()
                b.linear_affine[layer]=(a,bb)
    targets=[CacheBlock(ar.device) for _ in range(6)]
    q=torch.randn(6,1,2,dk,device='cuda',dtype=torch.bfloat16)*.1
    v=torch.randn(6,1,2,dv,device='cuda',dtype=torch.bfloat16)*.1
    alpha=torch.full((6,1,2),.97,device='cuda');beta=torch.full_like(alpha,.15)
    actual=[torch.empty_like(buf.initial) for _ in range(2)]
    def body():
        for l in range(2):
            actual[l].copy_(buf.compose(l))
            buf.capture(l,q,v,alpha,beta,1e-6)
    buf.prepare([[]],targets[:1]);graph=capture(body) if captured else None;buf.publish(False)
    a,b,c,d=blocks
    for chains in ([[a,b,c],[a,b],[a,b,c],[],[d,b]],[[d,c],[a],[]],[] ):
        selected=targets[:len(chains)]
        reference={}
        for target in selected:
            for l in range(2):
                old=target.linear_affine.get(l)
                reference[target,l]=(torch.eye(dk,device='cuda').repeat(1,2,1,1) if old is None else old[0].float(),
                    torch.zeros(1,2,dv,dk,device='cuda') if old is None else old[1].float())
        retained=[(t,t.clone()) for block in selected for pair in block.linear_affine.values() for t in pair]
        buf.prepare(chains,selected);buf.initial.fill_(float('nan'))
        if graph is None:body()
        else:graph.replay()
        buf.publish()
        for l in range(2):
            for w,chain in enumerate(chains):
                state=torch.zeros(1,2,dv,dk,device='cuda',dtype=torch.bfloat16)
                for block in chain:
                    pair=block.linear_affine.get(l)
                    if pair is not None:state=(state.float()@pair[0].float()+pair[1].float()).bfloat16()
                torch.testing.assert_close(actual[l][w],state[0].transpose(-1,-2),atol=.002,rtol=.008)
                key=q[w:w+1,0].float();key*=torch.rsqrt((key*key).sum(-1,keepdim=True)+1e-6)
                aa,bb=reference[selected[w],l]
                expected=update_affine_summary(A_hat=aa,B_hat=bb,k=key,v=v[w:w+1,0].float(),
                                              alpha=alpha[w:w+1,0],beta=beta[w:w+1,0])
                for got,want in zip(selected[w].linear_affine[l],expected):
                    assert got.dtype==torch.bfloat16
                    torch.testing.assert_close(got,want.bfloat16(),atol=.002,rtol=.008)
                    assert got.untyped_storage().nbytes()==got.numel()*2*2  # two layers per block slab
            assert torch.count_nonzero(actual[l][len(chains):])==0
        for t,saved in retained:torch.testing.assert_close(t,saved,atol=0,rtol=0)
    torch.cuda.synchronize();buf.retire_completed()
    assert not buf._pending
