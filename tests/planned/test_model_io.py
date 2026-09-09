from dataclasses import replace
import pytest

from minisgl.planned.forward_plan import BlockState,PrefillRequest,DecodeRequest,PlanCapacity,prepare_forward
from minisgl.planned.model_io import prepare_inputs


def forward():
    return prepare_forward({i:BlockState(i,i) for i in range(3)},capacity=PlanCapacity(2,8,2,4,8),
        prefill=[PrefillRequest((),0,3)],decode=[DecodeRequest((),1)])


def test_logical_tokens_and_images_pack_across_physical_hole():
    f=forward()
    p=prepare_inputs(f,[5,7,9,11],vocab_size=32,image_rows=(2,0))
    assert p.token_ids==(5,7,9,0,0,0,0,0,11,0)
    assert p.image_indices==(1,-1,0,-1,-1,-1,-1,-1,-1,-1)
    assert p.image_count==2


@pytest.mark.parametrize("tokens,images",[([1,2],()),([1,2,3,32],()),([1,2,3,4],(3,)),
                                         ([1,2,3,4],(1,1)),([1,2,3,4],(-1,))])
def test_bad_input_metadata_rejected(tokens,images):
    with pytest.raises(ValueError):prepare_inputs(forward(),tokens,vocab_size=32,image_rows=images)


@pytest.mark.parametrize("hidden",[128,4096])
def test_selected_final_norm_preserves_fp32_add_before_normalization(hidden):
    import torch
    import triton
    if not torch.cuda.is_available():pytest.skip("CUDA required")
    from flashinfer import gemma_fused_add_rmsnorm
    from minisgl.planned.io_kernels import selected_final_norm
    with torch.inference_mode():
        torch.manual_seed(125)
        x,r=[torch.randn(6,hidden,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
        w=torch.randn(hidden,device="cuda",dtype=x.dtype)*.1
        source=torch.tensor([3,1,-1,5],device="cuda")
        out=torch.empty(4,hidden,device="cuda",dtype=x.dtype)
        a,b=x.clone(),r.clone()
        gemma_fused_add_rmsnorm(a,b,w,1e-6,enable_pdl=False)
        selected_final_norm[(4,)](x,r,w,source,out,H=hidden,EPS=1e-6,BH=triton.next_power_of_2(hidden),
                                  num_warps=4,enable_fp_fusion=False)
        want=a[torch.tensor([3,1,0,5],device="cuda")];want[2]=0
        torch.testing.assert_close(out,want,rtol=.008,atol=.001)
        assert float((out.float()-want.float()).norm()/want.float().norm())<.001
        assert torch.equal(x,a) is False  # source x wasn't replaced by normalized x
