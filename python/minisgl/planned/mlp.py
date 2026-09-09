"""Bounded dense/shared/routed Qwen MLPs, one workspace across model layers."""
import torch
import triton

from .linear import BoundLinear,LinearWorkspace
from .moe import BoundExperts,ExpertsWorkspace
from .moe_kernels import silu_experts


class MLPWorkspace:
    def __init__(self,rows,hidden,intermediate,active,*,device,dtype=torch.bfloat16,
                 experts=0,linear_workspace=None,expert_quant_workspace=None):
        self.rows,self.h,self.i,self.active=rows,hidden,intermediate,active
        self.device,self.dtype=torch.device(device),dtype
        self.linear=linear_workspace
        self.expert_quant=expert_quant_workspace
        self.gate_up=torch.empty(rows,2*intermediate,device=device,dtype=dtype)
        self.activated=torch.empty(rows,intermediate,device=device,dtype=dtype)
        self.shared=torch.empty(rows,hidden,device=device,dtype=dtype) if experts else None
        self.shared_gate=torch.empty(rows,1,device=device,dtype=dtype) if experts else None
        self.router=torch.empty(rows,experts,device=device,dtype=dtype) if experts else None
        self.experts=ExpertsWorkspace() if experts else None


class BoundGatedMLP:
    def __init__(self,layer,workspace):
        from minisgl.layers import silu_and_mul
        self.layer,self.ws=layer,workspace
        if (layer.act_fn is not silu_and_mul or getattr(layer.down_proj,"_tp_size",1)!=1
                or layer.gate_up_proj.weight.shape!=(2*workspace.i,workspace.h)
                or layer.down_proj.weight.shape!=(workspace.h,workspace.i)):
            raise ValueError("dense/shared MLP profile mismatch")
        if any(p.weight_scale_inv is not None for p in (layer.gate_up_proj,layer.down_proj)) and workspace.linear is None:
            workspace.linear=LinearWorkspace(workspace.rows,max(workspace.h,workspace.i),device=workspace.gate_up.device)
        self.up=BoundLinear(layer.gate_up_proj,workspace.rows,dtype=workspace.dtype,
                             workspace=workspace.linear,active=workspace.active)
        self.down=BoundLinear(layer.down_proj,workspace.rows,dtype=workspace.dtype,
                             workspace=workspace.linear,active=workspace.active)

    def run(self,x,output):
        w=self.ws
        self.up.run(x,w.gate_up)
        silu_experts[(w.rows,triton.cdiv(w.i,256))](w.gate_up,w.active,w.activated,
            N=w.rows,I=w.i,BLOCK=256,IDS_MASK=False,num_warps=4,enable_fp_fusion=False)
        return self.down.run(w.activated,output)


class BoundMoeMLP:
    def __init__(self,layer,workspace):
        self.ws,self.layer=workspace,layer
        if workspace.experts is None or workspace.router.shape[1]!=layer.experts.num_experts:
            raise ValueError("MoE MLP workspace mismatch")
        self.shared=BoundGatedMLP(layer.shared_expert,workspace)
        self.gate=BoundLinear(layer.shared_expert_gate,workspace.rows,dtype=workspace.dtype,
                             workspace=workspace.linear,active=workspace.active)
        self.router=BoundLinear(layer.gate,workspace.rows,dtype=workspace.dtype,
                               workspace=workspace.linear,active=workspace.active)
        self.experts=BoundExperts(layer.experts,workspace.rows,workspace.active,dtype=workspace.dtype,
                                  workspace=workspace.experts,quant_workspace=workspace.expert_quant)

    def run(self,x,output):
        w=self.ws
        self.shared.run(x,w.shared)
        self.gate.run(x,w.shared_gate)
        self.router.run(x,w.router)
        return self.experts.run(x,w.router,output,shared=w.shared,gate=w.shared_gate)
