"""Complete Qwen full-attention sublayer with reusable profile scratch."""
import torch
import triton

from .linear import BoundLinear,LinearWorkspace
from . import attention_kernels as kernels


class AttentionWorkspace:
    def __init__(self,core,active,*,linear_workspace=None):
        self.core,self.active,self.linear_workspace=core,active,linear_workspace
        m,h,hk,d=core.total,core.h,core.hk,core.d
        self.width=(2*h+2*hk)*d
        self.raw=torch.empty(m,self.width,device=core.device,dtype=core.dtype)
        self.q=torch.empty(m,h,d,device=core.device,dtype=core.dtype)
        self.k=torch.empty(m,hk,d,device=core.device,dtype=core.dtype)
        self.value=self.raw[:,(2*h+hk)*d:].view(m,hk,d)
        self.gate=self.raw[:,:2*h*d].view(m,h,2*d)[...,d:]
        self.output=torch.empty_like(self.q)
        self.output_flat=self.output.view(m,h*d)


class BoundAttentionLayer:
    def __init__(self,layer,workspace):
        self.layer,self.ws=layer,workspace
        core=workspace.core
        if (layer.num_qo_heads!=core.h or layer.num_kv_heads!=core.hk or layer.head_dim!=core.d
                or layer.rotary_dim!=core.rd or not 0<=layer._kv_idx<len(core.keys)
                or layer.q_norm.eps!=layer.k_norm.eps):raise ValueError("Attention layer/profile mismatch")
        if getattr(layer.o_proj,"_tp_size",1)!=1:raise ValueError("planned runtime initially supports TP1")
        if any(p.weight_scale_inv is not None for p in (layer.qkv_proj,layer.o_proj)):
            if workspace.linear_workspace is None:
                workspace.linear_workspace=LinearWorkspace(core.total,
                    max(layer.qkv_proj.full_input_size,layer.o_proj.full_input_size),device=core.device)
        self.input=BoundLinear(layer.qkv_proj,core.total,dtype=core.dtype,
            workspace=workspace.linear_workspace,active=workspace.active)
        self.output=BoundLinear(layer.o_proj,core.total,dtype=core.dtype,
            workspace=workspace.linear_workspace,active=workspace.active)

    def run(self,x,output):
        l,w=self.layer,self.ws
        self.input.run(x,w.raw)
        kernels.normalize_qk[(w.core.total,w.core.h+w.core.hk)](
            w.raw,l.q_norm.weight,l.k_norm.weight,w.active,w.q,w.k,HQ=w.core.h,HK=w.core.hk,
            D=w.core.d,EPS=l.q_norm.eps,STRIDE=w.width,BD=triton.next_power_of_2(w.core.d),
            num_warps=1)
        w.core.run(l._kv_idx,w.q,w.k,w.value,w.output,gate=w.gate)
        return self.output.run(w.output_flat,output)
