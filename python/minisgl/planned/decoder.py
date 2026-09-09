"""Complete fixed-capacity decoder body; session/retirement are outside it.

One profile owns scratch shared across layers and consumers. prepare() uploads
new data; run() is the same body for eager and a full CUDA Graph capture. Vision,
tokenization, sampling and retained-output ownership belong to the outer runner.
"""
from types import SimpleNamespace
import torch
import triton

from .gdn_device import DeviceRows,_DeviceTables
from .gdn_layer import bind_gdn_layers
from .shared_attention import BoundSharedAttention
from .attention_layer import AttentionWorkspace,BoundAttentionLayer
from .linear import LinearWorkspace,BoundLinear
from .mlp import MLPWorkspace,BoundMoeMLP,BoundGatedMLP
from . import io_kernels


class _InputTables(_DeviceTables):
    fields=("token_ids","image_indices")


def _quant_uses(layers,rows):
    uses=[]
    def linear(p):
        if p.weight_scale_inv is not None:uses.append((rows,p.weight.shape[-1]))
    for l in layers:
        a=l.linear_attn if l._is_linear else l.self_attn
        for name in (("in_proj_qkv","in_proj_z","in_proj_a","in_proj_b","out_proj") if l._is_linear else ("qkv_proj","o_proj")):
            linear(getattr(a,name))
        mlp=l.mlp
        shared=mlp.shared_expert if hasattr(mlp,"experts") else mlp
        linear(shared.gate_up_proj);linear(shared.down_proj)
        if hasattr(mlp,"experts"):
            linear(mlp.gate);linear(mlp.shared_expert_gate)
            e=mlp.experts
            if e.gate_up_proj_scale_inv is not None:
                uses.extend(((rows,e.gate_up_proj.shape[-1]),(rows*e.top_k,e.down_proj.shape[-1])))
    return uses


class DecoderProgram:
    def __init__(self,model,gdn_pool,k_pool,v_pool,forward,attention,inputs,*,features=None):
        from flashinfer import gemma_rmsnorm,gemma_fused_add_rmsnorm
        self.rmsnorm,self.add_rmsnorm=gemma_rmsnorm,gemma_fused_add_rmsnorm
        self.model=model
        self.layers=tuple(model.model.layers.op_list)
        self.capacity=forward.capacity
        self.m=self.capacity.prefill_tokens+self.capacity.decode_workers
        self.outputs=self.capacity.prefill_requests+self.capacity.decode_workers
        self.embedding=model.model.embed_tokens
        if not self.layers or self.embedding.tp_size!=1 or self.outputs<1:
            raise ValueError("decoder requires nonempty TP1 profile/model")
        self.device,self.dtype=self.embedding.weight.device,self.embedding.weight.dtype
        self.h=self.embedding.weight.shape[1]
        self.vocab=self.embedding.num_embeddings
        if self.dtype not in (torch.bfloat16,torch.float16):raise ValueError("unsupported embedding dtype")
        self.rows=DeviceRows(forward.rows,self.device)
        self.inputs=_InputTables(inputs,self.device)
        self.features=torch.empty(max(1,self.capacity.prefill_tokens),self.h,device=self.device,dtype=self.dtype)
        self.hidden=torch.empty(self.m,self.h,device=self.device,dtype=self.dtype)
        self.residual=torch.empty_like(self.hidden)
        self.selected=torch.empty(self.outputs,self.h,device=self.device,dtype=self.dtype)
        self.logits=torch.empty(self.outputs,self.vocab,device=self.device,dtype=self.dtype)
        uses=_quant_uses(self.layers,self.m)
        self.quant=None
        if uses:
            self.quant=LinearWorkspace(max(r for r,k in uses),max(k for r,k in uses),device=self.device,
                                        max_elements=max(r*k for r,k in uses))
        glayers=[l.linear_attn for l in self.layers if l._is_linear]
        alayers=[l.self_attn for l in self.layers if not l._is_linear]
        self.gdn=None
        if glayers:self.gdn=bind_gdn_layers(glayers,gdn_pool,forward,rows=self.rows,linear_workspace=self.quant)
        self.graph_compatible=self.gdn is None or self.gdn.graph_compatible
        self.attention=self.attention_workspace=None
        if alayers:
            a=alayers[0]
            self.attention=BoundSharedAttention(k_pool,v_pool,forward,attention,
                num_qo_heads=a.num_qo_heads,rotary_dim=a.rotary_dim,rope_base=a._rope_base,mrope_section=a._mrope_section)
            self.attention_workspace=AttentionWorkspace(self.attention,self.rows.active,linear_workspace=self.quant)
        self.bound_attention=[]
        self.bound_mlp=[]
        self.mlp_workspaces={}
        for l in self.layers:
            self.bound_attention.append(None if l._is_linear else BoundAttentionLayer(l.self_attn,self.attention_workspace))
            mlp=l.mlp
            moe=hasattr(mlp,"experts")
            shared=mlp.shared_expert if moe else mlp
            i=shared.gate_up_proj.weight.shape[0]//2
            e=mlp.experts.num_experts if moe else 0
            key=(i,e)
            if key not in self.mlp_workspaces:
                self.mlp_workspaces[key]=MLPWorkspace(self.m,self.h,i,self.rows.active,device=self.device,dtype=self.dtype,
                    experts=e,linear_workspace=self.quant,expert_quant_workspace=self.quant)
            cls=BoundMoeMLP if moe else BoundGatedMLP
            self.bound_mlp.append(cls(mlp,self.mlp_workspaces[key]))
        head=model.lm_head.tied_embedding or model.lm_head
        self.head=BoundLinear(SimpleNamespace(weight=head.weight,weight_scale_inv=None,bias=model.lm_head.bias),
                              self.outputs,dtype=self.dtype)
        self.ready=False
        self.prepare(forward,attention,inputs,features)

    def prepare(self,forward,attention,inputs,features=None):
        self.ready=False
        if forward.capacity!=self.capacity or len(inputs.token_ids)!=self.m:
            raise ValueError("decoder input/profile mismatch")
        if (len(inputs.image_indices)!=self.m or type(inputs.image_count) is not int or inputs.image_count<0
                or any(type(i) is not int or not 0<=i<self.vocab for i in inputs.token_ids)):
            raise ValueError("invalid decoder token/image metadata")
        selected=[]
        for row,index in enumerate(inputs.image_indices):
            if type(index) is not int or index < -1:raise ValueError("invalid image index")
            if index>=0:
                if row>=self.capacity.prefill_tokens or not forward.rows.active[row]:
                    raise ValueError("image index on inactive/non-prefill row")
                selected.append(index)
        if sorted(selected)!=list(range(inputs.image_count)):
            raise ValueError("image map does not match supplied feature count")
        if inputs.image_count:
            if (features is None or features.shape!=(inputs.image_count,self.h) or features.device!=self.device
                    or features.dtype!=self.dtype or inputs.image_count>self.features.shape[0]):
                raise ValueError("precomputed image features do not match input map")
        elif features is not None and features.numel():raise ValueError("features supplied without image rows")
        self.rows.upload(forward.rows)
        if self.gdn is not None:self.gdn.upload(forward)
        if self.attention is not None:self.attention.prepare(forward,attention)
        self.inputs.upload(inputs)
        if inputs.image_count:self.features[:inputs.image_count].copy_(features)
        self.ready=True

    def run(self):
        if not self.ready:raise RuntimeError("decoder input preparation failed or missing")
        if not self.graph_compatible and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("unsupported backend requires eager execution of the whole decoder")
        io_kernels.embed_inputs[(self.m,triton.cdiv(self.h,256))](
            self.inputs.token_ids,self.inputs.image_indices,self.rows.active,self.embedding.weight,
            self.features,self.residual,H=self.h,BLOCK=256,num_warps=4)
        for i,l in enumerate(self.layers):
            norm=l.input_layernorm
            if i==0:self.rmsnorm(self.residual,norm.weight,norm.eps,out=self.hidden,enable_pdl=False)
            else:self.add_rmsnorm(self.hidden,self.residual,norm.weight,norm.eps,enable_pdl=False)
            if l._is_linear:self.gdn.run(l.linear_attn._lin_idx,self.hidden,self.hidden)
            else:self.bound_attention[i].run(self.hidden,self.hidden)
            norm=l.post_attention_layernorm
            self.add_rmsnorm(self.hidden,self.residual,norm.weight,norm.eps,enable_pdl=False)
            self.bound_mlp[i].run(self.hidden,self.hidden)
        norm=self.model.model.norm
        io_kernels.selected_final_norm[(self.outputs,)](
            self.hidden,self.residual,norm.weight,self.rows.output_rows,self.selected,
            H=self.h,EPS=norm.eps,BH=triton.next_power_of_2(self.h),num_warps=4,enable_fp_fusion=False)
        return self.head.run(self.selected,self.logits)
