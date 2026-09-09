"""Bounded routed-expert consumer with unchanged existing expert GEMMs.

The legacy compatibility path is not modified. Router backend/normalization
match the existing selector (vLLM, sgl-kernel, or fixed-output Torch). Dynamic
bincount/repeat_interleave are replaced with bounded device counters/assignments.
Shared-expert contribution can be fused into the final routed reduction.
"""
import torch
import triton

from minisgl.kernel.moe_impl import fused_moe_kernel_triton
from minisgl.moe.fused import try_get_optimal_moe_config
from .linear import LinearWorkspace,quantize_active_groups
from . import moe_kernels as kernels


class ExpertsWorkspace:
    """Profile-owned resources reused sequentially across all MoE layers."""
    def __init__(self):
        self.key=None
        self.tensors={}
        self.quant=None


class BoundExperts:
    def __init__(self,layer,rows,active,*,dtype=torch.bfloat16,quant_workspace=None,workspace=None):
        self.layer,self.rows,self.active,self.dtype=layer,rows,active,dtype
        self.e,self.twice_i,self.h=layer.gate_up_proj.shape
        self.i,self.k=self.twice_i//2,layer.top_k
        self.device=layer.gate_up_proj.device
        if (layer.tp_size!=1 or layer.activation!="silu" or rows<1
                or layer.down_proj.shape!=(self.e,self.h,self.i) or not 1<=self.k<=self.e
                or dtype not in (torch.bfloat16,torch.float16) or not layer.gate_up_proj.is_cuda
                or active.shape!=(rows,) or active.dtype!=torch.bool or active.device!=self.device):
            raise ValueError("unsupported planned MoE recipe")
        from minisgl.moe.fused import _use_torch_moe_fallback,_vllm_custom_moe_ops
        self.router_backend="torch"
        self.router_op=None
        if _use_torch_moe_fallback(self.device):
            ops=_vllm_custom_moe_ops()
            if ops is not None:self.router_backend,self.router_op="vllm",ops.topk_softmax
        else:
            from sgl_kernel import topk_softmax
            self.router_backend,self.router_op="sgl",topk_softmax
        self.fp8=layer.gate_up_proj.dtype==torch.float8_e4m3fn
        for w,s in ((layer.gate_up_proj,layer.gate_up_proj_scale_inv),(layer.down_proj,layer.down_proj_scale_inv)):
            if not w.is_contiguous() or w.device!=self.device:raise ValueError("invalid expert weights")
            if self.fp8:
                if (w.dtype!=torch.float8_e4m3fn or w.shape[1]%128 or w.shape[2]%128
                        or s is None or s.shape!=(self.e,w.shape[1]//128,w.shape[2]//128)
                        or s.dtype!=torch.float32 or s.device!=w.device or not s.is_contiguous()):
                    raise ValueError("invalid FP8 expert scales")
            elif w.dtype!=dtype or s is not None:raise ValueError("invalid dense expert dtype/scales")
        self.workspace=workspace if workspace is not None else ExpertsWorkspace()
        key=(rows,self.e,self.twice_i,self.h,self.k,self.device,dtype,self.router_backend)
        if self.workspace.key not in (None,key):raise ValueError("MoE workspace shape/profile mismatch")
        self.workspace.key=key
        def buf(name,*shape,dtype=dtype):
            if name not in self.workspace.tensors:
                self.workspace.tensors[name]=torch.empty(shape,device=self.device,dtype=dtype)
            result=self.workspace.tensors[name]
            if result.shape!=shape or result.dtype!=dtype:raise ValueError("MoE workspace recipe mismatch")
            return result
        self.config=try_get_optimal_moe_config(layer.gate_up_proj.shape,layer.down_proj.shape,self.k,rows)
        if self.fp8:self.config={**self.config,"BLOCK_SIZE_K":128}
        self.bm=self.config["BLOCK_SIZE_M"]
        self.n=rows*self.k
        self.aligned=self.n+(self.e+1)*(self.bm-1)
        self.nblocks=triton.cdiv(self.aligned,self.bm)
        self.sorted=buf("sorted",self.aligned,dtype=torch.int32)
        self.expert_ids=buf("experts",self.nblocks,dtype=torch.int32)
        self.counts=buf("counts",self.e,dtype=torch.int32)
        self.starts=buf("starts",self.e+1,dtype=torch.int32)
        self.num_padded=self.starts[-1:]
        self.raw_weights=buf("raw_weights",rows,self.k,dtype=torch.float32)
        self.raw_ids=buf("raw_ids",rows,self.k,dtype=torch.int64 if self.router_backend=="torch" else torch.int32)
        self.token_expert_indices=(buf("token_expert_indices",rows,self.k,dtype=torch.int32)
                                   if self.router_backend=="vllm" else None)
        self.weights=buf("weights",rows,self.k,dtype=torch.float32)
        self.ids=buf("ids",rows,self.k,dtype=torch.int32)
        self.logits32=buf("logits32",rows,self.e,dtype=torch.float32) if self.router_backend!="sgl" else None
        self.probs=buf("probs",rows,self.e,dtype=torch.float32) if self.router_backend=="torch" else None
        self.cache=buf("cache",self.n*max(self.twice_i,self.h))
        self.first=self.cache[:self.n*self.twice_i].view(rows,self.k,self.twice_i)
        self.third=self.cache[:self.n*self.h].view(rows,self.k,self.h)
        self.second=buf("second",self.n,self.i)
        self.xq=self.xs=self.yq=self.ys=None
        if self.fp8:
            if torch.cuda.get_device_capability(self.device)<(8,9):raise ValueError("native FP8 MoE unsupported")
            if quant_workspace is None:quant_workspace=self.workspace.quant
            if quant_workspace is None:quant_workspace=LinearWorkspace(self.n,max(self.h,self.i),device=self.device)
            self.workspace.quant=quant_workspace
            self.quant_workspace=quant_workspace
            self.xq,self.xs=quant_workspace.views(rows,self.h)
            self.yq,self.ys=quant_workspace.views(self.n,self.i)

    def route(self,logits):
        if logits.shape!=(self.rows,self.e) or logits.device!=self.device:
            raise ValueError("invalid router logits")
        if self.router_backend=="sgl":
            self.router_op(self.raw_weights,self.raw_ids,logits,self.layer.renormalize)
        else:
            self.logits32.copy_(logits)
            if self.router_backend=="vllm":
                self.router_op(self.raw_weights,self.raw_ids,self.token_expert_indices,
                               self.logits32,self.layer.renormalize)
            else:
                torch.softmax(self.logits32,dim=-1,out=self.probs)
                torch.topk(self.probs,self.k,dim=-1,out=(self.raw_weights,self.raw_ids))
        # Preserve the old helper's backend-specific normalization boundaries.
        self.align(renormalize=self.layer.renormalize and self.router_backend!="vllm")

    def align(self,*,renormalize=False):
        kernels.init_alignment[(triton.cdiv(max(self.aligned,self.e,self.nblocks),256),)](
            self.sorted,self.expert_ids,self.counts,N=self.n,A=self.aligned,NB=self.nblocks,E=self.e,BLOCK=256)
        kernels.finish_routes[(triton.cdiv(self.rows,16),)](
            self.raw_weights,self.raw_ids,self.active,self.weights,self.ids,self.counts,
            M=self.rows,K=self.k,E=self.e,RENORM=renormalize,BR=16,BK=triton.next_power_of_2(self.k),
            BE=triton.next_power_of_2(self.e+1),num_warps=4)
        kernels.prefix_counts[(1,)](self.counts,self.starts,E=self.e,BM=self.bm,
                                    BE=triton.next_power_of_2(self.e),num_warps=4)
        kernels.scatter_assignments[(max(self.nblocks,triton.cdiv(self.n,256)),)](
            self.ids,self.starts,self.counts,self.sorted,self.expert_ids,N=self.n,E=self.e,NB=self.nblocks,
            BM=self.bm,BE=triton.next_power_of_2(self.e),BLOCK=256,num_warps=4)

    def run_experts(self,x,output,*,shared=None,gate=None):
        if (x.shape!=(self.rows,self.h) or output.shape!=x.shape
                or any(t.dtype!=self.dtype or t.device!=self.device or not t.is_contiguous() for t in (x,output))):
            raise ValueError("invalid planned expert inputs")
        if (shared is None)!=(gate is None):raise ValueError("shared output and gate must be provided together")
        if shared is not None and (shared.shape!=x.shape or gate.shape!=(self.rows,1)
                or any(t.dtype!=self.dtype or t.device!=self.device or not t.is_contiguous() for t in (shared,gate))):
            raise ValueError("invalid shared expert buffers")
        l=self.layer
        if self.fp8:
            quantize_active_groups[(self.rows*(self.h//128),)](
                x,self.xq,self.xs,self.active,K=self.h,HAS_MASK=True,num_warps=4)
        fused_moe_kernel_triton(self.xq if self.fp8 else x,l.gate_up_proj,self.first,self.weights,self.ids,
            self.sorted,self.expert_ids,self.num_padded,l.apply_router_weight_on_input,self.k,self.config,
            compute_type=self.dtype,a_scale=self.xs,b_scale=l.gate_up_proj_scale_inv)
        kernels.silu_experts[(self.n,triton.cdiv(self.i,256))](self.first,self.ids,self.second,
            N=self.n,I=self.i,BLOCK=256,num_warps=4,enable_fp_fusion=False)
        if self.fp8:
            quantize_active_groups[(self.n*(self.i//128),)](
                self.second,self.yq,self.ys,self.ids,K=self.i,HAS_MASK=True,IDS_MASK=True,num_warps=4)
        fused_moe_kernel_triton(self.yq if self.fp8 else self.second,l.down_proj,self.third,self.weights,self.ids,
            self.sorted,self.expert_ids,self.num_padded,not l.apply_router_weight_on_input,1,self.config,
            compute_type=self.dtype,a_scale=self.ys,b_scale=l.down_proj_scale_inv)
        kernels.reduce_experts[(self.rows,triton.cdiv(self.h,256))](self.third,self.ids,output,
            output if shared is None else shared,output if gate is None else gate,self.active,
            M=self.rows,K=self.k,H=self.h,BLOCK=256,HAS_SHARED=shared is not None,num_warps=4,
            enable_fp_fusion=False)
        return output

    def run(self,x,logits,output,*,shared=None,gate=None):
        self.route(logits)
        return self.run_experts(x,output,shared=shared,gate=gate)
