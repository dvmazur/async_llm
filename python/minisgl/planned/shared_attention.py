"""Four bounded shared Attention consumers, prepared once for all layers.

The input q/k are already head-normalized. Projection/norm/output gate belong
to the complete model-layer adapter. No session allocation/commit is done here.
"""
import torch
import triton

from .gdn_device import _DeviceTables
from .paged_attention import BoundPagedAttention
from . import attention_kernels as kernels


class _SubTables(_DeviceTables):
    fields=("source_rows","positions")

    def __init__(self,plan,device):
        super().__init__(plan,device)
        self.positions_flat=self.positions.view(-1)


class _AttentionTables(_DeviceTables):
    fields=("write_token_slots","key_positions","merge_sources")

    def __init__(self,plan,device):
        super().__init__(plan,device)
        self.key_positions_flat=self.key_positions.view(-1)


class BoundSharedAttention:
    def __init__(self,k_pool,v_pool,forward,plan,*,num_qo_heads,rotary_dim,rope_base,
                 mrope_section=None,float_workspace=None):
        self.keys,self.values=tuple(k_pool),tuple(v_pool)
        if not self.keys or len(self.keys)!=len(self.values):raise ValueError("invalid KV layer pool")
        k=self.keys[0]
        if k.ndim!=4 or not k.is_cuda:raise ValueError("KV layers must be CUDA [pages,P,Hk,D]")
        self.np,self.p,self.hk,self.d=k.shape
        self.device,self.dtype=k.device,k.dtype
        if any(x.shape!=k.shape or x.device!=k.device or x.dtype!=k.dtype or not x.is_contiguous()
               for x in (*self.keys,*self.values)):raise ValueError("inconsistent KV pool views")
        self.flat_keys=tuple(x.view(self.np*self.p,1,self.hk,self.d) for x in self.keys)
        self.flat_values=tuple(x.view(self.np*self.p,1,self.hk,self.d) for x in self.values)
        self.h,self.rd=num_qo_heads,rotary_dim
        self.cap,self.attcap=forward.capacity,plan.capacity
        if (self.h%self.hk or not 0<rotary_dim<=self.d or rotary_dim%2
                or self.p!=self.attcap.page_size or self.np!=self.attcap.page_pool_capacity):
            raise ValueError("incompatible shared Attention recipe/pool")
        self.sh,self.sw=(mrope_section[1],mrope_section[2]) if mrope_section is not None else (0,0)
        self.inv=1./(rope_base**(torch.arange(0,rotary_dim,2,device=k.device,dtype=torch.float32)/rotary_dim))
        self.total=self.cap.prefill_tokens+self.cap.decode_workers
        self.meta=_AttentionTables(plan,k.device)
        self.sub_names=("prefill_context","prefill_self","decode_main","decode_aux")
        self.subs=tuple(getattr(plan,n) for n in self.sub_names)
        self.tables=tuple(_SubTables(s,k.device) for s in self.subs)
        self.qcounts=tuple(len(s.source_rows) for s in self.subs)
        n=sum(self.qcounts)
        self.queries=torch.empty(n,self.h,self.d,device=k.device,dtype=k.dtype)
        self.partial=torch.empty_like(self.queries)
        self.lse=torch.empty(n,self.h,device=k.device,dtype=torch.float32)
        self.qviews=self.queries.split(self.qcounts)
        self.oviews=self.partial.split(self.qcounts)
        self.lviews=self.lse.split(self.qcounts)
        placeholder=torch.empty(256,device=k.device,dtype=torch.uint8)
        ops=[]
        for i,(sub,qrows) in enumerate(zip(self.subs,self.qcounts)):
            aux=i==3
            op=None
            if len(sub.query_lengths):
                op=BoundPagedAttention(request_capacity=len(sub.query_lengths),row_capacity=qrows,
                    page_ref_capacity=plan.capacity.page_references,num_qo_heads=self.h,num_kv_heads=self.hk,
                    head_dim=self.d,page_size=1 if aux else self.p,page_pool_capacity=self.np*self.p if aux else self.np,
                    device=k.device,dtype=k.dtype,causal=i==1,decode=i>=2,float_workspace=placeholder,defer_workspace=True)
            ops.append(op)
        self.ops=tuple(ops)
        required=max((op.required_float_bytes for op in ops if op is not None),default=0)
        if float_workspace is None:
            float_workspace=torch.empty(max(256,(required+255)//256*256),device=k.device,dtype=torch.uint8)
        for op in ops:
            if op is not None:op.bind_workspace(float_workspace)
        self.prepare(forward,plan)

    def prepare(self,forward,plan):
        self.ready=False
        if forward.capacity!=self.cap or plan.capacity!=self.attcap:
            raise ValueError("shared Attention requires a new capacity profile")
        for name,table,op in zip(self.sub_names,self.tables,self.ops):
            sub=getattr(plan,name)
            table.upload(sub)
            if op is not None:op.plan(sub.query_lengths,sub.pages,sub.last_page_lengths)
        self.meta.upload(plan)
        self.ready=True

    def run(self,layer,q,k,v,output,*,gate=None):
        if not self.ready:raise RuntimeError("all Attention subplans must be ready before execution")
        if (q.shape!=(self.total,self.h,self.d) or k.shape!=(self.total,self.hk,self.d) or v.shape!=k.shape
                or output.shape!=q.shape or not 0<=layer<len(self.keys)):
            raise ValueError("invalid shared Attention input/output")
        if (any(x.device!=self.device or x.dtype!=self.dtype or x.stride(-1)!=1 for x in (q,k,v,output))
                or not output.is_contiguous()):raise ValueError("incompatible Attention activation buffers")
        if gate is not None and (gate.shape!=output.shape or gate.dtype!=self.dtype
                                 or gate.device!=self.device or gate.stride(-1)!=1):
            raise ValueError("invalid Attention output gate")
        p=self.cap.prefill_tokens
        for begin,count,indices in ((0,p,(0,1)),(p,self.cap.decode_workers,(2,3))):
            if not count:continue
            # All writes of this phase precede its reads. Crucially, the next
            # decode publication is not moved above the prefill attention.
            kernels.store_rotated_kv[(count,self.hk)](
                k,v,self.meta.write_token_slots,self.meta.key_positions_flat,self.inv,self.keys[layer],self.values[layer],
                BASE=begin,M=self.total,H=self.hk,D=self.d,RD=self.rd,SH=self.sh,SW=self.sw,
                KS0=k.stride(0),KS1=k.stride(1),VS0=v.stride(0),VS1=v.stride(1),
                BD=triton.next_power_of_2(self.d),num_warps=4,enable_fp_fusion=False)
            for i in indices:
                if self.ops[i] is None:continue
                t=self.tables[i]
                kernels.gather_rope[(self.qcounts[i],self.h)](
                    q,t.source_rows,t.positions_flat,self.inv,self.qviews[i],Q=self.qcounts[i],H=self.h,
                    D=self.d,RD=self.rd,SH=self.sh,SW=self.sw,QS0=q.stride(0),QS1=q.stride(1),
                    BD=triton.next_power_of_2(self.d),num_warps=4,enable_fp_fusion=False)
                kp,vp=(self.flat_keys[layer],self.flat_values[layer]) if i==3 else (self.keys[layer],self.values[layer])
                self.ops[i].run_capacity(self.qviews[i],kp,vp,self.oviews[i],self.lviews[i])
            kernels.merge_segments[(count,self.h)](
                self.partial,self.lse,self.meta.merge_sources,output,output if gate is None else gate,
                BASE=begin,H=self.h,D=self.d,
                S=self.attcap.context_segments+1,BD=triton.next_power_of_2(self.d),
                HAS_GATE=gate is not None,GS0=0 if gate is None else gate.stride(0),
                GS1=0 if gate is None else gate.stride(1),
                num_warps=4,enable_fp_fusion=False)
        return output
