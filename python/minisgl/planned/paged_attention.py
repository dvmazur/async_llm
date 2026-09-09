"""Narrow FlashInfer 0.6.17 FA2 capacity adapter (not shared Attention yet).

Unlike wrapper.run(), q has capacity rows, while true qo_indptr stays truthful.
Native FA2 graph planning already stores actual rows in device workspace.
The native recipe and address ownership are checked on every outside-graph
plan; replay calls the same native paged operator with fixed buffers/scalars.
"""
import importlib.metadata

import torch


class AttentionRecipeChanged(RuntimeError):
    pass


class BoundPagedAttention:
    def __init__(self,*,request_capacity,row_capacity,page_ref_capacity,num_qo_heads,
                 num_kv_heads,head_dim,page_size,page_pool_capacity,device,dtype=torch.bfloat16,
                 causal=False,float_workspace=None,decode=False,defer_workspace=False,fast_plan=True):
        if importlib.metadata.version("flashinfer-python")!="0.6.17":
            raise RuntimeError("FA2 capacity adapter requires audited flashinfer-python 0.6.17")
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper,BatchDecodeWithPagedKVCacheWrapper
        if (min(request_capacity,row_capacity,page_ref_capacity,num_qo_heads,num_kv_heads,
                head_dim,page_size,page_pool_capacity)<1 or row_capacity<request_capacity
                or num_qo_heads%num_kv_heads or dtype not in (torch.float16,torch.bfloat16)):
            raise ValueError("invalid bounded FA2 recipe")
        self.r,self.t,self.refs=request_capacity,row_capacity,page_ref_capacity
        self.h,self.hk,self.d,self.p=num_qo_heads,num_kv_heads,head_dim,page_size
        self.page_pool_capacity=page_pool_capacity
        self.dtype,self.causal=dtype,causal
        self.decode,self.tensor_cores=decode,num_qo_heads//num_kv_heads>=4
        if decode and (causal or row_capacity!=request_capacity):
            raise ValueError("decode recipe requires one capacity row per request, non-causal")
        device=torch.device(device)
        if device.type!="cuda":raise ValueError("FA2 capacity adapter requires CUDA")
        if device.index is None:device=torch.device("cuda",torch.cuda.current_device())
        self.device=device
        def ints(n):return torch.empty(n,device=device,dtype=torch.int32)
        self.indptr,self.indices,self.last=ints(self.r+1),ints(self.refs),ints(self.r)
        supplied_workspace=float_workspace is not None
        if float_workspace is None:float_workspace=torch.empty(256,device=device,dtype=torch.uint8)
        if (float_workspace.device!=device or float_workspace.dtype!=torch.uint8
                or not float_workspace.is_contiguous()):raise ValueError("invalid FA2 workspace")
        # TC decode already uses native FA2 paged prefill. Use its explicit
        # qo_indptr interface: BatchDecode manufactures arange(r+1), claiming
        # dummy capacity rows are real requests and changing split-KV choices.
        # The non-TC kernel has its own one-row/request native ABI.
        if decode and not self.tensor_cores:
            self.wrapper=BatchDecodeWithPagedKVCacheWrapper(float_workspace,kv_layout="NHD",use_cuda_graph=True,
                use_tensor_cores=self.tensor_cores,paged_kv_indptr_buffer=self.indptr,
                paged_kv_indices_buffer=self.indices,paged_kv_last_page_len_buffer=self.last,backend="fa2")
            self.qo=getattr(self.wrapper,"_qo_indptr_buf",None)
        else:
            self.qo=ints(self.r+1)
            self.wrapper=BatchPrefillWithPagedKVCacheWrapper(float_workspace,kv_layout="NHD",use_cuda_graph=True,
                qo_indptr_buf=self.qo,paged_kv_indptr_buf=self.indptr,paged_kv_indices_buf=self.indices,
                paged_kv_last_page_len_buf=self.last,backend="fa2")
        # Native planner consumes this as its constant worst-case row capacity;
        # actual row count remains qo_indptr[-1] and is separately uploaded.
        self.wrapper._max_total_num_rows=self.t
        self._recipe=None
        self._ready=False
        self._dma=None
        self.fast_plan=fast_plan
        # CPU staging is profile-owned just like the device metadata. The DMA
        # event below protects reuse. No per-forward scalar CPU Torch loops.
        self._host={name:torch.empty(n,dtype=torch.int32,pin_memory=True)
                    for name,n in (("qo",self.r+1),("indptr",self.r+1),
                                   ("indices",self.refs),("last",self.r),("lengths",self.r))}
        self._addresses=self._address_key()
        self.required_float_bytes,self.required_int_bytes=self._workspace_requirement()
        if not defer_workspace:
            if not supplied_workspace:
                float_workspace=torch.empty(max(256,(self.required_float_bytes+255)//256*256),device=device,dtype=torch.uint8)
            self.bind_workspace(float_workspace)

    def _workspace_requirement(self):
        """Ask the audited native planner, without materializing a plan.

        In graph mode capacities determine the split buffers, even when no
        queries are active. Hard-coded 32 MiB was insufficient for real Qwen
        Hq=16,D=256 and segmented PF profiles. Use this public sizing API once
        at setup, not an extra planning pass per layer/forward.
        """
        indptr=torch.arange(self.r+1,dtype=torch.int32)
        indices=torch.zeros(self.r,dtype=torch.int32)
        last=torch.ones(self.r,dtype=torch.int32)
        kwargs=dict(pos_encoding_mode="NONE",q_data_type=self.dtype,kv_data_type=self.dtype)
        if self.decode and not self.tensor_cores:
            return self.wrapper.workspace_size(indptr,indices,last,self.h,self.hk,self.d,self.p,**kwargs)
        return self.wrapper.workspace_size(torch.zeros(self.r+1,dtype=torch.int32),indptr,indices,last,
                    self.h,self.hk,self.d,self.p,causal=self.causal,**kwargs)

    def bind_workspace(self,float_workspace):
        """Setup only: share the maximum arena among sequential sub-wrappers."""
        if self._recipe is not None:raise RuntimeError("cannot rebind a planned/captured Attention workspace")
        if (float_workspace.device!=self.device or float_workspace.dtype!=torch.uint8
                or not float_workspace.is_contiguous() or float_workspace.numel()<self.required_float_bytes):
            raise ValueError(f"Attention workspace needs {self.required_float_bytes} bytes")
        ints=self.wrapper._int_workspace_buffer
        if ints.numel()*ints.element_size()<self.required_int_bytes:
            ints=torch.empty((self.required_int_bytes+255)//256*256,device=self.device,dtype=torch.uint8)
        self.wrapper.reset_workspace_buffer(float_workspace,ints)
        self._addresses=self._address_key()

    def _address_key(self):
        w=self.wrapper
        return tuple(x.data_ptr() for x in (self.qo,self.indptr,self.indices,self.last,
            w._float_workspace_buffer,w._int_workspace_buffer,w._pin_memory_int_workspace_buffer) if x is not None)

    def validate_recipe(self):
        w=self.wrapper
        key=tuple(w._plan_info)
        valid=(len(key)==10 and key[8]==1) if self.decode and not self.tensor_cores else (
            len(key)==15 and key[1]==self.t and key[13]==1)
        if not valid:
            raise AttentionRecipeChanged("unexpected FA2 native graph-plan ABI")
        if self._recipe is not None and key!=self._recipe:
            raise AttentionRecipeChanged(f"FA2 scalar recipe changed: {self._recipe} -> {key}")
        if self._address_key()!=self._addresses:
            raise AttentionRecipeChanged("FA2 workspace/metadata address changed")
        return key

    def plan(self,query_lengths,pages,last_page_lengths):
        """CPU-only inputs. Called once before model replay, not per layer.

        Include zero-Q padded requests explicitly. They get one valid dummy KV
        page; merge/output maps must exclude their nonexistent query rows.
        """
        self._ready=False
        if not (len(query_lengths)==len(pages)==len(last_page_lengths)==self.r):
            raise ValueError("request metadata must have capacity length")
        if any(type(n) is not int or n<0 for n in query_lengths) or sum(query_lengths)>self.t:
            raise ValueError("query rows exceed capacity")
        if self.decode and any(n not in (0,1) for n in query_lengths):
            raise ValueError("decode accepts only single-token or inactive requests")
        if sum(map(len,pages))>self.refs:raise ValueError("page references exceed capacity")
        qo,indptr,indices,lengths=[0],[0],[],[]
        for n,ids,last in zip(query_lengths,pages,last_page_lengths):
            if (not ids or any(type(i) is not int or not 0<=i<self.page_pool_capacity for i in ids)
                    or type(last) is not int or not 1<=last<=self.p):
                raise ValueError("every request needs valid page references/last length")
            if self.causal and n>(len(ids)-1)*self.p+last:
                raise ValueError("causal query suffix is longer than its KV sequence")
            qo.append(qo[-1]+n);indices.extend(ids);indptr.append(len(indices))
            lengths.append((len(ids)-1)*self.p+last)
        if self._dma is not None:self._dma.synchronize()
        cpu=lambda values:torch.tensor(values,dtype=torch.int32)
        try:
            if self.decode and not self.tensor_cores:
                self.wrapper.plan(cpu(indptr),cpu(indices),cpu(last_page_lengths),
                    self.h,self.hk,self.d,self.p,pos_encoding_mode="NONE",data_type=self.dtype,
                    q_data_type=self.dtype,kv_data_type=self.dtype,sm_scale=self.d**-.5,non_blocking=False)
            elif self.fast_plan and self._recipe is not None:
                self._plan_native(qo,indptr,indices,last_page_lengths,lengths,query_lengths)
            else:
                self.wrapper.plan(cpu(qo),cpu(indptr),cpu(indices),cpu(last_page_lengths),
                    self.h,self.hk,self.d,self.p,causal=self.causal,pos_encoding_mode="NONE",
                    q_data_type=self.dtype,kv_data_type=self.dtype,sm_scale=self.d**-.5,
                    fixed_split_size=None,disable_split_kv=False,non_blocking=False)
        finally:
            # Native planning can already have submitted DMA before rejecting a
            # recipe. Its pinned staging still cannot be reused immediately.
            if self._dma is None:self._dma=torch.cuda.Event()
            self._dma.record(torch.cuda.current_stream(self.device))
        self._recipe=self.validate_recipe()
        self.native=(self.wrapper._cached_module.run if self.decode and not self.tensor_cores
                     else self.wrapper._cached_module.paged_run)
        self._native_recipe=list(self._recipe)
        self.actual_rows=qo[-1]
        self._ready=True

    def _plan_native(self,qo,indptr,indices,last,lengths,query_lengths):
        """Audited FA2 0.6.17 plan ABI; unchanged native planner and scalars.

        The first public wrapper.plan initialized its module/backend once.
        Replanning only uploads metadata and calls that same native planner.
        Non-tensor-core decode keeps its separate upstream path.
        """
        w=self.wrapper
        values=dict(qo=qo,indptr=indptr,indices=indices,last=last,lengths=lengths)
        for name,v in values.items():self._host[name].numpy()[:len(v)]=v
        w._qo_indptr_last=qo[-1]
        w._max_q_len=max(query_lengths)
        w._max_kv_len=max(lengths)
        for name,dst in (("qo",self.qo),("indptr",self.indptr),("last",self.last)):
            dst.copy_(self._host[name],non_blocking=True)
        self.indices[:len(indices)].copy_(self._host['indices'][:len(indices)],non_blocking=True)
        w._kv_lens_buffer[:self.r].copy_(self._host['lengths'],non_blocking=True)
        w._plan_info=w._cached_module.plan(
            w._float_workspace_buffer,w._int_workspace_buffer,w._pin_memory_int_workspace_buffer,
            self._host['qo'],self._host['indptr'],self._host['lengths'],
            self.t,self.r,self.h,self.hk,self.p,True,self.d,self.d,self.causal,-1,
            -1,False,0,0)

    def run_capacity(self,q,k,v,out,lse):
        """Write real query rows; LSE is base 2 as in FlashInfer merge_states.

        Unused output rows are unspecified and must be excluded by merge maps.
        """
        if not self._ready:raise RuntimeError("valid capacity plan required before execution")
        if (q.shape!=(self.t,self.h,self.d) or out.shape!=q.shape or lse.shape!=(self.t,self.h)
                or k.ndim!=4 or k.shape[1:]!=(self.p,self.hk,self.d) or v.shape!=k.shape
                or k.shape[0]!=self.page_pool_capacity):raise ValueError("invalid capacity Q/KV/output shapes")
        if (any(x.device!=self.device or not x.is_contiguous() for x in (q,k,v,out,lse))
                or any(x.dtype!=self.dtype for x in (q,k,v,out)) or lse.dtype!=torch.float32):
            raise ValueError("invalid capacity Attention dtype/layout")
        w=self.wrapper
        if self.decode and not self.tensor_cores:
            self.native(w._float_workspace_buffer,w._int_workspace_buffer,self._native_recipe,
                q,k,v,self.indptr,self.indices,self.last,out,lse,0,-1,False,None,0.,self.d**-.5,1.,1.e4)
            return out,lse
        # Same FA2 operator used by the upstream wrapper, restricted to no
        # ALiBi/custom masks/scales/PDL. No lying about _qo_indptr_last, no slices.
        self.native(w._float_workspace_buffer,w._int_workspace_buffer,self._native_recipe,
            q,k,v,self.qo,self.indptr,self.indices,self.last,out,lse,int(self.causal),0,-1,False,
            None,None,None,None,None,None,0.,self.d**-.5,None,None,None,1.,1.e4,0,
            w._workspace_size)
        return out,lse
