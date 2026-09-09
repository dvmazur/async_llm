"""Bounded FLA 0.5.2 augmented prefill recipe with all-layer pool outputs.

Normalizes/expands values, runs the same chunk algorithms, publishes only A/B
and optional conv windows to their slots. No unused final-S/output-affine buffer.
No metadata preparation, allocation or autotuning inside the GPU body.
"""
import torch
import triton

from . import fla_io_kernels as io


class BoundFLA:
    def __init__(self,pool,phase,rows,capacity,*,key_heads,dtype=torch.bfloat16,
                 old_layer_rounding=False):
        # Import/choose backend at setup, not inside captured body.
        from . import fla_forward_kernels
        self.kernels=fla_forward_kernels
        if capacity.chunk_size!=64 or capacity.prefill_requests<1 or capacity.prefill_tokens<1:
            raise ValueError("bounded FLA currently requires a nonempty BT64 profile")
        self.pool,self.phase,self.rows=pool,phase,rows
        # Low-level augmented-scan tests use pure FLA semantics. The real Qwen
        # layer additionally retains the original one-token dispatch rounding.
        self.old_layer_rounding=old_layer_rounding
        self.layers,self.slots,pair,self.hv,self.d,other=pool.shape
        self.h=key_heads
        if (pair!=2 or self.d!=other or self.d>256 or key_heads<1 or self.hv%key_heads
                or not pool.is_cuda or pool.dtype!=torch.float32 or not pool.is_contiguous()
                or phase.width!=capacity.prefill_requests):
            raise ValueError("invalid FLA pool/profile")
        self.t,self.r,self.dtype=capacity.prefill_tokens,capacity.prefill_requests,dtype
        self.chunks=(self.t+63)//64+self.r
        self.chunk_count=rows.chunk_offsets[-1:]  # fixed view; values change on upload
        def buf(*shape,fp32=False):
            return torch.empty(shape,device=pool.device,dtype=torch.float32 if fp32 else dtype)
        self.qn=buf(1,self.t,self.h,self.d)
        self.kn=buf(1,self.t,self.h,self.d)
        self.va=buf(1,self.t,self.hv,3*self.d)
        self.gc=buf(1,self.t,self.hv,fp32=True)
        self.solved=buf(1,self.t,self.hv,64)
        self.w=buf(1,self.t,self.hv,self.d)
        self.u=buf(1,self.t,self.hv,3*self.d)
        self.hstates=buf(1,self.chunks,self.hv,3*self.d,self.d)
        self.vnew=buf(1,self.t,self.hv,3*self.d)
        self.initial_aug=buf(self.r,self.hv,3*self.d,self.d,fp32=True)

    def run(self,layer,q,k,v,g,beta,initial,output,*,conv=None):
        if not 0<=layer<self.layers:
            raise ValueError("layer outside pool")
        if (q.shape!=(1,self.t,self.h,self.d) or k.shape!=q.shape
                or v.shape!=(1,self.t,self.hv,self.d) or output.shape!=v.shape
                or g.shape!=(1,self.t,self.hv) or beta.shape!=g.shape
                or initial.shape!=(self.r,self.hv,self.d,self.d)):
            raise ValueError("invalid FLA shapes")
        if any(x.device!=self.pool.device for x in (q,k,v,g,beta,initial,output)):
            raise ValueError("FLA inputs must share a device")
        if any(x.dtype!=self.dtype for x in (q,k,v,output)) or initial.dtype!=torch.float32:
            raise ValueError("invalid FLA dtypes")
        if any(x.stride(-1)!=1 or x.stride(-2)!=self.d for x in (q,k,v)):
            raise ValueError("FLA qkv require contiguous head rows")
        if any(not x.is_contiguous() for x in (g,beta,initial,output)):
            raise ValueError("FLA gates/states/output must be contiguous")
        if conv is not None and (conv.phase is not self.phase or not conv.prefill
                                 or conv.slots!=self.slots or conv.layers!=self.layers):
            raise ValueError("conv must use the same prefill phase")
        t,h,hv,d=self.t,self.h,self.hv,self.d
        io.pack_qkv[(triton.cdiv(t,8),hv)](
            q,k,v,self.qn,self.kn,self.va,output,self.rows.active,
            T=t,H=h,HV=hv,D=d,QSTRIDE=q.stride(1),KSTRIDE=k.stride(1),VSTRIDE=v.stride(1),
            BD=triton.next_power_of_2(d),BT=8,num_warps=1)
        io.gather_initial[(self.r,hv,triton.cdiv(3*d*d,256))](
            initial,self.pool,self.phase.write_slots,self.phase.write_fresh,self.phase.active,
            self.initial_aug,LAYER=layer,SLOTS=self.slots,H=hv,D=d,BLOCK=256)
        kernels=self.kernels
        kernels.bounded_cumsum[(self.chunks,hv)](
            self.chunk_count,self.rows.prefill_recipe,self.old_layer_rounding,
            g,self.gc,1.4426950408889634,self.rows.prefill_offsets,
            self.rows.chunk_indices,T=t,B=1,H=hv,BT=64,REVERSE=False,HAS_SCALE=True,
            IS_VARLEN=True,HEAD_FIRST=False,num_warps=4)
        kernels.bounded_kkt[(self.chunks,hv)](
            self.chunk_count,self.rows.prefill_recipe,self.old_layer_rounding,
            self.kn,self.gc,beta,self.solved,self.rows.prefill_offsets,
            self.rows.chunk_indices,T=t,H=h,HV=hv,K=d,BT=64,BC=16,BK=32,
            USE_G=True,IS_VARLEN=True,num_warps=4)
        kernels.bounded_wu[(self.chunks,hv)](
            self.chunk_count,self.rows.prefill_recipe,self.old_layer_rounding,
            self.kn,self.va,beta,self.w,self.u,self.solved,self.gc,
            self.rows.prefill_offsets,self.rows.chunk_indices,T=t,H=h,HV=hv,K=d,V=3*d,
            BT=64,BK=64,BV=64,USE_G=True,IS_VARLEN=True,num_warps=4,num_stages=3)
        kernels.bounded_h[(triton.cdiv(3*d,32),self.r*hv)](
            self.phase.active,self.rows.prefill_recipe,self.old_layer_rounding,self.phase.write_slots,
            self.pool if conv is None else conv.pool,
            self.initial_aug if conv is None else conv.new_window,
            LAYER=layer,SLOT_CAP=self.slots,HAS_CONV=conv is not None,
            C=0 if conv is None else conv.channels,CK=0 if conv is None else conv.window_size,
            k=self.kn,v=self.u,w=self.w,v_new=self.vnew,g=self.gc,gk=None,h=self.hstates,
            h0=self.initial_aug,ht=self.pool,cu_seqlens=self.rows.prefill_offsets,
            chunk_offsets=self.rows.chunk_offsets,T=t,H=h,HV=hv,K=d,V=3*d,BT=64,BV=32,
            USE_G=True,USE_GK=False,USE_INITIAL_STATE=True,STORE_FINAL_STATE=True,
            SAVE_NEW_VALUE=True,STATE_V_FIRST=True,IS_VARLEN=True,num_warps=2,num_stages=2)
        kernels.bounded_o[(triton.cdiv(d,64),self.chunks,hv)](
            self.chunk_count,self.rows.prefill_recipe,self.old_layer_rounding,
            self.qn,self.kn,self.vnew,self.hstates,self.gc,None,output,
            self.rows.prefill_offsets,self.rows.chunk_indices,scale=d**-.5,T=t,H=h,HV=hv,
            K=d,V=3*d,OUT_V=d,BT=64,BK=64,BV=64,USE_G=True,USE_G_GAMMA=False,
            STATE_V_FIRST=True,IS_VARLEN=True,num_warps=4,num_stages=3)
        if self.old_layer_rounding:
            from .gdn_kernels import recurrent_capture
            recurrent_capture[(triton.cdiv(d,8),self.r*hv)](
                q,k,v,g,beta,g,initial,self.pool,self.phase.write_slots,self.phase.write_fresh,
                self.phase.active,output,self.pool if conv is None else conv.pool,
                initial if conv is None else conv.new_window,
                LAYER=layer,SLOTS=self.slots,H=h,HV=hv,D=d,KD=triton.next_power_of_2(d),ROWS=8,
                HAS_CONV=conv is not None,C=0 if conv is None else conv.channels,
                CK=0 if conv is None else conv.window_size,
                QSTRIDE=q.stride(1),KSTRIDE=k.stride(1),VSTRIDE=v.stride(1),
                prefill_recipe=self.rows.prefill_recipe,PREFILL_SPECIAL=True,
                num_warps=1,num_stages=3)
