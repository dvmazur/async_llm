"""Prepare bounded augmented FLA inputs without per-request Python or cat."""
import triton
import triton.language as tl


@triton.jit
def pack_qkv(q,k,v,qn,kn,va,output,active,T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,
             D:tl.constexpr,QSTRIDE:tl.constexpr,KSTRIDE:tl.constexpr,VSTRIDE:tl.constexpr,
             BD:tl.constexpr,BT:tl.constexpr):
    t=tl.program_id(0)*BT+tl.arange(0,BT)
    head=tl.program_id(1)
    c=tl.arange(0,BD)
    real=tl.load(active+t,t<T,other=0)
    mask=real[:,None] & (c[None,:]<D)
    if head % (HV//H) == 0:
        kh=head//(HV//H)
        qv=tl.load(q+t[:,None]*QSTRIDE+kh*D+c[None,:],mask,other=0.).to(tl.float32)
        kv=tl.load(k+t[:,None]*KSTRIDE+kh*D+c[None,:],mask,other=0.).to(tl.float32)
        qr=1/tl.sqrt(tl.sum(qv*qv,1)+1.e-6)
        kr=1/tl.sqrt(tl.sum(kv*kv,1)+1.e-6)
        dst=(t[:,None]*H+kh)*D+c[None,:]
        full=(t[:,None]<T)&(c[None,:]<D)
        tl.store(qn+dst,qv*qr[:,None],full)
        tl.store(kn+dst,kv*kr[:,None],full)
    value=tl.load(v+t[:,None]*VSTRIDE+head*D+c[None,:],mask,other=0.)
    dst=(t[:,None]*HV+head)*3*D+c[None,:]
    full=(t[:,None]<T)&(c[None,:]<D)
    tl.store(va+dst,value,full)
    tl.store(va+dst+D,0.,full)
    tl.store(va+dst+2*D,value,full)
    # Only dummy model rows need initialization. Real rows are produced by O.
    tl.store(output+(t[:,None]*HV+head)*D+c[None,:],0.,full&~real[:,None])


@triton.jit(do_not_specialize=["LAYER"])
def gather_initial(initial,pool,writes,fresh,active,aug,LAYER,
                   SLOTS:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr):
    worker,head=tl.program_id(0),tl.program_id(1)
    offset=tl.program_id(2)*BLOCK+tl.arange(0,BLOCK)
    component=offset//(D*D)
    within=offset%(D*D)
    mask=offset<3*D*D
    enabled=tl.load(active+worker)
    slot=tl.load(writes+worker).to(tl.int64)
    is_fresh=tl.load(fresh+worker)
    source=tl.load(initial+(worker*H+head)*D*D+within,mask&enabled&(component==0),other=0.)
    old_ptr=pool+((LAYER.to(tl.int64)*SLOTS+slot)*2*H+head)*D*D+(component-1)*H*D*D+within
    old=tl.load(old_ptr,mask&enabled&~is_fresh&(component>0),other=0.)
    identity=(within//D)==(within%D)
    old=tl.where(is_fresh&(component==1)&identity,1.,old)
    value=tl.where(component==0,source,old)
    tl.store(aug+(worker*H+head)*3*D*D+offset,tl.where(enabled,value,0.),mask)
