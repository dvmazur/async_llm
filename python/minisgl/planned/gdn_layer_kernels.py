"""Tensor-only fixed-output gates/norm; no block or profile planning."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def gates(a,b,a_log,dt_bias,active,g,beta_pf,beta_dec,alpha_dec,
          P:tl.constexpr,D:tl.constexpr,H:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row,head=i//H,i%H
    valid=i<(P+D)*H
    real=tl.load(active+row,valid,other=0)
    av=tl.load(a+i,valid&real,other=0.).to(tl.float32)
    bv=tl.load(b+i,valid&real,other=0.).to(tl.float32)
    dt=tl.load(dt_bias+head,valid,other=0.).to(tl.float32)
    al=tl.load(a_log+head,valid,other=0.).to(tl.float32)
    x=av+dt
    # The original softplus threshold is 20, including its linear upper branch.
    # Match the original torch.compile lowering: ATen exp uses libdevice.exp,
    # not Triton's approximate exp2-based tl.exp. Tiny gate differences can
    # cross BF16 boundaries and then change MoE routing over many layers.
    sp=tl.where(x>20.,x,libdevice.log1p(libdevice.exp(x)))
    gv=-libdevice.exp(al)*sp
    bet=1./(1.+tl.exp(-bv))
    tl.store(g+i,tl.where(real,gv,0.),valid)
    # PF beta rounds to activation dtype; decode beta stays FP32.
    tl.store(beta_pf+i,tl.where(real,bet,0.),valid&(row<P))
    tl.store(beta_dec+i-P*H,tl.where(real,bet,0.),valid&(row>=P))
    tl.store(alpha_dec+i-P*H,tl.where(real,libdevice.exp(gv),1.),valid&(row>=P))


@triton.jit
def _norm_tile(core,z,weight,active,output,start,R:tl.constexpr,H:tl.constexpr,
               D:tl.constexpr,EPS:tl.constexpr,BD:tl.constexpr,ROWS:tl.constexpr):
    group=start+tl.arange(0,ROWS)
    row=group//H
    cols=tl.arange(0,BD)
    real=tl.load(active+row,group<R*H,other=0)
    mask=real[:,None]&(cols[None,:]<D)
    x=tl.load(core+group[:,None]*D+cols[None,:],mask,other=0.).to(tl.float32)
    gate=tl.load(z+group[:,None]*D+cols[None,:],mask,other=0.).to(tl.float32)
    w=tl.load(weight+cols,cols<D,other=0.).to(tl.float32)
    norm=x*tl.rsqrt(tl.sum(x*x,axis=1)/D+EPS)[:,None]
    result=(norm*w[None,:])*(gate*tl.sigmoid(gate))
    tl.store(output+group[:,None]*D+cols[None,:],tl.where(real[:,None],result,0.),
             (group[:,None]<R*H)&(cols[None,:]<D))


@triton.jit
def gated_norm(core,z,weight,active,output,R:tl.constexpr,H:tl.constexpr,
               D:tl.constexpr,EPS:tl.constexpr,BD:tl.constexpr,ROWS:tl.constexpr=4,
               recipes=None,P:tl.constexpr=0):
    # Each CTA still owns four head rows. The old norm's reduction layout
    # changes with its real (not padded) phase size: one/two/four rows per tile.
    # Plan-provided recipes select that arithmetic inside the same captured
    # kernel, without new graph variants, buffers, or host per-layer dispatch.
    start=tl.program_id(0)*ROWS
    if recipes is None:
        _norm_tile(core,z,weight,active,output,start,R,H,D,EPS,BD,ROWS)
    else:
        recipe=tl.load(recipes+tl.where(start<P*H,0,1))
        if recipe==1:
            for offset in tl.static_range(ROWS):
                _norm_tile(core,z,weight,active,output,start+offset,R,H,D,EPS,BD,1)
        elif recipe==2:
            for offset in tl.static_range(0,ROWS,2):
                _norm_tile(core,z,weight,active,output,start+offset,R,H,D,EPS,BD,2)
        else:
            _norm_tile(core,z,weight,active,output,start,R,H,D,EPS,BD,ROWS)
