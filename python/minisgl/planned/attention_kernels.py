"""Fused query gather/RoPE, rotated KV publication, and direct segment merge."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def rotate_load(x,positions,inv,row,pos_row,head,cols,valid,
                XS0:tl.constexpr,XS1:tl.constexpr,POS_ROWS:tl.constexpr,
                D:tl.constexpr,RD:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr):
    raw=tl.load(x+row*XS0+head*XS1+cols,valid&(cols<D),other=0.).to(tl.float32)
    frequency=cols%(RD//2)
    axis=tl.where((frequency%3==1)&(frequency<3*SH),1,
                  tl.where((frequency%3==2)&(frequency<3*SW),2,0))
    pos=tl.load(positions+axis*POS_ROWS+pos_row,valid&(cols<RD),other=0).to(tl.float32)
    invf=tl.load(inv+frequency,cols<RD,other=0.)
    angle=pos*invf
    pair=tl.where(cols<RD//2,cols+RD//2,cols-RD//2)
    other=tl.load(x+row*XS0+head*XS1+pair,valid&(cols<RD),other=0.).to(tl.float32)
    rotated=raw*libdevice.cos(angle)+tl.where(cols<RD//2,-other,other)*libdevice.sin(angle)
    return tl.where(cols<RD,rotated,raw)


@triton.jit
def gather_rope(q,sources,positions,inv,output,Q:tl.constexpr,H:tl.constexpr,
                D:tl.constexpr,RD:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
                QS0:tl.constexpr,QS1:tl.constexpr,BD:tl.constexpr):
    row,head=tl.program_id(0),tl.program_id(1)
    cols=tl.arange(0,BD)
    source=tl.load(sources+row)
    value=rotate_load(q,positions,inv,source,row,head,cols,source>=0,
        XS0=QS0,XS1=QS1,POS_ROWS=Q,D=D,RD=RD,SH=SH,SW=SW)
    tl.store(output+(row*H+head)*D+cols,value,cols<D)


@triton.jit
def store_rotated_kv(k,v,slots,positions,inv,kpool,vpool,
                     BASE:tl.constexpr,M:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
                     RD:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
                     KS0:tl.constexpr,KS1:tl.constexpr,VS0:tl.constexpr,VS1:tl.constexpr,BD:tl.constexpr):
    row,head=BASE+tl.program_id(0),tl.program_id(1)
    slot=tl.load(slots+row).to(tl.int64)
    cols=tl.arange(0,BD)
    if slot>=0:
        key=rotate_load(k,positions,inv,row,row,head,cols,True,
            XS0=KS0,XS1=KS1,POS_ROWS=M,D=D,RD=RD,SH=SH,SW=SW)
        value=tl.load(v+row*VS0+head*VS1+cols,cols<D,other=0.)
        tl.store(kpool+(slot*H+head)*D+cols,key,cols<D)
        tl.store(vpool+(slot*H+head)*D+cols,value,cols<D)


@triton.jit
def merge_segments(partial,lse,sources,output,gate,BASE:tl.constexpr,H:tl.constexpr,
                   D:tl.constexpr,S:tl.constexpr,BD:tl.constexpr,
                   HAS_GATE:tl.constexpr,GS0:tl.constexpr,GS1:tl.constexpr):
    row,head=BASE+tl.program_id(0),tl.program_id(1)
    cols=tl.arange(0,BD)
    maximum=tl.full((),-5.e4,tl.float32)
    denominator=tl.full((),0.,tl.float32)
    value=tl.zeros((BD,),tl.float32)
    for seg in range(S):
        src=tl.load(sources+row*S+seg)
        if src>=0:
            score=tl.load(lse+src*H+head)
            v=tl.load(partial+(src*H+head)*D+cols,cols<D,other=0.).to(tl.float32)
            updated=tl.maximum(maximum,score)
            a,b=tl.exp2(maximum-updated),tl.exp2(score-updated)
            # FlashInfer state_t::merge uses CUDA fused multiply-add. Keep
            # that rounding even though RoPE/gating need unfused boundaries.
            value=tl.fma(value,a,v*b)
            denominator=tl.fma(denominator,a,b)
            maximum=updated
    result=value/tl.where(denominator>0.,denominator,1.)
    if HAS_GATE:
        g=tl.load(gate+row*GS0+head*GS1+cols,(cols<D)&(denominator>0.),other=0.).to(tl.float32)
        # Preserve the old BF16 Attention-output boundary before sigmoid gate.
        result=result.to(output.dtype.element_ty).to(tl.float32)*tl.sigmoid(g)
    tl.store(output+(row*H+head)*D+cols,result,cols<D)


@triton.jit
def normalize_qk(raw,q_weight,k_weight,active,q,k,HQ:tl.constexpr,HK:tl.constexpr,
                  D:tl.constexpr,EPS:tl.constexpr,STRIDE:tl.constexpr,BD:tl.constexpr):
    row,head=tl.program_id(0),tl.program_id(1)
    cols=tl.arange(0,BD)
    real=tl.load(active+row)
    is_q=head<HQ
    offset=tl.where(is_q,head*2*D,2*HQ*D+(head-HQ)*D)
    value=tl.load(raw+row*STRIDE+offset+cols,real&(cols<D),other=0.).to(tl.float32)
    qw=tl.load(q_weight+cols,cols<D,other=0.).to(tl.float32)
    kw=tl.load(k_weight+cols,cols<D,other=0.).to(tl.float32)
    weight=tl.where(is_q,qw,kw)
    # Preserve FlashInfer's default CuTe QKRMSNorm BF16/FP16 vec8 accumulation:
    # eight contiguous values per lane, followed by the warp reduction. A flat
    # tree over D values is close, but changed one learned Q value and then
    # downstream routing in the full A3B checkpoint.
    lane=tl.arange(0,32)
    partial=tl.zeros((32,),tl.float32)
    for chunk in tl.static_range(triton.cdiv(D,256)):
        for j in tl.static_range(8):
            c=chunk*256+lane*8+j
            v=tl.load(raw+row*STRIDE+offset+c,real&(c<D),other=0.).to(tl.float32)
            partial=tl.fma(v,v,partial)
    # CuTe norm/utils.py warp_reduce uses ascending XOR offsets 1,2,4,8,16
    # and keeps the sum per lane. Broadcasting one flat tl.sum or reversing
    # those offsets can differ by one FP32 ULP on learned activations.
    for shift in tl.static_range(5):
        partial+=tl.gather(partial,lane^(1<<shift),axis=0)
    rstd=tl.rsqrt(partial/D+EPS)
    normalized=value*tl.gather(rstd,(cols%256)//8,axis=0)*(1.+weight)
    if is_q:tl.store(q+(row*HQ+head)*D+cols,normalized,cols<D)
    else:tl.store(k+(row*HK+head-HQ)*D+cols,normalized,cols<D)
