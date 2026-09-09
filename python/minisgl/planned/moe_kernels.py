"""Bounded routing/alignment and masked expert intermediates; GEMM unchanged."""
import triton
import triton.language as tl


@triton.jit
def init_alignment(sorted_ids,expert_ids,counts,N:tl.constexpr,A:tl.constexpr,
                   NB:tl.constexpr,E:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(sorted_ids+i,N,i<A)
    tl.store(expert_ids+i,-1,i<NB)
    tl.store(counts+i,0,i<E)


@triton.jit
def finish_routes(raw_weights,raw_ids,active,weights,ids,counts,
                  M:tl.constexpr,K:tl.constexpr,E:tl.constexpr,RENORM:tl.constexpr,
                  BR:tl.constexpr,BK:tl.constexpr,BE:tl.constexpr):
    rows=tl.program_id(0)*BR+tl.arange(0,BR)
    cols=tl.arange(0,BK)
    real=tl.load(active+rows,rows<M,other=0)
    mask=(rows[:,None]<M)&(cols[None,:]<K)
    index=rows[:,None]*K+cols[None,:]
    selected=tl.load(raw_ids+index,mask&real[:,None],other=-1).to(tl.int32)
    valid=mask&real[:,None]&(selected>=0)&(selected<E)
    value=tl.load(raw_weights+index,valid,other=0.).to(tl.float32)
    if RENORM:value=value/(tl.sum(value,axis=1)[:,None]+1e-8)
    tl.store(weights+index,value,mask)
    tl.store(ids+index,tl.where(valid,selected,-1),mask)
    # Local histogram combines active masking, optional renormalization and
    # assignment counting in one producer, without a partial-histogram buffer.
    bins=tl.histogram(tl.reshape(tl.where(valid,selected,E),(BR*BK,)),BE)
    experts=tl.arange(0,BE)
    tl.atomic_add(counts+experts,bins,(experts<E)&(bins>0),sem="relaxed")


@triton.jit
def prefix_counts(counts,starts,E:tl.constexpr,BM:tl.constexpr,BE:tl.constexpr):
    experts=tl.arange(0,BE)
    count=tl.load(counts+experts,experts<E,other=0)
    padded=tl.cdiv(count,BM)*BM
    ends=tl.cumsum(padded)
    tl.store(starts+experts,ends-padded,experts<E)
    tl.store(starts+E,tl.sum(padded))
    # Histogram storage becomes per-expert insertion cursors after this kernel.
    tl.store(counts+experts,0,experts<E)


@triton.jit
def scatter_assignments(ids,starts,cursors,sorted_ids,expert_ids,N:tl.constexpr,
                        E:tl.constexpr,NB:tl.constexpr,BM:tl.constexpr,BE:tl.constexpr,BLOCK:tl.constexpr):
    block=tl.program_id(0)
    if block<NB:
        ends=tl.load(starts+1+tl.arange(0,BE),tl.arange(0,BE)<E,other=2147483647)
        expert=tl.sum((block*BM>=ends).to(tl.int32))
        total=tl.load(starts+E)
        tl.store(expert_ids+block,expert,block*BM<total)
    offset=block*BLOCK+tl.arange(0,BLOCK)
    expert=tl.load(ids+offset,offset<N,other=-1)
    valid=(offset<N)&(expert>=0)
    base=tl.load(starts+expert,valid,other=0)
    rank=tl.atomic_add(cursors+expert,1,valid,sem="relaxed")
    tl.store(sorted_ids+base+rank,offset,valid)


@triton.jit
def silu_experts(first,ids,second,N:tl.constexpr,I:tl.constexpr,BLOCK:tl.constexpr,IDS_MASK:tl.constexpr=True):
    row=tl.program_id(0)
    col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    flag=tl.load(ids+row)
    valid=flag>=0 if IDS_MASK else flag
    gate=tl.load(first+row*2*I+col,valid&(col<I),other=0.).to(tl.float32)
    up=tl.load(first+row*2*I+I+col,valid&(col<I),other=0.).to(tl.float32)
    tl.store(second+row*I+col,(gate*tl.sigmoid(gate))*up,col<I)


@triton.jit
def reduce_experts(values,ids,output,shared,gate,active,M:tl.constexpr,K:tl.constexpr,
                   H:tl.constexpr,BLOCK:tl.constexpr,HAS_SHARED:tl.constexpr):
    row=tl.program_id(0)
    cols=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    acc=tl.zeros((BLOCK,),tl.float32)
    for i in range(K):
        valid=tl.load(ids+row*K+i)>=0
        value=tl.load(values+(row*K+i)*H+cols,valid&(cols<H),other=0.).to(tl.float32)
        acc+=value
    if HAS_SHARED:
        real=tl.load(active+row)
        raw=tl.load(gate+row,real,other=0.).to(tl.float32)
        s=tl.load(shared+row*H+cols,real&(cols<H),other=0.).to(tl.float32)
        # Old Qwen shared branch rounds sigmoid and multiplication to BF16
        # before adding the already-rounded routed reduction. Preserve all three.
        sg=tl.sigmoid(raw).to(output.dtype.element_ty).to(tl.float32)
        product=(s*sg).to(output.dtype.element_ty).to(tl.float32)
        acc=acc.to(output.dtype.element_ty).to(tl.float32)+product
        acc=tl.where(real,acc,0.)
    tl.store(output+row*H+cols,acc,cols<H)
