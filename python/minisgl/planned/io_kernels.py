import triton
import triton.language as tl


@triton.jit
def embed_inputs(tokens,images,active,weights,features,output,H:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    real=tl.load(active+row)
    token=tl.load(tokens+row).to(tl.int64)
    feature=tl.load(images+row).to(tl.int64)
    word=tl.load(weights+token*H+col,real&(feature<0)&(col<H),other=0.)
    image=tl.load(features+feature*H+col,real&(feature>=0)&(col<H),other=0.)
    tl.store(output+row*H+col,tl.where(feature>=0,image,word),col<H)


@triton.jit
def selected_final_norm(x,residual,weight,output_rows,selected,H:tl.constexpr,EPS:tl.constexpr,BH:tl.constexpr):
    dest=tl.program_id(0)
    src=tl.load(output_rows+dest).to(tl.int64)
    cols=tl.arange(0,BH)
    valid=(src>=0)&(cols<H)
    # FlashInfer fused-add norm normalizes the FP32 sum, not the rounded
    # residual store. There is no downstream residual consumer after final norm.
    value=tl.load(x+src*H+cols,valid,other=0.).to(tl.float32)
    value+=tl.load(residual+src*H+cols,valid,other=0.).to(tl.float32)
    gamma=tl.load(weight+cols,cols<H,other=0.).to(tl.float32)
    result=value*tl.rsqrt(tl.sum(value*value)/H+EPS)*(1.+gamma)
    tl.store(selected+dest*H+cols,result,cols<H)
