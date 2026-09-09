"""Fixed-output Linear with one reusable FP8 activation arena per profile."""
import torch
import triton
import triton.language as tl

from minisgl.kernel.fp8 import _block_gemm


@triton.jit
def quantize_active_groups(x,q,scales,active,K:tl.constexpr,HAS_MASK:tl.constexpr,IDS_MASK:tl.constexpr=False):
    group=tl.program_id(0)
    cols=group*128+tl.arange(0,128)
    real=True
    if HAS_MASK:
        flag=tl.load(active+group//(K//128))
        real=flag>=0 if IDS_MASK else flag
    values=tl.load(x+cols,real,other=0.).to(tl.float32)
    maximum=tl.maximum(tl.max(tl.abs(values),0),1e-10)
    multiplier=tl.inline_asm_elementwise("div.approx.ftz.f32 $0, $1, $2;","=f,f,f",
        [448.,maximum],dtype=tl.float32,is_pure=True,pack=1)
    quant=tl.minimum(tl.maximum(values*multiplier,-448.),448.)
    tl.store(q+cols,quant)
    tl.store(scales+group,maximum*(1./448.))


class LinearWorkspace:
    def __init__(self,rows,max_input_size,*,device,max_elements=None):
        if rows<1 or max_input_size<128 or max_input_size%128:
            raise ValueError("invalid FP8 workspace capacity")
        self.rows,self.k=rows,max_input_size
        elements=rows*max_input_size if max_elements is None else max_elements
        if type(elements) is not int or elements<128 or elements%128:
            raise ValueError("invalid FP8 arena element capacity")
        self.quant=torch.empty(elements,device=device,dtype=torch.float8_e4m3fn)
        self.scales=torch.empty(elements//128,device=device,dtype=torch.float32)

    def views(self,rows,k):
        if not 0<rows<=self.rows or not 0<k<=self.k or k%128 or rows*k>self.quant.numel():
            raise ValueError("FP8 activation workspace capacity exceeded")
        return self.quant[:rows*k].view(rows,k),self.scales[:rows*(k//128)].view(rows,k//128)


class BoundLinear:
    def __init__(self,layer,rows,*,dtype,workspace=None,active=None):
        w=layer.weight
        self.weight,self.scales,self.rows,self.dtype=w,layer.weight_scale_inv,rows,dtype
        self.n,self.k=w.shape
        self.active=active
        if (layer.bias is not None or rows<1 or not w.is_cuda or not w.is_contiguous()
                or dtype not in (torch.float16,torch.bfloat16,torch.float32)):
            raise ValueError("unsupported planned Linear recipe")
        if active is not None and (active.shape!=(rows,) or active.dtype!=torch.bool or active.device!=w.device):
            raise ValueError("invalid Linear active-row mask")
        self.transposed=w.t()
        self.quant=self.input_scales=None
        if self.scales is None:
            if w.dtype!=dtype:raise ValueError("dense Linear weight/activation dtype mismatch")
        else:
            if (w.dtype!=torch.float8_e4m3fn or self.n%128 or self.k%128
                    or self.scales.shape!=(self.n//128,self.k//128) or self.scales.dtype!=torch.float32
                    or self.scales.device!=w.device or not self.scales.is_contiguous()):
                raise ValueError("invalid serialized block-FP8 weights/scales")
            if torch.cuda.get_device_capability(w.device)<(8,9):
                raise ValueError("native FP8 backend unsupported; select whole-forward compatibility before writes")
            if workspace is None:raise ValueError("FP8 Linear needs a profile-owned activation workspace")
            self.quant,self.input_scales=workspace.views(rows,self.k)
            if self.quant.device!=w.device:raise ValueError("FP8 workspace device mismatch")
        self.bm=16 if rows<32 else 32

    def run(self,x,output):
        if (x.shape!=(self.rows,self.k) or output.shape!=(self.rows,self.n)
                or any(t.device!=self.weight.device or t.dtype!=self.dtype or not t.is_contiguous()
                       for t in (x,output))):raise ValueError("invalid bound Linear input/output")
        if self.scales is None:torch.mm(x,self.transposed,out=output)
        else:
            quantize_active_groups[(self.rows*(self.k//128),)](
                x,self.quant,self.input_scales,x if self.active is None else self.active,
                K=self.k,HAS_MASK=self.active is not None,num_warps=4)
            _block_gemm[(triton.cdiv(self.rows,self.bm),triton.cdiv(self.n,64))](
                self.quant,self.weight,self.input_scales,self.scales,output,self.rows,self.n,self.k,self.bm,64)
        return output
