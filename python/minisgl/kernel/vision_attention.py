"""Non-causal vision attention; SGLang arithmetic on CUDA, SDPA fallback."""
from dataclasses import dataclass
from itertools import accumulate
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VisionAttentionPlan:
    lengths: tuple[int,...]
    starts: torch.Tensor
    sizes: torch.Tensor
    maximum: int


def prepare_vision_attention(lengths, device, cu_seqlens=None):
    lengths=tuple(int(n) for n in lengths)
    if not lengths or any(n<=0 for n in lengths):
        raise ValueError('Vision segments must be nonempty and positive')
    cu=(torch.tensor([0,*accumulate(lengths)],dtype=torch.int32,device=device)
        if cu_seqlens is None else cu_seqlens.to(device=device,dtype=torch.int32))
    if cu.numel()!=len(lengths)+1:
        raise ValueError('Invalid cumulative segment lengths')
    return VisionAttentionPlan(lengths,cu,cu[1:]-cu[:-1],max(lengths))


def vision_attention(q,k,v,plan):
    """[tokens,heads,dim]; images attend internally, never across boundaries."""
    if (q.ndim!=3 or q.shape!=k.shape or q.shape!=v.shape or
        q.dtype!=k.dtype or q.dtype!=v.dtype or q.device!=k.device or q.device!=v.device):
        raise ValueError('Expected matching Q/K/V tensors[tokens,heads,dim]')
    if sum(plan.lengths)!=q.shape[0] or plan.starts.device!=q.device:
        raise ValueError('Vision attention plan does not match token rows/device')
    if q.is_cuda and q.dtype in (torch.bfloat16,torch.float16):
        from .triton.vision_attention import context_attention_fwd
        # The upstream kernel permits arbitrary row/head strides, but D is unit stride.
        q,k,v=(x if x.stride(-1)==1 else x.contiguous() for x in (q,k,v))
        out=torch.empty_like(q,memory_format=torch.contiguous_format)
        context_attention_fwd(q,k,v,out,plan.starts,plan.sizes,plan.maximum,is_causal=False)
        return out
    parts=[];offset=0
    for length in plan.lengths:
        tensors=[x[offset:offset+length].transpose(0,1).unsqueeze(0) for x in (q,k,v)]
        parts.append(F.scaled_dot_product_attention(*tensors).squeeze(0).transpose(0,1))
        offset+=length
    return torch.cat(parts)
