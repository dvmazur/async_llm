from types import SimpleNamespace
import pytest,torch
from minisgl.shared_cache.attention import SharedCacheAttention

def wrapper(floats=8,ints=4):
 w=SimpleNamespace(_float_workspace_buffer=torch.empty(floats,dtype=torch.uint8),_int_workspace_buffer=torch.empty(ints,dtype=torch.uint8),calls=[])
 def reset(f,i):w._float_workspace_buffer=f;w._int_workspace_buffer=i
 w.reset_workspace_buffer=reset;w.workspace_size=lambda **kw:(32,16);w.plan=lambda **kw:w.calls.append(kw)
 return w

def test_eager_grows_before_plan_and_reuses_scratch_without_touching_graph_arena():
 a=SharedCacheAttention.__new__(SharedCacheAttention);a.device=torch.device('cpu');a._workspace=torch.empty(8,dtype=torch.uint8)
 original=a._workspace;w=wrapper()
 a._plan_eager_prefill(w,causal=False,non_blocking=True)
 assert w._float_workspace_buffer.numel()==32 and w._int_workspace_buffer.numel()==16
 assert w.calls==[dict(causal=False,non_blocking=True)]
 f,i=w._float_workspace_buffer,w._int_workspace_buffer
 a._plan_eager_prefill(w,causal=True,non_blocking=True)
 assert w._float_workspace_buffer is f and w._int_workspace_buffer is i and a._workspace is original

def test_eager_allocation_failure_keeps_both_old_bindings(monkeypatch):
 a=SharedCacheAttention.__new__(SharedCacheAttention);a.device=torch.device('cpu');w=wrapper()
 f,i=w._float_workspace_buffer,w._int_workspace_buffer;empty=torch.empty;count=0
 def allocation(*args,**kw):
  nonlocal count
  count+=1
  if count==2:raise torch.OutOfMemoryError('test')
  return empty(*args,**kw)
 monkeypatch.setattr(torch,'empty',allocation)
 with pytest.raises(torch.OutOfMemoryError):a._plan_eager_prefill(w,causal=False)
 assert w._float_workspace_buffer is f and w._int_workspace_buffer is i and not w.calls

@pytest.mark.skipif(not torch.cuda.is_available(),reason='real FlashInfer prefill workspace')
@torch.inference_mode()
def test_moe_1024_row_context_prefill_uses_reported_workspace():
 from flashinfer import BatchPrefillWithPagedKVCacheWrapper
 device=torch.device('cuda');a=SharedCacheAttention.__new__(SharedCacheAttention);a.device=device
 a._workspace=torch.empty(128*1024*1024,dtype=torch.uint8,device=device);arena=a._workspace
 w=BatchPrefillWithPagedKVCacheWrapper(arena,kv_layout='NHD',backend='fa2')
 kwargs=dict(qo_indptr=torch.tensor([0,1024],dtype=torch.int32),paged_kv_indptr=torch.tensor([0,1800],dtype=torch.int32),paged_kv_indices=torch.arange(1800,dtype=torch.int32,device=device),paged_kv_last_page_len=torch.ones(1,dtype=torch.int32),seq_lens=torch.tensor([1800],dtype=torch.int32),num_qo_heads=16,num_kv_heads=2,head_dim_qk=256,page_size=1,q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,max_token_per_sequence=1024,causal=False)
 required=w.workspace_size(**kwargs)
 a._plan_eager_prefill(w,**kwargs)
 assert w._float_workspace_buffer.numel()>=required[0] and w._int_workspace_buffer.numel()>=required[1] and a._workspace is arena
 q=torch.zeros(1024,16,256,device=device,dtype=torch.bfloat16)
 k=torch.zeros(1800,1,2,256,device=device,dtype=torch.bfloat16);v=torch.ones_like(k)
 torch.testing.assert_close(w.run(q,(k,v)),torch.ones_like(q),rtol=0,atol=.002)

 generator=torch.Generator(device=device).manual_seed(17)
 q=torch.randn(q.shape,device=device,dtype=q.dtype,generator=generator)*.1
 k=torch.randn(k.shape,device=device,dtype=k.dtype,generator=generator)*.1
 v=torch.randn(v.shape,device=device,dtype=v.dtype,generator=generator)
 expected=torch.nn.functional.scaled_dot_product_attention(q.transpose(0,1)[None],k[:,0].transpose(0,1)[None],v[:,0].transpose(0,1)[None],enable_gqa=True)[0].transpose(0,1)
 torch.testing.assert_close(w.run(q,(k,v)),expected,rtol=.02,atol=.002)
