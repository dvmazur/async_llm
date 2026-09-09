"""Unchanged old shared-attention wrappers; no new GPU/CPU-plan evaluator."""
from types import SimpleNamespace
import torch


class KVPool:
    def __init__(self,k,v):self.k,self.v=k,v
    def k_cache(self,layer):return self.k[layer]
    def v_cache(self,layer):return self.v[layer]
    def store_kv(self,k,v,locations,layer):
        heads,dim=self.k.shape[-2:]
        self.k[layer].view(-1,heads*dim).index_copy_(0,locations,k)
        self.v[layer].view(-1,heads*dim).index_copy_(0,locations,v)


class Block:
    def __init__(self,block,page_size,device):
        self.num_tokens,self.mrope_span=block.num_tokens,block.mrope_span
        self.pages,self.p,self.device=block.pages,page_size,device
    @property
    def num_pages(self):return (self.num_tokens+self.p-1)//self.p
    @property
    def last_page_len(self):return self.num_tokens-(self.num_pages-1)*self.p
    def page_numbers_tensor(self):
        return torch.tensor(self.pages[:self.num_pages],device=self.device,dtype=torch.int32)
    def next_slot(self):return self.pages[self.num_tokens//self.p]*self.p+self.num_tokens%self.p


def old_attention(op,kv_blocks,forward,q,k,v,layer,*,image_positions=None):
    from minisgl.shared_cache.attention import PrefillSpec
    dev=q.device
    blocks={i:Block(b,op.page_size,dev) for i,b in kv_blocks.items()}
    outputs=[]
    p=sum(r.length for r in forward.prefill_requests)
    if p:
        specs=[];positions=[[],[],[]];slots=[]
        for i,r in enumerate(forward.prefill_requests):
            b=blocks[r.write_to]
            rel=(image_positions or {}).get(i,(tuple(range(r.length)),)*3)
            end=b.num_tokens+r.length
            pages=b.pages[:(end+op.page_size-1)//op.page_size]
            specs.append(PrefillSpec(context=[blocks[x] for x in r.context if blocks[x].num_tokens],
                self_page_starts=torch.tensor(pages,device=dev)*op.page_size,num_new=r.length,
                self_prefix_len=b.num_tokens,self_prefix_span=b.mrope_span,
                mrope_rel=torch.tensor(rel,device=dev)))
            for a in range(3):positions[a].extend(b.mrope_span+t for t in rel[a])
            for t in range(r.length):
                index=b.num_tokens+t
                slots.append(b.pages[index//op.page_size]*op.page_size+index%op.page_size)
        meta=op.prepare_prefill_batch(specs)
        batch=SimpleNamespace(attn_metadata=meta,positions=torch.tensor(positions[0],device=dev),
            mrope_positions=torch.tensor(positions,device=dev),out_loc=torch.tensor(slots,device=dev))
        outputs.append(op.forward(q[:p].reshape(p,-1),k[:p].reshape(p,-1),v[:p].reshape(p,-1),layer,batch))
        for i,r in enumerate(forward.prefill_requests):
            rel=(image_positions or {}).get(i,(tuple(range(r.length)),)*3)
            blocks[r.write_to].num_tokens+=r.length
            blocks[r.write_to].mrope_span+=max(max(a) for a in rel)+1
    d=len(forward.decode_requests)
    if d:
        requests=forward.decode_requests
        group=SimpleNamespace(num_workers=d,cache_structure=[[blocks[b] for b in r.read_blocks] for r in requests],
                              write_to=[blocks[r.write_to] for r in requests])
        slots=torch.tensor([b.next_slot() for b in group.write_to],device=dev)
        new_pages={id(b):(b.pages[b.num_tokens//op.page_size]*op.page_size if b.num_tokens%op.page_size==0 else None)
                   for b in group.write_to}
        meta=op.prepare(group,new_pages,slots)
        positions=torch.tensor([b.mrope_span for b in group.write_to],device=dev)
        batch=SimpleNamespace(attn_metadata=meta,positions=positions,mrope_positions=positions[None].expand(3,-1),out_loc=slots)
        start=forward.capacity.prefill_tokens
        outputs.append(op.forward(q[start:start+d].reshape(d,-1),k[start:start+d].reshape(d,-1),
                                  v[start:start+d].reshape(d,-1),layer,batch))
    return torch.cat(outputs).reshape(p+d,q.shape[1],q.shape[2]) if outputs else q[:0]
