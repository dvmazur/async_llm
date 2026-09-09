"""Old layer adapter with independent Python/dictionary state and chain fold.

The production Qwen layer is left unchanged in the working copy. This fixture
exposes its original AR protocol, not the new planner's phase arrays/kernels.
"""
import torch


class OldState:
    def __init__(self,pool,blocks):
        self.blocks=blocks
        self.heads,self.dim=pool.shape.heads,pool.shape.dim
        self.affine={}
        self.conv={}
        for layer in range(pool.shape.layers):
            for b in blocks.values():
                if b.populated:
                    self.affine[layer,b.block_id]=tuple(x.clone()[None] for x in pool.affine[layer,b.slot])
                if b.has_conv:self.conv[layer,b.block_id]=pool.conv[layer,b.slot].clone()[None]
        self.device=pool.affine.device
        self.conv_shape=(pool.shape.conv_channels,pool.shape.conv_window)
        self.dtype=pool.conv.dtype

    def phase(self,requests,*,prefill):return OldPhase(self,requests,prefill=prefill)


class OldPhase:
    def __init__(self,state,requests,*,prefill):
        self.state,self.requests,self.prefill=state,requests,prefill
        self.prefill_segments=[r.length for r in requests] if prefill else None
        offsets=[0]
        for r in requests:offsets.append(offsets[-1]+(r.length if prefill else 1))
        self.prefill_cu_seqlens_cpu=torch.tensor(offsets,dtype=torch.long)
        self.prefill_cu_seqlens=self.prefill_cu_seqlens_cpu.to(state.device)

    def chain(self,r):
        return (*r.context,r.write_to) if self.prefill else r.read_blocks

    def prior_conv_states(self,lin):
        s=self.state
        windows=[]
        for r in self.requests:
            window=torch.zeros((1,*s.conv_shape),device=s.device,dtype=s.dtype)
            for b in self.chain(r):
                if (lin,b) in s.conv:window=s.conv[lin,b]
            windows.append(window)
        return torch.cat(windows)

    def compose_initial_recurrent_state(self,lin,dtype,*,state_v_first):
        assert state_v_first
        s=self.state
        states=[]
        for r in self.requests:
            acc=torch.zeros((1,s.heads,s.dim,s.dim),device=s.device,dtype=torch.float32)
            for b in self.chain(r):
                if (lin,b) in s.affine:
                    a,v=s.affine[lin,b]
                    acc=acc@a+v
            states.append(acc)
        return torch.cat(states).to(dtype)

    def _affine(self,lin,worker):
        s=self.state
        b=self.requests[worker].write_to
        if (lin,b) in s.affine:return s.affine[lin,b]
        a=torch.eye(s.dim,device=s.device)[None,None].expand(1,s.heads,s.dim,s.dim).clone()
        return a,torch.zeros_like(a)

    def _workers(self,workers):return list(range(len(self.requests))) if workers is None else workers

    def affine_scan_initial_state(self,lin,*,num_heads,d_k,d_v,workers=None):
        return torch.cat([torch.cat(self._affine(lin,w),dim=-2) for w in self._workers(workers)])

    def store_affine_scan_state(self,lin,state,*,d_k,workers=None):
        for row,w in enumerate(self._workers(workers)):
            self.state.affine[lin,self.requests[w].write_to]=(state[row:row+1,:,:d_k].clone(),
                                                           state[row:row+1,:,d_k:].clone())

    def set_conv_states(self,lin,conv,workers=None):
        for row,w in enumerate(self._workers(workers)):
            self.state.conv[lin,self.requests[w].write_to]=conv[row:row+1].clone()

    def capture_token_affines(self,lin,key,value,alpha,beta,workers=None):
        # Literal rank-one old update, deliberately not new fused evaluator.
        key=key.float()
        key=key*torch.rsqrt((key*key).sum(-1,keepdim=True)+1e-6)
        for row,w in enumerate(self._workers(workers)):
            A,B=self._affine(lin,w)
            for t in range(key.shape[1]):
                k=key[row:row+1,t]
                v=value[row:row+1,t].float()
                a=alpha[row:row+1,t].float()[...,None,None]
                b=beta[row:row+1,t].float()[...,None,None]
                A=a*A-a*b*(A*k[...,None,:]).sum(-1)[...,None]*k[...,None,:]
                B=a*B-a*b*(B*k[...,None,:]).sum(-1)[...,None]*k[...,None,:]+b*v[...,None]*k[...,None,:]
            self.state.affine[lin,self.requests[w].write_to]=(A,B)


def old_mixed(layer,x,state,prefill,decode):
    p=sum(r.length for r in prefill)
    if prefill and decode:
        from types import SimpleNamespace
        ar=SimpleNamespace(split=p,prefill=state.phase(prefill,prefill=True),
                           decode=state.phase(decode,prefill=False))
        return layer._forward_ar_mixed(x,ar)
    if prefill:return layer._forward_ar_prefill(x,state.phase(prefill,prefill=True))
    return layer._forward_ar_decode(x,state.phase(decode,prefill=False))


def raw_window_reference(before,raw,plan,prefill,decode):
    """Window bytes from *the same projected input*, independent of conv/store.

    Exact vs capacity BF16 GEMMs need not emit bit-identical raw projections.
    This tests copying/lifetime exactly, separately from old-model numerics.
    """
    result=before.clone()
    window=before.shape[-1]
    for phase,requests,pf in ((plan.prefill,prefill,True),(plan.decode,decode,False)):
        snapshot=result.clone()
        offset=0 if pf else plan.capacity.prefill_tokens
        for i,r in enumerate(requests):
            slot=phase.prior_conv_slots[i]
            prior=snapshot[slot] if slot>=0 else torch.zeros_like(snapshot[0])
            length=r.length if pf else 1
            new=torch.cat([prior,raw[offset:offset+length].t()],dim=-1)[:,-window:]
            result[phase.write_slots[i]]=new
            offset+=length
    return result
