"""Explicit eager compatibility over the same all-layer pool.

Selected for the complete GDN layer set before any writes. Uses unchanged old
layer/chunk/recurrent dispatch (including no-FLA arithmetic). Not graph capable,
not a hidden eager call within a captured planned forward.
"""
from types import SimpleNamespace

import torch


class _PoolPhase:
    def __init__(self,pool,phase,lengths):
        self.pool,self.phase,self.prefill_segments=pool,phase,lengths
        self.workers=sum(phase.active)
        offsets=[0]
        for n in lengths or ():offsets.append(offsets[-1]+n)
        self.prefill_cu_seqlens_cpu=torch.tensor(offsets,dtype=torch.long)
        self.prefill_cu_seqlens=self.prefill_cu_seqlens_cpu.to(pool.affine.device)

    def _workers(self,workers):return list(range(self.workers)) if workers is None else workers

    def _affine(self,lin,w):
        s=self.pool.shape
        if not self.phase.write_fresh[w]:
            a,b=self.pool.affine[lin,self.phase.write_slots[w]]
            return a[None],b[None]
        a=torch.eye(s.dim,device=self.pool.affine.device)[None,None].expand(1,s.heads,s.dim,s.dim)
        return a,torch.zeros_like(a)

    def prior_conv_states(self,lin):
        s=self.pool.shape
        zero=lambda:torch.zeros((s.conv_channels,s.conv_window),device=self.pool.conv.device,dtype=self.pool.conv.dtype)
        return torch.stack([self.pool.conv[lin,slot] if slot>=0 else zero()
                            for slot in self.phase.prior_conv_slots[:self.workers]])

    def compose_initial_recurrent_state(self,lin,dtype,*,state_v_first):
        assert state_v_first
        s=self.pool.shape
        zero=torch.zeros((s.heads,s.dim,s.dim),device=self.pool.affine.device,dtype=torch.float32)
        levels=[]
        for depth,level in enumerate(self.phase.trie.levels):
            values=[]
            for node in level:
                a,b=self.pool.affine[lin,node.slot]
                values.append(b if depth==0 else levels[depth-1][node.parent_row]@a+b)
            levels.append(values)
        return torch.stack([levels[depth-1][row] if depth else zero
                            for depth,row in self.phase.trie.terminals[:self.workers]]).to(dtype)

    def affine_scan_initial_state(self,lin,*,num_heads,d_k,d_v,workers=None):
        return torch.cat([torch.cat(self._affine(lin,w),dim=-2) for w in self._workers(workers)])

    def store_affine_scan_state(self,lin,state,*,d_k,workers=None):
        for row,w in enumerate(self._workers(workers)):
            target=self.pool.affine[lin,self.phase.write_slots[w]]
            target[0].copy_(state[row,:,:d_k]);target[1].copy_(state[row,:,d_k:])

    def set_conv_states(self,lin,conv,workers=None):
        for row,w in enumerate(self._workers(workers)):
            self.pool.conv[lin,self.phase.write_slots[w]].copy_(conv[row])

    def capture_token_affines(self,lin,key,value,alpha,beta,workers=None):
        from minisgl.shared_cache.gdn_affine import update_affine_summary
        key=key.float()
        key=key*torch.rsqrt((key*key).sum(-1,keepdim=True)+1e-6)
        for row,w in enumerate(self._workers(workers)):
            A,B=self._affine(lin,w)
            for t in range(key.shape[1]):
                A,B=update_affine_summary(A_hat=A,B_hat=B,k=key[row:row+1,t],
                    v=value[row:row+1,t].float(),alpha=alpha[row:row+1,t].float(),
                    beta=beta[row:row+1,t].float())
            target=self.pool.affine[lin,self.phase.write_slots[w]]
            target[0].copy_(A[0]);target[1].copy_(B[0])


class EagerGDNLayers:
    graph_compatible=False

    def __init__(self,layers,pool,plan,reason):
        self.layers,self.pool,self.capacity,self.reason=layers,pool,plan.capacity,reason
        self.upload(plan)

    def upload(self,plan):
        if plan.capacity!=self.capacity:raise ValueError("profile capacity changed")
        lengths=[b-a for a,b in zip(plan.rows.prefill_offsets,plan.rows.prefill_offsets[1:]) if b>a]
        self.pf=_PoolPhase(self.pool,plan.prefill,lengths)
        self.dec=_PoolPhase(self.pool,plan.decode,None)
        self.p=sum(lengths)
        active=[i for i,value in enumerate(plan.rows.active) if value]
        self.indices=torch.tensor(active,device=self.pool.affine.device,dtype=torch.long)

    def run(self,index,x,output):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("GDN compatibility requires eager execution of the whole forward")
        layer=self.layers[index]
        rows=self.capacity.prefill_tokens+self.capacity.decode_workers
        if x.shape!=(rows,layer.in_proj_qkv.full_input_size) or output.shape!=x.shape:
            raise ValueError("invalid GDN compatibility inputs")
        actual=x.index_select(0,self.indices)
        if self.pf.workers and self.dec.workers:
            value=layer._forward_ar_mixed(actual,SimpleNamespace(split=self.p,prefill=self.pf,decode=self.dec))
        elif self.pf.workers:value=layer._forward_ar_prefill(actual,self.pf)
        elif self.dec.workers:value=layer._forward_ar_decode(actual,self.dec)
        else:value=actual
        output.zero_()
        output.index_copy_(0,self.indices,value)
        return output
