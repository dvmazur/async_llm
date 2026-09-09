"""Session API over one planned decoder and fixed all-layer pools (TP1).

Scheduler policy, sampling and vision remain outside the captured decoder.
Every supported physical forward has one prepare, one whole-body execution,
then one completion/metadata commit. Existing async frontend is duck-typed.
"""
from dataclasses import replace
from functools import wraps
import time
import torch

from .block_store import BlockStore
from .catalogue import ProfileCatalogue
from .forward_plan import PrefillRequest,DecodeRequest
from .model_io import prepare_inputs
from .pools import GDNPool,PoolShape
from .decoder import DecoderProgram
from .runner import ProgramRunner


def _session_stream(method):
    """One runtime-owned CUDA stream; callers may submit from another stream.

    Consume caller inputs only after their queued producers. Public executions
    retain/commit after completion, so returned outputs are already ready. This
    does not allow two simultaneous model forwards in the same session.
    """
    @wraps(method)
    def run(self,*args,**kwargs):
        caller=torch.cuda.current_stream(self.device)
        if caller!=self.stream:self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):return method(self,*args,**kwargs)
    return run


class PlannedSession:
    def __init__(self,model,profiles,*,use_graph=True,allow_eager_overflow=True):
        self.model=model
        self.catalogue=ProfileCatalogue(profiles)
        profile=self.catalogue.profiles[0]
        f,a=profile.forward,profile.attention
        self.store=BlockStore(block_slots=f.block_slots,page_size=a.page_size,
                              page_capacity=a.page_pool_capacity,dummy_page=a.dummy_page)
        self.page_size=a.page_size
        self.device,self.dtype=model.model.embed_tokens.weight.device,model.model.embed_tokens.weight.dtype
        if self.device.type!="cuda" or model.model.embed_tokens.tp_size!=1:
            raise ValueError("planned session requires a loaded CUDA TP1 model")
        self.stream=torch.cuda.Stream(device=self.device)
        layers=tuple(model.model.layers.op_list)
        gl=[l.linear_attn for l in layers if l._is_linear]
        al=[l.self_attn for l in layers if not l._is_linear]
        if not gl or not al:raise ValueError("initial planned session supports hybrid Qwen GDN/Attention")
        g=gl[0];self._first_attention=al[0]
        if g.head_k_dim!=g.head_v_dim:raise ValueError("initial runtime uses square Qwen GDN state")
        self.gdn_pool=GDNPool(PoolShape(len(gl),f.block_slots,g.num_v_heads,g.head_k_dim,g.conv_dim,g.conv_kernel),
                              device=self.device,activation_dtype=self.dtype)
        shape=(len(al),a.page_pool_capacity,a.page_size,al[0].num_kv_heads,al[0].head_dim)
        self.k_pool=torch.empty(shape,device=self.device,dtype=self.dtype)
        self.v_pool=torch.empty_like(self.k_pool)
        self.k_pool[:,a.dummy_page].zero_();self.v_pool[:,a.dummy_page].zero_()
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.use_graph=use_graph
        self.allow_eager_overflow=allow_eager_overflow
        self.runners={}
        self._overflow_profile=self._overflow_runner=None
        self.overflow_forwards=0
        self.forward_count=self.prefill_input_tokens=self.decode_tokens=0
        self.prefill_output_rows=0
        self.last_forward=None

    def create_block(self):return self.store.create()
    def free_block(self,block):self.store.clear(block)

    def memory_report(self):
        from .accounting import memory_report
        return memory_report(self)

    def _requests(self,jobs,group):
        prefill=[PrefillRequest(tuple(self.store.block_id(b) for b in (j.context or [])),
                               self.store.block_id(j.block),int(j.input_ids.numel())) for j in jobs]
        decode=[] if group is None else [DecodeRequest(tuple(self.store.block_id(b) for b in w.cache_view),
                           self.store.block_id(w.output_block)) for w in group]
        return prefill,decode

    def can_mix(self,jobs,group):
        if not jobs or not len(group):return False
        pf,dec=self._requests(jobs,group)
        if {r.write_to for r in pf}&{r.write_to for r in dec}:return False
        try:self.catalogue.select(self.store._records,pf,dec)
        except ValueError:return False
        return True

    def _prepare_jobs(self,jobs):
        """Compute image features/relative coordinates once, outside decoder."""
        from minisgl.models.qwen3_5_mrope import get_rope_index
        ids=[];positions={};image_rows=[];features=[]
        for i,j in enumerate(jobs):
            part=j.input_ids.flatten().to(device="cpu",dtype=torch.int64).tolist()
            rel=j.mrope_rel
            feat=j.image_embeds
            types=None if j.mm_token_type_ids is None else j.mm_token_type_ids.flatten().cpu()
            if types is not None and len(types)!=len(part):raise ValueError("multimodal token-type length mismatch")
            if j.pixel_values is not None:
                if feat is not None:raise ValueError("provide pixels or precomputed features, not both")
                if types is None or j.image_grid_thw is None or not hasattr(self.model.model,"visual"):
                    raise ValueError("image prefill requires vision model, grid and token types")
                cfg=self.model.model.config.vision_config
                rel=get_rope_index(torch.tensor(part),types,cfg.spatial_merge_size,j.image_grid_thw.cpu())
                feat=self.model.model.visual.forward(j.pixel_values.to(self.device),j.image_grid_thw.to(self.device))
            elif j.image_grid_thw is not None and feat is None:
                raise ValueError("image grid without pixels/features")
            rows=[] if types is None else (types==1).nonzero().flatten().tolist()
            if rows:
                if feat is None or feat.shape!=(len(rows),self.model.model.embed_tokens.weight.shape[1]):
                    raise ValueError("image rows/features mismatch")
                if rel is None:raise ValueError("precomputed image features require mRoPE coordinates")
                features.append(feat.to(device=self.device,dtype=self.dtype))
                image_rows.extend(len(ids)+r for r in rows)
            elif feat is not None and feat.numel():raise ValueError("image features without image token rows")
            if rel is not None:
                positions[i]=tuple(tuple(axis) for axis in rel.to(device="cpu",dtype=torch.int64).tolist())
                if j.mrope_span is not None:
                    expected=j.block.mrope_span+max(0,int(rel.max())+1)
                    if j.mrope_span!=expected:raise ValueError("chunk's cumulative mRoPE span disagrees with positions")
            ids.extend(part)
        feature_tensor=torch.cat(features) if len(features)>1 else features[0] if features else None
        return ids,positions,image_rows,feature_tensor

    @torch.inference_mode()
    @_session_stream
    def _execute(self,jobs,group=None,input_ids=None):
        start=time.perf_counter()
        jobs=list(jobs)
        pf,dec=self._requests(jobs,group)
        if not pf and not dec:raise ValueError("empty session forward")
        overflow=False
        try:profile=self.catalogue.select(self.store._records,pf,dec)
        except ValueError:
            if not self.allow_eager_overflow:raise
            profile=self.catalogue.eager_overflow_profile(self.store._records,pf,dec)
            overflow=True
        ids,positions,image_rows,features=self._prepare_jobs(jobs)
        pf_tokens=len(ids)
        dec_ids=[] if input_ids is None else input_ids.flatten().to(device="cpu",dtype=torch.int64).tolist()
        if len(dec_ids)!=len(dec):raise ValueError("one input token required for each decode worker")
        ids.extend(dec_ids)
        tokens={};offset=0
        for r in pf:tokens[r.write_to]=tuple(ids[offset:offset+r.length]);offset+=r.length
        for r,t in zip(dec,dec_ids):tokens[r.write_to]=(t,)
        tx=self.store.begin_forward(capacity=profile.forward,attention_capacity=profile.attention,
            prefill=pf,decode=dec,tokens=tokens,prefill_mrope=positions)
        submitted=False
        try:
            inp=prepare_inputs(tx.forward,ids,vocab_size=self.model.model.embed_tokens.num_embeddings,image_rows=image_rows)
            if overflow:
                if self._overflow_profile!=profile:
                    # Prior _execute waits for all readers/retention/commit.
                    # Do not retain one overflow arena for every seen shape.
                    self._overflow_runner=None
                    self._overflow_profile=profile
                runner=self._overflow_runner
            else:runner=self.runners.get(profile)
            if runner is None:
                program=DecoderProgram(self.model,self.gdn_pool,self.k_pool,self.v_pool,tx.forward,tx.attention,inp,features=features)
                runner=ProgramRunner(program,use_graph=self.use_graph and not overflow)
                if overflow:self._overflow_runner=runner
                else:self.runners[profile]=runner
            runner.prepare_execution(tx.forward,tx.attention,inp,features)
            ready=time.perf_counter()
            self.store.mark_submitted(tx);submitted=True
            result=runner.execute_prepared()
            result.completion.synchronize()
            completed=time.perf_counter()
            assert self.store.finish(tx,result.completion)
        except BaseException:
            if not submitted:self.store.cancel(tx)
            else:
                # All writes may have partially executed. Drain before retiring
                # reservations; never return failed blocks as valid old state.
                self.store.mark_failed(tx)
                event=torch.cuda.Event();event.record(torch.cuda.current_stream(self.device))
                try:event.synchronize()
                finally:self.store.finish(tx,event)
            raise
        self.forward_count+=1;self.prefill_input_tokens+=pf_tokens;self.decode_tokens+=len(dec)
        self.prefill_output_rows+=len(pf)
        self.overflow_forwards+=overflow
        self.last_forward=dict(mode=tx.forward.mode,prefill_requests=len(pf),prefill_tokens=pf_tokens,decode_workers=len(dec),
            capacity_rows=profile.forward.prefill_tokens+profile.forward.decode_workers,used_graph=result.used_graph,
            eager_overflow=overflow,
            prepare_seconds=ready-start,execution_and_retain_seconds=completed-ready,wall_seconds=time.perf_counter()-start)
        return result.logits

    @torch.inference_mode()
    def prefill_block(self,block,input_ids,context=None,pixel_values=None,image_grid_thw=None,mm_token_type_ids=None):
        from minisgl.shared_cache.session import PrefillJob
        return self._execute([PrefillJob(block,input_ids,list(context or []),pixel_values,image_grid_thw,mm_token_type_ids)])

    @torch.inference_mode()
    def prefill_batch(self,jobs):
        jobs=list(jobs);logits=self._execute(jobs)
        return [logits[i:i+1] for i in range(len(jobs))]

    @torch.inference_mode()
    def decode_step(self,group,input_ids):return self._execute([],group,input_ids)

    @torch.inference_mode()
    def mixed_step(self,jobs,group,input_ids):
        jobs=list(jobs);logits=self._execute(jobs,group,input_ids)
        return [logits[i:i+1] for i in range(len(jobs))],logits[len(jobs):]

    def _token_slots(self,record):
        return [record.pages[t//self.page_size]*self.page_size+t%self.page_size for t in range(len(record.token_ids))]

    @torch.inference_mode()
    @_session_stream
    def _concatenate(self,left,right,destination):
        l,r,d=(self.store.block_id(b) for b in (left,right,destination))
        if left.num_tokens+right.num_tokens==0:self.store.clear(destination);return destination
        tx=self.store.begin_concatenation(left,right,destination)
        submitted=False
        try:
            states={b.block_id:b for b in tx.slots.plan.blocks}
            src=[b for b in (l,r) if states[b].populated]
            a,b=self.gdn_pool.affine[:,states[src[0]].slot].unbind(1)
            if len(src)==2:
                ar,br=self.gdn_pool.affine[:,states[src[1]].slot].unbind(1)
                affine=torch.stack((a@ar,b@ar+br),1)
            else:affine=self.gdn_pool.affine[:,states[src[0]].slot].clone()
            conv=self.gdn_pool.conv[:,states[src[-1]].slot].clone()
            source_slots=torch.tensor(self._token_slots(tx.before[l])+self._token_slots(tx.before[r]),device=self.device)
            destination_slots=torch.tensor(self._token_slots(tx.after[d]),device=self.device)
            flat_k=self.k_pool.flatten(1,2);flat_v=self.v_pool.flatten(1,2)
            # Copy before publish: required for destination==right/self-append,
            # where source/destination token ranges may overlap. Outside graph.
            keys=flat_k.index_select(1,source_slots);values=flat_v.index_select(1,source_slots)
            nleft=len(tx.before[l].token_ids);nright=len(tx.before[r].token_ids)
            if nright and tx.before[l].mrope_span:
                tail=keys[:,nleft:].reshape(-1,*keys.shape[-2:]).float()
                pos=torch.full((tail.shape[0],),tx.before[l].mrope_span,device=self.device,dtype=torch.int64)
                rotated=self._first_attention._apply_rope(tail,pos).to(self.dtype)
                keys[:,nleft:]=rotated.reshape_as(keys[:,nleft:])
            self.store.mark_submitted(tx);submitted=True
            target=states[d].slot
            self.gdn_pool.affine[:,target].copy_(affine)
            self.gdn_pool.conv[:,target].copy_(conv)
            flat_k.index_copy_(1,destination_slots,keys);flat_v.index_copy_(1,destination_slots,values)
            done=torch.cuda.Event();done.record();done.synchronize()
            assert self.store.finish(tx,done)
        except BaseException:
            if not submitted:self.store.cancel(tx)
            else:
                self.store.mark_failed(tx)
                done=torch.cuda.Event();done.record()
                try:done.synchronize()
                finally:self.store.finish(tx,done)
            raise
        return destination

    def append_block(self,left,right):return self._concatenate(left,right,left)

    def merge_blocks(self,left,right):
        destination=self.create_block()
        return self._concatenate(left,right,destination)
