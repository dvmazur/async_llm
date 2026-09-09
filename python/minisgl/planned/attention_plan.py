"""Pure CPU lowering of the common forward plan into four shared-KV views.

No tensors or per-layer work. Page lists include already reserved append pages;
num_tokens/mrope_span are committed old values. Planning predicts phase views
without publishing lengths, and never derives Attention visibility from GDN's
filtered affine trie (decode Attention has different peer-write semantics).
"""
from dataclasses import dataclass,replace


@dataclass(frozen=True,slots=True)
class KVBlock:
    block_id:int
    num_tokens:int
    mrope_span:int
    pages:tuple[int,...]

    def __post_init__(self):
        object.__setattr__(self,"pages",tuple(self.pages))
        if any(type(n) is not int or n<0 for n in (self.block_id,self.num_tokens,self.mrope_span,*self.pages)):
            raise ValueError("KV snapshot fields must be nonnegative integers")
        if bool(self.num_tokens)!=bool(self.mrope_span):
            raise ValueError("empty KV must have empty mRoPE span")


@dataclass(frozen=True,slots=True)
class AttentionCapacity:
    context_segments:int
    page_references:int  # per wrapper, counts references, not unique pages
    page_size:int
    page_pool_capacity:int
    dummy_page:int=0

    def __post_init__(self):
        if (any(type(getattr(self,f)) is not int or getattr(self,f)<0 for f in self.__dataclass_fields__)
                or min(self.page_references,self.page_size,self.page_pool_capacity)<1
                or self.dummy_page>=self.page_pool_capacity):
            raise ValueError("invalid Attention capacities")


@dataclass(frozen=True,slots=True)
class SubAttentionPlan:
    query_lengths:tuple[int,...]
    pages:tuple[tuple[int,...],...]
    last_page_lengths:tuple[int,...]
    source_rows:tuple[int,...]
    positions:tuple[tuple[int,...],tuple[int,...],tuple[int,...]]
    destinations:tuple[int,...]  # model physical row * merge_segments + segment

    @property
    def positions_flat(self):return tuple(p for axis in self.positions for p in axis)


@dataclass(frozen=True,slots=True)
class AttentionPlan:
    capacity:AttentionCapacity
    prefill_context:SubAttentionPlan
    prefill_self:SubAttentionPlan
    decode_main:SubAttentionPlan
    decode_aux:SubAttentionPlan
    write_token_slots:tuple[int,...]
    key_positions:tuple[tuple[int,...],tuple[int,...],tuple[int,...]]
    post_lengths:tuple[tuple[int,int,int],...]  # write block ID, tokens, span; commit later
    merge_sources:tuple[int,...]  # inverse destinations into four concatenated outputs

    @property
    def key_positions_flat(self):return tuple(p for axis in self.key_positions for p in axis)


class _SubBuilder:
    def __init__(self,requests,rows,cap):
        self.r,self.t,self.cap=requests,rows,cap
        self.lengths=[];self.pages=[];self.last=[]
        self.sources=[];self.positions=[[],[],[]];self.destinations=[]

    def add(self,source_rows,positions,block,segment,*,single_slot=None):
        c=self.cap
        if len(self.lengths)>=self.r:raise ValueError("Attention request/segment capacity exceeded")
        if len(self.sources)+len(source_rows)>self.t:raise ValueError("Attention query-row capacity exceeded")
        if single_slot is not None:
            pages=(single_slot,);last=1
        else:
            n=(block.num_tokens+c.page_size-1)//c.page_size
            if not n:raise ValueError("real Attention query cannot read empty KV")
            pages=block.pages[:n];last=block.num_tokens-(n-1)*c.page_size
        self.lengths.append(len(source_rows));self.pages.append(pages);self.last.append(last)
        self.sources.extend(source_rows)
        for axis in range(3):self.positions[axis].extend(positions[axis])
        self.destinations.extend(row*(c.context_segments+1)+segment for row in source_rows)

    def finish(self):
        pad=self.r-len(self.lengths)
        pages=self.pages+[(self.cap.dummy_page,)]*pad
        if sum(map(len,pages))>self.cap.page_references:raise ValueError("Attention page-reference capacity exceeded")
        tail=self.t-len(self.sources)
        return SubAttentionPlan(tuple(self.lengths+[0]*pad),tuple(pages),tuple(self.last+[1]*pad),
            tuple(self.sources+[-1]*tail),tuple(tuple(p+[0]*tail) for p in self.positions),
            tuple(self.destinations+[-1]*tail))


def prepare_attention(kv_blocks,forward,capacity,*,prefill_mrope=None):
    """One preparation for all attention layers; image positions are already CPU.

    prefill_mrope maps request index to three relative coordinate tuples; text
    defaults to (0..L-1) on each axis. Vision/tokenizer remain outside decoder.
    """
    c,f=capacity,forward.capacity
    snapshot={b.block_id:kv_blocks[b.block_id] for b in forward.blocks}
    for gdn in forward.blocks:
        kv=snapshot[gdn.block_id]
        if kv.block_id!=gdn.block_id or gdn.populated!=bool(kv.num_tokens):
            raise ValueError("GDN/KV committed snapshots disagree")
        if len(kv.pages)<(kv.num_tokens+c.page_size-1)//c.page_size:
            raise ValueError("committed KV has insufficient pages")
        if any(p>=c.page_pool_capacity for p in kv.pages):raise ValueError("KV page outside pool")
    pc=_SubBuilder(f.prefill_requests*c.context_segments,f.prefill_tokens*c.context_segments,c)
    ps=_SubBuilder(f.prefill_requests,f.prefill_tokens,c)
    dm=_SubBuilder(f.decode_workers*c.context_segments,f.decode_workers*c.context_segments,c)
    # Aux page IDs index flattened token slots, not real-size pages.
    aux_cap=replace(c,page_size=1,page_pool_capacity=c.page_pool_capacity*c.page_size,
                    dummy_page=c.dummy_page*c.page_size)
    da=_SubBuilder(f.decode_workers,f.decode_workers,aux_cap)
    total=f.prefill_tokens+f.decode_workers
    slots=[-1]*total;keypos=[[0]*total for _ in range(3)]
    image_positions=prefill_mrope or {}
    if any(type(i) is not int or not 0<=i<len(forward.prefill_requests) for i in image_positions):
        raise ValueError("unknown prefill image-position request")

    def grow(old,n,advance):
        new=replace(old,num_tokens=old.num_tokens+n,mrope_span=old.mrope_span+advance)
        if len(new.pages)<(new.num_tokens+c.page_size-1)//c.page_size:
            raise ValueError("append KV pages must be reserved before planning")
        return new

    def write(row,old,relative):
        token=old.num_tokens+relative
        slots[row]=old.pages[token//c.page_size]*c.page_size+token%c.page_size

    post_pf=dict(snapshot)
    offset=0
    for i,r in enumerate(forward.prefill_requests):
        old=snapshot[r.write_to]
        rel=image_positions.get(i,(tuple(range(r.length)),)*3)
        if (len(rel)!=3 or any(len(axis)!=r.length for axis in rel)
                or any(type(p) is not int or old.mrope_span+p<0 for axis in rel for p in axis)):
            raise ValueError("invalid prefill mRoPE coordinates")
        # A chunk inside an image may revisit earlier spatial coordinates.
        # Its relative positions can be negative after subtracting the already
        # committed span; the cumulative block span must not decrease.
        advance=max(0,max(max(axis) for axis in rel)+1)
        new=grow(old,r.length,advance)
        post_pf[r.write_to]=new
        rows=list(range(offset,offset+r.length))
        context=[snapshot[b] for b in r.context if snapshot[b].num_tokens]
        if len(context)>c.context_segments:raise ValueError("Attention context segment capacity exceeded")
        self_offset=sum(b.mrope_span for b in context)+old.mrope_span
        prefix=0
        for j,b in enumerate(context):
            pos=tuple(tuple(p+self_offset-prefix for p in axis) for axis in rel)
            pc.add(rows,pos,b,j)
            prefix+=b.mrope_span
        pos=tuple(tuple(p+old.mrope_span for p in axis) for axis in rel)
        ps.add(rows,pos,new,len(context))
        for t,row in enumerate(rows):
            write(row,old,t)
            for axis in range(3):keypos[axis][row]=pos[axis][t]
        offset+=r.length

    post_all=dict(post_pf)
    for r in forward.decode_requests:post_all[r.write_to]=grow(post_pf[r.write_to],1,1)
    for i,r in enumerate(forward.decode_requests):
        row=f.prefill_tokens+i
        old=post_pf[r.write_to]
        write(row,old,0)
        for axis in range(3):keypos[axis][row]=old.mrope_span
        context=[post_all[b] for b in r.read_blocks if post_all[b].num_tokens]
        if len(context)>c.context_segments:raise ValueError("Attention context segment capacity exceeded")
        total_span=sum(b.mrope_span for b in context)
        prefix=0
        for j,b in enumerate(context):
            pos=(total_span-prefix-1,)
            dm.add((row,),(pos,)*3,b,j)
            prefix+=b.mrope_span
        if r.write_to not in r.read_blocks:
            pos=(old.mrope_span,)
            da.add((row,),(pos,)*3,old,len(context),single_slot=slots[row])
    lengths=tuple((w.block_id,post_all[w.block_id].num_tokens,post_all[w.block_id].mrope_span)
                  for w in forward.writes)
    live_writes=[slot for slot in slots if slot>=0]
    if len(set(live_writes))!=len(live_writes):raise ValueError("new KV writes alias each other")
    used={}
    for b in snapshot.values():
        remaining=b.num_tokens
        for page in b.pages:
            if remaining<=0:break
            used[page]=max(used.get(page,0),min(c.page_size,remaining))
            remaining-=c.page_size
    if any(slot%c.page_size<used.get(slot//c.page_size,0) for slot in live_writes):
        raise ValueError("new KV write overlaps committed tokens; reserve copy-on-write pages first")
    subs=(pc.finish(),ps.finish(),dm.finish(),da.finish())
    merge=[-1]*(total*(c.context_segments+1))
    base=0
    for sub in subs:
        for row,dst in enumerate(sub.destinations):
            if dst>=0:
                if merge[dst]>=0:raise ValueError("duplicate Attention merge destination")
                merge[dst]=base+row
        base+=len(sub.source_rows)
    return AttentionPlan(c,*subs,tuple(slots),tuple(tuple(p) for p in keypos),lengths,tuple(merge))
