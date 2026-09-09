"""Owned block handles, KV page reservations and atomic all-layer metadata.

Pure CPU. The session submits GPU writes and a completion event. Public handles
never expose a mutable tensor/dictionary view; snapshots belong to the session.
KV pages are privately owned by one block (merge/append copy, as before).
"""
from collections import deque
from dataclasses import dataclass,replace

from .attention_plan import KVBlock,prepare_attention
from .slots import SlotRegistry


@dataclass(frozen=True,slots=True)
class BlockRecord:
    block_id:int
    token_ids:tuple[int,...]=()
    mrope_span:int=0
    pages:tuple[int,...]=()

    def kv(self):return KVBlock(self.block_id,len(self.token_ids),self.mrope_span,self.pages)


class BlockHandle:
    __slots__=("_owner","_id","__weakref__")

    def __init__(self,owner,block_id):self._owner,self._id=owner,block_id
    @property
    def num_tokens(self):return len(self._owner.record(self).token_ids)
    @property
    def token_ids(self):return list(self._owner.record(self).token_ids)
    @property
    def mrope_span(self):return self._owner.record(self).mrope_span
    @property
    def page_size(self):return self._owner.page_size
    @property
    def num_pages(self):return len(self._owner.record(self).pages)
    @property
    def revision(self):return self._owner.registry[self._id].revision
    def clear(self):self._owner.clear(self)
    def __repr__(self):return f"BlockHandle(id={self._id})"
    def __del__(self):
        # Destructors must not free memory after an unknown device failure.
        # Explicit API operations surface errors; teardown safely retains it.
        try:self._owner.release(self._id)
        except Exception:pass


@dataclass(slots=True)
class StorageTransaction:
    slots:object
    before:dict[int,BlockRecord]
    after:dict[int,BlockRecord]
    reserved_pages:tuple[int,...]
    attention:object=None

    @property
    def forward(self):return self.slots.plan


class BlockStore:
    def __init__(self,*,block_slots,page_size,page_capacity,dummy_page=0):
        if (any(type(x) is not int for x in (page_size,page_capacity,dummy_page))
                or page_size<1 or page_capacity<2 or not 0<=dummy_page<page_capacity):
            raise ValueError("invalid KV pool dimensions")
        self.registry=SlotRegistry(block_slots)
        self.page_size,self.page_capacity,self.dummy_page=page_size,page_capacity,dummy_page
        self._free_pages=deque(p for p in range(page_capacity) if p!=dummy_page)
        self._records={}
        self._pending=None
        self._dropped=set()

    @property
    def free_pages(self):return len(self._free_pages)

    def create(self):
        b=self.registry.create()
        self._records[b]=BlockRecord(b)
        return BlockHandle(self,b)

    def block_id(self,handle):
        if not isinstance(handle,BlockHandle) or handle._owner is not self:
            raise ValueError("block handle belongs to another runtime")
        self.registry[handle._id]
        return handle._id

    def record(self,handle):return self._records[self.block_id(handle)]

    def clear(self,handle):
        # Validate identity but allow clearing a block invalidated by a failed
        # forward. Registry checks health/pins before any page is released.
        if not isinstance(handle,BlockHandle) or handle._owner is not self:
            raise ValueError("block handle belongs to another runtime")
        b=handle._id
        self.registry.clear(b)
        self._free_pages.extend(self._records[b].pages)
        self._records[b]=BlockRecord(b)

    def release(self,b):
        if b not in self._records:return
        self.registry.release_handle(b)
        if self._pending is not None and b in self._pending.before:
            self._dropped.add(b)
        else:self._free_pages.extend(self._records.pop(b).pages)

    def _reserve_pages(self,counts):
        total=sum(counts.values())
        if total>self.free_pages:
            raise RuntimeError(f"KV pool exhausted: requested={total}, free={self.free_pages}, capacity={self.page_capacity-1}")
        return {b:tuple(self._free_pages.popleft() for _ in range(n)) for b,n in counts.items()}

    def _return_new_pages(self,pages):
        self._free_pages.extendleft(reversed(pages))

    def begin_forward(self,*,capacity,attention_capacity,prefill=(),decode=(),tokens,prefill_mrope=None):
        if self._pending is not None:raise RuntimeError("previous storage transaction is still in flight")
        c=attention_capacity
        if (c.page_size,c.page_pool_capacity,c.dummy_page)!=(self.page_size,self.page_capacity,self.dummy_page):
            raise ValueError("Attention profile disagrees with physical KV pool")
        tx=self.registry.begin(capacity=capacity,prefill=prefill,decode=decode)
        reserved=()
        try:
            before={b.block_id:self._records[b.block_id] for b in tx.plan.blocks}
            if set(tokens)!={w.block_id for w in tx.plan.writes}:raise ValueError("token writes mismatch")
            appended={b:tuple(ids) for b,ids in tokens.items()}
            if any(len(appended[w.block_id])!=w.added_tokens for w in tx.plan.writes):
                raise ValueError("wrong number of committed input tokens")
            if any(type(t) is not int or t<0 for ids in appended.values() for t in ids):
                raise ValueError("invalid token IDs")
            counts={w.block_id:max(0,(len(before[w.block_id].token_ids)+w.added_tokens+self.page_size-1)//self.page_size
                                  -len(before[w.block_id].pages)) for w in tx.plan.writes}
            pages=self._reserve_pages(counts)
            reserved=tuple(p for ps in pages.values() for p in ps)
            kv={b:replace(r.kv(),pages=r.pages+pages.get(b,())) for b,r in before.items()}
            att=prepare_attention(kv,tx.plan,c,prefill_mrope=prefill_mrope)
            after={}
            for b,n,span in att.post_lengths:
                ids=before[b].token_ids+appended[b]
                if len(ids)!=n:raise ValueError("planner/token commit lengths disagree")
                after[b]=BlockRecord(b,ids,span,kv[b].pages)
            self._pending=StorageTransaction(tx,before,after,reserved,att)
            return self._pending
        except Exception:
            self._return_new_pages(reserved)
            self.registry.cancel_before_launch(tx)
            raise

    def begin_concatenation(self,left,right,destination):
        """Snapshot operands before any write, including self-append.

        Destination may be either operand or a fresh empty handle. Filling the
        free tail page cannot overwrite a source token; storage is never shared.
        """
        if self._pending is not None:raise RuntimeError("previous storage transaction is still in flight")
        l,r,d=(self.block_id(x) for x in (left,right,destination))
        before={b:self._records[b] for b in dict.fromkeys((l,r,d))}
        if d not in (l,r) and before[d].token_ids:raise ValueError("merge destination must be empty or an operand")
        ids=before[l].token_ids+before[r].token_ids
        if not ids:raise ValueError("empty concatenation needs no GPU transaction")
        tx=self.registry.begin_mutation(readers=(l,r),destination=d,num_tokens=len(ids))
        reserved=()
        try:
            extra=max(0,(len(ids)+self.page_size-1)//self.page_size-len(before[d].pages))
            reserved=self._reserve_pages({d:extra})[d]
            after={d:BlockRecord(d,ids,before[l].mrope_span+before[r].mrope_span,before[d].pages+reserved)}
            self._pending=StorageTransaction(tx,before,after,reserved)
            return self._pending
        except Exception:
            self._return_new_pages(reserved);self.registry.cancel_before_launch(tx)
            raise

    def _check(self,tx):
        if tx is not self._pending:raise RuntimeError("stale/foreign storage transaction")

    def mark_submitted(self,tx):self._check(tx);self.registry.mark_submitted(tx.slots)
    def mark_failed(self,tx):self._check(tx);self.registry.mark_failed(tx.slots)

    def cancel(self,tx):
        self._check(tx)
        self.registry.cancel_before_launch(tx.slots)
        self._return_new_pages(tx.reserved_pages)
        self._pending=None
        self._retire_dropped()

    def finish(self,tx,completion):
        self._check(tx)
        if not self.registry.finish(tx.slots,completion):return False
        if tx.slots.stage=="committed":self._records.update(tx.after)
        else:self._return_new_pages(tx.reserved_pages)
        self._pending=None
        self._retire_dropped()
        return True

    def _retire_dropped(self):
        for b in self._dropped:self._free_pages.extend(self._records.pop(b).pages)
        self._dropped.clear()
