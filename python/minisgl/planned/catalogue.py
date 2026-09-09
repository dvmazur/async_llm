"""An explicit finite catalogue; topology/lengths never create graph keys."""
from dataclasses import dataclass

from .forward_plan import PlanCapacity
from .attention_plan import AttentionCapacity


@dataclass(frozen=True,slots=True)
class RuntimeProfile:
    forward:PlanCapacity
    attention:AttentionCapacity


class ProfileCatalogue:
    def __init__(self,profiles):
        self.profiles=tuple(profiles)
        if not self.profiles or len(set(self.profiles))!=len(self.profiles):
            raise ValueError("provide a nonempty explicit catalogue without duplicates")
        base=self.profiles[0]
        for p in self.profiles:
            if (p.forward.block_slots,p.attention.page_size,p.attention.page_pool_capacity,p.attention.dummy_page)!= (
                    base.forward.block_slots,base.attention.page_size,base.attention.page_pool_capacity,base.attention.dummy_page):
                raise ValueError("all profiles must share physical pools")
            if p.forward.prefill_tokens+p.forward.decode_workers<1:
                raise ValueError("empty capacity program is not a usable profile")

    def select(self,records,prefill,decode):
        """Capacity check without touching slots or performing any GPU work.

        Count Attention *references*, including bounded dummy requests, using
        predicted phase lengths; unique pages are not a sufficient bound.
        """
        pf,dec=tuple(prefill),tuple(decode)
        referenced=set(r.write_to for r in (*pf,*dec))
        referenced.update(b for r in pf for b in r.context)
        referenced.update(b for r in dec for b in r.read_blocks)
        before={b:len(records[b].token_ids) for b in referenced}
        postpf=dict(before)
        for r in pf:postpf[r.write_to]+=r.length
        postall=dict(postpf)
        for r in dec:postall[r.write_to]+=1
        depth=max([0]+[sum(bool(before[b]) for b in (*r.context,r.write_to)) for r in pf]
                       +[sum(bool(postpf[b]) for b in r.read_blocks) for r in dec])
        segments=max([0]+[sum(bool(before[b]) for b in r.context) for r in pf]
                         +[sum(bool(postall[b]) for b in r.read_blocks) for r in dec])
        fitting=[]
        for p in self.profiles:
            f,a=p.forward,p.attention
            if (len(pf)>f.prefill_requests or sum(r.length for r in pf)>f.prefill_tokens or len(dec)>f.decode_workers
                    or depth>f.chain_depth or segments>a.context_segments):continue
            pages=lambda n:(n+a.page_size-1)//a.page_size
            pc=[pages(before[b]) for r in pf for b in r.context if before[b]]
            ps=[pages(postpf[r.write_to]) for r in pf]
            dm=[pages(postall[b]) for r in dec for b in r.read_blocks if postall[b]]
            needed=max(sum(pc)+f.prefill_requests*a.context_segments-len(pc),
                       sum(ps)+f.prefill_requests-len(ps),
                       sum(dm)+f.decode_workers*a.context_segments-len(dm),f.decode_workers)
            if needed>a.page_references:continue
            # A bounded choice among user-provided profiles; never synthesize
            # a new profile from actual shapes, block IDs, or expert routes.
            fitting.append(p)
        if not fitting:raise ValueError("forward exceeds configured graph catalogue; no state writes submitted")
        return min(fitting,key=lambda p:(p.forward.prefill_tokens+p.forward.decode_workers,
                    p.forward.prefill_tokens*p.attention.context_segments,p.attention.page_references))

    def eager_overflow_profile(self,records,prefill,decode):
        """Exact-capacity eager-only escape, never inserted into the catalogue.

        Physical pools do not grow. A caller holds at most one such program,
        waits for its completion and discards it when the overflow shape changes.
        """
        pf,dec=tuple(prefill),tuple(decode)
        if not pf and not dec:raise ValueError("empty overflow forward")
        referenced=set(r.write_to for r in (*pf,*dec))
        referenced.update(b for r in pf for b in r.context)
        referenced.update(b for r in dec for b in r.read_blocks)
        before={b:len(records[b].token_ids) for b in referenced}
        postpf=dict(before)
        for r in pf:postpf[r.write_to]+=r.length
        postall=dict(postpf)
        for r in dec:postall[r.write_to]+=1
        depth=max([1]+[sum(bool(before[b]) for b in (*r.context,r.write_to)) for r in pf]
                         +[sum(bool(postpf[b]) for b in r.read_blocks) for r in dec])
        segments=max([0]+[sum(bool(before[b]) for b in r.context) for r in pf]
                         +[sum(bool(postall[b]) for b in r.read_blocks) for r in dec])
        base=self.profiles[0]
        size=base.attention.page_size
        pages=lambda n:(n+size-1)//size
        pc=[pages(before[b]) for r in pf for b in r.context if before[b]]
        dm=[pages(postall[b]) for r in dec for b in r.read_blocks if postall[b]]
        references=max(1,sum(pc)+len(pf)*segments-len(pc),sum(pages(postpf[r.write_to]) for r in pf),
                       sum(dm)+len(dec)*segments-len(dm),len(dec))
        from dataclasses import replace
        return RuntimeProfile(PlanCapacity(len(pf),sum(r.length for r in pf),len(dec),depth,base.forward.block_slots),
                              replace(base.attention,context_segments=segments,page_references=references))
