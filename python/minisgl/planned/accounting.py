"""Read-only setup/diagnostic accounting; never called inside decoder replay."""
import torch


def cuda_storages(root):
    """Count allocations once, not full storage size once for every tensor view.

    Walk only explicit runtime/native-adapter owners. Do not traverse JIT
    functions/modules and accidentally charge unrelated global compiler caches.
    """
    seen=set();storages={}
    def visit(x):
        if id(x) in seen:return
        seen.add(id(x))
        if isinstance(x,torch.Tensor):
            if x.is_cuda and x.numel():
                s=x.untyped_storage();storages[(x.device,s.data_ptr())]=s.nbytes()
        elif isinstance(x,dict):
            for v in x.values():visit(v)
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        elif type(x).__module__.startswith(("minisgl.planned.","flashinfer.")) and hasattr(x,"__dict__"):
            for v in vars(x).values():visit(v)
    visit(root)
    return storages


def memory_report(session):
    weights=cuda_storages(session.model.state_dict())
    pools=cuda_storages((session.gdn_pool.affine,session.gdn_pool.conv,session.k_pool,session.v_pool))
    owned=dict(weights);owned.update(pools)
    reports=[]
    programs=list(session.runners.items())
    if session._overflow_runner is not None:programs.append((session._overflow_profile,session._overflow_runner))
    for profile,runner in programs:
        storage=cuda_storages(runner.program)
        exclusive={k:v for k,v in storage.items() if k not in owned}
        owned.update(storage)
        reports.append(dict(prefill_requests=profile.forward.prefill_requests,prefill_rows=profile.forward.prefill_tokens,
            decode_workers=profile.forward.decode_workers,exclusive_explicit_bytes=sum(exclusive.values()),
            captures=0 if runner.body is None else runner.body.captures,replays=0 if runner.body is None else runner.body.replays,
            eager=runner.eager_count))
    s=session.store
    return dict(weight_bytes=sum(weights.values()),pool_bytes=sum(pools.values()),
        explicit_owned_bytes=sum(owned.values()),profiles=reports,
        gdn_reserved_bytes=session.gdn_pool.reserved_bytes,
        gdn_live_slots=s.registry.capacity-s.registry.free_slots,gdn_free_slots=s.registry.free_slots,
        kv_live_pages=s.page_capacity-1-s.free_pages,kv_free_pages=s.free_pages,
        process_cuda_allocated_bytes=torch.cuda.memory_allocated(session.device),
        process_cuda_reserved_bytes=torch.cuda.memory_reserved(session.device),
        note="Explicit storage accounting excludes opaque native/graph-private allocations; process totals include other owners.")
