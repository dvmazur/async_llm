"""Run the canonical notebook's unchanged Speleo policy on PlannedSession.

Only construction and obsolete memo-cache telemetry are adapted. Prompts,
decision helpers, actions, sampling, batching yield and environment code remain
the existing runner/notebook sources. Uses an already loaded model/session.
"""
import asyncio
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


def run_live(session,model_path,output):
    # Report-only plotting must not attach a GUI client to Craftium's private
    # Xvfb. Otherwise shutting down the environment can abort Python at exit
    # with XIO even after every action and summary completed successfully.
    import matplotlib
    matplotlib.use("Agg",force=True)
    from transformers import GenerationConfig
    import minisgl.llm
    from minisgl.planned.frontend import async_llm
    import speleo_async_parallel as runner
    config=GenerationConfig(do_sample=True,temperature=.8,top_k=20,top_p=.9)
    llm=async_llm(session,str(model_path),generation_config=config,enable_mixed_batch=True,batching_yield_rounds=3)
    args=SimpleNamespace(model=str(model_path),notebook=runner.DEFAULT_NOTEBOOK,results_dir=Path(output),
        seeds=(0,1,2),decisions=2,environment_max_steps=100,sampling_seed=20260903,mixed=True,
        max_running_req=16,num_pages=session.store.page_capacity,torch_profile=False,torch_profile_trace=False,
        legacy_prefill=False,batching_yield_rounds=3,evict_checkpoint_page_cache_after_load=False)
    original=minisgl.llm.AsyncLLM
    try:
        minisgl.llm.AsyncLLM=lambda *a,**kw:llm
        base,helpers,environments=runner.load_runtime(args)
    finally:minisgl.llm.AsyncLLM=original
    # The old reporting code assumes a persistent prefix cache object. This
    # runtime deliberately has none. Adapt only that one telemetry expression.
    source=inspect.getsource(runner.run)
    old="compose_cache = llm.async_engine.session.sc_gdn.compose_state_cache"
    assert source.count(old)==1
    source=source.replace(old,"compose_cache = None")
    namespace=dict(vars(runner))
    exec(compile(source,"<planned Speleo: memo-cache telemetry only>","exec"),namespace)
    path=asyncio.run(namespace["run"](args,base,helpers,environments))
    result=json.loads(path.read_text())
    return dict(summary_path=str(path),total_actions=result["total_actions"],wall_seconds=result["wall_seconds"],
        physical_forwards=result["batched_forwards"],notebook_sha256=hashlib.sha256(args.notebook.read_bytes()).hexdigest(),
        runtime_memory=session.memory_report(),overflow_forwards=session.overflow_forwards,
        note="Live wall may include first capture of additional configured profiles; this is not warmed target-GPU throughput.")
