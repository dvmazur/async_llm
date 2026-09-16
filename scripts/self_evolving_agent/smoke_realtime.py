"""GPU integration check: terminate an in-flight real reasoning call, then reuse LLM."""
import asyncio
import os
import torch
import transformers
from minisgl.llm import AsyncLLM
from run_budget_sweep import BudgetEngine, MODEL_REVISION
from tasks.health_gathering_env import HealthGatheringEnv
from tasks.runner import run_episodes
from warmup import warmup

async def main():
    model=f"/mnt/LLM/hub/models--Qwen--Qwen3.8-27B/snapshots/{MODEL_REVISION}"
    llm=AsyncLLM(model, dtype=torch.bfloat16, max_running_req=4, memory_ratio=.9,
                 generation_config=transformers.GenerationConfig(do_sample=True, temperature=.7, top_k=20, top_p=.9),
                 distributed_addr="tcp://127.0.0.1:2485")
    try:
        await warmup(llm)
        engine=BudgetEngine(llm, "health_gathering", 16384)
        env=HealthGatheringEnv(episode_timeout=70, seed=123)
        env.max_episodes=1
        result=await run_episodes(env,engine)
        print("CANCEL RESULT", result, "TRACE", engine.trace, flush=True)
        ep=result['episodes'][0]
        assert 'error' not in ep['info'], ep
        assert ep['info']['timeout'] and ep['steps']==0, ep
        assert engine.trace[-1]['cancelled'], engine.trace
        await warmup(llm)
        print("PASS: cancelled actual GPU inference safely, then reused model", flush=True)
    finally:
        await llm.close()

if __name__ == '__main__': asyncio.run(main())
