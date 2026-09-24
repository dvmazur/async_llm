"""Exercise actual GPU inference and both template modes before a long sweep."""

import _self_evolving_bootstrap  # noqa: F401
import asyncio
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/mnt/LLM")

import torch
import transformers
from minisgl.llm import AsyncLLM
from run_baseline import TASKS
from run_budget_sweep import BudgetEngine, MODEL_REVISION


async def main():
    model = f"/mnt/LLM/hub/models--Qwen--Qwen3.8-27B/snapshots/{MODEL_REVISION}"
    llm = AsyncLLM(model, dtype=torch.bfloat16, max_running_req=1, memory_ratio=.9,
                   max_seq_len_override=32768,
                   generation_config=transformers.GenerationConfig(
                       do_sample=True, temperature=.7, top_k=20, top_p=.9),
                   distributed_addr=f"tcp://127.0.0.1:{2390 + int(os.environ['CUDA_VISIBLE_DEVICES'])}")
    try:
        for task in TASKS:
            env = TASKS[task](seed=9182)
            try:
                obs = env.reset()
                for mode, budget in [("reasoning", 0), ("reasoning", 64), ("no_think", 64)]:
                    engine = BudgetEngine(llm, task, budget, mode)
                    start = time.monotonic()
                    action = await engine.act(obs)
                    print(task, mode, budget, action, time.monotonic()-start, engine.trace, flush=True)
            finally:
                env.env.close()
    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
