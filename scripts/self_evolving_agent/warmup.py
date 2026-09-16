"""Warm vision/prefill/decode kernels on a saved observation before timing games."""
async def warmup(llm):
    from tasks.doom_env import DoomEnv
    from run_budget_sweep import BudgetEngine
    env = DoomEnv(seed=9182)
    try:
        observation = env.reset()
    finally:
        env.close()
    for mode, budget in [("reasoning", 0), ("no_think", 8)]:
        await BudgetEngine(llm, "doom", budget, mode).act(observation)
    print("WARMUP complete (excluded from evaluation)", flush=True)
