"""GPU check: game clock pauses during real inference and forwards are counted."""

import _self_evolving_bootstrap  # noqa: F401
import asyncio
import json
import os

import torch
import transformers

from minisgl.llm import AsyncLLM
from action_efficiency import ForwardCounter
from run_budget_sweep import BudgetEngine, MODEL_REVISION
from tasks.health_gathering_env import HealthGatheringEnv
from tasks.runner import run_episodes
from warmup import warmup


async def main():
    if os.environ.get('SEA_GAME_MODE') != 'synchronous':
        raise ValueError('Set SEA_GAME_MODE=synchronous')
    model = f'/mnt/LLM/hub/models--Qwen--Qwen3.8-27B/snapshots/{MODEL_REVISION}'
    llm = AsyncLLM(model, dtype=torch.bfloat16, max_running_req=1, memory_ratio=.9,
                   max_seq_len_override=32768,
                   generation_config=transformers.GenerationConfig(
                       do_sample=True, temperature=.7, top_k=20, top_p=.9),
                   distributed_addr='tcp://127.0.0.1:2485')
    try:
        await warmup(llm)
        counter = ForwardCounter(llm)
        for mode, budget in [('reasoning', 0), ('no_think', 16), ('reasoning', 64)]:
            env = HealthGatheringEnv(episode_timeout=8, seed=123)
            env.max_episodes = 1
            env.max_steps_per_episode = 1  # Must not override the native horizon.
            class PausedClockEngine(BudgetEngine):
                async def act(self, observation, on_token=None):
                    before = env.env.game.get_episode_time()
                    action = await super().act(observation, on_token=on_token)
                    assert env.env.game.get_episode_time() == before
                    return action
            engine = PausedClockEngine(llm, 'health_gathering', budget, mode)
            before = counter.calls
            result = await run_episodes(env, engine)
            ep = result['episodes'][0]
            assert 'error' not in ep['info'], ep
            assert ep['steps'] == 2 and ep['info']['game_tics'] == 8 and ep['info']['timeout'], ep
            assert not any(t['cancelled'] for t in engine.trace)
            forwards = counter.calls - before
            assert forwards == sum(t['llm_forward_calls'] for t in engine.trace) > 0
            if budget == 0:
                assert forwards == 4 and all(t['generated_tokens'] == 0 for t in engine.trace)
            print(json.dumps(dict(mode=mode, budget=budget, episode=ep,
                                  forward_calls=forwards, forwards_per_env_step=forwards / ep['steps'])), flush=True)
        print('PASS: synchronous GPU inference pauses game time, native horizon and forward accounting verified', flush=True)
    finally:
        await llm.close()


if __name__ == '__main__':
    asyncio.run(main())
