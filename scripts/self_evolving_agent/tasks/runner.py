from __future__ import annotations

import logging
import traceback
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


async def run_episodes(
    env: Any,
    engine: Any,
    max_steps_per_episode: int = 8,
    on_episode_start: Optional[Callable[[], None]] = None,
    on_step: Optional[Callable[[], None]] = None,
    on_frame: Optional[Callable[[Any], None]] = None,
    on_episode_end: Optional[Callable[[], None]] = None,
) -> dict[str, Any]:
    """Fixed harness, not agent-editable: drives `env` (a doom_basic-style
    GymEnv -- reset()/step() -- see tasks/math_env.py, tasks/ttft_env.py,
    and tasks/doom_env.py's real GymEnv) against the agent's own
    `engine.act(observation, on_token=...)`. One broken episode (e.g. no
    act() defined yet) never kills the run -- it just scores 0 with the
    error recorded, same no-guardrail spirit as the rest of the harness.
    `on_episode_start`, if given, is called once at the top of every episode
    (e.g. to touch a liveness heartbeat during long task-scoring runs);
    `on_step`, if given, is called once per step within an episode too --
    math/ttft's single-shot episodes never needed this, but a real
    multi-step game episode (e.g. doom, dozens of steps) can take long
    enough between episode boundaries that a heartbeat only at the top of
    each episode risks a false "hung process" verdict from a caller's
    watchdog.
    `max_steps_per_episode` is only a default -- an env can define its own
    `max_steps_per_episode` attribute to override it (e.g. a real multi-step
    game episode needs far more than 8 steps to show anything meaningful,
    unlike math/ttft's single-shot prompt-then-done episodes).
    `on_frame`, if given, is called with every raw observation this episode
    sees (the reset() observation, then each step() observation) -- envs
    with image observations (doom, health_gathering, my_way_home) use this
    to let a caller record a replay; text-observation envs (math/ttft) just
    get called with their text, which a caller can ignore. `on_episode_end`,
    if given, fires once per episode right after its loop ends, before the
    next episode's on_episode_start -- the natural point to flush/save
    whatever `on_frame` accumulated."""
    episodes: list[dict[str, Any]] = []
    n_episodes = getattr(env, "max_episodes", 1)
    on_token = getattr(env, "on_token", None)
    steps_cap = getattr(env, "max_steps_per_episode", max_steps_per_episode)

    for _ in range(n_episodes):
        if on_episode_start is not None:
            on_episode_start()
        obs = env.reset()
        if on_frame is not None:
            on_frame(obs)
        total_reward = 0.0
        info: dict[str, Any] = {}
        done = False
        steps = 0
        try:
            while not done and steps < steps_cap:
                if on_step is not None:
                    on_step()
                action = await engine.act(obs, on_token=on_token)
                obs, reward, done, *rest = env.step(action)
                if on_frame is not None:
                    on_frame(obs)
                total_reward += reward
                info = rest[0] if rest else {}
                steps += 1
        except Exception:
            info = {"error": traceback.format_exc()}
            total_reward = 0.0
        if on_episode_end is not None:
            on_episode_end()
        episodes.append({"reward": total_reward, "info": info, "steps": steps})

    avg_reward = sum(e["reward"] for e in episodes) / len(episodes) if episodes else 0.0
    return {"env": getattr(env, "name", "task"), "avg_reward": avg_reward, "episodes": episodes}
