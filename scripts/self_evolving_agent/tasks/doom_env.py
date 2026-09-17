from __future__ import annotations

import random
import time
from typing import Any, Optional

from .realtime_vizdoom import make_vizdoom, realtime_options, game_mode, environment_doc, game_tic_limits

# Matches VizdoomDefendLine-v1's actual Discrete(4) button_map (index 0 = no
# button pressed, 1 = ATTACK, 2 = TURN_RIGHT, 3 = TURN_LEFT).
ACTION_NAMES = ["wait", "fire", "right", "left"]

# Surfaced verbatim into the round prompt via agent.py's task_env.doc hook.
# Prose, not code, on purpose: nothing here is enforced by the harness, and
# engine.py/prompt.py stay entirely the agent's own design surface.
DOC = (
    "`doom` observations are raw game screenshots (images), not text. Actions: "
    f"{ACTION_NAMES!r} (or their index, 0-3 in that order) -- wait / fire / turn right / turn "
    "left.\n"
    "Feeding an image to the model needs the multimodal chat-template/processor path, not the "
    "plain text tokenizer.\n"
    "One structure that has worked well: several separate, independently freeable cache blocks "
    "(system instructions built once; a running text history that only ever grows; a small "
    "current-frame block freed and rebuilt every step) instead of one re-encoded blob, and "
    "picking the action via a logit probe over just the action-name tokens instead of generating "
    "and parsing free text. Not required -- build it however you like.\n"
    "The game runs asynchronously at the configured tic rate (normally 35 tics/sec), including "
    "while act() awaits LLM inference. The active inference action policy is described below. "
    "Death and the native episode timeout end evaluation "
    "and cancel any pending decision. Latency therefore directly affects attainable reward. "
    "Use real LLM.forward() calls: evaluations with no forward calls are invalid.\n"
    "Reward is the only thing this experiment scores on. A fast policy that scores low is not an "
    "improvement over a slower one that scores high -- act() latency should only be a concern if "
    "it's genuinely blocking something that would raise reward, never a goal in its own right.\n"
    "That said, this experiment is about designing an LLM-driven inference structure, not about "
    "maximizing game score by whatever means happens to work. act() should actually invoke the "
    "model (a real forward pass through `llm(...)`/`llm.forward(...)`, whether directly or via "
    "generate()) for its decisions on close to every step -- not a purely scripted/heuristic "
    "action-selector that never touches the LLM at all, even one that happens to score as well as "
    "or better than a real inference pass would. Next round's prompt reports how many real LLM "
    "forward passes actually ran during the task versus how many env steps it took; a policy that "
    "mostly or entirely avoids calling the LLM is treated as a failed round regardless of its score "
    "and will not be reported as a working result, so don't route around real inference for speed "
    "or reliability -- a real, if imperfect, LLM-driven decision is what's being evaluated here."
)


class DoomEnv:
    """Wraps a real ViZDoom gymnasium env (see doom_basic/env.py's GymEnv for the
    pattern this mirrors) behind the same reset()/step()/restart() shape as
    tasks/math_env.py and tasks/ttft_env.py. Unlike those, an episode here is
    genuinely multi-step -- see max_steps_per_episode, read by tasks/runner.py."""

    real_time = True
    name = "doom"
    doc = DOC
    # Matches run_baseline.py's DEFAULT_EPISODES=5 and health_gathering_env.py's
    # max_episodes -- keeps evolution rounds comparable to baselines.
    max_episodes = 5
    max_steps_per_episode = 100

    def __init__(
        self,
        env_id: str = "VizdoomDefendLine-v1",
        frame_skip: int = 4,
        episode_timeout: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if episode_timeout is None:
            episode_timeout = game_tic_limits()["doom"]
        self.doc = environment_doc(DOC)
        self.real_time = game_mode() == "asynchronous"
        self.native_episode_limit = True
        self.env = make_vizdoom(env_id, frame_skip=frame_skip,
                                   game_tic_limit=episode_timeout, **realtime_options())
        # A fixed seed pins vizdoom's native RNG for every episode (useful to
        # reproduce a specific run); `seed=None` (default) means reset() below
        # draws a fresh seed each episode instead.
        self._fixed_seed = seed
        self.seed = seed if seed is not None else random.SystemRandom().randrange(1, 2**31 - 1)
        self.screen: Optional[Any] = None
        self._last_ts: Optional[float] = None
        self._latencies: list[float] = []

    def _to_action_index(self, action: Any) -> int:
        if isinstance(action, str):
            name = action.strip().lower()
            if name not in ACTION_NAMES:
                raise ValueError(f"unknown doom action {action!r}; expected one of {ACTION_NAMES}")
            return ACTION_NAMES.index(name)
        index = int(action)
        if not 0 <= index < len(ACTION_NAMES):
            raise ValueError(f"doom action index {index} out of range [0, {len(ACTION_NAMES)})")
        return index

    def reset(self) -> Any:
        # Fresh seed every episode (unless one was pinned at construction) --
        # otherwise every episode across every round of a run replays the
        # exact same scripted scenario.
        self.seed = (
            self._fixed_seed if self._fixed_seed is not None
            else random.SystemRandom().randrange(1, 2**31 - 1)
        )
        obs, _info = self.env.reset(seed=self.seed)
        self.screen = obs["screen"]
        self._last_ts = time.monotonic()
        self._latencies = []
        return self.screen

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        # Time since the last observation was handed back is (almost
        # entirely) time spent inside the agent's own act().
        now = time.monotonic()
        latency = (now - self._last_ts) if self._last_ts is not None else None
        if latency is not None:
            self._latencies.append(latency)
        obs, reward, done, info = self.env.step(self._to_action_index(action))
        self.screen = obs["screen"]
        info = dict(info)
        info["act_latency_s"] = latency
        info["avg_act_latency_s"] = (
            sum(self._latencies) / len(self._latencies) if self._latencies else None
        )
        self._last_ts = time.monotonic()
        return self.screen, float(reward), bool(done), info

    def poll(self):
        obs, reward, done, info = self.env.poll()
        self.screen = obs["screen"]
        info["avg_act_latency_s"] = (sum(self._latencies) / len(self._latencies)
                                     if self._latencies else None)
        observation = self.screen
        return observation, reward, done, info

    def close(self):
        self.env.close()

    def restart(self) -> None:
        """No cyclic prompt cursor to rewind (unlike MathEnv/TTFTEnv) -- every
        reset() already starts a brand new game episode, so there is nothing
        else to reset here. Present for interface parity with the other envs."""
