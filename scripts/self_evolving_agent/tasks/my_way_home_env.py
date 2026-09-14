from __future__ import annotations

import random
import time
from typing import Any, Optional

import gymnasium
import vizdoom.gymnasium_wrapper  # noqa: F401  registers the Vizdoom gymnasium env ids

# Confirmed via VizdoomMyWayHome-v1's actual Discrete(6) button_map:
# index 0 = no button pressed, 1 = TURN_LEFT, 2 = TURN_RIGHT, 3 = MOVE_FORWARD,
# 4 = MOVE_LEFT (strafe), 5 = MOVE_RIGHT (strafe). No ATTACK button, unlike
# tasks/doom_env.py's DefendLine.
ACTION_NAMES = ["wait", "turn_left", "turn_right", "forward", "strafe_left", "strafe_right"]

# Surfaced verbatim into the round prompt via agent.py's generic task_env.doc hook,
# same mechanism as tasks/doom_env.py's/health_gathering_env.py's DOC.
DOC = (
    "`my_way_home` observations are raw game screenshots (images), not text. Actions: "
    f"{ACTION_NAMES!r} (or their index, 0-5 in that order) -- wait / turn left / turn right / "
    "move forward / strafe left / strafe right. Like `health_gathering`, there is no fire/attack "
    "button here.\n"
    "Feeding an image to the model needs the multimodal chat-template/processor path, not the "
    "plain text tokenizer.\n"
    "Scenario: the agent spawns in a random room of a small multi-room maze and must find its way "
    "to a fixed goal room containing a vest, using only what it sees on screen -- there is no map "
    "or compass. Reward is sparse and terminal: +1 for reaching the vest, a small negative living "
    "reward (-0.0001) per tic otherwise, and the episode simply times out with 0 net progress if "
    "the vest is never found. This is a pure navigation/exploration task -- unlike `doom` (aim + "
    "shoot) and `health_gathering` (react to a continuously draining health bar), there is no "
    "moment-to-moment danger here, so what matters is remembering which rooms/paths have already "
    "been explored across steps rather than reacting fast to any single frame.\n"
    "One structure that has worked well for the sibling envs: several separate, independently "
    "freeable cache blocks (system instructions built once; a running text history that only ever "
    "grows; a small current-frame block freed and rebuilt every step) instead of one re-encoded "
    "blob, and picking the action via a logit probe over just the action-name tokens instead of "
    "generating and parsing free text. Not required -- build it however you like. Given how sparse "
    "the reward is here, a running text history of what's already been tried/seen (e.g. \"turned "
    "left twice, hit a dead end, backtracking\") may matter more for this env than for the others.\n"
    "The underlying ViZDoom scenario itself carries no notion of your decision speed -- reward is "
    "purely game score. This harness measures decision speed separately and puts it in the info "
    "dict step() returns: `act_latency_s` (wall-clock seconds between you receiving an observation "
    "and this step's action arriving -- i.e. how long your own act() took) and `avg_act_latency_s` "
    "(the running mean over the episode so far). This is visible to you next round as an average "
    "over the task run. It isn't part of the reward and nothing enforces a target -- purely "
    "diagnostic."
)


class MyWayHomeEnv:
    """Wraps VizdoomMyWayHome-v1 behind the same reset()/step()/restart() shape as
    tasks/doom_env.py's DoomEnv and tasks/health_gathering_env.py's HealthGatheringEnv
    -- same wrapper pattern, a third scenario (pure navigation/exploration to a fixed
    goal with sparse terminal reward, instead of aim+shoot or navigate+survive) to
    exercise yet another slice of the agent's own inference structure."""

    name = "my_way_home"
    doc = DOC
    max_episodes = 2
    max_steps_per_episode = 100

    def __init__(
        self,
        env_id: str = "VizdoomMyWayHome-v1",
        frame_skip: int = 4,
        episode_timeout: int = 2100,
        seed: Optional[int] = None,
    ) -> None:
        self.env = gymnasium.make(env_id, render_mode="rgb_array", frame_skip=frame_skip,
                                   episode_timeout=episode_timeout)
        # See tasks/doom_env.py's DoomEnv.__init__ for why this is randomized
        # rather than a fixed constant -- a hardcoded seed pins vizdoom's own
        # native RNG identically across process launches.
        self.seed = seed if seed is not None else random.SystemRandom().randrange(1, 2**31 - 1)
        self.screen: Optional[Any] = None
        self._last_ts: Optional[float] = None
        self._latencies: list[float] = []

    def _to_action_index(self, action: Any) -> int:
        if isinstance(action, str):
            name = action.strip().lower()
            if name not in ACTION_NAMES:
                raise ValueError(f"unknown my_way_home action {action!r}; expected one of {ACTION_NAMES}")
            return ACTION_NAMES.index(name)
        index = int(action)
        if not 0 <= index < len(ACTION_NAMES):
            raise ValueError(f"my_way_home action index {index} out of range [0, {len(ACTION_NAMES)})")
        return index

    def reset(self) -> Any:
        obs, _info = self.env.reset(seed=self.seed)
        self.screen = obs["screen"]
        self._last_ts = time.monotonic()
        self._latencies = []
        return self.screen

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        now = time.monotonic()
        latency = (now - self._last_ts) if self._last_ts is not None else None
        if latency is not None:
            self._latencies.append(latency)
        obs, reward, terminated, truncated, info = self.env.step(self._to_action_index(action))
        self.screen = obs["screen"]
        info = dict(info)
        info["act_latency_s"] = latency
        info["avg_act_latency_s"] = (
            sum(self._latencies) / len(self._latencies) if self._latencies else None
        )
        self._last_ts = time.monotonic()
        return self.screen, float(reward), bool(terminated or truncated), info

    def restart(self) -> None:
        """No cyclic prompt cursor to rewind (unlike MathEnv/TTFTEnv) -- every
        reset() already starts a brand new game episode, so there is nothing
        else to reset here. Present for interface parity with the other envs."""
