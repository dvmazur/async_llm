from __future__ import annotations

import random
import time
from typing import Any, Optional

from .realtime_vizdoom import make_vizdoom, realtime_options, game_mode, environment_doc, game_tic_limits

# Matches VizdoomHealthGathering-v1's actual Discrete(4) button_map: index 0 =
# no button pressed, 1 = MOVE_FORWARD, 2 = TURN_RIGHT, 3 = TURN_LEFT. No
# ATTACK button here, unlike tasks/doom_env.py's DefendLine.
ACTION_NAMES = ["wait", "forward", "right", "left"]

# Surfaced verbatim into the round prompt via agent.py's task_env.doc hook.
DOC = (
    "`health_gathering` observations are a dict `{'screen': <image>, 'health': <float>}`, not bare "
    f"text. Actions: {ACTION_NAMES!r} (or their index, 0-3 in that order) -- wait / move forward / "
    "turn right / turn left. Unlike the sibling `doom` (defend-the-line) task, there is no "
    "fire/attack button here.\n"
    "Feeding `observation['screen']` to the model needs the multimodal chat-template/processor "
    "path, not the plain text tokenizer.\n"
    "Scenario: the floor is poisonous and continuously drains health; medkits scattered around the "
    "room restore it when walked over. Reward is a +1 living reward per game tic survived, and a "
    "-100 penalty if health reaches 0 -- so the score is driven by finding and reaching medkits "
    "before health runs out, not by any single decisive action. `observation['health']` carries the "
    "agent's current HEALTH value directly as a plain float, a signal `doom`'s DefendLine env "
    "doesn't expose -- worth reading if you want a sharper act() than reasoning off the screenshot "
    "alone.\n"
    "Medkit appearance, confirmed by directly rendering frames from this exact env (not guessed): "
    "they are small, glowing GREEN canisters/vials standing upright on the floor, usually near a "
    "wall -- not a white box, not a red-and-white medical cross (that's the Doom HUD icon, which "
    "never appears in the 3D scene itself). A prior run's engine assumed 'white box' / 'white "
    "medical cross' and built its whole vision prompt and logit-probe classifier around finding "
    "that wrong shape+color; it never reliably found real medkits as a result and plateaued right "
    "at the do-nothing baseline described below. Describe or classify for the actual green-canister "
    "look, and check near the base of walls, not just open floor -- they're small and easy to miss "
    "at a distance against the mottled wall texture.\n"
    "One structure that has worked well for `doom`: several separate, independently freeable cache "
    "blocks (system instructions built once; a running text history that only ever grows; a small "
    "current-frame block freed and rebuilt every step) instead of one re-encoded blob, and picking "
    "the action via a logit probe over just the action-name tokens instead of generating and "
    "parsing free text. Not required -- build it however you like.\n"
    "The game runs asynchronously at the configured tic rate (normally 35 tics/sec), including "
    "while act() awaits LLM inference. The active inference action policy is described below. "
    "Death and the native episode timeout end evaluation "
    "and cancel any pending decision. Latency therefore directly affects attainable reward. "
    "Use real LLM.forward() calls: evaluations with no forward calls are invalid.\n"
    "Watch out for a degenerate local optimum here: doing nothing at all (e.g. always 'wait', or "
    "any policy that never seeks out a medkit) still nets a score around ~280-290 -- health drains "
    "to 0 on its own after a few hundred tics of living reward, and the resulting -100 death penalty "
    "only cancels part of what was accumulated passively before that. A score in that ~280-290 range "
    "is NOT evidence of a working health-seeking policy; it's what inaction alone already scores. "
    "Only treat a policy as actually working if it reliably beats that baseline by actively detecting "
    "and moving toward medkits (e.g. using the screen and/or the HEALTH game variable to steer "
    "towards visible medkits, or exploring when none are visible, rather than only reacting to "
    "imminent damage). A policy that's genuinely seeking medkits tends to land well above that "
    "baseline, commonly in the many-hundreds-to-low-thousands range -- so merely beating ~290 by a "
    "small margin isn't strong evidence either; look for a clear, large jump.\n"
    "A common failure mode here: detecting that a medkit is visible somewhere on screen and then "
    "always moving straight forward regardless of where in the frame it actually is. That only "
    "connects when the medkit happens to be dead-center; anything off to the left or right gets "
    "walked past. Once you detect a medkit, work out roughly which side of the frame it's on and "
    "turn toward it before or while moving forward, and keep re-checking its position on every "
    "step it stays visible rather than deciding a direction once and committing to it -- your own "
    "heading and the medkit's position in frame both shift as you move.\n"
    "If you run a background loop that keeps producing a scene description or classification, "
    "wire its output into the code path that actually picks your action. A concurrent stream that "
    "computes something on every step but whose result is never read by act() burns compute for "
    "nothing.\n"
    "Be careful what a blanket `except Exception:` in act() falls back to. A prior run's act() "
    "caught every exception and defaulted to 'forward' -- meaning any bug (a bad tensor shape, a "
    "processor mismatch, an index error) silently turned into the exact always-forward failure mode "
    "described above for the rest of that step, with no trace of the real error anywhere the model "
    "could see it. If you catch broadly, log or return the exception text somewhere you'll actually "
    "look at next round, and default to something neutral (e.g. a small turn, or 'wait') rather than "
    "a default that happens to match a known bad policy.\n"
    "Reward is the only thing this experiment scores on. A fast policy that scores low is not an "
    "improvement over a slower one that scores high -- act() latency should only be a concern if "
    "it's genuinely blocking something that would raise reward, never a goal in its own right.\n"
    "That said, this experiment is about designing an LLM-driven inference structure, not about "
    "maximizing game score by whatever means happens to work. act() should actually invoke the "
    "model (a real forward pass through `llm(...)`/`llm.forward(...)`, whether directly or via "
    "generate()) for its decisions on close to every step -- not a purely scripted/heuristic "
    "action-selector that never touches the LLM at all, even one that happens to score as well as "
    "or better than a real inference pass would (e.g. a pixel-based medkit detector wired straight "
    "to a hardcoded turn/explore state machine, with generate() left sitting unused). Next round's "
    "prompt reports how many real LLM forward passes actually ran during the task versus how many "
    "env steps it took; a policy that mostly or entirely avoids calling the LLM is treated as a "
    "failed round regardless of its score and will not be reported as a working result, so don't "
    "route around real inference for speed or reliability -- a real, if imperfect, LLM-driven "
    "decision is what's being evaluated here."
)


class HealthGatheringEnv:
    """Wraps VizdoomHealthGathering-v1 behind the same reset()/step()/restart() shape
    as tasks/doom_env.py's DoomEnv -- same wrapper pattern, a different scenario
    (navigate + survive instead of aim + shoot) to exercise a different slice of the
    agent's own inference structure (no fire action, reward is survival time)."""

    real_time = True
    name = "health_gathering"
    doc = DOC
    # Was 2 -- this env's quantized, highly luck-dependent reward (0, 284,
    # 332, ... 10000) made a 2-episode round score dominated by one-off
    # draws. Bumped to 5 to match run_baseline.py's DEFAULT_EPISODES.
    max_episodes = 5
    # 2500 harness steps * frame_skip=4 = 10,000 game tics -- matches
    # episode_timeout below. Previously 100 (400 tics), low enough that
    # "survive to the end" alone reliably hit the cap, making it impossible
    # to tell a working health-seeking policy from lucky survival.
    max_steps_per_episode = 2500

    def __init__(
        self,
        env_id: str = "VizdoomHealthGathering-v1",
        frame_skip: int = 4,
        episode_timeout: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if episode_timeout is None:
            episode_timeout = game_tic_limits()["health_gathering"]
        self.doc = environment_doc(DOC)
        self.real_time = game_mode() == "asynchronous"
        self.native_episode_limit = True
        self.env = make_vizdoom(env_id, frame_skip=frame_skip,
                                   game_tic_limit=episode_timeout, **realtime_options())
        # See tasks/doom_env.py's DoomEnv.__init__ -- same randomized-seed
        # rationale: a fixed seed pins vizdoom's native RNG identically
        # across launches, so `seed=None` (default) draws fresh each episode.
        self._fixed_seed = seed
        self.seed = seed if seed is not None else random.SystemRandom().randrange(1, 2**31 - 1)
        self.screen: Optional[Any] = None
        self._last_ts: Optional[float] = None
        self._latencies: list[float] = []

    def _to_action_index(self, action: Any) -> int:
        if isinstance(action, str):
            name = action.strip().lower()
            if name not in ACTION_NAMES:
                raise ValueError(f"unknown health_gathering action {action!r}; expected one of {ACTION_NAMES}")
            return ACTION_NAMES.index(name)
        index = int(action)
        if not 0 <= index < len(ACTION_NAMES):
            raise ValueError(f"health_gathering action index {index} out of range [0, {len(ACTION_NAMES)})")
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
        return {"screen": self.screen, "health": float(obs["gamevariables"][0])}

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
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
        observation = {"screen": self.screen, "health": float(obs["gamevariables"][0])}
        return observation, float(reward), bool(done), info

    def poll(self):
        obs, reward, done, info = self.env.poll()
        self.screen = obs["screen"]
        info["avg_act_latency_s"] = (sum(self._latencies) / len(self._latencies)
                                     if self._latencies else None)
        observation = {"screen": self.screen, "health": float(obs["gamevariables"][0])}
        return observation, reward, done, info

    def close(self):
        self.env.close()

    def restart(self) -> None:
        """No cyclic prompt cursor to rewind (unlike MathEnv/TTFTEnv) -- every
        reset() already starts a brand new game episode, so there is nothing
        else to reset here. Present for interface parity with the other envs."""
