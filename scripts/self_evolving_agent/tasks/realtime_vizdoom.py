"""Native real-time ViZDoom, with a dedicated thread owning all game I/O.

The simulator runs while Python/model inference is busy. A polling thread
refreshes native state, detects terminal events, and retains the final reward.
"""
from __future__ import annotations

import os
import threading
import time

import gymnasium
import vizdoom
import vizdoom.gymnasium_wrapper  # noqa: F401


def realtime_options():
    """Shared options for every game entry point; defaults are the eval protocol."""
    return {"ticrate": int(os.environ.get("SEA_GAME_TICRATE", "35")),
            "action_policy": os.environ.get("SEA_INFERENCE_ACTION", "hold_last")}


def game_mode():
    mode = os.environ.get("SEA_GAME_MODE", "asynchronous")
    if mode not in {"synchronous", "asynchronous"}:
        raise ValueError(f"Unknown game mode: {mode}")
    return mode


def game_tic_limits():
    limits = {"doom": int(os.environ.get("SEA_DOOM_TIC_LIMIT", "1000")),
              "health_gathering": int(os.environ.get("SEA_HEALTH_GATHERING_TIC_LIMIT", "10000"))}
    if any(value <= 0 for value in limits.values()):
        raise ValueError("Native game tic limits must be positive")
    return limits


def make_vizdoom(env_id, **options):
    if game_mode() == "synchronous":
        from .synchronous_vizdoom import SynchronousVizdoom
        return SynchronousVizdoom(env_id, **options)
    return RealtimeVizdoom(env_id, **options)


def environment_doc(doc):
    if game_mode() == "synchronous":
        start = doc.index("The game runs asynchronously")
        end = doc.index("Use real LLM.forward()", start)
        doc = doc[:start] + doc[end:]
    return doc + "\n" + action_policy_doc()


class RealtimeVizdoom:
    def __init__(self, env_id, *, frame_skip, ticrate, game_tic_limit, action_policy):
        if ticrate <= 0 or game_tic_limit <= 0 or frame_skip <= 0:
            raise ValueError("Positive tic rate, game horizon, and frame skip required")
        if action_policy not in {"hold_last", "wait"}:
            raise ValueError("Unknown action policy")
        self.env = gymnasium.make(env_id, render_mode="rgb_array", frame_skip=frame_skip)
        self.base = self.env.unwrapped
        self.game = self.base.game
        self.game.set_mode(vizdoom.Mode.ASYNC_PLAYER)
        self.game.set_ticrate(ticrate)
        self.game.set_episode_timeout(game_tic_limit + self.game.get_episode_start_time())
        self.frame_skip, self.ticrate = frame_skip, ticrate
        self.game_tic_limit, self.action_policy = game_tic_limit, action_policy
        # Native living reward only counts tics requested by advance_action,
        # not every autonomous tic. Integrate it from the native game clock.
        self.living_reward = self.game.get_living_reward()
        self.game.set_living_reward(0)
        if self.base.num_delta_buttons:
            raise ValueError("This adapter expects discrete binary-button actions")
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._thread = None

    def _check_error(self):
        if self._error is not None:
            raise RuntimeError("Real-time ViZDoom thread failed") from self._error

    def reset(self, *, seed=None):
        self.close()
        self._stop.clear()
        with self._cv:
            self._ready = False
            self._done = False
            self._timeout = False
            self._error = None
            self._buttons = list(self.base.button_map[0])
            self._pending_action = 0
            self._submitted = self._applied = 0
            self._applied_tic = 0
            self._tics = 0
            self._reward = self._delivered_reward = 0.0
            self._thread = threading.Thread(target=self._run, args=(seed,), daemon=True)
            self._thread.start()
            self._cv.wait_for(lambda: self._ready or self._error is not None)
            self._check_error()
            return self._obs, self._info()

    def _run(self, seed):
        try:
            obs, _ = self.env.reset(seed=seed)
            self.game.set_action(self.base.button_map[0])
            initial_tic = self.game.get_episode_time()
            start = time.monotonic()
            with self._cv:
                self._obs = obs
                self._start = start
                self._end = None
                self._ready = True
                self._cv.notify_all()
            last_applied = 0
            while not self._stop.is_set():
                with self._cv:
                    submitted, action = self._submitted, self._pending_action
                if submitted != last_applied:
                    self.game.set_action(self.base.button_map[action])
                    last_applied = submitted
                    with self._cv:
                        self._applied = submitted
                        self._applied_tic = self._tics
                elif self.action_policy == "wait" and self._tics - self._applied_tic >= self.frame_skip:
                    self.game.set_action(self.base.button_map[0])
                # Zero tics does not refresh native state, even in ASYNC_PLAYER.
                self.game.advance_action(1)
                actual_tics = self.game.get_episode_time() - initial_tic
                tics = min(actual_tics, self.game_tic_limit)
                native_done = self.game.is_episode_finished()
                done = native_done or actual_tics >= self.game_tic_limit
                self.base.state = self.game.get_state()
                # Use the installed wrapper's collector to retain its image
                # layout, game variables, and terminal-observation convention.
                obs = self.base._VizdoomEnv__collect_observations()
                obs = {key: value.copy() if hasattr(value, "copy") else value for key, value in obs.items()}
                reward = float(self.game.get_total_reward()) + self.living_reward * tics
                with self._cv:
                    self._buttons = list(self.game.get_last_action())
                    self._obs, self._tics, self._reward = obs, tics, reward
                    self._done = done
                    if done:
                        self._end = time.monotonic()
                    self._timeout = bool(self.game.is_episode_timeout_reached() or actual_tics >= self.game_tic_limit)
                    self._cv.notify_all()
                if done:
                    break
        except BaseException as error:
            with self._cv:
                self._error = error
                self._cv.notify_all()
        finally:
            self.env.close()

    def _info(self):
        return {"game_tics": self._tics, "game_tic_limit": self.game_tic_limit,
                "game_elapsed_s": self._tics / self.ticrate,
                "wall_elapsed_s": (self._end or time.monotonic()) - self._start,
                "episode_total_reward": self._reward,
                "real_time": True, "ticrate": self.ticrate,
                "action_policy": self.action_policy, "buttons": list(self._buttons),
                "timeout": getattr(self, "_timeout", False)}

    def _consume(self):
        delta = self._reward - self._delivered_reward
        self._delivered_reward = self._reward
        return self._obs, delta, self._done, self._info()

    def poll(self):
        with self._cv:
            self._check_error()
            return self._consume()

    def step(self, action):
        if not self.base.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        with self._cv:
            self._check_error()
            if not self._done:
                self._submitted += 1
                submitted = self._submitted
                self._pending_action = action
                self._cv.wait_for(lambda: self._error is not None or self._done or
                                  (self._applied >= submitted and
                                   self._tics - self._applied_tic >= self.frame_skip))
                self._check_error()
            return self._consume()

    def close(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("Real-time game thread did not stop")
            self._thread = None


def action_policy_doc():
    if game_mode() == "synchronous":
        return ("The game is synchronous and paused during inference. Each returned action "
                "advances the game by four tics (or until native death/timeout). "
                "Interactivity is measured as LLM.forward() calls per completed environment step.")
    policy = realtime_options()["action_policy"]
    if policy == "hold_last":
        return ("Active inference action policy: hold_last. The game keeps the last returned "
                "action pressed until a new decision arrives, including throughout inference. "
                "It waits before the first action. Each step waits at least frame_skip tics "
                "before returning an observation, but the action does not expire afterward.")
    return ("Active inference action policy: wait. Each returned action lasts frame_skip tics, "
            "then the game presses no buttons while inference is pending.")
