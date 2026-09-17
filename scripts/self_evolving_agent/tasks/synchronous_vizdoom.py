"""Native PLAYER mode: inference never advances game time."""
import time

import gymnasium
import vizdoom
import vizdoom.gymnasium_wrapper  # noqa: F401


class SynchronousVizdoom:
    def __init__(self, env_id, *, frame_skip, ticrate, game_tic_limit, action_policy):
        if min(frame_skip, ticrate, game_tic_limit) <= 0:
            raise ValueError("Positive frame skip, tic rate and horizon required")
        self.env = gymnasium.make(env_id, render_mode="rgb_array", frame_skip=frame_skip)
        self.base = self.env.unwrapped
        self.game = self.base.game
        self.game.set_mode(vizdoom.Mode.PLAYER)
        self.game.set_ticrate(ticrate)
        self.game.set_episode_timeout(game_tic_limit + self.game.get_episode_start_time())
        self.frame_skip, self.ticrate = frame_skip, ticrate
        self.game_tic_limit = game_tic_limit

    def reset(self, *, seed=None):
        self._obs, _ = self.env.reset(seed=seed)
        self._initial_tic = self.game.get_episode_time()
        self._start = time.monotonic()
        self._end = None
        self._done = False
        self._tics = 0
        self._reward = 0.0
        self._buttons = [float(value) for value in self.base.button_map[0]]
        return self._obs, self._info()

    def _info(self):
        return dict(game_tics=self._tics, game_tic_limit=self.game_tic_limit,
                    game_elapsed_s=self._tics / self.ticrate,
                    wall_elapsed_s=(self._end or time.monotonic()) - self._start,
                    episode_total_reward=self._reward, real_time=False,
                    game_mode="synchronous", ticrate=self.ticrate,
                    action_policy="step", buttons=self._buttons,
                    timeout=self._tics >= self.game_tic_limit)

    def poll(self):
        return self._obs, 0.0, self._done, self._info()

    def step(self, action):
        if not self.base.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if self._done:
            return self.poll()
        self._buttons = [float(value) for value in self.base.button_map[action]]
        reward = self.game.make_action(self._buttons, min(self.frame_skip, self.game_tic_limit - self._tics))
        self._tics = self.game.get_episode_time() - self._initial_tic
        self._reward += float(reward)
        self._done = self.game.is_episode_finished() or self._tics >= self.game_tic_limit
        self.base.state = self.game.get_state()
        self._obs = self.base._VizdoomEnv__collect_observations()
        if self._done:
            self._end = time.monotonic()
        return self._obs, float(reward), self._done, self._info()

    def close(self):
        self.env.close()
