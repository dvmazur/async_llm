"""Headless, deterministic ViZDoom checks; importing this module needs no GPU."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ACTIONS = ("wait", "forward", "right", "left")


@dataclass(frozen=True)
class Setup:
    seed: int = 105
    agent_seed: int = 7
    max_actions: int = 12
    min_actions: int = 4
    frame_skip: int = 4
    episode_timeout: int = 16000
    consolidation_steps: int = 3
    readiness_interval: int = 14
    thought_length: int = 50
    description_length: int = 8
    history_token_limit: int = 4096
    max_pixels: int = 65536
    kv_tokens: int = 16384
    max_seq_len: int = 8192
    prefill_rows: int = 128
    dtype: str = "float16"

    def validate(self):
        if (
            min(
                self.max_actions,
                self.min_actions,
                self.frame_skip,
                self.episode_timeout,
                self.consolidation_steps,
                self.readiness_interval,
                self.description_length,
                self.history_token_limit,
                self.kv_tokens,
            )
            < 1
        ):
            raise ValueError("All limits must be positive")
        if self.thought_length < 2 or self.min_actions > self.max_actions:
            raise ValueError("Use thought_length >= 2 and min_actions <= max_actions")


def frame_hash(frame):
    h = hashlib.sha256()
    h.update(str((frame.shape, str(frame.dtype))).encode())
    h.update(frame.tobytes())
    return h.hexdigest()


def make_env(cfg):
    import gymnasium as gym
    import vizdoom.gymnasium_wrapper  # noqa: F401

    cfg.validate()
    env = gym.make(
        "VizdoomHealthGathering-v1",
        render_mode="rgb_array",
        frame_skip=cfg.frame_skip,
        episode_timeout=cfg.episode_timeout,
    )
    buttons = [b.name for b in env.unwrapped.game.get_available_buttons()]
    mapping = tuple(
        tuple(buttons[i] for i, pressed in enumerate(row) if pressed)
        for row in env.unwrapped.button_map
    )
    expected = ((), ("MOVE_FORWARD",), ("TURN_RIGHT",), ("TURN_LEFT",))
    if mapping != expected:
        env.close()
        raise ValueError(f"Unexpected action mapping: {mapping}")
    return env


def snapshot(env):
    import vizdoom

    game = env.unwrapped.game
    return {
        "health": float(game.get_game_variable(vizdoom.GameVariable.HEALTH)),
        "game_tick": int(game.get_episode_time()),
        "return": float(game.get_total_reward()),
        "dead": bool(game.is_player_dead()),
    }


def provenance():
    import gymnasium
    import vizdoom

    folder = Path(vizdoom.__file__).parent
    files = {}
    for name in ("health_gathering.cfg", "health_gathering.wad"):
        path = folder / "scenarios" / name
        if not path.exists():
            raise FileNotFoundError(path)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "scenario": "VizdoomHealthGathering-v1",
        "vizdoom": vizdoom.__version__,
        "gymnasium": gymnasium.__version__,
        "scenario_files": files,
        "source_gist": "https://gist.github.com/timka-byzov/3cc4a875991b897eec6131b14869b49e",
        "gist_revision": "a9488564436610d5e95b6ba0450113d86f968e2d",
    }


def save_frame(frame, path):
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def environment_preflight(output, setup=None):
    """Real CPU environment checks. This is deliberately NOT a teacher rollout."""
    cfg = Setup(**(setup or {}))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sequences = []
    actions = (2, 1, 3, 0, 1, 1)
    for repeat in range(2):
        env = make_env(cfg)
        try:
            observation, _ = env.reset(seed=cfg.seed)
            frames = [frame_hash(observation["screen"])]
            states = [snapshot(env)]
            if repeat == 0:
                save_frame(observation["screen"], output / "initial.png")
            for action in actions:
                observation, reward, terminated, truncated, _ = env.step(action)
                frames.append(frame_hash(observation["screen"]))
                states.append(
                    {
                        **snapshot(env),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                if repeat == 0:
                    save_frame(
                        observation["screen"], output / f"cpu_step_{len(frames) - 1:02d}.png"
                    )
                if terminated or truncated:
                    break
            sequences.append({"frames": frames, "states": states})
        finally:
            env.close()
    if sequences[0] != sequences[1]:
        raise RuntimeError("Same seed/action sequence is not deterministic")
    result = {
        "kind": "cpu_environment_preflight",
        "teacher_inference": False,
        "deterministic": True,
        "setup": asdict(cfg),
        "provenance": provenance(),
        "scripted_actions": [ACTIONS[i] for i in actions],
        "checks": sequences[0],
    }
    (output / "environment.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(environment_preflight(a.output), indent=2))
