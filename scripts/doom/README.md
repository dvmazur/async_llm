# doom-demo

Qwen3.5 vision plays [ViZDoom](https://vizdoom.farama.org/) through minisgl's
shared-cache `AsyncLLM` frontend — no HTTP server, no RL training, just the
model looking at the screen and pressing keys.

Each tick the demo re-encodes the current frame into an updatable image block
(`refresh_block`) and, crucially, lets the model *reason* about it before
acting: it generates a short frame-grounded analysis, then presses a key-name
action (`LEFT` / `RIGHT` / `SPACE`) arg-maxed over that reasoning. This is the
configuration that clears `VizdoomBasic` on Qwen3.5-27B (~+69 mean reward over
seeds 0-3). The purely reactive single-token doer (`--no-reason`) is a constant
"fire" regardless of the frame.

Cache layout per tick:

| Block | Lifetime |
| --- | --- |
| system | prefilled once (role + corrected mechanics) |
| user | prefilled once (the request text) |
| frame queue | last `k` frames, oldest recycled in place via `refresh_block` |
| reason | one short analysis per action, then freed |
| thinker | optional persistent planner the doer also reads (`--thinker`) |

Two things made it work: a system prompt that corrects the mechanics the model
gets wrong on its own (you cannot move forward or back — strafe to centre the
monster under the fixed crosshair, then fire), and high enough resolution to
localize the monster.

## Install

`minisgl` (the parent repo) and `async-thoughts` (`scripts/async_thoughts`,
which provides the engine helpers) must already be installed in the
environment. Then, from the repo root:

```bash
uv pip install -e scripts/doom
```

## Run

As an installed console script (headless — the demo forces
`SDL_VIDEODRIVER=dummy`):

```bash
HF_HOME=/mnt/LLM CUDA_VISIBLE_DEVICES=0 doom-demo --steps 60 --out doom.gif
doom-demo --model /path/to/Qwen3.5-27B --seed 0 --steps 80
doom-demo --thinker --frame-hint          # add the persistent planner
doom-demo --no-reason                     # reactive baseline (fires constantly)
```

Or as a module (no install needed, run from `scripts/doom/`):

```bash
uv run python -m doom_demo
```

All defaults live in `doom_demo/config.py` (`DoomConfig`); the prompts live in
`doom_demo/prompt.py`. The model can also be set via the `MINISGL_DEMO_MODEL`
environment variable.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3.5-0.8B` | HF model path (or `MINISGL_DEMO_MODEL`). |
| `--env-id` | `VizdoomBasic-v1` | Gymnasium ViZDoom environment. |
| `--steps` | `60` | Agent steps in the episode. |
| `--seed` | random | Env reset seed (fixes the monster spawn). |
| `--frame-skip` | `4` | Env frames per agent step. |
| `--max-pixels` | `150000` | Frame is smart-resized to fit this budget. |
| `--k-frames` | `2` | Frames kept in context (image-block queue). |
| `--user-prompt` | built-in | Override the user request text. |
| `--no-reason` | off | Reactive baseline: one key, no reasoning. |
| `--reason-tokens` | `28` | Reasoning length before the key-name action. |
| `--reason-temp` | `0.0` | Reasoning temperature (0 = greedy). |
| `--thinker` | off | Also run a persistent thinker the doer reads. |
| `--thinker-temp` | `0.7` | Thinker sampling temperature. |
| `--thinker-tokens` | `8` | Thinker tokens generated per tick. |
| `--frame-hint` | off | Splice a "screen updated" note into the thinker. |
| `--memory-ratio` | `0.9` | Fraction of free GPU memory for weights + KV cache. |
| `--out` | `doom.gif` | GIF of the episode (empty to skip). |

### Notes

- Requires a CUDA-capable GPU; the demo exits if `torch.cuda` is unavailable.
- The default 0.8B model runs anywhere but does not clear the level; the
  reported result is Qwen3.5-27B.
- Lowering `--max-pixels` below ~150k makes the model unable to localize the
  monster, and it stops aiming.
