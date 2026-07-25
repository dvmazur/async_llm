# doom-basic

The plain VLM agent loop on [ViZDoom](https://vizdoom.farama.org/): each env
step, show the model the current frame, let it reason briefly about what has to
happen, then **probe its logits** for the keypress and step the environment.

```text
def action_probe(reasoning_prompt, reasoning_trace) -> action:
    full_prompt = cat(reasoning_prompt, reasoning_trace, PROBE_QUERY)
    logits      = model.encode(full_prompt)
    valid       = logits[possible_actions_mask]      # FIRE / RIGHT / LEFT only
    return action_of[possible_actions_mask[valid.argmax()]]


for each step:
    frame  = env screen

    reasoning_prompt = reasoning_system + user + <the frame> + "what has to happen?"
    reasoning_trace  = model.generate(reasoning_prompt, max_steps=MAX_REASONING_STEPS)

    action = action_probe(reasoning_prompt, reasoning_trace)

    obs = env.step(action)
```

The two halves are deliberately separate. The reasoning pass is asked for free,
*abstract* thought about the frame — where the monster sits relative to the
crosshair and what has to happen next — and is explicitly told **not** to name a
key and to be brief (one or two sentences, 30 words). The keypress is then read
off the model's own next-token distribution, masked down to the action words, so
there is no output format for the model to get wrong: every step yields a legal
action, and no parsing or fallback path is involved.

Each step prints the trace, then the probe's distribution over actions, then the
action taken:

```text
[  1] Relative to the centre crosshair, the monster is positioned to the left.
      You must strafe LEFT to bring it under the crosshair, then FIRE.
      probe: LEFT=0.99 RIGHT=0.01 FIRE=0.00  -> move left  r=-4 total=-8
```

Nothing is carried across steps: every step builds a fresh, self-contained
prompt and frees it afterwards — no frame queue, no in-place image refresh, no
background thinker. The only memory the model gets is the short list of its own
recent actions, rendered into the prompt as text (`--history`, `0` disables).
See `scripts/doom` for the shared-cache version that keeps the last *k* frames
and a running plan live in the KV cache.

Under the hood a step is three blocks: the reasoning prompt as one standalone
multimodal prefill (text, frame, text — an image can only be encoded in a
context-free prefill), the trace generated into a second block that reads it,
and the probe query as a third, throwaway prefill on top of both. All three are
freed at the end of the step.

The system prompt is the one from `scripts/doom`: it corrects the mechanics the
model gets wrong on its own (you cannot move forward or back — strafe until the
monster sits under the fixed centre crosshair, then fire). Without it the model
keeps trying to walk toward the monster and never shoots. The reasoning seed
(`"Relative to the centre crosshair, the monster is"`) and the probe query
(`"To put it under the crosshair and shoot, I press:"`) also come from there —
that phrasing commits to firing once centred, where a "Step 1 / Step 2" framing
made the model oscillate instead.

## Install

`minisgl` (the parent repo) must already be installed in the environment. Then,
from the repo root:

```bash
uv pip install -e scripts/doom_basic
```

## Run

As an installed console script (headless — the demo forces
`SDL_VIDEODRIVER=dummy`):

```bash
HF_HOME=/mnt/LLM CUDA_VISIBLE_DEVICES=0 doom-basic --steps 40 --out doom_basic.gif
doom-basic --model /path/to/Qwen3.5-27B --seed 0 --steps 60
doom-basic --reason-tokens 64      # let the trace run longer before the probe
doom-basic --history 0             # fully stateless: no action list in the prompt
doom-basic --temperature 0.7       # sample the reasoning instead of greedy
```

Or as a module (no install needed, run from `scripts/doom_basic/`):

```bash
uv run python -m doom_basic
```

All defaults live in `doom_basic/config.py` (`BasicConfig`); the prompts live in
`doom_basic/prompt.py`. The model can also be set via the `MINISGL_DEMO_MODEL`
environment variable.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3.5-0.8B` | HF model path (or `MINISGL_DEMO_MODEL`). |
| `--env-id` | `VizdoomBasic-v1` | Gymnasium ViZDoom environment. |
| `--steps` | `60` | Agent steps in the episode. |
| `--seed` | random | Env reset seed (fixes the monster spawn). |
| `--frame-skip` | `4` | Env frames per agent step. |
| `--max-pixels` | `150000` | Frame is smart-resized to fit this budget. |
| `--reason-tokens` | `32` | Cap on the trace before the probe runs. |
| `--temperature` | `0.0` | Reasoning temperature (0 = greedy). |
| `--top-p` | `0.95` | Nucleus cutoff when sampling. |
| `--history` | `4` | Recent actions listed in the prompt (0 = stateless). |
| `--memory-ratio` | `0.9` | Fraction of free GPU memory for weights + KV cache. |
| `--out` | `doom_basic.gif` | GIF of the episode (empty to skip). |

### Notes

- Requires a CUDA-capable GPU; the demo exits if `torch.cuda` is unavailable.
- The probe is only as good as the trace it reads. On Qwen3.5-4B it tracks the
  trace closely (a trace that says "strafe RIGHT" probes `RIGHT≈0.9`), so the
  run summary reports the mean probability of the chosen action as a confidence
  number. The 0.8B default runs anywhere but its trace rarely localizes the
  monster, and the probe then collapses to a near-constant `FIRE`.
- If the trace is cut off mid-sentence, raise `--reason-tokens`; the "BE BRIEF"
  instruction lands on the 4B (which finishes in ~35 tokens) but not reliably on
  the 0.8B.
- Lowering `--max-pixels` below ~150k makes the model unable to localize the
  monster, and it stops aiming.
