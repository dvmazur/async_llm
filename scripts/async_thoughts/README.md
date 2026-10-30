# async-thoughts

A standalone [AsyncReasoning](https://arxiv.org/abs/2512.10931) demo using
minisgl's shared cache. A thinker generates private reasoning while a writer
produces the answer. The terminal shows both streams in different colors and
logs when writing pauses or resumes.

The readiness probe reuses the live thinker and writer cache blocks and encodes
only a short yes/no question. It runs in a separate coroutine every 20 thinking
tokens or at a thinker paragraph boundary. A writer paragraph pauses writing
until the probe allows it to continue. When thinking ends or reaches its token
budget, the writer finishes up to its own budget. No additional user inputs or
server are needed.

## Install and run

Install the parent project first (see the [main README](../../README.md)), then
run these commands from the repository root:

```bash
uv pip install -e scripts/async_thoughts
uv run --no-sync async-thoughts
uv run --no-sync async-thoughts --model Qwen/Qwen3.5-0.8B \
  --problem "What is 17 * 23?" --max-think-tokens 192 --max-answer-tokens 192
```

The installed package also supports `uv run --no-sync python -m async_thoughts`.
A CUDA-capable GPU is required. The default model is `Qwen/Qwen3.5-27B` in BF16;
use a smaller model if GPU memory is limited. Decode and short probe prefills use
CUDA graphs, including for Qwen3.5's hybrid attention.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3.5-27B` | Model name or local path; also settable with `MINISGL_DEMO_MODEL`. |
| `--problem` | polynomial evaluation | Question for the assistant to solve. |
| `--max-think-tokens` | `1024` | Thinker token budget; `--max-steps` is a backward-compatible alias. |
| `--max-answer-tokens` | `1024` | Writer token budget. |
| `--probe-period` | `20` | Readiness check interval in thinking tokens; paragraph boundaries also trigger checks. |
| `--memory-ratio` | `0.85` | Engine GPU memory fraction; lower it to leave more headroom. |
| `--page-size` | `1` | Tokens per cache page. |
| `--trace-output` | off | Save request-batch traces and aligned sampled-token targets to a new JSON file. |

Defaults are defined in `async_thoughts/demo.py`.

## Record a demo and train from it

After installing the demo, record a short run from the repository root:

```bash
uv run --no-sync async-thoughts \
  --model Qwen/Qwen3.5-0.8B --memory-ratio 0.3 \
  --max-think-tokens 32 --max-answer-tokens 16 --probe-period 4 \
  --trace-output training_runs/async_thoughts/trace.json
```

The file records the actual scheduled prefill/decode requests, ordered cache
snapshots, probe requests, and block clears. Each thinker/writer sample becomes
the target of the input row whose logits produced it, including terminal tokens
and the final sample that has not yet been written to the cache. Targets use
`-100` for prefills, probe questions, and flush-only forwards. Probe decisions
are not training labels. Full vocabulary logits are not serialized; the file
stores each supervised target's raw, unmasked inference log-probability.

Recording wraps the session before prefill and observes `async_generate` results.
It does not run additional model forwards. Capturing target log-probabilities
adds synchronization overhead and can affect asynchronous scheduling; the trace
preserves the schedule that actually executed. A failed run does not export a
dataset, and an existing output file is never overwritten.

The JSON is accepted directly by the existing trainer. One update exercises
parallel forward, backward, and an optimizer step, then saves an HF checkpoint:

```bash
uv run --no-sync python -m minisgl.training.sft train \
  --data training_runs/async_thoughts/trace.json \
  --output training_runs/async_thoughts/checkpoint --steps 1 --lr 1e-6 \
  --gradient-checkpointing
```

The trainer checks recorded target log-probabilities against FP32 replay before
updating weights. Demo inference uses BF16, so this is a tolerance check rather
than a bitwise comparison. Start with a small checkpoint: full FP32 training,
gradients, and AdamW states need substantially more memory than demo inference.
Use fresh output paths for another run. More updates overfit this trajectory;
the command does not measure improvements in answer quality. The small learning
rate matters for short greedy rollouts: the default `3e-5` overshot on the
Qwen3.5-0.8B demo trace, while `1e-6` decreased its loss over five updates.

For direct access to forward/backward, after the inference process exits:

```python
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from minisgl.training.replay import Replay, forward
from minisgl.training.sft import supervised_loss
from minisgl.training.trace import Trace

data = json.loads(Path("training_runs/async_thoughts/trace.json").read_text())
example = data["examples"][0]
trace = Trace.from_dict(example["trace"])
replay = Replay.from_trace(trace, device="cuda")
targets = torch.tensor(example["targets"], device="cuda")
config = AutoConfig.from_pretrained(data["model"]).get_text_config()
model = AutoModelForCausalLM.from_pretrained(
    data["model"], config=config, dtype=torch.float32
).cuda().train()

# Inspect only supervised rows; returned rows follow original trace order.
with torch.no_grad():
    logits = forward(model, replay, rows=(targets != -100).nonzero().flatten())
print(logits.shape)
del logits

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.zero_grad(set_to_none=True)
loss = supervised_loss(model, replay, targets) / (targets != -100).sum()
loss.backward()
print("loss:", loss.item())
```

The supervised loss projects vocabulary logits in checkpointed chunks; it still
replays the context tokens needed by the supervised rows. It does not apply a
global next-token shift, which would misalign interleaved workers.
