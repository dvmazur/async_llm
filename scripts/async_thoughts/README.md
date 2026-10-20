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

Defaults are defined in `async_thoughts/demo.py`.
