# async-thoughts

AsyncReasoning-style "async thoughts" demo on minisgl's shared cache.

Port of yandex-research/AsyncReasoning's
[`notebooks/demo_async_thoughts.ipynb`](https://github.com/yandex-research/AsyncReasoning/blob/main/notebooks/demo_async_thoughts.ipynb)
to a standalone, installable package driven by minisgl's in-process async
cache API — no HTTP server, no notebook.  All inference goes through the
unified `AsyncLLM.forward` method: conditional
prefill for the cache-block setup, decode-mode custom-generate loops for the
two streams, and a throwaway prefill for the probe.

Two streams run concurrently against the same model:

- **thinker** — internal chain-of-thought reasoning between `<think>...</think>`
- **writer** — the user-facing answer, which sees the thinker's partial reasoning

A periodic mode-switching probe decides whether the thinker has produced enough
new reasoning to let the writer continue, flipping between a single-worker
`thinker_only` decode and a batched two-worker `thinker_and_writer` decode.

## Install

`minisgl` (the parent repo) must already be installed in the environment. Then,
from the repo root:

```bash
uv pip install -e scripts/async_thoughts
```

## Run

As an installed console script:

```bash
async-thoughts
async-thoughts --problem "What is 17 * 23?"
async-thoughts --max-steps 400 --probe-period 20
```

Or as a module (no install needed, run from `scripts/async_thoughts/`):

```bash
uv run python -m async_thoughts
```

All defaults live in `async_thoughts/config.py`. The model can also be set via
the `MINISGL_DEMO_MODEL` environment variable. Pick the GPU with
`CUDA_VISIBLE_DEVICES`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-32B` | HF model path (or `MINISGL_DEMO_MODEL`). |
| `--problem` | polynomial-eval problem | Question for the assistant to solve. |
| `--max-steps` | `800` | Hard cap on total decode steps. |
| `--probe-period` | `30` | Run the mode-switching probe every N steps. |
| `--memory-ratio` | `0.9` | Fraction of free GPU memory for weights + KV cache. |

### Notes

- Requires a CUDA-capable GPU; the demo exits if `torch.cuda` is unavailable.
- `Qwen/Qwen3-32B` at bf16 fits on a single 80 GiB GPU (~64 GiB weights). Lower
  `--memory-ratio` to ~`0.3` for 8B-class models if you also keep another model
  resident.
- Smaller Qwen3 models tend to starve the writer because the probe margin is too
  small to keep `thinker_and_writer` active for more than one decode step at a time.

### Qwen3.5 (hybrid Gated-DeltaNet) models

Qwen3.5 models are supported and run through the same `SharedCacheSession`: their
3-in-4 linear-attention layers compose across shared-cache blocks via the GDN
affine cache, while the 1-in-4 full-attention layers use the (partial-RoPE, gated)
shared-cache attention. They decode eagerly — CUDA graph is auto-disabled for
hybrid models. Even small variants keep the writer active, so they make quick demos:

```bash
async-thoughts --model /path/to/Qwen3.5-0.8B --max-steps 140 --probe-period 20
async-thoughts --model /path/to/Qwen3.5-27B --memory-ratio 0.8
```
