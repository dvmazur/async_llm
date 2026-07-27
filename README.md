# AsyncReasoning

>This code contains the anonymized patch to mini-sglang that implements AsyncReasoning. Per rules, we anonymized the change itself but kept the original code unchanged. Anonymous authors of AsyncReasoning are not related to the mini-sglang codebase itself, only the anonymized patch.
A standalone, installable implementation of AsyncReasoning inference based on mini-sglang.
---

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
uv pip install -e scripts/async_reasoning
```

## Run

As an installed console script:

```bash
async-reasoning
async-reasoning --problem "What is 17 * 23?"
async-reasoning --max-steps 400 --probe-period 20
```

Or as a module (no install needed, run from `scripts/async_reasoning/`):

```bash
uv run python -m async_reasoning
```

All defaults live in `async_reasoning/config.py`. The model can also be set via
the `MINISGL_DEMO_MODEL` environment variable. Pick the GPU with
`CUDA_VISIBLE_DEVICES`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-32B` | HF model path (or `MINISGL_DEMO_MODEL`). |
| `--problem` | polynomial-eval problem | Question for the assistant to solve. |
| `--max-steps` | `800` | Hard cap on total decode steps. |
| `--probe-period` | `30` | Run the mode-switching probe every N steps. |
| `--memory-ratio` | `0.9` | Fraction of free GPU memory for weights + KV cache. |
| `--page-size` | `1` | Number of tokens per KV-cache page. |

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
async-reasoning --model /path/to/Qwen3.5-0.8B --max-steps 140 --probe-period 20
async-reasoning --model /path/to/Qwen3.5-27B --memory-ratio 0.8
```
