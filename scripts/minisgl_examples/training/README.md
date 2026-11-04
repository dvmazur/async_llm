# Training examples

These repository scripts collect trajectories, optimize their recorded targets,
and save standard Hugging Face checkpoints. The library itself exposes recording
and differentiable replay; see the [training API](../../../python/minisgl/training/README.md).

- `sft.py`: collect short asynchronous arithmetic traces, or train from saved traces.
- `stress.py`: collect a seeded synthetic workload with complex cache views
  and mutations through `RecordedAsyncLLM` and its scheduler.
- `loss.py`: shared supervised loss and evaluation helpers. They select original
  trace rows and project vocabulary logits in checkpointed chunks.
- `student.py`: shared HF student loading, training, evaluation and checkpoint
  reload checks for [parallel thinking](parallel_thinking/README.md) and
  [distillation](distillation/README.md). Both demos accept an in-memory collected
  example or a saved trace, and expose `record`, `train`, `verify`, and `run` stages.
- `process.py`: run model stages in separate processes, preserving the caller's
  environment and releasing accelerator memory between stages.

## Data and checkpoints

The SFT and stress miners write JSON containing a model ID and an `examples` list. Each example
contains a format-v2 `trace`, row-aligned `targets` (`-100` for unsupervised rows),
and `logprobs` (`null` for unsupervised rows). Sampled labels belong to the forward
row whose logits produced them; a global next-token shift misaligns interleaved
workers. The async-thoughts demo exports the same schema.

For the SFT CLI below, replay checks the recorded target log-probabilities with a maximum
absolute error of `0.1`. Each AdamW update accumulates all examples, normalized by
their total supervised token count. Pixel traces select the full conditional
generation model so the vision tower also receives gradients; checkpoints include
its processor instead of a text-only tokenizer.

Mining uses FP16; the training CLI uses FP32, including the GDN core. Evaluation
runs every five updates and at the end. Training stops early when every example
has 100% target accuracy and loss below `0.01`. It saves a standard HF checkpoint
and tokenizer to `--output`, with `metrics.json` in its parent directory.

## Run

Run from the repository root. `uv` manages the environment:

```bash
uv sync --extra dev
uv pip install -e ./scripts
```

The CLIs require exactly one visible GPU. Output paths must be new; use a
separate run directory for each experiment's metrics.

Mine short Qwen3 trajectories, then overfit them:

```bash
uv run --no-sync python -m minisgl_examples.training.sft mine \
  --model Qwen/Qwen3-0.6B --max-tokens 8 \
  --output training_runs/qwen3_demo/trajectories.json

uv run --no-sync python -m minisgl_examples.training.sft train \
  --data training_runs/qwen3_demo/trajectories.json \
  --output training_runs/qwen3_demo/checkpoint --steps 80 --gradient-checkpointing
```

For asynchronous arithmetic trajectories with more workers and mutations, add
`--stress` to `sft mine`. For synthetic edge cases through the inference
scheduler, use the separate stress miner; its JSON trains through the same CLI:

```bash
uv run --no-sync python -m minisgl_examples.training.stress \
  --model Qwen/Qwen3.5-0.8B --large-size 16 \
  --output training_runs/qwen35_demo/trajectories.json

uv run --no-sync python -m minisgl_examples.training.sft train \
  --data training_runs/qwen35_demo/trajectories.json \
  --output training_runs/qwen35_demo/checkpoint --steps 80 --gradient-checkpointing
```

These are overfitting experiments: success measures learning the recorded targets,
not answer quality. For a CUDA GDN comparison, train into a fresh output directory
with `--gdn-read-mode factorized --gdn-read-backend triton`.

## Flags and tradeoffs

All commands support `-h` / `--help`. Defaults below are literal CLI defaults.

### `minisgl_examples.training.sft mine`

| Flag | Default | Meaning / tradeoff |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-0.6B` | HF model ID or local model path. |
| `--output` | `training_runs/qwen3_sft/trajectories.json` | New trajectory JSON file. |
| `--stress` | off | Six asynchronous workers with changing views, prefill extensions, copies, merges and clears; more coverage and cost than the default two-worker examples. |
| `--max-tokens` | `16` | Positive generation limit per worker; longer runs increase trace size and replay cost. |
| `--temperature` | `0.8` | Positive top-20 sampling temperature; higher values increase diversity. |
| `--seed` | `7` | Sampling seed; asynchronous scheduling can still vary. |

### `minisgl_examples.training.stress`

Uses synthetic token inputs and argmax targets. Exercises small/large blocks,
empty/reordered views, concurrent writes, self-copies, source extensions, merges,
clears, and block reuse; prints coverage statistics.

| Flag | Default | Meaning / tradeoff |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-0.6B` | HF model ID or local model path. |
| `--output` | required | New JSON file in the same format as `sft mine`. |
| `--large-size` | `128` | Seed/extension block size cap, `1..256`; larger values increase context and memory. Copies can produce larger blocks. |
| `--seed` | `7` | Seed for deterministic token inputs and workload layout. |

### `minisgl_examples.training.sft train`

The model comes from the data file. There is no separate training model flag.

| Flag | Default | Meaning / tradeoff |
| --- | --- | --- |
| `--data` | `training_runs/qwen3_sft/trajectories.json` | Mined or synthetic trajectory JSON. |
| `--output` | `training_runs/qwen3_sft/checkpoint` | New HF checkpoint directory; metrics go in its parent. |
| `--steps` | `80` | Positive maximum optimizer updates; early stopping may shorten the run. |
| `--lr` | `3e-5` | Positive AdamW learning rate; larger values may fit faster but destabilize training. |
| `--seed` | `7` | Torch seed. |
| `--gradient-checkpointing` | off | Recompute decoder layers in backward: less activation memory, more compute. |
| `--attention-checkpointing` / `--no-attention-checkpointing` | on | Recompute full-attention chunks: less saved attention memory, more compute. Independent of layer checkpointing. |
| `--loss-checkpointing` / `--no-loss-checkpointing` | on | Recompute vocabulary projection/loss chunks: less saved logits memory, more compute. Independent of layer checkpointing. |
| `--cache-replay-plans` / `--no-cache-replay-plans` | on | Retain device metadata between forwards: faster repeated replay, memory retained per visited trace. Off rebuilds next forward; layers and backward still share the active plan. |
| `--query-chunk-size` | `128` | Positive full-attention query rows per chunk. Smaller reduces score workspace but adds gathers/launches; never truncates history. |
| `--loss-chunk-size` | `128` | Positive supervised rows per vocabulary projection. Smaller reduces logits workspace but adds overhead. |
| `--gdn-segment-size` | `8` | Positive compact segment bound; `1` disables short-segment compaction. Larger values can save dense summaries but increase local triangular work and reconstruction. |
| `--gdn-read-mode` | `expanded` | `expanded` reads compact segments token by token; `factorized` shortens read paths but adds factor caches and gathers. Speed and peak memory depend on layout. |
| `--gdn-cache-reused` / `--no-gdn-cache-reused` | on | Cache heavily reused dense GDN transforms within each forward: less reconstruction, more memory. Separate from metadata plan caching. |
| `--gdn-read-backend` | `torch` | `torch` or `triton`. Triton fuses CUDA forward reads, reducing launches and intermediate IO; adds first-use JIT cost. Summary construction and backward remain Torch. |

GDN flags affect only Qwen3.5 linear-attention layers. None of the optional
tradeoffs is a universal speed or memory win; compare on the intended trace.
Full attention can still require quadratic work in visible context. GDN avoids
flattening copied history, but dense summaries, saved read vectors, and deep merge
dependencies still cost memory/time. Chunking and checkpointing do not reduce
model weights or optimizer state.

## Validation and limits

The scripts require one visible GPU. This is a single-GPU overfitting harness,
with no mixed precision, distributed training, or resume flag. Success measures
learning the recorded targets, not answer quality. Tests cover mining and data
alignment, chunked loss/gradients, replay options, image model loading, and
checkpoint output:

```bash
uv run --no-sync pytest -o addopts='' tests/training
```
