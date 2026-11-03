# Training from AsyncLLM traces

Replay asynchronous inference as a differentiable forward pass, then overfit its
recorded targets. Supports dense text **Qwen3** and text/image **Qwen3.5** using
Hugging Face layers with live AsyncLLM weights or an HF model. The inference
engine is unchanged.

## Pipeline

1. **Record.** Run agents inside `with RecordedAsyncLLM(llm) as recorded_llm`
   and retrieve their history with `recorded_llm.get_trace()`.
   Each input token records its ID, scheduled step, prefill/decode kind, block
   position, and ordered view with visible lengths. Copy/merge operations store
   compact snapshot references; clearing starts a new live block version while
   old snapshots retain their gradients. Numeric KV/GDN states are never saved.
2. **Replay.** `replay_trace(model, trace)` compiles the history into shared block
   spans and indices cached on the trace. Full attention reconstructs visibility
   and block-relative RoPE, with one softmax over each complete view. Prefill
   prefixes, concurrent decodes, reordered/repeated blocks, copies, merges, and
   clears retain their semantics. Qwen3.5 GDN rebuilds differentiable affine
   summaries `(A, B)` from current activations, composes them in view order, and
   reconstructs convolution history from source indices. Cached plans contain
   metadata only.
3. **Train.** Mining writes JSON containing the model ID, traces, aligned targets,
   and inference log-probabilities. Targets belong to original trace rows;
   unsupervised rows use `-100`. A global next-token shift is incorrect for
   interleaved workers. SFT first checks replay against inference (maximum target
   log-probability error must be ≤ `0.1`), then trains all weights with AdamW.
   Each update accumulates all examples, normalized by supervised token count.

Image requests retain pixels, grids, token types, and three-axis positions in the
same trace format. Replay re-encodes saved pixels through the Qwen3.5 vision tower,
so it receives gradients. Image embeddings are recomputed, never stored in traces.
Cache visibility still uses token counts, while rotary coordinates use
image-aware spans, including copied/merged blocks. Use the full HF conditional
generation model for pixel traces; the SFT loader selects it automatically and
saves its processor with the checkpoint.

Use `replay_trace(model, trace)` for decoder hidden states without the vocabulary
projection. Set `hidden_only=False` to return logits; use `logits.softmax(-1)` for
probabilities. `model` can be an `AsyncLLM` or a compatible HF model. Optional
`rows` accepts a list/tuple of integers or a one-dimensional integer tensor,
indexing the original trace rows; output follows the requested order, including
duplicates, while retaining the context needed to compute those rows. Pass the
recorded `Trace` directly, or use `trace.to(device, **options)` to choose the
device and replay tuning options.

Traces store request batches in format-v2 JSON. Token-based construction,
format-v1 JSON, and `BlockUpdate.sources` are no longer supported. Record through
`RecordedAsyncLLM` or construct `Trace(batches=...)`; cache mutations use block
`refs`, including `refs=()` for a clear. `trace.tokens` is a derived view for
replay and inspection, not a serialized field.

## Record and backpropagate

This text example records a single causal prefill, then backpropagates a
next-token loss. It uses the same live model weights for inference and replay:

```python
import asyncio

import minisgl
import torch
import torch.nn.functional as F


async def main():
    llm = minisgl.AsyncLLM(
        "Qwen/Qwen3-0.6B",
        dtype=torch.bfloat16,
        attention_backend="fi",
        cuda_graph_max_bs=0,
        num_page_override=1024,
        max_seq_len_override=512,
    )
    try:
        ids = llm.tokenizer.encode("The capital of France is Paris.", add_special_tokens=False)
        with minisgl.training.RecordedAsyncLLM(llm) as recorded:
            block = await recorded.create_block()
            await recorded.forward(ids[:-1], write_to=block, return_logits=False)
            await recorded.free_block(block)
        trace = recorded.get_trace()

        # Default replay returns hidden states; request logits for this loss.
        logits = minisgl.training.replay_trace(llm, trace, hidden_only=False)
        targets = torch.tensor(ids[1:], device=logits.device)
        loss = F.cross_entropy(logits.float(), targets)
        loss.backward()
        print("loss:", loss.item())
    finally:
        await llm.close()


asyncio.run(main())
```

Multiple recording scopes can share one engine and still batch their requests:

```python
from minisgl.training import RecordedAsyncLLM, Trace, replay_trace


async def collect_trace(llm) -> Trace:
    with RecordedAsyncLLM(llm) as recorded:
        await run_agent(recorded)
    return recorded.get_trace()


async def train_agents(llm):
    traces = await asyncio.gather(*(collect_trace(llm) for _ in range(5)))
    for trace in traces:
        logits = replay_trace(llm, trace, hidden_only=False)
        compute_loss(logits).backward()
```

Here `run_agent` and `compute_loss` are your application functions.
Each scope records only calls through its wrapper, including
`prefill_block`, `forward`, `async_generate`, and
cache mutations. Unused blocks that are only created and freed are omitted.
Agents in one scope may share blocks; separate scopes must use independent
blocks and record their history from the start. Reading a populated unrecorded
block or using another active scope's block raises an error.
Await all calls and close generation streams before leaving the scope. Exiting
restores the engine after the last recorder and leaves the LLM open; exceptional
exit cancels only that scope's queued requests. Sampled labels and file export
remain application responsibilities. Use `RecordedAsyncLLM` for recording; the
older whole-session `record(llm)` API has been removed.

The AsyncLLM adapter supports unquantized dense models on one rank. It enables
gradients on the current engine weights without reloading a checkpoint. Finish
all inference requests before replay, and free cached blocks before resuming
inference after weight updates. The next-token shift above applies to this
single contiguous prefill; interleaved traces need their recorded row targets.

Mining uses FP16; the training CLI uses FP32, including the GDN core. Evaluation
runs every five updates and at the end. Training stops early when every example
has 100% target accuracy and loss below `0.01`. It saves a standard HF checkpoint
and tokenizer to `--output`, with `metrics.json` in its parent directory.

## Run

The `minisgl.training.sft` and `minisgl.training.stress` CLI entry points are
scheduled for future deprecation. They remain functional and emit a
`FutureWarning` when invoked. No replacement command or removal date is set.

Run from the repository root. `uv` manages the environment:

```bash
uv sync --extra dev
```

The CLIs require exactly one visible GPU. Output paths must be new; use a
separate run directory for each experiment's metrics.

Mine short Qwen3 trajectories, then overfit them:

```bash
uv run --no-sync python -m minisgl.training.sft mine \
  --model Qwen/Qwen3-0.6B --max-tokens 8 \
  --output training_runs/qwen3_demo/trajectories.json

uv run --no-sync python -m minisgl.training.sft train \
  --data training_runs/qwen3_demo/trajectories.json \
  --output training_runs/qwen3_demo/checkpoint --steps 80 --gradient-checkpointing
```

For asynchronous arithmetic trajectories with more workers and mutations, add
`--stress` to `sft mine`. For synthetic edge cases through the real inference
session, use the separate stress miner; its JSON trains through the same CLI:

```bash
uv run --no-sync python -m minisgl.training.stress \
  --model Qwen/Qwen3.5-0.8B --large-size 16 \
  --output training_runs/qwen35_demo/trajectories.json

uv run --no-sync python -m minisgl.training.sft train \
  --data training_runs/qwen35_demo/trajectories.json \
  --output training_runs/qwen35_demo/checkpoint --steps 80 --gradient-checkpointing
```

These are overfitting experiments: success measures learning the recorded targets,
not answer quality. For a CUDA GDN comparison, train into a fresh output directory
with `--gdn-read-mode factorized --gdn-read-backend triton`.

## Flags and tradeoffs

All commands support `-h` / `--help`. Defaults below are literal CLI defaults.

### `minisgl.training.sft mine`

| Flag | Default | Meaning / tradeoff |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-0.6B` | HF model ID or local model path. |
| `--output` | `training_runs/qwen3_sft/trajectories.json` | New trajectory JSON file. |
| `--stress` | off | Six asynchronous workers with changing views, prefill extensions, copies, merges and clears; more coverage and cost than the default two-worker examples. |
| `--max-tokens` | `16` | Positive generation limit per worker; longer runs increase trace size and replay cost. |
| `--temperature` | `0.8` | Positive top-20 sampling temperature; higher values increase diversity. |
| `--seed` | `7` | Sampling seed; asynchronous scheduling can still vary. |

### `minisgl.training.stress`

Uses synthetic token inputs and argmax targets. Exercises small/large blocks,
empty/reordered views, concurrent writes, self-copies, source extensions, merges,
clears, and block reuse; prints coverage statistics.

| Flag | Default | Meaning / tradeoff |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-0.6B` | HF model ID or local model path. |
| `--output` | required | New JSON file in the same format as `sft mine`. |
| `--large-size` | `128` | Seed/extension block size cap, `1..256`; larger values increase context and memory. Copies can produce larger blocks. |
| `--seed` | `7` | Seed for deterministic token inputs and workload layout. |

### `minisgl.training.sft train`

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

Tests cover regular HF versus replay forward/backward on one block, sequential
references, complex mutations/views, overfitting, checkpointing, and Torch/Triton
parity. Run selected tests without creating coverage artifacts:

```bash
uv run --no-sync pytest -o addopts='' \
  tests/core/test_training.py tests/core/test_training_gdn.py tests/core/test_training_sft.py
```

CUDA cases are skipped when CUDA is unavailable. Numerical parity uses tolerances
because replay changes reduction order and mining uses FP16.

Supported configurations require default RoPE, zero attention dropout, full
attention (no sliding windows), and SiLU GDN convolution. Video inputs, MoE,
externally modified recurrent states, and higher-order GDN gradients are
unsupported. This CLI is a single-GPU overfitting harness; it has no mixed
precision, distributed training, or resume flag.
