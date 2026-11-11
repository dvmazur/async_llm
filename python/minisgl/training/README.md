# Training from AsyncLLM traces

Record asynchronous inference and replay it as a differentiable forward pass.
Supports dense text **Qwen3** and text/image **Qwen3.5** using Hugging Face layers
with live AsyncLLM weights or an HF model. The inference engine is unchanged.

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
3. **Train.** Choose your own loss and optimizer, then call `loss.backward()`.
   Targets belong to original trace rows; a global next-token shift is incorrect
   for interleaved workers. Saving datasets, evaluating, and checkpointing remain
   application responsibilities. The [training examples](../../../scripts/minisgl_examples/training/README.md)
   provide collection and training workflows, including a chunked supervised loss.

Image requests retain pixels, grids, token types, and three-axis positions in the
same trace format. Replay re-encodes saved pixels through the Qwen3.5 vision tower,
so it receives gradients. Image embeddings are recomputed, never stored in traces.
Cache visibility still uses token counts, while rotary coordinates use
image-aware spans, including copied/merged blocks. Use the full HF conditional
generation model for pixel traces so the vision tower is present during replay.

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

## Full attention backends

Select a backend with `trace.to(device, attention_backend=...)`:

- `torch` (default): dense FP32 reference, returning the original V dtype.
- `torch_optimized`: the same precision, sharing K/V preparation and K RoPE
  across query chunks.
- `fa4`: masked varlen attention with a differentiable LSE merge; requires
  Hopper (SM90) and FP16/BF16 activations.

For FA4, install the optional dependencies in the training environment:

```bash
pip install 'flash-attn-4[cu13]==4.0.0b37' \
  'nvidia-cutlass-dsl[cu13]==4.6.2' 'quack-kernels==0.6.4'
```

FA4 and merge kernels compile on their first forward/backward. See the
[methodology and measurements](../../../docs/efficient-attention-for-async-llm-training-task.pdf)
for implementation details.

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

## Validation and limits

Tests cover regular HF versus replay forward/backward on one block, sequential
references, complex mutations/views, overfitting, checkpointing, and Torch/Triton
parity. Run selected tests without creating coverage artifacts:

```bash
uv run --no-sync pytest -o addopts='' tests/training
```

CUDA cases are skipped when CUDA is unavailable. Numerical parity uses tolerances
because replay changes reduction order.

Supported configurations require default RoPE, zero attention dropout, full
attention (no sliding windows), and SiLU GDN convolution. Video inputs, MoE,
externally modified recurrent states, and higher-order GDN gradients are
unsupported.
