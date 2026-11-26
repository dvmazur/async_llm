# Correction trace collection

This collector records the async-input protocol from
`MrDarkTesla/math_500_sharded` on top of the current `training` APIs. At K=0,
both shards are concatenated in the initial user prompt before generation.
At K>0, the second shard is appended once to the prompt KV block as
`ADDITIONAL INFORMATION` after K model-generated tokens. Decoding then
continues in the existing output block. The resulting format-v2
traces, row-aligned targets, and inference log probabilities are checkpointed as
individual one-example JSON files. Each file can be consumed directly by
`minisgl_examples.training.sft train`.

Use `--two-correction-writes` to reproduce the historical protocol. At any K,
including K=0, the collector appends the same correction to the prompt KV
block and to the live reasoning KV block before sampling the next token.

The default path is intended for Ampere-or-newer GPUs and preserves the normal BF16
configuration. It contains no Turing compatibility path.

```bash
uv run --no-sync python -m minisgl_examples.training.corrections \
  --model Qwen/Qwen3.5-4B \
  --start-index 0 --count 16 \
  --k-steps 0 16 32 64 128 256 512 --max-tokens 1024 \
  --concurrency 16 \
  --output training_runs/corrections

uv run --no-sync python -m minisgl_examples.training.sft train \
  --data training_runs/corrections/traces/0000-k0016.json \
  --output training_runs/corrections/checkpoint
```

The dataset revision is pinned. Existing completed trace files are skipped when the
same command is restarted. The selected protocol (`prompt_only` by default,
`prompt_and_output` with the flag) is recorded in the run configuration and
each example. Use a separate output directory for each protocol. Earlier trace
files still support replay. Every example keeps
all boxed-answer decisions, correct/incorrect/mixed labels, pre-correction text and
grading, and EOS positions. Use `--no-cuda-graphs` only when debugging graph-specific
failures; it does not change trace semantics. `--concurrency` controls independent
rollouts sharing one model instance and defaults to one for the large-model path.
