# Correction trace collection

This collector records the async-input protocol from
`MrDarkTesla/math_500_sharded` on top of the current `training` APIs. A model starts
from `problem_shards[0]`; after `--k-steps` generated tokens,
`problem_shards[1]` arrives as `ADDITIONAL INFORMATION`. The resulting format-v2
traces, row-aligned targets, and inference log probabilities are checkpointed as
individual one-example JSON files. Each file can be consumed directly by
`minisgl_examples.training.sft train`.

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
same command is restarted; a differing `config.json` is rejected. Every example keeps
all boxed-answer decisions, correct/incorrect/mixed labels, pre-correction text and
grading, and EOS positions. Use `--no-cuda-graphs` only when debugging graph-specific
failures; it does not change trace semantics. `--concurrency` controls independent
rollouts sharing one model instance and defaults to one for the large-model path.
