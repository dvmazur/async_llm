# Async distillation

An async-thoughts teacher solves **17 × 23 = 391**. An HF student learns its
recorded targets, reloads its checkpoint, then solves the problem independently.
Use the [shared setup and Python API](../README.md), or open
`simple_distillation.ipynb` with the project environment as its kernel.

```bash
uv run --no-sync python -m minisgl_examples.training.distillation run \
  --teacher Qwen/Qwen3-4B --student Qwen/Qwen3-0.6B \
  --output training_runs/distillation
```

Add `--trace path/to/trace.json` to reuse a saved teacher example. Its recorded
prompt is reused for student generation. Output paths must be new.

Run individual stages:

```bash
uv run --no-sync python -m minisgl_examples.training.distillation record \
  Qwen/Qwen3-4B training_runs/teacher
uv run --no-sync python -m minisgl_examples.training.distillation train \
  training_runs/teacher/trace.json Qwen/Qwen3-0.6B training_runs/student
uv run --no-sync python -m minisgl_examples.training.distillation verify \
  training_runs/teacher/trace.json training_runs/student/checkpoint \
  training_runs/reloaded.json
```

Defaults: 128 thinker tokens, 192 answer tokens, a probe every 4 tokens, up to
100 optimizer steps, and learning rate `1e-6`. For a short execution check, use
`record --think-tokens 8 --answer-tokens 8 --no-cuda-graphs`, then `train --steps 1`.
Use each command's `--help` for resource controls.

Training accepts one text-only Qwen3 trace with compatible teacher/student
tokenizers. BF16 forward/backward uses FP32 master weights and AdamW state.
Checkpoint reload must reproduce selected replay logits exactly.

The complete `run` workflow requires a correct, complete teacher answer and
**100% replay accuracy with loss < 0.01** before student generation. The separate
`train` command performs training only. `trained/metrics.json` reports fitting;
`checks.json` separately checks the student's answer, async behavior and trace
structure. Learning the recorded targets does not guarantee independent answers.

`demo.py` contains the workflow and CLI; `rollout.py` records and checks traces.
