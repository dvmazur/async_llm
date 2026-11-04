# Async distillation

The teacher solves **17 × 23 = 391** through the async-thoughts schedule. Train an
HF student on its recorded inputs and targets, verify the checkpoint, then let the
student solve the same task independently. The full workflow rejects an incorrect
or truncated teacher answer before training and requires overfit before student generation.

Install the examples package with `uv pip install -e ./scripts` in the project
environment. The demo has the same structure as parallel thinking:

- `demo.py`: collection/training/verification commands and the complete workflow.
- `rollout.py`: async-thoughts collection, answer checks and trace comparisons.
- `simple_distillation.ipynb`: the original experiment, calling these Python functions.

Both demos use `training.student` for trace loading, HF student training and checkpoint
reload, and `training.process` to run model stages in separate processes.

## Collect now or load later

Run collection, training, checkpoint verification and independent generation:

```bash
uv run --no-sync python -m minisgl_examples.training.distillation run \
  --teacher Qwen/Qwen3-4B --student Qwen/Qwen3-0.6B \
  --output training_runs/distillation
```

To reuse a saved teacher example, add `--trace path/to/trace.json`. The recorded
prompt and teacher model metadata are used; the teacher is not loaded for generation.
The same answer and overfit acceptance checks still apply.

An in-memory example can be passed directly to training:

```python
from minisgl_examples.training.distillation.demo import collect_trace, train_student
from minisgl_examples.training.process import isolated

example = isolated(collect_trace, "Qwen/Qwen3-4B")
report = isolated(train_student, "Qwen/Qwen3-0.6B", example, "training_runs/student")
# A saved trace path is accepted in place of example.
```

`train_student` is a training stage; the complete `run` workflow additionally grades
the teacher answer. Each model stage uses a fresh process, preserving the caller's
accelerator selection and releasing its model memory on exit. Use a new output directory.

Separate commands are available for short execution checks:

```bash
uv run --no-sync python -m minisgl_examples.training.distillation record \
  Qwen/Qwen3-0.6B training_runs/distillation-smoke/teacher \
  --think-tokens 8 --answer-tokens 8 --no-cuda-graphs
uv run --no-sync python -m minisgl_examples.training.distillation train \
  training_runs/distillation-smoke/teacher/trace.json Qwen/Qwen3-0.6B \
  training_runs/distillation-smoke/trained --steps 1
uv run --no-sync python -m minisgl_examples.training.distillation verify \
  training_runs/distillation-smoke/teacher/trace.json \
  training_runs/distillation-smoke/trained/checkpoint \
  training_runs/distillation-smoke/reloaded.json
```

## Preserved experiment settings

This demo accepts one text-only Qwen3 example and requires compatible teacher/student
tokenizers. `RecordedAsyncLLM` captures actual requests and cache views; sampled targets
are aligned to original trace rows. Prompt and probe rows have no direct loss.

Forward uses BF16, while master weights, gradients and AdamW state remain FP32.
Backward stays inside parameter substitution so checkpoint recomputation uses the same
weights. Gradient checkpointing, gradient clipping at 1.0, seed 7, and evaluation every
five updates are retained. Overfit requires **100% accuracy and loss < 0.01**.

The workflow checks exact replay logits after saving/reloading and evaluates the saved
checkpoint again in a fresh process. `checks.json` keeps independent results for boxed
answer correctness, async behavior, and exact tokens/targets/cache history after block-ID
renaming. Matching replay targets does not guarantee matching independent generation.

`--think-tokens`, `--answer-tokens`, `--probe-period`, `--memory-ratio`, `--max-seq-len`
and `--no-cuda-graphs` bound generation resources without weakening acceptance criteria.
