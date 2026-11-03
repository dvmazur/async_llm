# Sanity check: distillation on one example

Task: **17 × 23 = 391**. First, the 4B teacher generates a rollout and must produce
one numeric `\boxed{391}` answer. Then train the 0.6B student and verify the saved
checkpoint. Finally, let the student solve the task independently.
An incorrect or truncated teacher answer stops the experiment before training.

Install the examples package with `uv pip install -e ./scripts` after setting up
the root environment. Open `simple_distillation.ipynb`, or run the same workflow
from the repository root:

```bash
uv run --no-sync python -m minisgl_examples.training.distillation.sanity run \
  --teacher Qwen/Qwen3-4B --student Qwen/Qwen3-0.6B \
  --output training_runs/distillation
```

Select one GPU before starting. The demo inherits device, thread and model-cache settings.
The CLI saves results under `--output`; the notebook creates a new
`training_runs/async_distillation_sanity_check/simple_multiply_*` directory.
Each model runs in a separate process and releases its GPU memory when it exits.

Training uses `minisgl.training.Trace` and `replay_trace` through the shared
`minisgl_examples.training.loss` helpers. Generation reuses the original `async_thoughts` schedule
and its `RecordedAsyncLLM` recorder.

The demo supports one text-only Qwen3 trace and one numeric answer.
Forward uses BF16; master weights and AdamW states remain FP32. Backward runs inside
the BF16 parameter substitution scope so checkpoint recomputation uses the same weights.
Fixed settings: seed 7, gradient clipping 1.0, gradient checkpointing, and evaluation every 5 updates.
Overfit success: accuracy 100% and loss < 0.01.

`--think-tokens`, `--answer-tokens`, `--probe-period`, `--memory-ratio`,
`--max-seq-len`, and `--no-cuda-graphs` control generation resources. Smaller budgets
can truncate the teacher answer or prevent overlapping streams; they do not relax acceptance.
For execution smoke tests, each stage is also available separately:

```bash
uv run --no-sync python -m minisgl_examples.training.distillation.sanity record \
  Qwen/Qwen3-0.6B training_runs/distillation-smoke/teacher \
  --think-tokens 8 --answer-tokens 8 --no-cuda-graphs
uv run --no-sync python -m minisgl_examples.training.distillation.sanity train \
  training_runs/distillation-smoke/teacher/trace.json Qwen/Qwen3-0.6B \
  training_runs/distillation-smoke/trained --steps 1
uv run --no-sync python -m minisgl_examples.training.distillation.sanity verify \
  training_runs/distillation-smoke/teacher/trace.json \
  training_runs/distillation-smoke/trained/checkpoint \
  training_runs/distillation-smoke/reloaded.json
```

These stage commands test execution; only the complete workflow checks teacher correctness
before training and requires overfit and checkpoint agreement before independent generation.

The final cell saves `checks.json` with independent results:

1. Boxed answer correctness against the reference value 391.
2. Structural async behavior: valid cache dependencies, writer activity before the last
   thinker samples, probe requests before writer activity, and a growing thinker context
   visible to the writer. These checks do not assess the optimality of each `yes/no` decision.
3. Exact equality of tokens, targets, and cache history after renaming block IDs.

The third check also compares scheduling: identical generated tokens can still have different
batch ordering and cache views.

Read `sanity.py` in execution order:

- `run_process`, `record_rollout` — run the original demo and record a trace.
- `load_student`, `train_from_trace`, `evaluate_student` — load the model and run the training loop.
- `TrainingStep`, `TraceEvaluation`, `run_bf16` — small precision adapters for checkpointing.
- `check_answer`, `check_async_behavior`, `normalize_trace`, `compare_rollouts` — check the results.

Only dense text Qwen3 is supported. The training counter directly counts completed
optimizer updates. Schedule checks use original token rows; trace comparison uses
straightforward block ID renaming.
