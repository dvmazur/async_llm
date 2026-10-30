# Sanity check: distillation on one example

This folder contains **three files**:

- `simple_distillation.ipynb` — six short cells to run from top to bottom.
- `sanity.py` — model launching, training on one trace, and three checks.
- `README.md` — this guide.

Task: **17 × 23 = 391**. First, the 4B teacher generates a rollout and must produce
one numeric `\boxed{391}` answer. Then train the 0.6B student and verify the saved
checkpoint. Finally, let the student solve the task independently.
An incorrect or truncated teacher answer stops the experiment before training.

Open the notebook and select an idle GPU in the first cell.
At least 12000 MiB of free memory is required.
Results are saved in a new `training_runs/async_distillation_sanity_check/simple_multiply_*` directory.
Each model runs in a separate process and releases its GPU memory when it exits.

Training uses the existing `Trace`, `Replay`, `supervised_loss`, and `evaluate` from
`minisgl.training`. Generation uses the original `async_thoughts` without modifications.

The demo supports one text-only Qwen3 trace and one numeric answer.
Forward uses BF16; master weights and AdamW states remain FP32. Backward runs inside
the BF16 parameter substitution scope so checkpoint recomputation uses the same weights.
Fixed settings: seed 7, gradient clipping 1.0, gradient checkpointing, and evaluation every 5 updates.
Overfit success: accuracy 100% and loss < 0.01.

The final cell saves `checks.json` with independent results:

1. Boxed answer correctness against the reference value 391.
2. Structural async behavior: valid cache dependencies, writer activity before the last
   thinker samples, probe requests before writer activity, and a growing thinker context
   visible to the writer. These checks do not assess the optimality of each `yes/no` decision.
3. Exact equality of tokens, targets, and cache history after renaming block IDs.

Read `sanity.py` in execution order:

- `run_process`, `record_rollout` — run the original demo and record a trace.
- `load_student`, `train_from_trace`, `evaluate_student` — load the model and run the training loop.
- `TrainingStep`, `TraceEvaluation`, `run_bf16` — small precision adapters for checkpointing.
- `check_answer`, `check_async_behavior`, `normalize_trace`, `compare_rollouts` — check the results.

Only dense text Qwen3 is supported. The training counter directly counts completed
optimizer updates. Schedule checks use original token rows; trace comparison uses
straightforward block ID renaming.
