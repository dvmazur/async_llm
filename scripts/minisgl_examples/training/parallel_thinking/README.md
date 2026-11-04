# Parallel thinking

Alice and Bob solve a problem through shared cache blocks. Completed steps are
published to their common history; each worker then clears its private block.
An HF student learns the recorded targets and generates its own solution.
Use the [shared setup and Python API](../README.md).

```bash
uv run --no-sync python -m minisgl_examples.training.parallel_thinking run \
  --teacher Qwen/Qwen3-4B --student Qwen/Qwen3-1.7B \
  --output training_runs/parallel-thinking
```

Add `--mode no-barrier --bob-delay 0.05 --delay-every 2` for independent workers,
or `--trace path/to/trace.json` to reuse a saved teacher example. The recorded
prompt is reused for student generation. Output paths must be new.

Run individual stages:

```bash
uv run --no-sync python -m minisgl_examples.training.parallel_thinking record \
  Qwen/Qwen3-4B training_runs/teacher
uv run --no-sync python -m minisgl_examples.training.parallel_thinking train \
  training_runs/teacher/trace.json Qwen/Qwen3-1.7B training_runs/checkpoint
uv run --no-sync python -m minisgl_examples.training.parallel_thinking verify \
  training_runs/teacher/trace.json training_runs/checkpoint training_runs/reloaded.json
```

Defaults: 384 tokens per worker, 64 answer tokens, up to 80 optimizer steps, and
learning rate `3e-5`. For a short execution check, add `--max-steps 4 --answer-steps 2
--steps 1 --no-cuda-graphs` to `run`. Use each command's `--help` for resource controls.

Teacher and student tokenizers must agree. BF16 forward/backward runs on the
accelerator; BF16 master weights and AdamW state stay on CPU. Checkpoint reload
must reproduce selected replay logits exactly. `report.json` contains the teacher
trajectory, training metrics and independent student output. Inspect answer quality
before accepting a trajectory as supervision.

The notebooks retain the original experiments: `train_small_model.ipynb` uses
synchronized workers; `no_barrier.ipynb` compares normal and delayed Bob. Select
the project environment as their kernel. Both call the same Python functions.

`demo.py` contains the workflow and CLI; `rollout.py` contains the cache schedule
and prompts.
