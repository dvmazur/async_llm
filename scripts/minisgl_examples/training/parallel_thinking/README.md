# Parallel thinking

Alice and Bob solve a problem using shared cache blocks. Completed steps are
published to their common history, then each worker clears its private block.
Synchronized and independent schedules share the same collection and HF student
training code.

Install the examples package with `uv pip install -e './scripts[notebooks]'` after
setting up the root environment. Run the package or open either existing notebook
with the project environment as its kernel:

- `train_small_model.ipynb`: synchronized collection, training, checkpoint reload
  and independent student generation.
- `no_barrier.ipynb`: independent workers, delayed Bob, training, then normal and
  delayed student generation.

`demo.py` exposes collection, training, verification and workflow entry points;
`rollout.py` contains the prompts and cache schedule. Both demos use the shared
`training.student` and `training.process` modules.

```bash
uv run --no-sync python -m minisgl_examples.training.parallel_thinking run \
  --teacher Qwen/Qwen3-4B --student Qwen/Qwen3-1.7B \
  --output training_runs/parallel-thinking
```

Add `--mode no-barrier --bob-delay 0.05 --delay-every 2` for independent progress.
For a short execution check, use `--max-steps 4 --answer-steps 2 --steps 1
--no-cuda-graphs`. Generated answers are not guaranteed; inspect the teacher
trajectory before accepting it as supervision.

Collection can feed training immediately or be saved for a later run:

```python
from minisgl_examples.training.parallel_thinking.demo import collect_trace, train_student
from minisgl_examples.training.process import isolated
from minisgl_examples.training.student import save_example

example = isolated(collect_trace, "Qwen/Qwen3-4B", max_steps=384)
save_example(example, "training_runs/teacher.json")
report = isolated(train_student, "Qwen/Qwen3-1.7B", example, "training_runs/checkpoint")
# The same training function also accepts "training_runs/teacher.json".
```

Separate CLI stages use the same functions:

```bash
uv run --no-sync python -m minisgl_examples.training.parallel_thinking record \
  Qwen/Qwen3-4B training_runs/teacher
uv run --no-sync python -m minisgl_examples.training.parallel_thinking train \
  training_runs/teacher/trace.json Qwen/Qwen3-1.7B training_runs/checkpoint
uv run --no-sync python -m minisgl_examples.training.parallel_thinking verify \
  training_runs/teacher/trace.json training_runs/checkpoint training_runs/reloaded.json
uv run --no-sync python -m minisgl_examples.training.parallel_thinking run \
  --trace training_runs/teacher/trace.json --student Qwen/Qwen3-1.7B \
  --output training_runs/from-saved-trace
```

Use fresh output paths. The complete workflow saves `trace.json`, `checkpoint/`
and `report.json`, and reuses the recorded prompt for independent student generation.
Teacher and student tokenizers must agree. `RecordedAsyncLLM` captures the executed
cache views; sampled targets are aligned to trace rows, with prefills and flushes masked.

Training uses `replay_trace` and a chunked vocabulary loss. BF16 master weights and
AdamW state stay on CPU, forward/backward run on the selected accelerator, and
checkpoint reload must reproduce the selected replay logits exactly. Collection,
training and independent generation run in fresh processes to release model memory
between stages. The original `demo` module command and notebook `solve` /
`train_student` calls remain available.
