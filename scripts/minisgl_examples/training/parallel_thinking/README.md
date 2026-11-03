# Parallel thinking

Alice and Bob solve a problem using shared cache blocks. Completed steps are
appended to the common history, then each worker clears its private block and
starts a new step. Both notebooks call the shared `demo.py` implementation:

- `train_small_model.ipynb`: synchronized generation, teacher/student training,
  checkpoint save/reload and free student generation.
- `no_barrier.ipynb`: independent worker loops, delayed Bob, the same training
  workflow, then free generation with normal and delayed schedules. An exclusive
  publication gate protects cache mutations while allowing concurrent forwards.

Run `bash scripts/minisgl_examples/training/parallel_thinking/setup_env.sh`, select **AsyncLLM (uv)** and open
one notebook. This uses the repository's environment and preserves the caller's
accelerator selection. Teacher and student must use the same tokenizer vocabulary.
The defaults are Qwen3-4B and Qwen3-1.7B; model IDs or local checkpoint paths can be
changed. Generated answers are not guaranteed to be correct; inspect the teacher
trajectory before using it as supervision.

From the repository root, run the entire workflow without a notebook:

```bash
python -m minisgl_examples.training.parallel_thinking.demo \
  --teacher Qwen/Qwen3-4B --student Qwen/Qwen3-1.7B \
  --output training_runs/parallel-thinking
```

For a short smoke run, use `--teacher Qwen/Qwen3.5-0.8B
--student Qwen/Qwen3.5-0.8B --max-steps 4 --answer-steps 2 --train-steps 1
--no-cuda-graphs`. To exercise independent progress, add `--mode no-barrier
--bob-delay 0.05 --delay-every 2`. Each run needs a new output directory.

The run saves `trace.json`, `checkpoint/`, and `report.json`. `RecordedAsyncLLM`
records actions, while shared functions associate sampled targets with input rows.
Replay uses `Trace.to` and `replay_trace`; the reusable chunked loss lives in
`minisgl_examples.training.loss`. Training keeps BF16 master weights and AdamW state on
CPU, performs forward/backward on the accelerator, and checks exact replay logits
after saving and reloading the checkpoint. Collection, training and free generation
run in separate processes so each inference engine starts with a fresh CUDA context.
