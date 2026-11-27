# Streaming video overfit

A Qwen3.5-4B teacher watches rolling previous/current frame pairs. A persistent
thinker reasons about the stream; a yes/no gate decides when a writer should
answer. Qwen3.5-0.8B learns the recorded trajectory and repeats the task independently.
Use the [shared setup](../README.md), then install the example dependencies:

```bash
uv pip install -e './scripts[notebooks,video-understanding]' 'transformers==5.14.1'
```

Open `video_understanding_overfit.ipynb` with the project environment as its kernel.
Select one GPU with `CUDA_VISIBLE_DEVICES`. Outputs go to
`training_runs/video_understanding`; set `OVERFIT_RUN` to use a fresh directory.
The notebook includes executed outputs and embedded plots.

`prepare.py` downloads the annotation and one clip from the pinned
[ProactiveVideoQA](https://huggingface.co/datasets/wangyueqian/ProactiveVideoQA)
VAD archive using HTTP ranges. It verifies the video SHA256 and extracts frames
at 2, 5, 12, and 14 seconds. Downloads and decoded frames stay in the HF cache.
Only the question, frame pixels, and timestamps enter inference.

`experiment.py` records AsyncLLM requests and cache views, checks replay on the
teacher and baseline student, trains the student's language weights with the
vision tower frozen, and checks checkpoint reload. The trained student generates
its own thinker tokens, gate decisions, and answer. A frozen-frame control keeps
timestamps unchanged while repeating pre-event pixels.

Defaults: greedy decoding, seed 7, 16 thinker tokens per observation, a 96-token
writer limit, and up to 80 AdamW updates at `3e-6`. Replay uses `torch` attention
with query chunks of 64; settings are saved with each rollout and inherited by
training and evaluation. Cached reports must match those settings.

Run individual stages from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m minisgl_examples.training.video_understanding.experiment collect \
  --model Qwen/Qwen3.5-4B --output training_runs/video_understanding/teacher.json
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m minisgl_examples.training.video_understanding.experiment check \
  --source training_runs/video_understanding/teacher.json --dtype float16 \
  --output training_runs/video_understanding/teacher_parity.json
```

Use each command's `--help` for options. Output files cannot be overwritten.
This is a one-clip overfit check with sampled image pairs and task-specific
content/timing checks; it does not evaluate native video tensors or a full benchmark.
