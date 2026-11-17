# Streaming video overfit

Run `video_understanding_overfit.ipynb` on caradhras with the project Python
kernel. It records a genuine asynchronous sampled-frame stream, validates
training replay, updates a small student, reloads its checkpoint, and compares
free-running thinker tokens, gate actions, answer text and response timing.

Change `RUN` in the setup cell to a new path to rerun inference and training.
Existing files in the selected run directory are explicitly reused; stages never
silently overwrite results. All figures are embedded as PNG, including when the
kernel uses the Agg backend.

## Verified source interfaces

This is a separate worktree, branch `codex/video-understanding-overfit`, based on
`training-internal-refactor` at `33136eddd5d3b5f646ec88f5602baeae1a432cf0`.
The image-steering worktree and main checkout are unchanged.

The supplied demo at https://github.com/vovanbart/AsyncLLM-demo was inspected at
`967a9074409566f2356aad98c9657ce0c6223521` (its only public branch is main).
That checkout contains a nuScenes frame source, rolling previous/current image
pairs, a persistent thinker, independent yes/no gate and a writer branch. It
uses older direct `block.clear()` calls; they do not record the cache mutation.
No SoccerNet or ProactiveVideoQA adapter is present in that public checkout.

Here every cache operation uses the current public `RecordedAsyncLLM` wrapper.
Observation changes use `free_block` and `prefill_block`, retaining thinker KV.
At 16 emitted thinker tokens per observation, the gate decides autonomously
whether to answer. The gate reads `[prompt, observation, gate]`; the thinker reads
`[prompt, observation, thinker]`; the writer reads
`[prompt, observation, thinker, writer]`. The first yes starts free answer
generation, ending at EOS or its token cap. Gate decisions are model outputs,
not an externally replayed teacher schedule.

The demo's continuous observer masks end-of-turn/control tokens; this notebook
does the same. Gate selection takes the highest raw logit over one-token
spellings of yes/no. All selected targets retain their original, unmasked,
full-vocabulary logprobs. Raw argmax accuracy and policy reproduction therefore
measure different things and are reported separately.

This is a stream of images representing video, not a native `pixel_values_videos`
request. Current replay explicitly does not support native video tensors. Each
rolling pair is encoded as two images, exactly the input representation in the
demo. Trace format v2 retains both images' pixels, grids, token types, MRoPE
coordinates, rotary spans, ordered visible cache lengths and block clears.
The existing differentiable forward reconstructs attention and GDN history and
re-encodes the saved images. It never flattens the trace into a conventional
causal sequence. No library changes or custom kernels are needed.

## One real dataset case

The task is ProactiveVideoQA/VAD, `Explosion016_x264.mp4`, dataset revision
`7f59bce2a6cdf40b8a895767aac7a52a92136c5f`:
https://huggingface.co/datasets/wangyueqian/ProactiveVideoQA

The source clip is 32.1 seconds, 4,523,207 bytes. It was extracted as a single
member from the public ZIP via HTTP ranges, not by downloading the full dataset.
`prepare.py` reproduces this optional CPU-only preparation using `remotezip`
and ffmpeg. The notebook itself needs only the bundled PNG frames and manifest.
Original video/annotation provenance and SHA256 hashes are in `assets/task.json`.
The upstream video belongs to its original source; retain provenance and use
these research-evaluation artifacts consistently with upstream terms.

The original question asks about suspicious/harmful activities or destructive
accidents. Frames at 2, 5, 12 and 14 seconds are supplied in order; the selected
reference reply window is [6, 16] seconds. The first sampled visible event
occurs at frame 2, 12 seconds. The expected gate sequence is no/no/yes.
Question and frame timestamps enter inference. Filenames, reference answer,
reply windows, source labels and expected gate timing do not.

Content checks require smoke and a building. Fire/explosion language is reported
separately. This deliberately avoids claiming that a two-fact check is a general
semantic judge. Caption text, thinker stages and the original annotation remain
visible for manual inspection. This small diagnostic does not calculate aggregate
PAUC or evaluate an entire video at full frame rate.

SoccerNet-Caption was investigated as the other mentioned source. Its official
raw-video distribution requires accepted access terms for protected videos:
https://www.soccer-net.org/data . No SoccerNet video was present in this account,
and this deliverable does not claim a SoccerNet run. It focuses on one accessible
ProactiveVideoQA trajectory rather than making a synthetic replacement dataset.

## Training and checks

Defaults: Qwen3.5-4B teacher, Qwen3.5-0.8B student, greedy decoding, seed 7,
16 thinker tokens/frame, 96 writer-token cap, one GPU, FP16 inference, no CUDA
graphs. Training uses FP32 language weights, a frozen vision tower, AdamW with
LR 3e-6, clipping and at most 80 steps. Real parameter deltas and finite nonzero
gradients are mandatory. Stop early at target accuracy 100% and full teacher
sequence probability at least 0.95.

Shared code is imported from existing `training.student.load_student`,
`training.loss`, `training.recording.record_sample`, `training.process.isolated`,
and the public recorder/replay API. This folder contains only the task-specific
stream, inspection/metrics and a short optimization loop.

Before training, compare teacher inference to teacher replay and the baseline
student's own inference to student replay. The maximum target-logprob error
must not exceed 0.1. Comparing teacher probabilities directly against a student
with different weights would be invalid. Results are grouped by frame and role.
User/image/prefix/flush rows use loss target -100. Generated thinker, gate and
writer targets belong to the forward rows that actually produced them.

Evaluation reloads the saved HF checkpoint into real AsyncLLM and supplies only
external frames. The no-event control freezes actual input pixels at the last
pre-event frame, while preserving timestamps and update schedule. It must keep
waiting through all frames. Comparison verifies the processed tensors against
known pre-event images, not only a metadata flag. This catches a student that
memorized a response at a fixed frame index.

## Server and local workflow

Local source:
`/Users/warri/codex-remote/video-understanding-overfit`.
Server source:
`/home/mrozaeva/experiments/video-understanding-overfit/repo`.
The server `.venv` is a symlink to the compatible image-steering environment;
that environment was reused without editing its packages. `run_remote.sh` and
the notebook put this worktree's python/scripts directories first on PYTHONPATH,
including in isolated subprocesses. Installed versions are torch 2.13.0+cu130
and transformers 5.14.1. Do not run `uv sync` against the shared environment.

Sync local changes without deleting server data or copying weights/caches:

```bash
rsync -az --exclude=.git --exclude=.venv --exclude=__pycache__ \
  --exclude=training_runs --exclude=uv.lock --exclude='/assets' \
  /Users/warri/codex-remote/video-understanding-overfit/ \
  caradhras:experiments/video-understanding-overfit/repo/
```

Select the `.venv/bin/python` kernel in this server worktree. Before a GPU run,
inspect `nvidia-smi` and use a free GPU with CUDA_VISIBLE_DEVICES. Never stop
another user's processes. A standalone training repeat on the saved teacher:

```bash
cd /home/mrozaeva/experiments/video-understanding-overfit/repo
nvidia-smi
CUDA_VISIBLE_DEVICES=0 bash scripts/minisgl_examples/training/video_understanding/run_remote.sh train \
  --teacher training_runs/video_understanding/teacher.json \
  --student Qwen/Qwen3.5-0.8B \
  --baseline training_runs/video_understanding/student_before.json \
  --output training_runs/video_understanding/manual_train --steps 80 --lr 0.000003
```

`manual_train` must be new. For free-running before/after and the visual control,
execute the notebook. Stage CLI commands `collect`, `check`, and `train` also
support `--help`. All results/logs go under `training_runs/video_understanding`.
Only diagnostic artifacts are copied back locally; large checkpoint weights
remain on the server. Actual metrics and limitations are in `RUN_REPORT.md`.
