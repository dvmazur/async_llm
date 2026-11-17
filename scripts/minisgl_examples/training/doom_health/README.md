# Doom Health Gathering H+Delta overfit

Open `doom_health_overfit.ipynb` from this worktree and run top to bottom. This is a
short, closed-loop health-recovery experiment, not a full-episode survival benchmark.
**Real GPU trajectory reproduction passed; visual dependence was not established.**
The blank-image control reproduced the same short run. See `RUN_REPORT.md`.

Source: [Health Gathering H+Delta gist](https://gist.github.com/timka-byzov/3cc4a875991b897eec6131b14869b49e),
revision `a9488564436610d5e95b6ba0450113d86f968e2d`. My Way Home was not used.
Repository base: `origin/training-internal-refactor`,
`33136eddd5d3b5f646ec88f5602baeae1a432cf0`.

## What is reproduced

H is consolidated text KV history; Delta is recent text KV history. At each action,
the agent receives the previous/current real game frames. Descriptor and thinker
generate concurrently. Actor checkpoints read stable cache prefixes, while newly
generated tails are excluded from that checkpoint's view. A `continue` checkpoint
does not advance the game. The chosen game action advances ViZDoom by four tics.
Every three actions Delta is appended to H and cleared. The source used 100 actions
between consolidations; three makes append/clear/read visible in a short overfit run.

Teacher: Qwen3.5-4B; student: Qwen3.5-0.8B; deterministic greedy selection; seed 105;
at most 12 game actions. A successful teacher must recover health, remain alive,
consolidate Delta, and subsequently read the enlarged H. If 4B fails, the notebook
stops before training: a larger teacher or a different explicit setup must then be
tested, with fresh artifact paths. The original notebook default was a 27B model;
4B recovered health in the recorded seven-action run.

The real `RecordedAsyncLLM` saves views, block append/clear operations, image tensors,
MRoPE positions, and inference inputs. The small adapter records loss targets only
for emitted descriptor/thinker tokens and consumed actor decisions. Unconsumed
lookahead predictions and all input tokens stay masked. Replay uses the existing
multimodal training forward; it does not flatten this into causal text training.
Raw, unconstrained target logprobs must match inference within 0.1 maximum absolute
error before training. Per-step/per-role errors are saved. Actor choice is constrained
to action token IDs, but parity compares its original unmasked distribution.

Training uses the existing `train_student`: full parameter updates including vision,
FP32, learning rate 3e-6, up to 80 steps. Small parameter deltas and reloaded loss are
checked. Both student evaluations run the same live agent; **their own actions drive
their own environment**. Descriptions, thoughts, history summaries, and frames are
generated afresh, never copied from the teacher. Compare role token sequences,
checkpoint schedules, actions, health, and exact frame hashes. Action/physics
reproduction and exact token reproduction are reported separately.

## caradhras

Repository: `/home/mrozaeva/experiments/doom-health-overfit/repo`.
Additional dependencies: sibling `deps/`, isolated from the shared venv. `.venv`
points to the existing image experiment environment; no packages were changed there.
The extra packages are vizdoom 1.3.0, gymnasium 1.2.3, pygame 2.6.1,
cloudpickle 3.1.2, farama-notifications 0.0.6.

Check `nvidia-smi` first. Following the user's clarification, the notebook permits
shared use with `DOOM_ALLOW_SHARED_GPU=1`: each stage requires 48 GiB free (32 GiB
allocator budget plus 16 GiB reserve). Our allocator is capped at 32 GiB; native
inference sets the cap after Engine construction, whose model and KV allocation is
already explicit. An idle device is also accepted. Each stage checks again. Restart
the kernel after changing GPU visibility. Existing artifacts are reused;
use a fresh RUN directory when changing models, seed, prompts, or settings.

From the repository root, run individual GPU stages as follows, replacing `1` with
a physical device that has enough free VRAM. Use a fresh output directory when
rerunning; the saved successful experiment already occupies these artifact paths:

```bash
export CUDA_VISIBLE_DEVICES=1
export DOOM_ALLOW_SHARED_GPU=1 DOOM_GPU_LIMIT_GIB=32
bash scripts/minisgl_examples/training/doom_health/run_remote.sh collect \
  --model Qwen/Qwen3.5-4B --output training_runs/doom_health/teacher.json
bash scripts/minisgl_examples/training/doom_health/run_remote.sh check \
  --source training_runs/doom_health/teacher.json \
  --output training_runs/doom_health/teacher_parity.json --dtype float16
bash scripts/minisgl_examples/training/doom_health/run_remote.sh collect \
  --model Qwen/Qwen3.5-0.8B --output training_runs/doom_health/student_before.json
bash scripts/minisgl_examples/training/doom_health/run_remote.sh train \
  --teacher training_runs/doom_health/teacher.json --student Qwen/Qwen3.5-0.8B \
  --baseline training_runs/doom_health/student_before.json --output training_runs/doom_health/trained
```

Use the notebook for checkpoint reload, free-running evaluation, plots, and comparison.
It also contains a real black-frame visual-input control. In the saved experiment
this control produced identical tokens and actions, so visual dependence was not
demonstrated despite exact free-running trajectory reproduction.
To run CPU checks without touching a GPU:

```bash
export PYTHONPATH="$PWD/python:$PWD/scripts:$PWD/../deps"
export CUDA_VISIBLE_DEVICES="" SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
.venv/bin/python -m pytest -o addopts='' -q tests/scripts/test_doom_health.py tests/scripts/test_doom_health_scheduler.py
.venv/bin/python -m minisgl_examples.training.doom_health.preflight --help
```

Artifacts live in `training_runs/doom_health`: CPU environment/processor diagnostics,
then (only after actual GPU runs) teacher/before/after JSON traces with frame folders,
parity results, training/checkpoint outputs, and comparison. PNG notebook displays are
embedded, so reading results does not depend on a browser image renderer.
