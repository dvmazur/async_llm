# Parallel thinking: one-trajectory training

`train_small_model.ipynb` restores the Alice/Bob cube example on a 16 GiB GPU.
It records a fresh teacher rollout, checks two untrained baselines, trains on that
one recorded trajectory, reloads the checkpoint, and compares independent student
generation. Defaults: **Qwen3-4B-Thinking-2507 teacher, Qwen3-1.7B student**.

This is a fresh run of the previous experiment, not a copy of its QwQ-32B training
artifacts. The old artifacts remained on pretrain, which was unavailable during
restoration. No weights, responses or metrics from that experiment are fabricated.

## Setup

From the repository root on Linux x86_64, with `uv` installed and CUDA available:

```bash
bash scripts/parallel_thinking/setup_env.sh
.venv/bin/hf download Qwen/Qwen3-4B-Thinking-2507 --local-dir models/Qwen3-4B-Thinking-2507
.venv/bin/hf download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B
```

The setup script creates a Python 3.12 uv environment and installs the repository's
locked dependencies. It uses the original official PyTorch CDN because the R2
mirror is unreachable from imladris. Versions and the repository lockfile stay
unchanged; Transformers 5.14.1 matches the previous notebook environment. Use
`uv run --no-sync` or `.venv/bin/python` afterward: plain `uv sync` selects R2 again.

Choose a free Ampere-or-newer GPU with at least 14,000 MiB free (check
`nvidia-smi`). The notebook defaults to physical GPU 2 on imladris. Change
`PARALLEL_GPU` before starting the kernel. Model paths can be changed via
`PARALLEL_TEACHER` and `PARALLEL_STUDENT` or the setup cell. Use the **AsyncLLM (uv)**
kernel and Run All. Do not select an 8 GiB Quadro RTX 4000 for this BF16 recipe.

To access Jupyter from your computer:

```bash
ssh -L 8890:127.0.0.1:8890 imladris
cd ~/async_llm
PARALLEL_GPU=2 .venv/bin/jupyter lab --no-browser --ip=127.0.0.1 --port=8890
```

Open the localhost URL printed by Jupyter, including its authentication token.
The server remains bound to loopback. Alternatively open the repository via
VS Code Remote SSH and select `.venv/bin/python` as the notebook kernel.

The notebook resolves the selected physical GPU index to its UUID before launching
children. CUDA ordinal ordering differs from nvidia-smi on this mixed-GPU host.
The helper sets `TVM_FFI_CUDA_ARCH_LIST` from the selected physical GPU before
CUDA initialization. This is needed on mixed-GPU machines: TVM FFI otherwise
probes GPU 0, ignoring `CUDA_VISIBLE_DEVICES`.

## What the experiment checks

- The native-prompt baseline checks that the untrained student can solve the task.
  A separate matched-prompt baseline uses the exact teacher prompt and is reported
  even if wrong. Only that matched baseline is used for the before/after comparison.
  No answer or solution is inserted into any prompt.
- Both workers read the other's unfinished block. Two concurrent one-token
  requests are followed by a round barrier; this is not independent-speed execution.
- Finished blocks are flushed before being moved into common history; merges on
  ties follow Alice/Bob order. No periodic `insert_s1_prompt_every_tokens` is used.
- `DemoRecorder` from `scripts/async_thoughts/async_thoughts/training.py` records
  actual batches, visible cache lengths, copies and clears. Targets belong to
  the decode row that predicted them; other rows are masked with `-100`.
- RoPE frequency buffers remain FP32; both training and checkpoint reload assert this.
- `python/minisgl/training/replay.py` reconstructs visibility and block-relative
  RoPE. `supervised_loss` and `evaluate` are imported from
  `python/minisgl/training/sft.py` in **this same checkout**.
- FP32 master weights, copied gradients and AdamW states live on CPU. BF16 compute
  weights and forward/backward live on GPU. After clipping and a CPU optimizer
  update, masters are copied back to GPU. The checkpoint stores FP32 master values.
  This fits the 1.7B student on 16 GiB and requires roughly 35 GiB of host RAM.
  Unlike PR #6's GPU `functional_call` recipe, it trades transfer/CPU time for VRAM.
- Query and loss chunks of 64 reduce workspace, without truncating the trace.
  Full-weight AdamW uses lr 3e-5, seed 7 and gradient clipping at 1.0.
- A fresh process reloads the checkpoint and evaluates the fixed trace. Another
  process generates independently with the teacher's prompt but no target tokens.
  The comparison reports answer correctness, text, token IDs and ordered events
  `(round, worker, step, tokens)` separately. It does not compare internal scheduler
  batches or wall-clock timing, and does not establish generalization.

Every run writes logs, teacher trace, results, metrics and checkpoint to a fresh
`training_runs/parallel_thinking/<UTC timestamp>/` directory. The teacher and
native-prompt student must answer correctly before training. The matched-prompt
baseline is an observation, not a gate; its failures remain in the final comparison. Training must reach 100%
token accuracy and mean CE < 0.01. A failed gate is left visible for inspection.

## Files and provenance

- `train_small_model.ipynb`: the complete experiment.
- `setup_env.sh`: reproducible environment setup for Linux x86_64.
- `parallel.py`: the same shared-cache solver, training and CLI helpers used by the notebook.
- `hogwild_formatting.py`, `HOGWILD_LICENSE`: original formatter and Apache-2.0 license.

The solver was ported from the earlier pretrain notebook. Original task and
formatting: [Hogwild full_example](https://github.com/eqimp/hogwild_llm/blob/aad3712a8c36144a5bc5c4948155f7df5ef7b6f4/full_example.ipynb).
Copyright 2025 Hogwild! Inference paper authors. Layout follows
[async distillation PR #6](https://github.com/dvmazur/async_llm/pull/6).

Qwen3-0.6B was also tested during restoration: its untrained rollout returned 144,
so it failed the baseline gate and was not used as the default student.

The original Qwen3-4B teacher was tested too and returned 588 on this schedule.
Its incorrect trajectory was rejected before training.

The Thinking-2507 native prompt adds `<think>` and the untrained 1.7B returned 588
on that prompt. Forcing the student template on the teacher instead made the teacher
return 125. The notebook therefore reports the native capability check and matched
before/after comparison separately; it never labels the native baseline as a
successful matched-prompt baseline.

## Verified run on imladris (2026-11-01)

Run: `training_runs/parallel_thinking/20261101T115331740144Z/`.
GPU: NVIDIA RTX A4000, physical index 2 (selected explicitly by UUID in the final setup).
Python 3.12.16, Torch 2.13.0+cu130, Transformers 5.14.1, FlashInfer 0.6.17.

| Check | Result |
| --- | --- |
| Thinking-2507 teacher | Answer 294; 6 published steps |
| Untrained 1.7B, native prompt | Answer 294; 20 published steps |
| Untrained 1.7B, teacher prompt | Answer 588; 48 published steps |
| Full-weight training | 15 updates; CE 0.169990 → 0.000210809; accuracy 95.03% → 100% |
| Reloaded checkpoint | Identical CE and 100% accuracy |
| Trained 1.7B, teacher prompt | Answer 294; exact transcript, token IDs and all 6 publication events |
| GPU UUID selection check | Repeated rollout identical to the verified student rollout |

Training took 571.7 seconds; peak allocated GPU memory was 7.06 GiB.
The checkpoint contains FP32 master weights; inference uses BF16.
All code cells completed successfully. The only subsequent notebook code change
resolves the same physical GPU by UUID; its selection and repeated rollout were
verified separately. Outputs retain the original CUDA-ordinal display, explained
in the notebook introduction. This is memorization of one example, not evidence
of generalization or robustness to different asynchronous schedules.
