# Real-time evaluation protocol (version 3)

This replaces the synchronous September 15 campaign. Old results are preserved
but are not pooled with these experiments.

## Environment and execution

All game entry points use native ViZDoom `ASYNC_PLAYER`, with a dedicated thread
refreshing observations, rewards, and terminal state during model inference.
Normal speed is 35 game tics/second. Global options are `SEA_GAME_TICRATE` and
`SEA_INFERENCE_ACTION` (`wait`, the default, or `hold_last`). In wait mode a
completed action lasts four tics, then no buttons are pressed until another
answer arrives. In hold-last mode that action remains pressed during inference.

Keep the repo's configured native limits: Defend the Line 1,000 tics, Health
Gathering 10,000 tics, My Way Home 2,100 tics. Decision-count limits are ignored
for real-time games. Death or timeout cancels the pending decision. The one
already submitted GPU request is drained before freeing its cache; its answer
is discarded. Game duration and cleanup duration are recorded separately.
Living rewards are integrated over elapsed native game tics, including tics
that advance autonomously between polls. Terminal penalties remain native.

Use dalaran GPUs **1, 5 and 6**, one model per GPU, `HF_HOME=/mnt/LLM`, and the
existing uv-managed Python environment. Qwen/Qwen3.8-27B BF16 is pinned to
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Vision/prefill/decode warm-up uses a
saved frame with its game already closed, outside scored episodes.

## Order and policies

1. **Minimal self-evolution:** 10 independent runs per environment, five valid
   evolution rounds per run, five episodes per round. Each run starts from
   pristine engine and minimal prompt seeds. Existing task descriptions remain
   available. Prompts now describe live time, cancellation and validity rules.
   Compile failures, no fresh evaluation, zero `LLM.forward()` calls, and runtime
   errors do not count as valid rounds and cannot update best-score tracking.
   A genuine inference that times out before returning any action remains valid.
   There is a 20-attempt safety cap; exhausting it is reported as incomplete,
   never as five successful rounds. Failed runs require inspection before resume.
2. **Baselines:** each condition gets fresh 10 runs × 5 episodes in each env:
   - **Full reasoning:** open `<think>`, maximum 16,384 generated tokens,
     including reasoning and answer. Early closing of thought and early answers
     are allowed; act only when a legal `\boxed{action}` appears after `</think>`.
   - **Direct logits:** closed `<think></think>`, append `Action:`, choose the
     highest-logit legal action token, without text generation.
   - **Thinking-disabled generation:** closed `<think></think>`, generate up to
     64, 128, 512 or 1,024 tokens, and wait for a legal `\boxed{action}`. Ordinary
     text may contain explanation or reasoning despite the closed think segment.
   - **Uniform random:** independently choose among all four legal actions,
     including wait, after each four-tic action; no model calls.
3. Review performance before deciding whether detailed self-evolution or prompt
   adjustments are needed. Do not silently change speed or waiting policy.

Text generation stops on a legal boxed answer, EOS, its token budget, or episode
termination. EOS/budget exhaustion without a legal answer produces `wait` and is
explicitly logged. Unfinished decisions at death/timeout are discarded. Every
attempt records actual token usage, text, stop reason, latency and cancellation.
Baselines have no history or persistent cache between decisions. Health Gathering
also receives the current health value. Sampling: temperature .7, top-k 20,
top-p .9, repetition penalty 1.0. Evolution retains its existing penalty 1.15.

## Replication and uncertainty

Game seeds are deterministic and paired across conditions: each run has five
distinct seeds, and the same run's seed set is reused across evolution rounds.
Evolution generation has an independent run seed. Baseline sampling is seeded
per episode. Timing and asynchronous scheduling can still vary between executions.

Report the mean of ten five-episode run means with a two-sided 95% Student-t CI:
`mean ± t(.975, 9) × sample_sd(run_means) / sqrt(10)`. The 50 episodes are not
50 independent run replicates. Evolution gets a separate result per valid round,
not a selected-best-round estimate. Partial reports state the completed-run
count and use its degrees of freedom; one completed run has no error bar.

There are 250 scored episodes per environment for the five evolution checkpoints
and 350 per environment for the seven baseline conditions. Invalid attempts may
add diagnostic episodes but cannot enter valid-round summaries. No old results
are reused. Error bars describe uncertainty in the mean, not individual rewards.

Episode files preserve seeds, rewards, elapsed game time, actions, forward-call
counts (evolution), and inference diagnostics. Baselines resume atomically saved
episodes; errors stop the campaign for investigation. Evolution saves attempted
rounds and valid-round numbers separately. Warm-up is excluded from forward-call
validity counts. `status.json` records the current phase. All 20 independent minimal runs must
finish before any baseline starts; random can run alongside the GPU baselines.

## Run and outputs

```bash
source /home/yakushev-ga/Projects/Doom/.venv/bin/activate
export HF_HOME=/mnt/LLM CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PWD/python"
python -u scripts/self_evolving_agent/run_async_campaign.py \
  --output scripts/self_evolving_agent/eval_runs/async_campaign_20260916 --gpus 5 6 1
```

`evolution_summary.csv` / `evolution_report.md` summarize each valid round.
Each baseline directory contains episode JSON, config, source hashes, package
versions, `summary.csv`, `report.md`, and a 95% CI plot once enough runs finish.
`baseline_summary.csv` combines all baseline conditions after completion.
The earlier `run_baseline_campaign.py` remains a legacy wider-budget launcher;
use `run_async_campaign.py` for this revised sequence and condition set.

## Actions per forward

`action_efficiency.csv` and `action_efficiency.md` report completed environment
actions divided by `LLM.forward()` API calls, including both prefills and decodes.
Each run's ratio uses totals over its five episodes; report the mean ratio and
95% Student-t CI across runs. Raw action and forward totals are also shown.
Calls made during decisions cancelled at episode end count in the denominator;
evolution/code generation and warm-up do not. This is calls, not tokens or GPU
batch launches. Random policy has zero forwards and an undefined ratio (N/A).
The live reporter recovers existing evolution counts without restarting workers:

```bash
python scripts/self_evolving_agent/action_efficiency.py \
  --root scripts/self_evolving_agent/eval_runs/async_campaign_20260916 --watch
```

GPU workers claim entire evolution runs with file locks. The launcher can adopt
existing worker PIDs during a supervisor handover, preserving live engines and
in-progress generations. GPU 1 was added after the initial two-GPU launch.
