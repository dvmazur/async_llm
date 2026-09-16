# Fresh baseline budget sweep protocol

## Conditions

All conditions use Qwen/Qwen3.8-27B in BF16, one current screenshot and the
existing task description. Health Gathering also receives its current health.
There is no observation history, self-editing, or cache reuse between decisions.
The checkpoint is pinned to revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

- **Non-reasoning (0):** chat template explicitly disables thinking. Append
  `Action:` and select the highest-logit legal action token.
- **Budgeted reasoning:** enable thinking, with the same `medium` reasoning-effort
  template and task prompt at all budgets. Sample up to 64, 128, 512, 1024, 4096,
  8192, or 16384 tokens, stopping early on `</think>` or EOS. Close the thinking
  segment, append `Action:`, and select the highest-logit legal action token.
  The decision uses the generated reasoning in its cache. Budget counts sampled
  reasoning tokens (including a sampled stopping token), excludes input/cue tokens
  and the final action selection. At the cap, the final sampled non-stop token is
  consumed before the decision cue. No action is extracted from unfinished prose.
- **Thinking-disabled generation (`no_think`):** the assistant prefix contains an
  already-closed `<think>\n\n</think>` segment. Sample ordinary answer text for up
  to 64, 128, 512, or 1024 tokens, stopping on EOS. Append `Action:` and choose
  the highest-logit legal action in the context of that generated answer.
  This differs from the direct zero-token baseline: it permits ordinary answer
  generation, but does not open a model thinking segment. The model can still
  explain its choice in ordinary text, so the name describes the template mode,
  not a guarantee that its answer contains no verbal reasoning.

All three paths use constrained action selection after their respective
prefix/generation, so arbitrary parsing fallbacks cannot drive the comparison.
The 16384-token limit is a maximum; early `</think>` or EOS is allowed.

Sampling uses temperature 0.7, top-k 20, top-p 0.9, with no repetition penalty.
Budgets are maxima; actual consumption and cap-exhaustion rates are reported.
The exact prompt is stored in the experiment configuration.

## Evaluation and uncertainty

Fresh 10 runs × 5 episodes for every condition in each environment: 1200 episodes
(one direct condition, seven reasoning budgets, four thinking-disabled budgets).
Defend the Line has a 100-step cap; Health Gathering has a 2500-step cap.
Frame skip and native episode timeouts retain the environment defaults.
Rewards and per-step execution use the existing `tasks.runner.run_episodes`.
Each run/episode has a distinct deterministic environment and sampling seed;
the same seeds are paired across budget conditions. Conditions are shuffled
within each run to distribute effects of execution order. Following the user's
updated GPU authorization, use physical GPUs 5 and 6 on dalaran via
`--gpus 5 6`; `HF_HOME=/mnt/LLM`. Each worker loads one model on its single
visible device. GPU 5 starts with reasoning and GPU 6 starts with
thinking-disabled generation. Each worker then helps with the other mode.
The launcher also accepts other explicitly authorized GPU lists; it does not
automatically select additional GPUs. OS file locks prevent two workers from evaluating the same
episode and automatically release on process exit. GPU identity is recorded
for each episode. Sampling seeds are reset per episode in each worker.

The plotted value is the mean of the 10 run means. Error bars are two-sided
95% Student-t confidence intervals: mean ± t(0.975, 9) × sample SD / sqrt(10).
The five episodes within a run do not count as five independent run replicates.
Intervals describe uncertainty in the mean, not the spread of individual rewards.
Partial reports explicitly show completed-run counts and use the corresponding
degrees of freedom. No existing baseline results contribute.

Episode reward, seed, steps, latency, token counts, stopping reasons, and actions
are written atomically after each episode. Resume skips completed episodes.
An inference exception is recorded and stops the sweep for investigation;
failed runs are never silently treated as valid zero-reward evaluations.
Result and report writes retry every 30 seconds on a full filesystem, keeping
computed episode data in memory instead of losing it to an output error. This
addresses the September 15 shared-disk failure; it does not protect against
process termination or host restarts before an episode is saved.
`capped_episodes` counts episodes reaching the harness limit, which can coincide
with native termination on that step.

The Python environment was created with `uv venv --python 3.12 .venv` and
`uv pip install --python .venv/bin/python -e . gymnasium vizdoom scipy matplotlib`.
Package versions and evaluation-source hashes are recorded with the results.

## Commands

```bash
# Full two-GPU campaign (activate .venv so ninja is available to CUDA JIT):
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME=/mnt/LLM
python -u scripts/self_evolving_agent/run_baseline_campaign.py \
  --output scripts/self_evolving_agent/eval_runs/baseline_campaign_20260915 --gpus 5 6

# Individual mode or report-only commands:
CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/LLM .venv/bin/python \
  scripts/self_evolving_agent/run_budget_sweep.py \
  --output scripts/self_evolving_agent/eval_runs/budget_sweep_20260915

.venv/bin/python scripts/self_evolving_agent/run_budget_sweep.py \
  --output scripts/self_evolving_agent/eval_runs/budget_sweep_20260915 --report-only

CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/LLM .venv/bin/python \
  scripts/self_evolving_agent/run_budget_sweep.py --mode no_think \
  --budgets 64 128 512 1024 \
  --output scripts/self_evolving_agent/eval_runs/no_think_sweep_20260915
```

Outputs include `config.json`, `environment_gpu*.json`, `packages_gpu*.txt`, episode JSON
records, `summary.csv`, `report.md`, and `reward_ci.png`. A run in progress is
not a completed evaluation. Final reports must have 10/10 valid runs for all
24 environment/condition combinations across the two mode directories.
The campaign supervisor writes a combined report, chart, status JSON, and one
log per GPU, and stops the campaign if a worker fails. The report names the
highest observed mean per environment only after all conditions finish; this
ranking does not claim statistically significant superiority after selection.
