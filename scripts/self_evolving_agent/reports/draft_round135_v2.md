# Round 1/3/5 eval, all-GPU batch (2026-09-12/13) — post best-score-regression-prompting fix

Batch: `run_repeat_batch_round135_allgpu.sh`, 12 cells (3 reps x {doom, health_gathering} x
{detailed, minimal}), 8-way GPU-parallel in 2 waves. Includes both fixes from
`reports/draft_round135.md`/`.claude/plans/fuzzy-snuggling-graham.md`: the circuit-breaker
pitfall added to both seed prompts, and the new "best score so far" regression-awareness signal
in `agent.py`'s `build_env_turn()`. `max_episodes=5` both envs, `TARGET_VALID_STEPS=5`,
`MAX_ATTEMPTS=12`. Plots: `reports/doom_round135_scatter.png`,
`reports/health_gathering_round135_scatter.png`.

**Update (2026-09-14):** `_load_valid_rows()` (`generate_evolution_report.py`) now also excludes
rounds where `act()` was never freshly called that round — detected by comparing each raw CSV
row's `(score, avg_act_latency_ms)` pair against the previous raw row's pair (an exact repeat, or
a blank score, means `run_persistent.py` just re-emitted the carried-forward value, not a fresh
task result). Previously such "no-op" rounds were counted as valid, which meant round 1 was
almost always a blank/unplotted no-op round for doom — this is why doom had no round-1 point in
the original version of this report. All numbers below are recomputed under the corrected filter;
both scatter plots have been regenerated. The substantive conclusions are unchanged, but doom now
has real round-1 data and a few cells' run/point counts shifted (noted inline).

**Bottom line: mixed, same as before. One clean success (doom/minimal, now with a real round-1
baseline point showing genuine round-over-round improvement), one still-inconclusive cell
(doom/detailed, now down to exactly 1 usable rep instead of 2), and both health_gathering variants
still show a round-1 -> round-3 win that decays by round 5 — the "no regression with rounds"
requirement is still not met there.**

## doom

| variant | round 1 | round 3 | round 5 | baseline (no_reasoning / reasoning) |
|---|---|---|---|---|
| minimal (n=3 / n=3 / n=1) | 2.9 @ ~20/min | **7.5** @ ~17/min | 8.0 @ ~19/min | 4.15 / 4.85 |
| detailed (n=1 of 2) | 6.6 @ ~11/min | 3.6 @ ~28/min | none | 4.15 / 4.85 |

- **minimal**: still the one cell that fully satisfies the governing directive, and the corrected
  filter makes the case *stronger*: round 1 now shows a real (poor) baseline of 2.9 — two of the
  three reps' round-1 attempts crashed inside `act()` every episode (score 0.0, no latency
  recorded since the exception fired before any timing), the third scored 8.6 — and by round 3
  all three reps are producing real, healthy scores (6.6-8.8) that hold or improve through round 5
  (8.0, n=1 — the other two reps didn't reach a 5th valid round, one because a round 5 attempt hit
  the outlier fallback-latency filter). Interactivity stayed in the real-inference range
  (~17-20 act()/min) throughout. Beats both baselines comfortably by round 3.
- **detailed**: one of the batch's two usable reps (`..._slot0`) has its *entire* valid-row history
  excluded outright — every valid round it reached ran at fallback-latency (>1.7M "actions"/min,
  literally microseconds per call), so `_checkpoints_score_and_rate()`'s outlier filter drops all
  of it, leaving zero contribution to any checkpoint. (A third rep, `..._mutable_slot0` from wave
  2, crashed before its first round ever completed — no `round_metrics.csv` at all — and is
  excluded from the run list entirely, likely a port-reuse race between waves.) That leaves
  exactly 1 rep (`..._slot4`) contributing any data, and it regresses round 1 -> round 3
  (6.6 -> 3.6) with no round-5 data (only 4 valid rounds total). **Still inconclusive: with only
  1 of 3 reps producing any usable checkpoint data, this isn't a confirmed finding** — but the
  offline-fallback failure mode is still clearly present in another third of doom/detailed's reps
  (the fully-excluded `..._slot0`).

## health_gathering

| variant | round 1 | round 3 | round 5 | baseline (no_reasoning / reasoning) |
|---|---|---|---|---|
| detailed (n=3 / n=2 / n=1) | 551 @ ~10/min | **1404** @ ~21/min | 514 @ ~20/min | 405.6 / 360.8 |
| minimal (n=3 / n=2 / n=2) | 95 @ ~10/min | 358 @ ~142/min* | 342 @ ~134/min* | 405.6 / 360.8 |

\* one of the two contributing reps' rate at round 3 and round 5 (`..._slot3` wave 2) is ~250-270
act()/min — fast but under the 10,000/min outlier cutoff, so it isn't excluded, though it's worth
flagging as borderline-fast rather than squarely in the ~2000-6000ms/call range the other reps show.

- Both variants still show a real, substantial LLM-driven improvement from round 1 to round 3
  (detailed 551->1404, minimal 95->358) — healthy, non-outlier act()-latency values back this as
  genuine policy improvement, not measurement noise. Round 1 is now visibly *worse* than before
  the filter fix for both variants (two of three reps in each variant crashed inside `act()` on
  every episode in round 1, scoring a real 0.0 with no latency recorded) — a real, if unflattering,
  round-1 baseline rather than a blank one.
- **Both variants still decay from round 3 to round 5** (detailed 1404->514, minimal 358->342).
  This directly violates the "score must not degrade with rounds" requirement, even with the
  agent's own best-score-comparison feedback live throughout this batch.
- Net result at round 5: detailed lands only marginally above both baselines (514 vs 405.6/360.8);
  minimal lands *below* the no_reasoning baseline (342 vs 405.6) and only barely above the
  reasoning baseline (360.8). Neither variant delivers a clear, durable win over baseline by
  round 5 — the round-3 peak is still the only point in either curve that would count as an
  unambiguous success.
- `health_gathering_detailed_20260912_113946_mutable_slot2` (1 of the 3 detailed reps) only
  reached 4 of its now-stricter-filtered valid rounds before hitting the `MAX_ATTEMPTS=12` cap in
  a repeating `SyntaxError` self-edit loop — its round-1 and round-3 checkpoints (both 284.0,
  carried over from an early crashed-then-recovered state) are real data but it never reaches
  round 5.
- Intermittent (not persistent) offline-fallback still shows up and gets correctly excluded as an
  outlier in individual rounds/checkpoints (e.g. `health_gathering_minimal_..._slot3` (wave 1)'s
  round 3 at 3.8M "actions"/min, and `doom_minimal_20260913_082450_mutable_slot1`'s round 5 at
  2.1M/min). The regression-awareness prompt reduces how often this poisons an entire run
  end-to-end (contrast with the fully-poisoned `doom_detailed_..._slot0`), but does not eliminate
  the failure mode itself — it still recurs within otherwise-healthy runs.

## Takeaways

1. The circuit-breaker prompting fix + best-score regression signal produced one unambiguous
   success (doom/minimal) and measurably reduced (but did not eliminate) whole-run
   offline-fallback poisoning elsewhere. Correcting the valid-round filter to require a genuine
   fresh `act()` call this round only strengthens this read: doom/minimal's real round-1 baseline
   (2.9, including two reps that crashed every episode) makes its round-3/round-5 improvement look
   like real learning rather than an artifact of skipping a bad first attempt.
2. A different, previously-identified failure mode remains the dominant one for health_gathering:
   strong round-1 -> round-3 gains that decay by round 5, seen in *both* variants, with latency
   staying healthy/real throughout the decay (a real policy regression, not a fallback). Not yet
   addressed by any existing prompt guidance.
3. doom/detailed's result is still confounded — now by an even starker split (1 of 2 usable reps
   contributes zero data at all, being fallback-poisoned end-to-end) — and should not be treated
   as a confirmed finding either way.
4. Per the governing directive's requirements (non-degrading scores, beats baseline, genuine LLM
   usage, both envs): **only doom/minimal fully satisfies all of them in this batch**, and does so
   more convincingly under the corrected filter than it appeared to before.

Historical launcher note: `run_repeat_batch_round135_allgpu.sh` has been removed.
These results describe the original batch; use `run_async_campaign.py` for new campaigns.
