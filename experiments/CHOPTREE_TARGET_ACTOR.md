# ChopTree: visual target planner and motor actor

On-device Planner -> Actor v8, with a self-contained 1x500x100 experiment.

## Policy

Each action has two sampled model calls:

1. **Target planner** sees the current full frame, enlarged to 448px with a red
   aiming cross, recent public action/reward outcomes and its previous short plan.
   It chooses a wood face, route or search target, normalized image coordinates,
   apparent proximity and body obstacle. These are model hypotheses, not sensors.
2. **Motor actor** gets only the structured target coordinates/type/proximity/
   obstacle and last action. It does not receive the planner's potentially
   misleading prose. It first classifies alignment, then samples one of all eight
   controls and predicts its effect.

Python does not map coordinates to a forced action, correct a model decision, read
block types from the simulator, or replace a failed generation with a fallback
action. JSON grammar constrains response format; all controls remain available.

Memory is bounded: twelve public action/reward records, one previous model plan,
and twelve small RGB thumbnails for a pixel-change/revisit summary. No growing chat
or hidden-state observations. Each role frees its own KV block before the next role
or environment action. The planner samples at T=0.5, raised to T=0.9 after twelve
actions without reward; that counter alone is not proof of stagnation. Motor T=0.7.

## Run

Edit the paths and GPU list directly in the experiment file, then run from this
repository with base Python:

```bash
python3 experiments/choptree_1x500_r100_planner_actor.py
```

This runs **one active world, 500 actions per episode, 100 sequential episodes**,
seeds 0–99. The model stays loaded between episodes. A native terminal state can
end an episode before the action limit. The file carries its complete config and
does not import settings from another experiment. `VENV` selects the existing
engine environment; it needs the project's `assessment` dependency group
(`xgrammar==0.2.7`) for the generated roles. Install it once into that same venv:
`uv pip install --python /path/to/minisgl-venv/bin/python xgrammar==0.2.7`.
Set `MODEL`, `CRAFTIUM`, `RESULTS` and
`GPUS` to the deployed locations. Counts, role temperatures, output token budget,
physics parameters and recording settings are also in that file. The Runner API
is unchanged. Use a new `RESULTS` directory for each run.

Native controls are identical: 224px observations, frameskip 4, pmul 2, turns 7°,
fixed noon, settled reset, 0.1-second action delay. The reset wrapper retains the
validated initial NOP; its one-participant start barrier does not wait for another
world. If concurrency is increased explicitly, it gates episode starts only.
This is **not** a fixed-game-time control: observed native frame dtime can differ.

GIF recording is on, individual PNG dumping off. Full public decisions, sampled
role text, rewards, timings and native diagnostics are recorded. Native position
diagnostics are for analysis, not inputs to the policy.

## Evidence so far

The frozen v8 improved both short comparisons on RTX:

| Episodes | Actions | Probe mean wood | Planner + actor mean wood |
| --- | ---: | ---: | ---: |
| Seeds 0–4 | 40 | 1.6 | 3.0 |
| Seeds 20–24 | 60 | 1.8 | 4.0 |
| 10 completed paired episodes | 200 | 3.7 | 7.4 |

The long run was stopped at the user's request: ten full episodes, five partial
episodes excluded from this table, five not started. Completed seeds:
100,101,104,105,108,109,112,113,116,117. The candidate won nine of those ten pairs;
its median score was7 and maximum14. These are action-budget comparisons, not
wall-clock speedups. The new **1x500x100** configuration has not been run in full.

It won each paired short episode. Remaining visible failures include stale target
coordinates, mistaking snow left after a removed stump for wood, and repeated
targeting of high suspended blocks. The policy has not eliminated these failures.
