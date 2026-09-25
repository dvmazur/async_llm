# ChopTree: two single-conversation controls

Both use the text Planner/Actor prompts, sampled generation and separate sampled
action readout. No JSON grammar or model-output parser. Counts, temperatures,
paths, engine and environment settings are explicitly in each experiment file.

```bash
python3 experiments/choptree_1x500_r100_sequential_step.py
python3 experiments/choptree_1x500_r100_sequential_episode.py
```

Run them separately; each loads one model on its configured GPU. Edit `VENV`,
`MODEL`, `CRAFTIUM`, `RESULTS`, `GPUS` before running. Base Python must be >=3.11.
No XGrammar installation is needed. Both default to one active pipeline, up to
500 actions per episode, 100 repeats, seeds0–99. PNG off, GIF on.

## What changes

| Variant | Shared within one action | Between actions |
| --- | --- | --- |
| Original text Planner/Actor | Separate role contexts; action gets plan and assessment | Bounded external memory, no persistent chat |
| `sequential_step` | One chat: current image → Planner → Actor → action | Chat freed; same bounded external memory as original |
| `sequential_episode` | One chat: current image → Planner → Actor → action | Entire transcript and images retained in KV until episode end |

All already execute the roles in order: this is a **context-visibility control**,
not merely replacing concurrent calls with sequential calls. In both new variants,
Actor can attend to the image and Planner's full response in the shared chat. The
same Planner/Actor instructions live in one system message, and user turns select
the active stage. This is not a claim of numerically identical model inputs.

The episode version appends the latest action/reward and image-change evidence.
Previous plans and older action/reward records are already in the conversation;
it does not repeatedly paste the entire transcript or the previous plan. Each role
retains its last generated token even when capped, closes its assistant turn, and
the actual sampled button is also appended before the next observation.

Native settings remain224px, frameskip4, pmul2,7° turns, noon, settled reset and
0.1s action delay. Planner sees the same marked448px frame as the text baseline.
Temperatures remain0.5/0.9 for Planner and0.7 for Actor/readout; role limit768.

## Capacity is explicit

`sequential_step` uses65536 KV slots (1.25GiB on Qwen35B-A3B).
`sequential_episode` uses294912 slots (5.625GiB), covering the model's native
262144-token context plus spare pages for this one-pipeline configuration.
Increasing concurrency may require a larger pool; each episode owns its own chat.

There is **no automatic history trimming, reset, summarization or RoPE extension**.
Before a prefill, the episode checks retained tokens + new inputs + generation/
turn-close reserve against the model's native context limit. Exceeding it raises
an explicit `ContextLimitError` before that GPU operation, rather than silently
turning the full-history control into a windowed control. The failed episode/run
is not reported as completed. The500-action setting is a maximum, not a guarantee
that any possible sequence of768-token role replies fits262144 tokens.

RTX smoke: both modes completed12 actions, with24 role completions and12 action
readouts each. The step mode's retained context before every action was0; the
episode mode grew continuously to6150 tokens. Persistent KV was freed at episode
end. These were seeds0/1, functionality checks rather than paired quality tests.

Neither new1x500x100 configuration has been run in full. The smoke's growing chat
added about429 tokens/action after its first turn; extrapolating that short sample
gives roughly216k tokens at500 actions, below262k, but longer replies can exceed it.
Do not treat this estimate as a guarantee of capacity or quality for100 episodes.
