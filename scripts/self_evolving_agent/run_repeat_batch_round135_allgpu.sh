#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Round-1/3/5 eval, rerun after:
#   1) the silent-LLM-fallback-circuit-breaker fix (see reports/draft_round135.md,
#      .claude/plans/fuzzy-snuggling-graham.md), and
#   2) the new "best score so far" regression-awareness signal added to
#      agent.py's build_env_turn + matching prompt_seed.py/prompt_seed_minimal.py
#      guidance, targeting the score-must-not-degrade-with-rounds requirement.
#
# Unlike run_repeat_batch_round135_both.sh (GPU-2-only, sequential, 12 cells),
# this uses the full authorized GPU fleet in parallel: up to 8 cells run at
# once (one process per GPU -- each run needs ~78GB of a GPU's ~82GB, so more
# than one live run per GPU is not safe), in waves, until all 12 cells (3 runs
# x {doom, health_gathering} x {detailed, minimal}) are done.
#
# Cell order interleaves all 4 conditions within each wave (rather than doing
# all reps of one condition before moving to the next) so a full first pass
# across every condition finishes as early as possible.

GPUS=(0 1 2 3 4 5 6 7)
NGPU=${#GPUS[@]}

TARGET_VALID_STEPS=5
MAX_ATTEMPTS=12

CELLS=()
for rep in 1 2 3; do
  CELLS+=("doom|detailed" "doom|minimal" "health_gathering|detailed" "health_gathering|minimal")
done

echo "=== round135_allgpu: ${#CELLS[@]} cells total, up to ${NGPU} concurrent ==="

run_cell() {
  local gpu="$1" slot="$2" port="$3" task="$4" variant="$5"
  local mdir="mutable_slot${slot}"
  echo "=== EXPERIMENT (gpu ${gpu}, ${task}/${variant}, slot ${slot}): starting ==="
  CUDA_VISIBLE_DEVICES="$gpu" SEA_MUTABLE_DIR="$mdir" SEA_LLM_PORT="$port" \
    uv run python reset_engine.py --prompt "$variant" --mutable-dir "$mdir"
  CUDA_VISIBLE_DEVICES="$gpu" SEA_MUTABLE_DIR="$mdir" SEA_LLM_PORT="$port" \
    uv run python run_persistent.py "$TARGET_VALID_STEPS" 32000 "$task" "$variant" "$MAX_ATTEMPTS"
  echo "=== EXPERIMENT (gpu ${gpu}, ${task}/${variant}, slot ${slot}): finished ==="
}

i=0
n=${#CELLS[@]}
wave=0
while [ "$i" -lt "$n" ]; do
  wave=$((wave + 1))
  pids=()
  wave_start=$i
  for ((g = 0; g < NGPU && i < n; g++, i++)); do
    cell="${CELLS[$i]}"
    task="${cell%%|*}"
    variant="${cell##*|}"
    gpu="${GPUS[$g]}"
    slot="$g"
    port=$((2370 + g))
    run_cell "$gpu" "$slot" "$port" "$task" "$variant" &
    pids+=("$!")
  done
  echo "=== round135_allgpu: wave ${wave} launched (cells ${wave_start}..$((i - 1)), pids: ${pids[*]}) ==="
  wait "${pids[@]}"
  echo "=== round135_allgpu: wave ${wave} complete ==="
done

echo "=== EXPERIMENT: round135_allgpu batch (${#CELLS[@]} cells) complete ==="
