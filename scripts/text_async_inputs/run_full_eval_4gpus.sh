#!/bin/bash
# 4-GPU runner with unique ports for each GPU
# GPU 0 (port 2333): k=-1, k=0
# GPU 1 (port 2334): k=16, k=32
# GPU 2 (port 2335): k=64, k=128
# GPU 3 (port 2336): k=256, k=512

set -e

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
DATASET_PATH="${DATASET_PATH:-$HOME/Projects/mdt-doom-stuff/math500_shards_full}"
RESULTS_DIR="${RESULTS_DIR:-./eval_results/math-500-async-prompt-append}"
BUDGET="${BUDGET:-2048}"
TOTAL_SAMPLES=500

echo "========================================="
echo "Math-500 Async Input Full Evaluation"
echo "========================================="
echo "Model: $MODEL_NAME"
echo "Total samples: $TOTAL_SAMPLES"
echo "Using 4 GPUs in parallel with separate ports"
echo "Injection: append shard directly to prompt block; view stays [prompt, output]"
echo "========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/math500_async_eval.py"
mkdir -p "$RESULTS_DIR"

# Function to run k-values on a specific GPU with unique port
run_gpu() {
    local gpu=$1
    local port=$2
    shift 2
    local k_values=("$@")

    for k in "${k_values[@]}"; do
        echo "[GPU $gpu Port $port] Starting k=$k..."
        CUDA_VISIBLE_DEVICES=$gpu python "$EVAL_SCRIPT" \
            --model-name "$MODEL_NAME" \
            --dataset_path "$DATASET_PATH" \
            --path-to-results "$RESULTS_DIR" \
            --budget "$BUDGET" \
            --k-steps "$k" \
            --start 0 \
            --end "$TOTAL_SAMPLES" \
            --memory-ratio 0.8 \
            --distributed-port "$port" \
            > "${RESULTS_DIR}/k_${k}_gpu${gpu}.log" 2>&1
        echo "[GPU $gpu] Completed k=$k"
    done
    echo "[GPU $gpu] All k-values completed"
}

# Start all 4 GPUs in parallel with unique ports
run_gpu 0 2333 -1 0 &
run_gpu 1 2334 16 32 &
run_gpu 2 2335 64 128 &
run_gpu 3 2336 256 512 &

# Wait for all GPUs to finish
wait

echo "========================================="
echo "All evaluations completed!"
echo "========================================="
echo
echo "Results summary:"
for K in -1 0 16 32 64 128 256 512; do
    SUMMARY_FILE="$RESULTS_DIR/k_$K/summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        ACC=$(python3 -c "import json; d=json.load(open('$SUMMARY_FILE')); print(f\"{d['accuracy']:.3f} ({d['correct']}/{d['total']})\")" 2>/dev/null || echo "error")
        echo "  k=$K: $ACC"
    else
        echo "  k=$K: not found"
    fi
done
