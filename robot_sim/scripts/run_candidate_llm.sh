#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source robot_sim/scripts/runtime_config.sh
if [[ ${ROBOT_LLM_HARDWARE_PROFILE:-legacy} == h100 ]]; then
    # The simulator keeps its single-CPU lock; a Hopper model gets a CPU pool.
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
    export MAX_JOBS=4
    unset OMP_THREAD_LIMIT ROBOT_CPU_LOCKED KMP_AFFINITY GOMP_CPU_AFFINITY OMP_PLACES
else
    source robot_sim/scripts/resource_limits.sh
fi
export PYTHONPATH="$ROOT/python:$ROOT"
export HF_HUB_OFFLINE=1 HF_HOME="$ROBOT_HF_HOME"
export CUDA_VISIBLE_DEVICES="${ROBOT_LLM_GPU:-0,1}"
cache_profile="${ROBOT_LLM_HARDWARE_PROFILE:-legacy}"
export TRITON_CACHE_DIR="$ROOT/robot_sim/.runtime/triton-$cache_profile"
export TORCH_EXTENSIONS_DIR="$ROOT/robot_sim/.runtime/torch_extensions-$cache_profile"
export CUDA_CACHE_PATH="$ROOT/robot_sim/.runtime/cuda_cache-$cache_profile"
export TVM_FFI_CACHE_DIR="$ROOT/robot_sim/.runtime/tvm_ffi-$cache_profile"
export PATH="$(dirname "$ROBOT_LLM_PYTHON"):/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
MODEL="${MODEL:-$ROOT/robot_sim/.runtime/models/Qwen3.8-27B-FP8}"
TP_SIZE="${ROBOT_LLM_TP_SIZE:-2}"
if [[ "$TP_SIZE" != 1 && "$TP_SIZE" != 2 ]]; then echo "ROBOT_LLM_TP_SIZE must be 1 or 2" >&2; exit 2; fi
runner=("$ROBOT_LLM_PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$TP_SIZE"
        -m robot_sim.candidate_nav.llm --model "$MODEL" "$@")
if [[ ${ROBOT_LLM_HARDWARE_PROFILE:-legacy} == h100 ]]; then
    exec "${runner[@]}"
else
    exec /usr/bin/python3 -m robot_sim.resource_limits "${runner[@]}"
fi
