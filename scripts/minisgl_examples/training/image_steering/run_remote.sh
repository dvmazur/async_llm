#!/usr/bin/env bash
# Run from the repository root with CUDA_VISIBLE_DEVICES set to one available GPU.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to a free GPU}"
export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
exec .venv/bin/python -m minisgl_examples.training.image_steering.experiment "$@"
