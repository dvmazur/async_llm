#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Choose a free GPU after checking nvidia-smi}"
export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$PWD/python:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=/usr/local/cuda
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
exec .venv/bin/python -m minisgl_examples.training.video_understanding.experiment "$@"
