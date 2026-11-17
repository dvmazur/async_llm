#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Choose a GPU with enough free VRAM after checking nvidia-smi}"
export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$PWD/python:$PWD/scripts:$PWD/../deps${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=/usr/local/cuda
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec .venv/bin/python -m minisgl_examples.training.doom_health.experiment "$@"
