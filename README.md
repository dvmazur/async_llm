# LLMs are general asynchronous agents

This repository extends Mini-SGLang with an asynchronous inference API. It supports concurrent generation streams
that reuse model state, as well as multimodal interaction through `AsyncLLM`.

## Environment setup

Use Linux with an NVIDIA GPU and enough GPU memory for your chosen model. The
project requires Python 3.10 or newer; the commands below use Python 3.12.
Install `uv` before proceeding.

The dependencies in [pyproject.toml](pyproject.toml) use CUDA 13.0 wheels (`cu130`).
Install a compatible NVIDIA driver and the CUDA 13.0 Toolkit, including `nvcc`,
for JIT compilation of CUDA kernels. Check that both are available:

```bash
nvidia-smi
nvcc --version
```

If the toolkit is installed outside the default location, set `CUDA_HOME` to its
installation directory and add `$CUDA_HOME/bin` to `PATH`. The CUDA version shown
by `nvidia-smi` describes driver support; use `nvcc --version` to check the installed
toolkit.

From the root of this repository, create the environment and install the project
with the dependencies recorded in [uv.lock](uv.lock):

```bash
uv sync --locked --python 3.12
source .venv/bin/activate
```

The project is installed in editable mode, so changes under `python/` take effect
without reinstalling. The CUDA package indexes are configured in `pyproject.toml`.
For development tools and test dependencies, use:

```bash
uv sync --locked --python 3.12 --extra dev
```

Verify that the environment can import the package and access a GPU:

```bash
python - <<'PYTHON'
import minisgl
import torch

print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
assert torch.cuda.is_available(), "CUDA is unavailable; check the GPU and driver"
print(f"GPU: {torch.cuda.get_device_name(0)}")
PYTHON
```

This checks package import and GPU visibility. Follow the demo instructions below
to install any additional demo dependencies and run model inference.

## Demos

- [Async thoughts](scripts/async_thoughts/README.md): run a thinker and a writer
  concurrently, allowing the writer to use the thinker's partial reasoning.
  Includes installation instructions and command-line examples.
- [Doom Basic](scripts/doom_basic/README.md): a notebook-style example of a
  vision-language model observing game frames and choosing actions in ViZDoom.

See the [original Mini-SGLang README](README.mini-sglang.md) for background on the
base inference framework and its environment requirements.

## Datasets

- [datasets/sharded_vqa.parquet](datasets/sharded_vqa.parquet): a dataset of 513 image pairs from math and VisualQA tasks, each containing an original image and an edited version with answer-changing visual errors, for evaluating how VLM agents revise ongoing reasoning after visual corrections.
