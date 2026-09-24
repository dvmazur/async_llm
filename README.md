# Asynchronous Reasoning with Mini-SGLang

This repository extends Mini-SGLang with an asynchronous inference API. It supports concurrent generation streams
that reuse model state, as well as multimodal interaction through `AsyncLLM`.

## Demos

- [Async thoughts](scripts/async_thoughts/README.md): run a thinker and a writer
  concurrently, allowing the writer to use the thinker's partial reasoning.
  Includes installation instructions and command-line examples.
- [Doom Basic](scripts/doom_basic/README.md): a notebook-style example of a
  vision-language model observing game frames and choosing actions in ViZDoom.

See the [original Mini-SGLang README](README.mini-sglang.md) for background on the
base inference framework and its environment requirements.
