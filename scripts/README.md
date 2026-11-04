# Minisgl examples

Install the library and its example packages in the same environment:

```bash
uv pip install -e . -e ./scripts
```

All Python imports use `minisgl_examples`. Run the installed commands below, or
their modules with `python -m minisgl_examples.<package>.<module>`.

| Package | Command | Purpose |
|---|---|---|
| `demos.async_thoughts` | `async-thoughts` | Concurrent thinker/writer, with optional training trace export |
| `training.parallel_thinking` | `parallel-thinking` | Synchronized or independent workers, training and checkpoint reload |
| `training.distillation` | `async-distillation` | Teacher validation, trace distillation and independent student generation |
| `training.sft` | `minisgl-sft` | Collect traces or train saved examples |
| `training.stress` | `minisgl-stress` | Record cache mutation workloads through the async scheduler |

The parallel-thinking and distillation packages each contain `demo.py` for
`record`, `train`, `verify`, and `run` stages, and `rollout.py` for their generation
schedule. Both accept collected examples directly or saved traces. Their notebooks
call the same Python implementations.

Training workflows share `training.recording`, `training.loss`, and the HF student
loop in `training.student`; `training.process` isolates model stages. The library
recorder is always `minisgl.training.RecordedAsyncLLM`; sampled targets and file
export remain application responsibilities. The Doom example is documentation
under `demos/doom_basic`.

Training tests live in `tests/training`, with CUDA numerical/integration tests
under `tests/training/gpu`. Script tests live in `tests/scripts`.
