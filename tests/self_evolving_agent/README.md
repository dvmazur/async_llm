# Self-evolving agent tests

Regression tests and manual GPU smoke checks live here. Evaluation entry points
remain in `scripts/self_evolving_agent/`.

Run CPU/native ViZDoom tests from the repository root (no CUDA device needed):

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover \
  -s tests/self_evolving_agent -p 'test_*.py'
```

Or use pytest:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/self_evolving_agent -q
```

Use the project's uv-managed environment with its test dependencies installed.
ViZDoom tests require native process/shared-memory access.

GPU checks are manual and are not collected by the commands above. Run only on
an explicitly authorized, available GPU. Examples using GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/LLM python tests/self_evolving_agent/smoke_budget_sweep.py
CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/LLM python tests/self_evolving_agent/smoke_realtime.py
CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/LLM SEA_GAME_MODE=synchronous \
  python tests/self_evolving_agent/smoke_synchronous.py
```

The shared import bootstrap resolves the repository's source and script modules
for direct execution, unittest discovery, and pytest.

## Evaluation launchers

- `scripts/self_evolving_agent/run_persistent.py`: one evolution run for a selected task.
- `scripts/self_evolving_agent/run_async_campaign.py`: isolated repeated campaigns.
- `scripts/self_evolving_agent/reset_engine.py`: initialize a private engine/prompt directory.

The obsolete `run_experiment_3x.sh` and
`run_repeat_batch_round135_allgpu.sh` wrappers were removed. Main Python
evaluation scripts and their launch arguments are unchanged by this cleanup.
