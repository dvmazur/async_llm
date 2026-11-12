from contextlib import nullcontext

import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_profiling(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())
