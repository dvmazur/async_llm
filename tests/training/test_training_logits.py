import pytest
from minisgl.training import replay_trace

from tests.training.fixtures import cuda_delta_math as cuda_delta_math
from tests.training.models import live_llm, tiny_model
from tests.training.workloads import mutation_trace


@pytest.mark.parametrize("unsupported", ["tensor_parallel", "quantization", "pending_work"])
def test_async_llm_replay_rejects_unsupported_engine_state(unsupported):
    llm, _, _ = live_llm(tiny_model())
    if unsupported == "tensor_parallel":
        llm.engine.config.tp_info.size = 2
    elif unsupported == "quantization":
        llm.engine.config.quantization = "fp8"
    else:
        llm.async_engine.has_work = True
    with pytest.raises((NotImplementedError, RuntimeError), match="AsyncLLM"):
        replay_trace(llm, mutation_trace())
