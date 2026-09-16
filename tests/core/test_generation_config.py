from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine import EngineConfig
import torch
from transformers import GenerationConfig


def test_default_generation_config_is_loaded_once_and_params_are_fresh(monkeypatch):
    calls = []

    def load(path):
        calls.append(path)
        return GenerationConfig(do_sample=True, temperature=.7, top_k=20, top_p=.9)

    monkeypatch.setattr(GenerationConfig, "from_pretrained", load)
    config = EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16)
    first = config.get_default_sampling_params()
    first.temperature = 9
    assert config.get_default_sampling_params().temperature == .7
    assert calls == ["unused"]


def test_explicit_generation_config_does_not_load_model_config(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("explicit generation config must not hit the hub")

    monkeypatch.setattr(GenerationConfig, "from_pretrained", fail)
    config = EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16,
                          generation_config=GenerationConfig(do_sample=False))
    assert config.get_default_sampling_params().is_greedy


def test_missing_generation_config_uses_default_once(monkeypatch):
    calls = []

    def missing(path):
        calls.append(path)
        raise OSError("no generation config")

    monkeypatch.setattr(GenerationConfig, "from_pretrained", missing)
    config = EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16)
    assert isinstance(config.get_default_sampling_params(), SamplingParams)
    assert config.get_default_sampling_params().is_greedy
    assert calls == ["unused"]
