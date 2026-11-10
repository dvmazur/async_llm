"""Gemma retains the HF decoder/MLP instead of copying their fused implementations."""
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm, Gemma4TextDecoderLayer, Gemma4TextMLP

from minisgl.models.config import ModelConfig
from minisgl.models.gemma4 import Gemma4ForCausalLM
from test_gemma4_config import tiny_config


def test_fp8_keeps_transformers_decoder_mlp_norm_and_head(monkeypatch):
    from minisgl.distributed import DistributedInfo
    import minisgl.distributed.info as info
    monkeypatch.setattr(info, '_TP_INFO', DistributedInfo(0, 1))
    config = ModelConfig.from_hf(tiny_config())
    with torch.device('meta'):
        model = Gemma4ForCausalLM(config)
    assert all(type(layer) is Gemma4TextDecoderLayer for layer in model.model.layers)
    assert all(type(layer.mlp) is Gemma4TextMLP for layer in model.model.layers)
    assert type(model._hf.lm_head) is torch.nn.Linear
    assert model._hf.lm_head.forward.__func__ is torch.nn.Linear.forward
    norms = [m for m in model.model.modules() if isinstance(m, Gemma4RMSNorm)]
    assert norms and all(type(m) is Gemma4RMSNorm for m in norms)
    assert all(m._norm.__func__ is Gemma4RMSNorm._norm for m in norms)
