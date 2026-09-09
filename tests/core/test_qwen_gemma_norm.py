"""Raw Qwen norm weights and FP32 +1, against the mathematical reference."""
from types import SimpleNamespace

import pytest
import torch

from minisgl.layers.norm import GemmaRMSNorm, GemmaRMSNormFused, RMSNorm


def reference(x, weight, eps=1e-6):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
            * (1 + weight.float())).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('shape', [(7, 1024), (5, 2, 256)])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_gemma_norm_avoids_folded_low_precision_weight(shape, dtype):
    torch.manual_seed(714)
    x = torch.randn(shape, device='cuda', dtype=dtype)
    weight = (torch.randn(shape[-1], device='cuda') * .1).to(dtype)
    layer = GemmaRMSNorm(shape[-1], 1e-6)
    layer.weight = weight
    before = x.clone()
    actual = layer.forward(x)
    expected = reference(x, weight)
    old = RMSNorm(shape[-1], 1e-6)
    old.weight = weight + 1
    old_output = old.forward(x)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=.005, atol=2e-5)
    error = (actual.float() - expected.float()).abs().mean()
    old_error = (old_output.float() - expected.float()).abs().mean()
    assert error < old_error * .1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_gemma_fused_residual_and_inplace_contract():
    torch.manual_seed(19)
    x = torch.randn(9, 1024, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(1024, device='cuda', dtype=torch.bfloat16) * .1
    value = x.float() + residual.float()
    expected = reference(value, weight).to(x.dtype)
    layer = GemmaRMSNormFused(1024, 1e-6)
    layer.weight = weight
    output, returned_residual = layer.forward(x, residual)
    assert output is x and returned_residual is residual
    torch.testing.assert_close(residual, value.to(residual.dtype), rtol=0, atol=0)
    torch.testing.assert_close(output, expected, rtol=.01, atol=.002)


def test_qwen_loader_does_not_round_plus_one_into_bf16(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from test_qwen3_5_moe import _tiny_hf_config
    from minisgl.distributed import DistributedInfo
    import minisgl.distributed.info as dist
    from minisgl.engine.engine import Engine

    monkeypatch.setattr(dist, '_TP_INFO', DistributedInfo(0, 1))
    _tiny_hf_config().save_pretrained(tmp_path)
    names = ['layers.0.input_layernorm', 'layers.0.post_attention_layernorm',
             'layers.0.self_attn.q_norm', 'layers.0.self_attn.k_norm', 'norm',
             'layers.0.linear_attn.norm']
    original = torch.tensor([.001953125, -.001953125, .00390625], dtype=torch.bfloat16)
    save_file({f'model.language_model.{name}.weight': original.clone() for name in names},
              tmp_path / 'model.safetensors')
    fake = SimpleNamespace(device=torch.device('cpu'), dtype=torch.bfloat16)
    loaded = Engine._load_weight_state_dict(fake, SimpleNamespace(
        use_dummy_weight=False, model_path=str(tmp_path)))
    for name in names:
        torch.testing.assert_close(loaded[f'model.{name}.weight'], original, rtol=0, atol=0)
