"""Rounding contracts for Qwen; no scheduler/cache or fusion changes."""
import pytest
import torch

from minisgl.layers.norm import RMSNorm, RMSNormFused


def reference_norm(x, weight, eps, offset=1.0):
    xf = x.float()
    normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (normalized * (weight.float() + offset)).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('size', [128, 256, 2048])
def test_qwen_norm_keeps_plus_one_in_fp32(size):
    torch.manual_seed(817)
    x = torch.randn(23, size, device='cuda', dtype=torch.bfloat16)
    weight = (torch.randn(size, device='cuda') * .015).bfloat16()
    norm = RMSNorm(size, 1e-6, weight_plus_one=True)
    norm.weight = weight
    expected = reference_norm(x, weight, norm.eps)
    actual = norm.forward(x)
    torch.testing.assert_close(actual, expected, rtol=.008, atol=.001)
    assert (actual == expected).float().mean() > .999
    # The original load-time BF16 fold irreversibly drops information.
    folded = RMSNorm(size, norm.eps)
    folded.weight = weight + 1.
    assert (folded.forward(x) != expected).float().mean() > .1
    inplace = x.clone()
    norm.forward_inplace(inplace)
    torch.testing.assert_close(inplace, actual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_qwen_residual_norm_and_default_norm():
    from flashinfer import fused_add_rmsnorm, gemma_fused_add_rmsnorm, rmsnorm
    torch.manual_seed(171)
    x = torch.randn(13, 256, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = (torch.randn(256, device='cuda') * .02).bfloat16()
    norm = RMSNormFused(256, 1e-6, weight_plus_one=True)
    norm.weight = weight
    expected, expected_residual = x.clone(), residual.clone()
    gemma_fused_add_rmsnorm(expected, expected_residual, weight, norm.eps)
    actual, actual_residual = norm.forward(x.clone(), residual.clone())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    actual, actual_residual = norm.forward(x)
    torch.testing.assert_close(actual, reference_norm(x, weight, norm.eps), rtol=.008, atol=.001)
    assert actual_residual is x
    plain = RMSNormFused(256, norm.eps)
    assert plain.rmsnorm is rmsnorm and plain.fused_add_rmsnorm is fused_add_rmsnorm


def test_loader_does_not_fold_qwen_norm_weights(tmp_path, monkeypatch):
    from test_fp8_loading import config_for_checkpoint
    from minisgl.distributed import DistributedInfo
    import minisgl.distributed.info as dist
    from minisgl.models.weight import load_weight
    from safetensors.torch import save_file
    monkeypatch.setattr(dist, '_TP_INFO', DistributedInfo(0, 1))
    config = config_for_checkpoint(tmp_path)
    del config.quantization_config
    config.save_pretrained(tmp_path)
    raw = torch.tensor([.001, .003, -.002], dtype=torch.bfloat16)
    save_file({'model.language_model.layers.0.input_layernorm.weight': raw},
              tmp_path/'model.safetensors')
    loaded = dict(load_weight(str(tmp_path), torch.device('cpu')))
    torch.testing.assert_close(loaded['model.layers.0.input_layernorm.weight'], raw, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('dim', [64, 72, 80])
def test_vision_rotary_matches_transformers(dtype, dim):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision
    from minisgl.models.qwen3_5_vision import _vision_rope
    torch.manual_seed(771)
    q = torch.randn(3, 17, dim, dtype=dtype).transpose(0, 1)
    k = torch.randn_like(q)
    phase = torch.randn(17, dim)
    cos, sin = phase.cos(), phase.sin()
    expected_q, expected_k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    torch.testing.assert_close(_vision_rope(q, cos, sin), expected_q, rtol=0, atol=0)
    torch.testing.assert_close(_vision_rope(k, cos, sin), expected_k, rtol=0, atol=0)


@pytest.mark.parametrize('length', [1, 3, 7])
@pytest.mark.parametrize('has_history', [False, True])
def test_prefill_conv_scalar_reference(length, has_history):
    from minisgl.models.qwen3_5_delta import _prefill_conv_silu
    torch.manual_seed(617)
    x = torch.randn(2, 5, length, dtype=torch.bfloat16)
    weight = torch.randn(5, 1, 4, dtype=x.dtype)
    history = torch.randn(2, 5, 3, dtype=x.dtype) if has_history else torch.zeros(2, 5, 3, dtype=x.dtype)
    joined = torch.cat([history, x], -1)
    expected = []
    for t in range(length):
        value = torch.zeros(2, 5)
        for tap in range(4):
            value += (joined[..., t+tap] * weight[:, 0, tap]).float()
        expected.append(torch.nn.functional.silu(value).to(x.dtype))
    expected = torch.stack(expected, -1)
    actual = _prefill_conv_silu(joined, weight, padding=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if not has_history:
        actual = _prefill_conv_silu(x, weight, padding=3)[..., :length]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('length', [1, 7, 167])
@pytest.mark.parametrize('has_history', [False, True])
def test_prefill_conv_matches_sglang(length, has_history):
    reference = pytest.importorskip('sglang.kernels.ops.mamba.causal_conv1d_triton')
    from minisgl.models.qwen3_5_delta import _prefill_conv_silu
    torch.manual_seed(617)
    x = torch.randn(length, 256, device='cuda', dtype=torch.bfloat16).T
    weight = torch.randn(256, 4, device='cuda', dtype=x.dtype)
    history = torch.randn(1, 256, 3, device='cuda', dtype=x.dtype) if has_history else torch.zeros(1, 256, 3, device='cuda', dtype=x.dtype)
    expected = reference.causal_conv1d_fn(x, weight, None, history.clone(),
        torch.tensor([0, length], device='cuda', dtype=torch.int32), [length],
        has_initial_state=torch.tensor([has_history], device='cuda'))
    actual = _prefill_conv_silu(torch.cat([history, x[None]], -1), weight[:, None], 0)[0]
    torch.testing.assert_close(actual, expected, rtol=.008, atol=.001)
    assert (actual == expected).float().mean() > .999


def test_chunk_beta_storage_is_fp32_without_changing_values(monkeypatch):
    import minisgl.models.qwen3_5_delta as delta
    beta = torch.tensor([[[.1, .6], [.8, .3]]], dtype=torch.bfloat16)
    qkv = torch.zeros(1, 2, 2, 8, dtype=torch.bfloat16)
    g = torch.zeros(1, 2, 2)
    sentinel = object()
    expected = beta.float()
    def reference_contract(q, k, v, *, beta, **kwargs):
        assert beta.dtype == torch.float32
        torch.testing.assert_close(beta, expected, rtol=0, atol=0)
        return sentinel
    monkeypatch.setattr(delta, '_fla_chunk', reference_contract)
    assert delta._chunk_delta(qkv, qkv, qkv, g, beta) is sentinel


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_shared_expert_gate_preserves_intermediate_precision(dtype):
    from minisgl.models.qwen3_5_moe import _shared_expert_gate
    torch.manual_seed(815)
    x = torch.randn(31, 256, dtype=dtype)
    weight = (torch.randn(1, 256) * .3).to(dtype)
    expected = (x.double() * weight.double()).sum(-1, keepdim=True).sigmoid()
    actual = _shared_expert_gate(x, weight)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('size', [256, 2048])
def test_shared_expert_sum_matches_sglang(size):
    reference = pytest.importorskip('sglang.kernels.ops.elementwise.elementwise')
    from minisgl.models.qwen3_5_moe import _shared_expert_gate
    torch.manual_seed(719)
    x = torch.randn(31, size, device='cuda', dtype=torch.bfloat16)
    weight = (torch.randn(1, size, device='cuda') * .03).bfloat16()
    shared, routed = torch.randn_like(x), torch.randn_like(x)
    gate = _shared_expert_gate(x, weight)
    actual = (routed.float() + gate*shared.float()).to(x.dtype)
    expected = routed.clone()
    reference.fused_gate_sigmoid_mul_add(x, weight.squeeze(), shared, expected)
    torch.testing.assert_close(actual, expected, rtol=.008, atol=.001)
    assert (actual == expected).float().mean() > .999


def test_shared_gate_reads_before_inplace_experts():
    from types import SimpleNamespace
    from minisgl.models.qwen3_5_moe import Qwen3_5MoeMLP, _shared_expert_gate
    torch.manual_seed(79)
    x = torch.randn(7, 32, dtype=torch.bfloat16)
    weight = torch.randn(1, 32, dtype=x.dtype)
    shared, routed = torch.randn_like(x), torch.randn_like(x)
    expected = (routed.float() + _shared_expert_gate(x, weight)*shared.float()).to(x.dtype)
    op = object.__new__(Qwen3_5MoeMLP)
    op.shared_expert = SimpleNamespace(forward=lambda value: shared)
    op.shared_expert_gate = SimpleNamespace(weight=weight)
    op.gate = SimpleNamespace(forward=lambda value: torch.zeros(7, 4))
    def overwrite(value, logits):
        value.copy_(routed)
        return value
    op.experts = SimpleNamespace(forward=overwrite)
    torch.testing.assert_close(op.forward(x), expected, rtol=0, atol=0)


def test_attention_gate_has_no_intermediate_bf16_rounding():
    from minisgl.models.qwen3_5_attn import _sigmoid_output_gate
    torch.manual_seed(177)
    x = torch.randn(13, 8, 256, dtype=torch.bfloat16)
    gate = torch.randn_like(x) * 3
    expected = (x.double() * gate.double().sigmoid()).to(x.dtype)
    actual = _sigmoid_output_gate(x, gate)
    torch.testing.assert_close(actual, expected, rtol=.008, atol=.001)
    assert (actual == expected).float().mean() > .999
    assert (x * gate.sigmoid() != expected).float().mean() > .1


@pytest.mark.parametrize('width', [2, 3, 4])
def test_decode_conv_last_window_and_precision(width):
    from minisgl.models.qwen3_5_delta import _decode_conv_silu
    torch.manual_seed(179)
    x = torch.randn(3, 256, width+1, dtype=torch.bfloat16)
    weight = torch.randn(256, 1, width, dtype=x.dtype)
    saved = x.clone()
    expected = torch.nn.functional.silu((x[..., -width:].double()*weight[:,0].double()).sum(-1)).to(x.dtype)
    actual = _decode_conv_silu(x, weight)
    torch.testing.assert_close(actual, expected, rtol=.008, atol=.001)
    assert (actual == expected).float().mean() > .999
    assert torch.equal(x, saved), 'Arithmetic must not mutate convolution history'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('batch', [1, 5, 32])
def test_decode_conv_matches_sglang_cuda(batch):
    reference = pytest.importorskip('sglang.srt.layers.attention.mamba.causal_conv1d')
    from minisgl.models.qwen3_5_delta import _decode_conv_silu
    torch.manual_seed(179)
    x = torch.randn(batch, 256, device='cuda', dtype=torch.bfloat16)
    history = torch.randn(batch, 256, 3, device='cuda', dtype=x.dtype)
    weight = torch.randn(256, 4, device='cuda', dtype=x.dtype)
    actual = _decode_conv_silu(torch.cat([history, x[...,None]],-1), weight[:,None])
    expected = reference.causal_conv1d_update(x.clone(), history.clone(), weight, activation='silu')
    torch.testing.assert_close(actual, expected, rtol=.008, atol=.001)
    assert (actual == expected).float().mean() > .999
