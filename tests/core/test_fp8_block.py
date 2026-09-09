"""Serialized block weights: storage, loading, GEMM and MoE reference tests."""
import pytest
import torch
import torch.nn.functional as F
from minisgl.kernel.fp8 import block_fp8_linear, quantize_fp8_groups
from minisgl.moe.fused import fused_experts_impl


def random_weight(shape):
    w = (torch.randn(shape, device="cuda") * 20).to(torch.float8_e4m3fn)
    scales = torch.rand((*shape[:-2], shape[-2]//128, shape[-1]//128), device="cuda")*.001 + .001
    return w, scales


def dequant_weight(w, s):
    return w.float() * s.repeat_interleave(128, -2).repeat_interleave(128, -1)


def dequant_input(x):
    q, s = quantize_fp8_groups(x)
    return q.float() * s.repeat_interleave(128, -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 6, 19, 64, 137])
def test_dense(rows):
    torch.manual_seed(5)
    x = torch.randn(rows, 384, device="cuda", dtype=torch.bfloat16)
    w, s = random_weight((256, 384))
    actual = block_fp8_linear(x, w, s)
    expected = F.linear(dequant_input(x), dequant_weight(w, s)).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.004)
    assert float((actual.float()-expected.float()).norm()/expected.float().norm()) < .002


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zero_and_empty_inputs():
    x = torch.zeros(3, 256, device="cuda", dtype=torch.bfloat16)
    q, s = quantize_fp8_groups(x)
    assert not q.float().count_nonzero()
    assert torch.isfinite(s).all() and (s > 0).all()
    w, scales = random_weight((128, 256))
    assert block_fp8_linear(x[:0], w, scales).shape == (0, 128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_scale_association_matches_cutlass_rounding_boundary():
    # Two block contributions straddle a BF16 halfway point. CUTLASS computes
    # dot * (input_scale * weight_scale), then accumulates. The former chained
    # (dot * input_scale) * weight_scale produces 1.0078125 instead of 1.0.
    # The constants are exact FP32 values from the isolated association probe.
    x = torch.zeros(1, 256, device="cuda", dtype=torch.bfloat16)
    x[0, 0], x[0, 128] = 1.5, .625
    w = torch.zeros(128, 256, device="cuda").to(torch.float8_e4m3fn)
    w[:, 0], w[:, 128] = 1, 1
    scales = torch.tensor([[0.4874778985977173, 0.436303049325943]], device="cuda")
    actual = block_fp8_linear(x, w, scales)
    torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows,on_input", [(1, False), (6, False), (33, False), (6, True)])
def test_moe(rows, on_input):
    torch.manual_seed(8)
    e, h, d, topk = 8, 256, 128, 3
    x = torch.randn(rows, h, dtype=torch.bfloat16, device="cuda")
    w1, s1 = random_weight((e, 2*d, h))
    w2, s2 = random_weight((e, h, d))
    ids = torch.stack([torch.randperm(e, device="cuda")[:topk] for _ in range(rows)]).int()
    scores = torch.softmax(torch.randn(rows, topk, device="cuda"), -1)
    qx, dw1, dw2 = dequant_input(x), dequant_weight(w1,s1), dequant_weight(w2,s2)
    expected = torch.zeros_like(x, dtype=torch.float32)
    for row in range(rows):
        for j in range(topk):
            expert = int(ids[row, j])
            hidden = F.linear(qx[row], dw1[expert])
            if on_input:
                hidden *= scores[row,j]
            gate, up = hidden.to(x.dtype).float().chunk(2)
            hidden = (F.silu(gate)*up).to(x.dtype)
            out = F.linear(dequant_input(hidden[None])[0], dw2[expert])
            if not on_input:
                out *= scores[row,j]
            expected[row] += out.to(x.dtype).float()
    actual = fused_experts_impl(x.clone(), w1, w2, scores, ids,
                                apply_router_weight_on_input=on_input, w1_scale=s1, w2_scale=s2)
    torch.testing.assert_close(actual, expected.to(x.dtype), rtol=.06, atol=.006)
    assert float((actual.float()-expected).norm()/expected.norm()) < .02


def test_scale_key_merge():
    from minisgl.models.weight import _get_expert_stack_info, _get_merge_info
    key = "model.layers.0.mlp.experts.2.gate_proj.weight_scale_inv"
    merged, slot, slots = _get_merge_info(key)
    assert slot == "gate"
    assert _get_expert_stack_info(merged) == ("model.layers.0.mlp.experts.gate_up_proj_scale_inv", 2)


def test_serialized_loader_preserves_weight_bytes_and_scales(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from test_qwen3_5_moe import _tiny_hf_config
    from minisgl.distributed import DistributedInfo
    import minisgl.distributed.info as dist
    from minisgl.engine.engine import Engine
    monkeypatch.setattr(dist, "_TP_INFO", DistributedInfo(0,1))
    config = _tiny_hf_config()
    config.quantization_config = {"quant_method":"fp8","fmt":"e4m3",
                                  "activation_scheme":"dynamic","weight_block_size":[128,128]}
    config.save_pretrained(tmp_path)
    tensors={}
    prefix="model.language_model.layers.0.mlp.experts"
    # Distinct values for every expert/projection expose gate/up/scale swaps.
    for e in range(4):
        for j, part in enumerate(("gate_proj","up_proj","down_proj")):
            shape=(128,256) if part!="down_proj" else (256,128)
            tensors[f"{prefix}.{e}.{part}.weight"]=torch.full(shape,e+j+1).to(torch.float8_e4m3fn)
            tensors[f"{prefix}.{e}.{part}.weight_scale_inv"]=torch.full(
                (shape[0]//128,shape[1]//128), .125*(e+j+1),dtype=torch.bfloat16)
    save_file(tensors,tmp_path/"model.safetensors")
    fake=SimpleNamespace(device=torch.device("cpu"),dtype=torch.bfloat16)
    loaded=Engine._load_weight_state_dict(fake,SimpleNamespace(use_dummy_weight=False,model_path=str(tmp_path)))
    key="model.layers.0.mlp.experts.gate_up_proj"
    expected=torch.stack([torch.cat([tensors[f"{prefix}.{e}.{p}.weight"] for p in ("gate_proj","up_proj")]) for e in range(4)])
    assert torch.equal(loaded[key].view(torch.uint8),expected.view(torch.uint8))
    expected_s=torch.stack([torch.cat([tensors[f"{prefix}.{e}.{p}.weight_scale_inv"] for p in ("gate_proj","up_proj")]) for e in range(4)])
    torch.testing.assert_close(loaded[key+"_scale_inv"],expected_s.float(),rtol=0,atol=0)
    assert loaded[key].dtype==torch.float8_e4m3fn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_layer_loading():
    from minisgl.layers.linear import LinearReplicated
    from minisgl.layers.moe import MoELayer
    from minisgl.distributed import set_tp_info
    set_tp_info(0,1)
    with torch.device("meta"):
        linear = LinearReplicated(256,128,False)
        moe = MoELayer(8,2,256,128)
    w,s = random_weight((128,256))
    state = {"weight":w,"weight_scale_inv":s}
    linear.load_state_dict(state)
    assert not state and linear.weight is w and linear.weight_scale_inv is s
    a,sa = random_weight((8,256,256))
    b,sb = random_weight((8,256,128))
    state = {"gate_up_proj":a,"gate_up_proj_scale_inv":sa,"down_proj":b,"down_proj_scale_inv":sb}
    moe.load_state_dict(state)
    assert not state and moe.gate_up_proj is a
    assert sum(t.numel()*t.element_size() for t in moe.state_dict().values()) < 8*256*384*1.01
