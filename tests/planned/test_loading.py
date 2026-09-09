import json
import pytest
import torch

from test_decoder import model
from minisgl.planned.loading import load_model


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("fp8",[False,True])
def test_real_serialized_loader_and_full_session_preserve_weight_bytes(tmp_path,monkeypatch,fp8):
    from safetensors.torch import save_file
    from minisgl.planned.session import PlannedSession
    from minisgl.planned.catalogue import RuntimeProfile
    from minisgl.planned.forward_plan import PlanCapacity
    from minisgl.planned.attention_plan import AttentionCapacity
    source=model(monkeypatch,fp8)
    c=source.model.config
    config=dict(model_type="qwen3_5_moe_text",architectures=["Qwen3_5MoeForConditionalGeneration"],
        num_hidden_layers=c.num_layers,num_attention_heads=c.num_qo_heads,num_key_value_heads=c.num_kv_heads,
        head_dim=c.head_dim,hidden_size=c.hidden_size,vocab_size=c.vocab_size,intermediate_size=c.intermediate_size,
        rms_norm_eps=c.rms_norm_eps,hidden_act="silu",tie_word_embeddings=False,
        num_experts=c.num_experts,num_experts_per_tok=c.num_experts_per_tok,
        moe_intermediate_size=c.moe_intermediate_size,shared_expert_intermediate_size=c.shared_expert_intermediate_size,
        norm_topk_prob=True,layer_types=list(c.layer_types),max_position_embeddings=1024,
        linear_num_key_heads=c.linear_num_key_heads,linear_num_value_heads=c.linear_num_value_heads,
        linear_key_head_dim=c.linear_key_head_dim,linear_value_head_dim=c.linear_value_head_dim,
        linear_conv_kernel_dim=c.linear_conv_kernel_dim,
        rope_parameters=dict(rope_type="default",rope_theta=1e4,partial_rotary_factor=.5,mrope_section=[6,5,5]))
    if fp8:config["quantization_config"]=dict(quant_method="fp8",fmt="e4m3",weight_block_size=[128,128],activation_scheme="dynamic")
    (tmp_path/"config.json").write_text(json.dumps(config))
    original={name:t.clone() for name,t in source.state_dict().items()}
    save_file({name:t.cpu() for name,t in original.items()},tmp_path/"model.safetensors")
    # The planned loader must not construct an old Engine or its duplicate
    # serving GDN/KV pools as an incidental implementation step.
    from minisgl.engine import Engine
    monkeypatch.setattr(Engine,"__init__",lambda *a,**k:pytest.fail("old Engine allocated"))
    restored=load_model(tmp_path)
    assert original.keys()==restored.state_dict().keys()
    for name,t in restored.state_dict().items():
        assert t.dtype==original[name].dtype
        assert torch.equal(t.view(torch.uint8),original[name].view(torch.uint8)),name
    profile=RuntimeProfile(PlanCapacity(1,8,2,4,4),AttentionCapacity(3,32,4,32))
    s=PlannedSession(restored,[profile]);b=s.create_block()
    logits=s.prefill_block(b,torch.tensor([1,2,3]))
    assert logits.shape==(1,128) and torch.isfinite(logits).all()
    assert b.token_ids==[1,2,3] and s.last_forward["used_graph"]


def test_loader_never_downloads_implicitly(tmp_path):
    with pytest.raises(ValueError,match="local checkpoint"):load_model(tmp_path/"missing")
