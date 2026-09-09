from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from minisgl.distributed import DistributedInfo
from minisgl.models.weight import load_weight, _shard_tensor, _get_expert_stack_info


def config_for_checkpoint(path):
    from test_qwen3_5_moe import _tiny_hf_config
    config=_tiny_hf_config()
    config.quantization_config=dict(quant_method='fp8',fmt='e4m3',activation_scheme='dynamic',
                                    weight_block_size=[128,128])
    config.save_pretrained(path)
    return config


def test_loader_preserves_serialized_expert_bytes(tmp_path,monkeypatch):
    import minisgl.distributed.info as dist
    from minisgl.engine.engine import Engine
    monkeypatch.setattr(dist,'_TP_INFO',DistributedInfo(0,1))
    config_for_checkpoint(tmp_path)
    prefix='model.language_model.layers.0.mlp.experts'
    tensors={}
    for expert in range(4):
        for j,part in enumerate(['gate_proj','up_proj','down_proj']):
            shape=(256,128) if part=='down_proj' else (128,256)
            tensors[f'{prefix}.{expert}.{part}.weight']=torch.full(shape,expert+j+1).to(torch.float8_e4m3fn)
            tensors[f'{prefix}.{expert}.{part}.weight_scale_inv']=torch.full(
                (shape[0]//128,shape[1]//128), .125*(expert+j+1),dtype=torch.bfloat16)
    save_file(tensors,tmp_path/'model.safetensors')
    loaded=Engine._load_weight_state_dict(SimpleNamespace(device=torch.device('cpu'),dtype=torch.bfloat16),
        SimpleNamespace(model_path=str(tmp_path),use_dummy_weight=False,quantization='fp8'))
    k='model.layers.0.mlp.experts.gate_up_proj'
    expected=torch.stack([torch.cat([tensors[f'{prefix}.{e}.{p}.weight'] for p in ['gate_proj','up_proj']]) for e in range(4)])
    assert torch.equal(loaded[k].view(torch.uint8),expected.view(torch.uint8))
    scales=torch.stack([torch.cat([tensors[f'{prefix}.{e}.{p}.weight_scale_inv'] for p in ['gate_proj','up_proj']]) for e in range(4)])
    torch.testing.assert_close(loaded[k+'_scale_inv'],scales.float(),rtol=0,atol=0)


def test_explicit_opt_in_and_bad_format(tmp_path,monkeypatch):
    import minisgl.distributed.info as dist
    monkeypatch.setattr(dist,'_TP_INFO',DistributedInfo(0,1))
    config=config_for_checkpoint(tmp_path)
    with pytest.raises(ValueError,match='explicit'):list(load_weight(str(tmp_path),torch.device('cpu')))
    with pytest.raises(ValueError,match='Unsupported'):list(load_weight(str(tmp_path),torch.device('cpu'),quantization='int4'))
    bad_path=tmp_path/'bad';bad_path.mkdir()
    config.quantization_config['activation_scheme']='static';config.save_pretrained(bad_path)
    with pytest.raises(ValueError,match='serialized'):list(load_weight(str(bad_path),torch.device('cpu'),quantization='fp8'))


def test_scale_sharding_matches_blocks():
    for r in range(2):
        scales=torch.arange(2*4*4).reshape(2,4,4)
        values=scales.repeat_interleave(128,-2).repeat_interleave(128,-1)
        for name in ['experts.gate_up_proj','experts.down_proj']:
            a=_shard_tensor('model.layers.0.mlp.'+name,values,r,2,2)
            b=_shard_tensor('model.layers.0.mlp.'+name+'_scale_inv',scales,r,2,2)
            assert torch.equal(a,b.repeat_interleave(128,-2).repeat_interleave(128,-1))
    assert _get_expert_stack_info('model.layers.0.mlp.experts.2.gate_up_proj.weight_scale_inv') == ('model.layers.0.mlp.experts.gate_up_proj_scale_inv',2)


def test_linear_state_and_bf16_path(monkeypatch):
    from minisgl.layers.linear import LinearReplicated
    from minisgl.kernel import fp8
    linear=LinearReplicated(256,128,True)
    w=torch.randn(128,256);b=torch.randn(128)
    state={'weight':w,'bias':b}
    linear.load_state_dict(state)
    monkeypatch.setattr(fp8,'block_fp8_linear',lambda *a:pytest.fail('FP8 called on unquantized weights'))
    x=torch.randn(5,256)
    torch.testing.assert_close(linear.forward(x),torch.nn.functional.linear(x,w,b),rtol=0,atol=0)
    quant=LinearReplicated(256,128,False)
    qw=torch.ones(128,256).to(torch.float8_e4m3fn);s=torch.ones(1,2)
    state={'weight':qw,'weight_scale_inv':s}
    quant.load_state_dict(state)
    assert not state and quant.weight is qw and quant.weight_scale_inv is s


def test_python_fp8_option():
    from minisgl.engine import EngineConfig
    options=dict(model_path='unused-local-model',tp_info=DistributedInfo(0,1),dtype=torch.bfloat16)
    assert EngineConfig(**options).quantization is None
    assert EngineConfig(**options,quantization='fp8').quantization=='fp8'
