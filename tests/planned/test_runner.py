import pytest
import torch

from test_decoder import setup
from minisgl.planned.runner import ProgramRunner
from minisgl.planned.forward_plan import prepare_forward
from minisgl.planned.attention_plan import prepare_attention
from minisgl.planned.model_io import prepare_inputs

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@torch.inference_mode()
def test_full_runner_capture_does_not_mutate_states_and_retains_old_outputs(monkeypatch):
    _,blocks,kv,f,ap,inputs,features,im,gdn,k,v,_,_,_,_,program=setup(monkeypatch,fp8=True)
    runner=ProgramRunner(program)
    originals=[x.clone() for x in (gdn.affine,gdn.conv,k,v)]
    runner.prepare_execution(f,ap,inputs,features)
    assert runner.body.captures==1 and runner.forward_count==0
    for t,old in zip((gdn.affine,gdn.conv,k,v),originals):
        torch.testing.assert_close(t,old,rtol=0,atol=0,equal_nan=True)
    result=runner.execute_prepared();result.completion.synchronize()
    retained=result.logits.clone()
    assert result.logits.data_ptr()!=program.logits.data_ptr()
    for i in range(3):
        now=prepare_forward(blocks,capacity=f.capacity,prefill=f.prefill_requests[:1],decode=f.decode_requests[:1])
        a=prepare_attention(kv,now,ap.capacity)
        inp=prepare_inputs(now,[7+i]*sum(now.rows.active),vocab_size=program.vocab)
        for t,old in zip((gdn.affine,gdn.conv,k,v),originals):t.copy_(old)
        runner.prepare_execution(now,a,inp)
        current=runner.execute_prepared();current.completion.synchronize()
        torch.testing.assert_close(result.logits,retained,rtol=0,atol=0)
    assert runner.body.captures==1 and runner.body.replays==4
    assert runner.forward_count==4 and runner.eager_count==0
    with pytest.raises(RuntimeError,match="already consumed"):runner.execute_prepared()


@torch.inference_mode()
def test_no_fla_full_decoder_chooses_eager_before_state_writes(monkeypatch):
    from minisgl.models import qwen3_5_delta as delta
    monkeypatch.setattr(delta,"_fla_chunk",None);monkeypatch.setattr(delta,"_fla_recurrent",None)
    net,blocks,kv,f,ap,inputs,features,im,gdn,k,v,ref,att,_,_,program=setup(monkeypatch,fp8=False)
    assert not program.graph_compatible
    runner=ProgramRunner(program)
    before=[x.clone() for x in (gdn.affine,gdn.conv,k,v)]
    runner.prepare_execution(f,ap,inputs,features)
    for t,old in zip((gdn.affine,gdn.conv,k,v),before):
        torch.testing.assert_close(t,old,rtol=0,atol=0,equal_nan=True)
    from test_decoder import old_decoder
    wanted=old_decoder(net,f,inputs,features,kv,im,ref,att)
    result=runner.execute_prepared();result.completion.synchronize()
    assert runner.body is None and runner.eager_count==1 and not result.used_graph
    assert torch.isfinite(result.logits).all()
    torch.testing.assert_close(result.logits,wanted,rtol=.06,atol=.008)
    assert float((result.logits.float()-wanted.float()).norm()/wanted.float().norm())<.025


@torch.inference_mode()
def test_failed_prepare_cannot_execute_stale_plan(monkeypatch):
    from dataclasses import replace
    # A malformed input must be rejected before any body/retention work.
    data=setup(monkeypatch)
    runner=ProgramRunner(data[-1],use_graph=False)
    f,ap,inputs,features=data[3:7]
    runner.prepare_execution(f,ap,inputs,features)
    with pytest.raises(ValueError,match="token/image metadata"):
        runner.prepare_execution(f,ap,replace(inputs,token_ids=(999,)*len(inputs.token_ids)),features)
    with pytest.raises(RuntimeError,match="not prepared"):runner.execute_prepared()
