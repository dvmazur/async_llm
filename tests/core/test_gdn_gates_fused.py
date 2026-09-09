"""Gate-only fusion vs literal old expression; other GDN math is unchanged."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import minisgl.models.qwen3_5_delta as delta


def old_gates(self, a, b):
    # Literal pre-change method, retained as an independent test reference.
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
    return beta, g


def inputs(shape, dtype, device, strided=False):
    gen = torch.Generator(device=device).manual_seed(734)
    if strided:
        a = torch.randn(shape[0]*2, shape[-1], device=device, dtype=dtype, generator=gen)[::2]
        b = torch.randn(shape[0]*2, shape[-1], device=device, dtype=dtype, generator=gen)[::2]
    else:
        a = torch.randn(shape, device=device, dtype=dtype, generator=gen) * 5
        b = torch.randn(shape, device=device, dtype=dtype, generator=gen) * 5
    state = SimpleNamespace(
        A_log=torch.randn(shape[-1], device=device, dtype=dtype, generator=gen),
        dt_bias=torch.randn(shape[-1], device=device, dtype=dtype, generator=gen),
    )
    return state, a, b


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
def test_cpu_remains_exact_and_does_not_compile(monkeypatch, dtype):
    state, a, b = inputs((7, 4), dtype, 'cpu')
    def unexpected(*a, **kw):
        raise AssertionError('CPU must not invoke the compiled CUDA path')
    monkeypatch.setattr(delta, '_compiled_gdn_gates', unexpected)
    actual = delta.Qwen3_5GatedDeltaNet._gates(state, a, b)
    expected = old_gates(state, a, b)
    assert all(torch.equal(x,y) for x,y in zip(actual, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_real_shapes_layouts_dtypes_and_parameter_changes():
    # 3D one-token extend is a distinct graph; batch-one is also specialized.
    # Avoid resetting compiler state: this tests production dynamic graph reuse.
    cases = [((n,32), torch.bfloat16, False) for n in (3,1,2,6,18,36,91,138,265,1024)]
    cases += [((6,1,32), torch.bfloat16, False), ((18,32), torch.bfloat16, True),
              ((18,32), torch.float16, False), ((18,32), torch.float32, False)]
    for shape, dtype, strided in cases:
        state,a,b = inputs(shape,dtype,'cuda',strided)
        before = [x.clone() for x in (a,b,state.A_log,state.dt_bias)]
        actual = delta.Qwen3_5GatedDeltaNet._gates(state,a,b)
        expected = old_gates(state,a,b)
        assert actual[0].dtype == b.dtype and actual[1].dtype == torch.float32
        torch.testing.assert_close(actual[0],expected[0],rtol=2e-6,atol=2e-7)
        torch.testing.assert_close(actual[1],expected[1],rtol=2e-6,atol=2e-6)
        assert all(torch.equal(x,y) for x,y in zip((a,b,state.A_log,state.dt_bias),before))
        # No stale precomputed constants when parameters are loaded/replaced.
        state.A_log = state.A_log + .25
        state.dt_bias = state.dt_bias - .125
        updated = delta.Qwen3_5GatedDeltaNet._gates(state,a,b)
        for x,y in zip(updated,old_gates(state,a,b)):
            torch.testing.assert_close(x,y,rtol=2e-6,atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_gate_extremes_and_softplus_threshold():
    a = torch.tensor([-100.,-30.,-1.,0.,1.,19.999,20.,20.001,80.,100.],device='cuda').view(-1,1).expand(-1,32).contiguous()
    b = a.clone()
    state = SimpleNamespace(A_log=torch.zeros(32,device='cuda'),dt_bias=torch.zeros(32,device='cuda'))
    actual = delta.Qwen3_5GatedDeltaNet._gates(state,a,b)
    expected = old_gates(state,a,b)
    for x,y in zip(actual,expected):
        assert torch.isfinite(x).all()
        torch.testing.assert_close(x,y,rtol=2e-6,atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_compilation_is_independent_of_fla(monkeypatch):
    state,a,b = inputs((6,32),torch.bfloat16,'cuda')
    monkeypatch.setattr(delta,'_fla_recurrent',None)
    monkeypatch.setattr(delta,'_fla_chunk',None)
    actual = delta.Qwen3_5GatedDeltaNet._gates(state,a,b)
    for x,y in zip(actual,old_gates(state,a,b)):
        torch.testing.assert_close(x,y,rtol=2e-6,atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_no_fla_recurrence_old_vs_compiled_gates(monkeypatch):
    monkeypatch.setattr(delta,'_fla_recurrent',None)
    # Compare a sequence of recurrent updates, not just the gate outputs.
    # Rectangular state also checks the V-first fallback convention.
    gen=torch.Generator(device='cuda').manual_seed(884)
    params,a,b=inputs((6,32),torch.bfloat16,'cuda')
    old_state=torch.randn(6,32,12,8,device='cuda',generator=gen)*.1
    new_state=old_state.clone()
    for _ in range(16):
        a=torch.randn(a.shape,device='cuda',dtype=a.dtype,generator=gen)
        b=torch.randn(b.shape,device='cuda',dtype=b.dtype,generator=gen)
        q,k=[torch.randn(6,1,32,8,device='cuda',generator=gen) for _ in range(2)]
        v=torch.randn(6,1,32,12,device='cuda',generator=gen)
        old_beta,old_g=old_gates(params,a,b)
        new_beta,new_g=delta.Qwen3_5GatedDeltaNet._gates(params,a,b)
        old_out,old_state=delta._recurrent_delta(q,k,v,old_g[:,None],old_beta[:,None],
                                                old_state,state_v_first=True)
        new_out,new_state=delta._recurrent_delta(q,k,v,new_g[:,None],new_beta[:,None],
                                                new_state,state_v_first=True)
        torch.testing.assert_close(new_out,old_out,rtol=1e-5,atol=2e-6)
        torch.testing.assert_close(new_state,old_state,rtol=1e-5,atol=2e-6)
