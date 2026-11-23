"""Compiler policy is lazy, local, and never hides compilation failures."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from minisgl.training.attention._compile import _inductor_backend, compile_attention


def test_cpu_attention_stays_eager_and_differentiable(monkeypatch):
    def unexpected_compile(*args, **kwargs):
        raise AssertionError("CPU attention must not invoke the compiler")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    function = compile_attention(lambda query: query.square())
    query = torch.tensor([2.0, -3.0], requires_grad=True)
    output = function(query)
    torch.testing.assert_close(output, query.square())
    torch.testing.assert_close(torch.autograd.grad(output.sum(), query)[0], 2 * query)


@pytest.mark.parametrize("fail", [False, True])
def test_cuda_policy_is_lazy_and_restored_after_each_call(monkeypatch, fail):
    from torch._dynamo import config
    from torch._dynamo.exc import FailOnRecompileLimitHit

    creations = []
    query = SimpleNamespace(device=torch.device("cuda"))

    def eager(query):
        raise AssertionError("A CUDA compiler failure must not silently use eager")

    def compiled(query):
        assert config.accumulated_recompile_limit == 1024
        if fail:
            raise FailOnRecompileLimitHit("test limit")
        return query

    def create(function, **options):
        assert function is eager
        creations.append(options)
        return compiled

    monkeypatch.setattr(torch, "compile", create)
    function = compile_attention(eager)
    assert not creations
    with config.patch(accumulated_recompile_limit=23):
        for _ in range(2):
            if fail:
                with pytest.raises(FailOnRecompileLimitHit, match="test limit"):
                    function(query)
            else:
                assert function(query) is query
            assert config.accumulated_recompile_limit == 23
    assert creations == [
        {
            "backend": _inductor_backend,
            "fullgraph": True,
            "dynamic": False,
            "recompile_limit": 1024,
        }
    ]


@pytest.mark.parametrize("fail", [False, True])
def test_backward_autocast_policy_is_restored_after_compilation(monkeypatch, fail):
    from torch import _inductor
    from torch._functorch import config

    graph, inputs, result = object(), [], object()

    def compile(graph_arg, inputs_arg):
        assert graph_arg is graph and inputs_arg is inputs
        assert config.backward_pass_autocast == "off"
        if fail:
            raise RuntimeError("compiler failed")
        return result

    monkeypatch.setattr(_inductor, "compile", compile)
    with config.patch(backward_pass_autocast="same_as_forward"):
        if fail:
            with pytest.raises(RuntimeError, match="compiler failed"):
                _inductor_backend(graph, inputs)
        else:
            assert _inductor_backend(graph, inputs) is result
        assert config.backward_pass_autocast == "same_as_forward"


def test_specialization_policy_does_not_leak_to_another_thread(monkeypatch):
    from torch._dynamo import config

    entered, release = Event(), Event()
    query = SimpleNamespace(device=torch.device("cuda"))

    def compiled(query):
        assert config.accumulated_recompile_limit == 1024
        entered.set()
        assert release.wait(timeout=5), "Test failed to release the compiler worker"
        return query

    monkeypatch.setattr(torch, "compile", lambda *args, **kwargs: compiled)
    function = compile_attention(lambda query: query)

    def worker():
        with config.patch(accumulated_recompile_limit=23):
            assert function(query) is query
            assert config.accumulated_recompile_limit == 23

    with config.patch(accumulated_recompile_limit=17), ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker)
        try:
            assert entered.wait(timeout=5), "Compiler worker did not start"
            assert config.accumulated_recompile_limit == 17
        finally:
            release.set()
        future.result(timeout=5)
