"""Prove the installed FLA varlen kernels support capacity metadata replay.

No vendored kernels or global cache patches. This prototype deliberately bypasses
the fwd_h wrapper that caches chunk_offsets by tensor identity.
"""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FLA CUDA bindings')
def test_guards_reference_installed_jit_bodies_and_keep_original_configs():
    import importlib
    pytest.importorskip('fla.ops.gated_delta_rule')
    from minisgl.kernel.gdn_fla_guards import kernels
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction
    bound = kernels()
    assert bound is kernels()
    assert len(bound) == 5
    for launch, body in bound:
        assert isinstance(body, JITFunction)
        assert body.fn.__module__.startswith('fla.')
        assert launch.configs
        assert 'BODY' in launch.arg_names
        original = getattr(importlib.import_module(body.fn.__module__), body.fn.__name__)
        found = False
        while not isinstance(original, JITFunction):
            if isinstance(original, Autotuner):
                assert launch.configs is original.configs
                assert launch.keys == original.keys
                found = True
            original = original.fn
        assert found and original is body


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FLA CUDA graph contract')
@pytest.mark.parametrize('skip_empty', [False, True])
@pytest.mark.parametrize('gate_dtype', [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_one_capture_changes_lengths_chunks_and_request_count(skip_empty, gate_dtype, monkeypatch):
    pytest.importorskip('fla.ops.gated_delta_rule')
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(213)
    rows, workers, heads, dim = 256, 4, 2, 64
    chunks = rows // 64 + workers
    q, k, v = [torch.randn(1, rows, heads, dim, device='cuda', dtype=torch.bfloat16) * .1
               for _ in range(3)]
    g = (-torch.rand(1, rows, heads, device='cuda') * .05).to(gate_dtype)
    beta = torch.rand(1, rows, heads, device='cuda')
    initial = torch.randn(workers, heads, dim, dim, device='cuda') * .1
    cu = torch.zeros(workers + 1, device='cuda', dtype=torch.int32)
    chunk_indices = torch.zeros(chunks, 2, device='cuda', dtype=torch.int32)
    offsets = torch.zeros(workers + 1, device='cuda', dtype=torch.int32)
    original_empty = torch.Tensor.new_empty

    def poisoned_empty(tensor, *args, **kwargs):
        value = original_empty(tensor, *args, **kwargs)
        if tuple(value.shape) == (1, chunks, heads, dim, dim):
            value.fill_(float('nan'))  # H must produce every chunk that O reads.
        elif tuple(value.shape) == tuple(initial.shape) and value.dtype == torch.float32:
            value.fill_(123)  # Inactive final states must not be touched in skip mode.
        return value

    monkeypatch.setattr(torch.Tensor, 'new_empty', poisoned_empty)

    def prepare(lengths):
        boundaries, starts, indices = [0], [0], []
        for w, length in enumerate(lengths):
            boundaries.append(boundaries[-1] + length)
            count = (length + 63) // 64
            indices.extend((w, c) for c in range(count))
            starts.append(starts[-1] + count)
        # Invalid sentinels prove guards happen BEFORE any table/sequence read.
        chunk_indices.fill_(2**30)
        if indices:
            chunk_indices[:len(indices)].copy_(torch.tensor(indices, device='cuda', dtype=torch.int32))
        for dst, source in ((cu, boundaries), (offsets, starts)):
            dst.copy_(torch.tensor(source, dtype=dst.dtype, device=dst.device))

    prepare([0, 0, 0, 0])  # Even first warmup/capture may have zero actual work.
    def forward():
        from minisgl.kernel.gdn_prefill import chunk_gdn
        return chunk_gdn(q, k, v, g, beta, initial, cu, chunk_indices, offsets,
                          output_final_state=True, skip_empty_states=skip_empty)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, final = forward()
    from minisgl.kernel.gdn_fla_guards import kernels
    cache_sizes = [len(launch.cache) for launch, _ in kernels()]
    pointers = (cu.data_ptr(), chunk_indices.data_ptr(), offsets.data_ptr())
    for lengths in ([65, 20, 0, 0], [0, 63, 0, 65], [63, 65, 19, 0], [128, 1, 0, 0],
                    [64, 64, 64, 64], [1, 0, 0, 0], [0, 0, 0, 0]):
        prepare(lengths)
        graph.replay()
        actual_output, actual_final = output.clone(), final.clone()
        start = 0
        for worker, length in enumerate(lengths):
            if length:
                section = slice(start, start + length)
                want, want_final = chunk_gated_delta_rule(
                    q[:, section], k[:, section], v[:, section], g=g[:, section],
                    beta=beta[:, section], initial_state=initial[worker:worker+1],
                    output_final_state=True, use_qk_l2norm_in_kernel=True)
                torch.testing.assert_close(actual_output[:, section], want, atol=2e-4, rtol=2e-3)
                torch.testing.assert_close(actual_final[worker:worker+1], want_final, atol=2e-5, rtol=2e-4)
            else:
                if skip_empty:
                    assert torch.all(actual_final[worker] == 123)
                else:
                    torch.testing.assert_close(actual_final[worker], initial[worker], atol=0, rtol=0)
            start += length
        assert torch.count_nonzero(actual_output[:, start:]) == 0
        assert pointers == (cu.data_ptr(), chunk_indices.data_ptr(), offsets.data_ptr())
    assert cache_sizes == [len(launch.cache) for launch, _ in kernels()]
