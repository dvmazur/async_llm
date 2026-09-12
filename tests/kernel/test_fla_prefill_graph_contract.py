"""Prove the installed FLA varlen kernels support capacity metadata replay.

No vendored kernels or global cache patches. This prototype deliberately bypasses
the fwd_h wrapper that caches chunk_offsets by tensor identity.
"""

import pytest
import torch


def candidate_chunk(*args):
    from minisgl.kernel.gdn_prefill import chunk_gdn
    return chunk_gdn(*args, output_final_state=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FLA CUDA graph contract')
@torch.inference_mode()
def test_one_capture_changes_lengths_chunks_and_request_count():
    pytest.importorskip('fla.ops.gated_delta_rule')
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(213)
    rows, workers, heads, dim = 256, 4, 2, 64
    chunks = rows // 64 + workers
    q, k, v = [torch.randn(1, rows, heads, dim, device='cuda', dtype=torch.bfloat16) * .1
               for _ in range(3)]
    g = -torch.rand(1, rows, heads, device='cuda') * .05
    beta = torch.rand(1, rows, heads, device='cuda')
    initial = torch.randn(workers, heads, dim, dim, device='cuda') * .1
    cu = torch.zeros(workers + 1, device='cuda', dtype=torch.int32)
    chunk_indices = torch.zeros(chunks, 2, device='cuda', dtype=torch.int32)
    offsets = torch.zeros(workers + 1, device='cuda', dtype=torch.int32)

    def prepare(lengths):
        boundaries, starts, indices = [0], [0], []
        for w, length in enumerate(lengths):
            boundaries.append(boundaries[-1] + length)
            count = (length + 63) // 64
            indices.extend((w, c) for c in range(count))
            starts.append(starts[-1] + count)
        indices += [(0, chunks - 1)] * (chunks - len(indices))
        for dst, source in ((cu, boundaries), (chunk_indices, indices), (offsets, starts)):
            dst.copy_(torch.tensor(source, dtype=dst.dtype, device=dst.device))

    prepare([64, 64, 64, 64])
    def forward():
        return candidate_chunk(q, k, v, g, beta, initial, cu, chunk_indices, offsets)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, final = forward()
    pointers = (cu.data_ptr(), chunk_indices.data_ptr(), offsets.data_ptr())
    for lengths in ([65, 20, 0, 0], [63, 65, 19, 0], [128, 1, 0, 0],
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
                torch.testing.assert_close(actual_final[worker], initial[worker], atol=0, rtol=0)
            start += length
        assert torch.count_nonzero(actual_output[:, start:]) == 0
        assert pointers == (cu.data_ptr(), chunk_indices.data_ptr(), offsets.data_ptr())
