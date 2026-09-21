"""CUDA capture of compatibility routing/alignment and unchanged expert GEMMs."""

import pytest
import torch

from minisgl.moe.fused import fused_experts_impl, fused_topk, moe_align_block_size


def alignment_reference(ids, block, experts):
    # Reference uses stable ordering; within-expert order is not required.
    flat = ids.cpu().flatten().tolist()
    sorted_ids, expert_ids = [], []
    for expert in range(experts):
        tokens = [i for i, value in enumerate(flat) if value == expert]
        padded = ((len(tokens) + block - 1) // block) * block
        sorted_ids.extend(tokens + [len(flat)] * (padded - len(tokens)))
        expert_ids.extend([expert] * (padded // block))
    return sorted_ids, expert_ids


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_alignment_known_capacity(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    for rows, block in ((1, 4), (5, 16), (32, 8)):
        ids = torch.randint(0, 8, (rows, 2), device=device, dtype=torch.int32)
        actual, experts, count = moe_align_block_size(ids, block, 8)
        expected, expected_experts = alignment_reference(ids, block, 8)
        assert count.item() == len(expected)
        actual_ids = actual[:len(expected)].tolist()
        offset = 0
        for expert in range(8):
            size = expected_experts.count(expert) * block
            assert sorted(actual_ids[offset:offset + size]) == sorted(expected[offset:offset + size])
            offset += size
        assert experts[:len(expected_experts)].tolist() == expected_experts
        assert (actual[len(expected):] == ids.numel()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph capture')
@pytest.mark.parametrize('fp8', [False, True])
@torch.inference_mode()
def test_routing_alignment_and_experts_capture(monkeypatch, fp8):
    import minisgl.moe.fused as module
    # Exercise the compatibility branch even on GPUs that have sgl_kernel.
    monkeypatch.setattr(module, '_use_torch_moe_fallback', lambda device: True)
    torch.manual_seed(801)
    rows, hidden, intermediate, experts = 5, 256, 128, 4
    x = torch.randn(rows, hidden, device='cuda', dtype=torch.bfloat16)
    gate = torch.randn(rows, experts, device='cuda')
    w1 = torch.randn(experts, 2 * intermediate, hidden, device='cuda', dtype=torch.bfloat16) * .02
    w2 = torch.randn(experts, hidden, intermediate, device='cuda', dtype=torch.bfloat16) * .02
    kwargs = {}
    if fp8:
        w1 = (w1.float() * 128).to(torch.float8_e4m3fn)
        w2 = (w2.float() * 128).to(torch.float8_e4m3fn)
        kwargs = dict(w1_scale=torch.full((experts, 2, 2), 1/128, device='cuda'),
                      w2_scale=torch.full((experts, 2, 1), 1/128, device='cuda'))

    def forward():
        scores, ids = fused_topk(x, gate, 2, True)
        aligned = moe_align_block_size(ids, 16, experts)
        output = fused_experts_impl(x.clone(), w1, w2, scores, ids, **kwargs)
        return ids, aligned, output

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids, aligned, output = forward()
    for step in range(4):
        gate.copy_(torch.randn_like(gate) * (step + 1))
        x.copy_(torch.randn_like(x))
        expected_ids, expected_alignment, expected_output = forward()
        graph.replay()
        torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
        for actual, expected in zip(aligned, expected_alignment):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(output, expected_output, atol=0, rtol=0)
