"""CUDA correctness of bounded compose and buffer-free recurrent/capture fusion.

No checkpoint or performance claims; replay tests are address/mask checks.
"""
from dataclasses import replace
import subprocess
import sys

import pytest
import torch

from minisgl.planned.forward_plan import BlockState, DecodeRequest, PlanCapacity, prepare_forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def scenario(dim, heads=4, width=8):
    from minisgl.planned.gdn_device import DevicePhase, BoundGDN
    torch.manual_seed(321)
    cap = PlanCapacity(0, 0, width, 6, 24)
    pool = torch.randn(2, 24, 2, heads, dim, dim, device="cuda", dtype=torch.float32) * .03
    pool[:, :, 0] += torch.eye(dim, device="cuda") * .7
    # Deliberately poison fresh slots; correct masks must not consume them.
    pool[:, 9].fill_(float("nan"))
    blocks = {i: BlockState(i, i, i != 9, i != 9) for i in range(10)}
    requests = [DecodeRequest(c, w) for c, w in zip(
        [(0, 1, 6), (0, 1, 6, 7), (0,), (0, 1, 9), ()], [6, 7, 8, 9, 5])]
    plan = prepare_forward(blocks, capacity=cap, decode=requests)
    meta = DevicePhase(plan.decode, pool.device)
    bound = BoundGDN(pool, meta)
    return bound, plan, requests, blocks


def compose_reference(pool, layer, requests, blocks):
    rows = []
    for r in requests:
        acc = torch.zeros_like(pool[layer, 0, 1])
        first = True
        for b in r.read_blocks:
            if blocks[b].populated:
                A, B = pool[layer, blocks[b].slot]
                acc = B.clone() if first else acc @ A + B
                first = False
        rows.append(acc)
    return torch.stack(rows)


def recurrence_reference(q, k, v, g, beta, initial):
    q, k, v = q.float(), k.float(), v.float()
    q = q / ((q*q).sum(-1, keepdim=True) + 1.e-6).sqrt()
    k = k / ((k*k).sum(-1, keepdim=True) + 1.e-6).sqrt()
    ratio = v.shape[2] // k.shape[2]
    q, k = q.repeat_interleave(ratio, 2)[:, 0], k.repeat_interleave(ratio, 2)[:, 0]
    q *= q.shape[-1] ** -.5
    decayed = initial * g[:, 0].exp()[..., None, None]
    innovation = beta[:, 0, :, None] * (v[:, 0] - (decayed*k[..., None, :]).sum(-1))
    after = decayed + innovation[..., None] * k[..., None, :]
    return (after * q[..., None, :]).sum(-1)[:, None]


def capture_reference(pool, layer, requests, blocks, key, value, alpha, beta):
    result = pool.clone()
    heads, d = pool.shape[3], pool.shape[-1]
    key = key.float().repeat_interleave(heads // key.shape[2], 2)[:, 0]
    key *= torch.rsqrt((key * key).sum(-1, keepdim=True) + 1.e-6)
    for i, req in enumerate(requests):
        block = blocks[req.write_to]
        if block.populated:
            A, B = pool[layer, block.slot]
        else:
            A = torch.eye(d, device=pool.device).expand(heads, d, d)
            B = torch.zeros_like(A)
        k = key[i]
        a, b = alpha[i, 0, :, None, None], beta[i, 0, :, None, None]
        new_A = a*A - a*b*(A*k[:, None]).sum(-1)[..., None]*k[:, None]
        new_B = a*B - a*b*(B*k[:, None]).sum(-1)[..., None]*k[:, None] + b*value[i, 0].float()[..., None]*k[:, None]
        result[layer, block.slot, 0], result[layer, block.slot, 1] = new_A, new_B
    return result


@pytest.mark.parametrize("dim", [5, 32, 128])
@torch.inference_mode()
def test_compose_terminal_scatter_matches_chain_reference(dim):
    bound, plan, requests, blocks = scenario(dim)
    before = bound.pool.clone()
    for layer in range(2):
        bound.initial.fill_(float("nan"))
        for buf in bound.frontiers: buf.fill_(float("nan"))
        actual = bound.compose(layer)
        expected = compose_reference(before, layer, requests, blocks)
        torch.testing.assert_close(actual[:len(requests)], expected, rtol=2e-5, atol=2e-6)
        assert torch.equal(actual[len(requests):], torch.zeros_like(actual[len(requests):]))
    torch.testing.assert_close(bound.pool, before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dim,heads,key_heads,dtype", [
    (5, 4, 2, torch.float32), (32, 4, 4, torch.bfloat16), (128, 32, 16, torch.bfloat16)])
@torch.inference_mode()
def test_recurrent_capture_writes_existing_pool_without_final_s(dim, heads, key_heads, dtype):
    bound, plan, requests, blocks = scenario(dim, heads)
    width = bound.meta.width
    q, k = [torch.randn(width, 1, key_heads, dim, device="cuda", dtype=dtype) for _ in range(2)]
    v = torch.randn(width, 1, heads, dim, device="cuda", dtype=dtype)
    g = -torch.rand(width, 1, heads, device="cuda") * .4
    beta = torch.rand_like(g); alpha = g.exp()
    output = torch.empty_like(v)
    before = bound.pool.clone()
    bound.compose(1)
    initial = bound.initial.clone()
    n = len(requests)
    expected_output = recurrence_reference(q[:n], k[:n], v[:n], g[:n], beta[:n], initial[:n]).to(dtype)
    expected_pool = capture_reference(before, 1, requests, blocks, k, v, alpha, beta)
    # Float inputs in inactive rows need not be safe: branches must skip them.
    for x in (q, k, v, g, beta, alpha): x[n:].fill_(float("nan"))
    bound.decode(1, q, k, v, g, beta, alpha, output)
    torch.testing.assert_close(output[:n], expected_output, rtol=.008 if dtype == torch.bfloat16 else 2e-5,
                               atol=1e-4 if dtype == torch.bfloat16 else 2e-6)
    torch.testing.assert_close(bound.pool, expected_pool, rtol=3e-5, atol=3e-6, equal_nan=True)
    torch.testing.assert_close(bound.initial, initial, rtol=0, atol=0)
    assert torch.equal(output[n:], torch.zeros_like(output[n:]))
    # No output state allocation is part of the API or the bound recipe.
    assert not hasattr(bound, "final_state")


@torch.inference_mode()
def test_same_graph_replays_changed_topology_and_fresh_flags():
    bound, plan, requests, blocks = scenario(32)
    n, h, d = bound.meta.width, bound.heads, bound.dim
    q, k, v = [torch.randn(n, 1, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    g = -torch.rand(n, 1, h, device="cuda")*.2
    beta, alpha = torch.rand_like(g), g.exp()
    out = torch.empty_like(v)
    original_pool = bound.pool.clone()
    def body():
        bound.compose(0)
        bound.decode(0, q, k, v, g, beta, alpha, out)
    body(); torch.cuda.synchronize()
    bound.pool.copy_(original_pool)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): body()
    addresses = [getattr(bound.meta, name).data_ptr() for name in bound.meta.fields]
    for reads in [((0,1,6), (0,1,6,7)), ((),), ((1,0), (0,), (0,1,9))] * 3:
        reqs = [DecodeRequest(c, w) for c, w in zip(reads, [6,7,9])]
        next_plan = prepare_forward(blocks, capacity=plan.capacity, decode=reqs)
        bound.meta.upload(next_plan.decode)
        bound.pool.copy_(original_pool)
        q.normal_(); k.normal_(); v.normal_()
        initial = compose_reference(original_pool, 0, reqs, blocks)
        want_out = recurrence_reference(q[:len(reqs)], k[:len(reqs)], v[:len(reqs)],
                                       g[:len(reqs)], beta[:len(reqs)], initial).to(out.dtype)
        want_pool = capture_reference(original_pool, 0, reqs, blocks, k, v, alpha, beta)
        graph.replay()
        torch.testing.assert_close(out[:len(reqs)], want_out, rtol=.008, atol=1e-4)
        torch.testing.assert_close(bound.pool, want_pool, rtol=3e-5, atol=3e-6, equal_nan=True)
    assert addresses == [getattr(bound.meta, name).data_ptr() for name in bound.meta.fields]


def test_gpu_binding_does_not_require_fla_import():
    subprocess.run([sys.executable, "-c", "import sys; sys.modules['fla']=None; "
                    "from minisgl.planned.gdn_device import BoundGDN"], check=True)


@torch.inference_mode()
def test_output_matches_fla_and_decode_creates_no_torch_buffers():
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    bound, plan, requests, blocks = scenario(32)
    w, h, d = bound.meta.width, bound.heads, bound.dim
    q, k, v = [torch.randn(w, 1, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    g = -torch.rand(w, 1, h, device="cuda") * .2
    beta, alpha = torch.rand_like(g), g.exp()
    out = torch.empty_like(v)
    saved = bound.pool.clone()
    bound.compose(0)
    expected, _ = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta,
        initial_state=bound.initial, output_final_state=True,
        use_qk_l2norm_in_kernel=True, state_v_first=True)
    bound.decode(0, q, k, v, g, beta, alpha, out)  # warm kernel
    torch.cuda.synchronize()
    bound.pool.copy_(saved)
    before = torch.cuda.memory_allocated()
    bound.decode(0, q, k, v, g, beta, alpha, out)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    torch.testing.assert_close(out[:len(requests)], expected[:len(requests)], rtol=.008, atol=1e-4)


@torch.inference_mode()
def test_repeated_decode_compares_all_affines_over_32_steps():
    bound, plan, requests, blocks = scenario(32)
    w, h, d = bound.meta.width, bound.heads, bound.dim
    reference = bound.pool.clone()
    q, k, v = [torch.randn(w, 1, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    g = -torch.rand(w, 1, h, device="cuda")*.2
    beta, alpha = torch.rand_like(g), g.exp()
    out = torch.empty_like(v)
    for step in range(32):
        q.normal_(); k.normal_(); v.normal_()
        current_plan = prepare_forward(blocks, capacity=plan.capacity, decode=requests)
        bound.meta.upload(current_plan.decode)
        initial = compose_reference(reference, 0, requests, blocks)
        expected_output = recurrence_reference(q[:len(requests)], k[:len(requests)], v[:len(requests)],
                                               g[:len(requests)], beta[:len(requests)], initial).to(out.dtype)
        reference = capture_reference(reference, 0, requests, blocks, k, v, alpha, beta)
        bound.compose(0); bound.decode(0, q, k, v, g, beta, alpha, out)
        torch.testing.assert_close(out[:len(requests)], expected_output, rtol=.01, atol=2e-4)
        torch.testing.assert_close(bound.pool, reference, rtol=5e-5, atol=5e-6, equal_nan=True)
        for r in requests:
            b = blocks[r.write_to]
            blocks[r.write_to] = replace(b, populated=True, has_conv=True, revision=b.revision+1)
