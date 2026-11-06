import asyncio
import re

import pytest
import torch
from minisgl.shared_cache import AsyncContext
from minisgl.training import RecordedAsyncLLM, Trace
from minisgl.training.gdn import GDNReplay, delta
from minisgl.training.trace import BlockUpdate, ForwardBatch, ForwardRequest
from torch.utils._python_dispatch import TorchDispatchMode

from tests.training.fixtures import native_torch_kernels as native_torch_kernels
from tests.training.helpers import Session, generate_stress_example, make_llm
from tests.training.references import (
    Float64DeltaMath,
    sequential_delta,
)
from tests.training.workloads import (
    mutation_trace,
    repeated_view_trace,
    single_block_trace,
    uneven_history_trace,
)


def test_triton_saved_vector_offsets_are_64_bit(tmp_path, monkeypatch):
    triton = pytest.importorskip("triton")
    from minisgl.training.gdn_kernel import _read_paths
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    constants = {
        "N": 8192,
        "H": 16,
        "DK": 128,
        "DV": 128,
        "DEPTH": 129,
        "WIDTH": 0,
        "SAVE": True,
        "BK": 128,
        "BV": 128,
        "BS": 1,
    }
    assert (constants["DEPTH"] - 1) * constants["N"] * constants["H"] * constants["DK"] >= 2**31
    signature = {
        name: "*i64" if name in {"Lookup", "Paths"} else "*fp32"
        for name in _read_paths.arg_names[: _read_paths.arg_names.index("N")]
    }
    compiled = triton.compile(
        ASTSource(_read_paths, signature, constants),
        target=GPUTarget("cuda", 80, 32),
        options={"num_warps": 4, "enable_fp_fusion": False},
    )
    ir = compiled.asm["ttir"]
    definitions = dict(re.findall(r"^\s*(%\w+)\s*=\s*(.*)$", ir, re.MULTILINE))
    vector_stores = []
    for pointer in re.findall(r"tt.store (%\w+),", ir):
        pending, dependencies = [pointer], set()
        while pending:
            name = pending.pop()
            if name not in dependencies:
                dependencies.add(name)
                pending.extend(re.findall(r"%\w+", definitions.get(name, "")))
        if "%Vectors" in dependencies:
            vector_stores.append(dependencies)
    assert len(vector_stores) == 1
    for name in vector_stores[0]:
        assert not re.search(r"arith\.(addi|muli).*:\s*i32\b", definitions.get(name, ""))


@pytest.mark.parametrize("backend", ["invalid", "triton"])
def test_delta_rejects_invalid_or_cpu_triton_backend(backend):
    trace = mutation_trace()
    n = len(trace.tokens)
    with pytest.raises(ValueError, match="backend|CUDA|cuda"):
        replay = GDNReplay.from_trace(trace, torch.arange(n), read_backend=backend)
        delta(
            torch.randn(n, 1, 3),
            torch.randn(n, 1, 3),
            torch.randn(n, 1, 4),
            torch.rand(n, 1),
            torch.rand(n, 1),
            replay,
        )


@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
@pytest.mark.parametrize("segment_size", [1, 2, 8, 16])
@pytest.mark.parametrize("trace_kind", ["single", "mutation", "uneven", "stress", "repeated"])
def test_delta_fp64_outputs_and_gradients(trace_kind, segment_size, read_mode, record_property):
    if trace_kind == "stress":
        data = generate_stress_example()
        trace = Trace.from_dict(data["trace"])
    elif trace_kind == "uneven":
        trace = uneven_history_trace()
    elif trace_kind == "repeated":
        trace = repeated_view_trace()
    else:
        trace = single_block_trace(True) if trace_kind == "single" else mutation_trace()
    torch.manual_seed(7)
    n = len(trace.tokens)
    order = torch.randperm(n)
    replay = GDNReplay.from_trace(trace, order, segment_size=segment_size, read_mode=read_mode)
    inputs = tuple(
        (
            x.requires_grad_()
            for x in (
                torch.randn(n, 2, 3, dtype=torch.float64),
                torch.randn(n, 2, 3, dtype=torch.float64),
                torch.randn(n, 2, 4, dtype=torch.float64),
                torch.rand(n, 2, dtype=torch.float64),
                torch.rand(n, 2, dtype=torch.float64),
            )
        )
    )
    expected = sequential_delta(*inputs, trace, order)
    weights = torch.randn_like(expected)
    expected_grads = torch.autograd.grad((expected * weights).sum(), inputs)
    with Float64DeltaMath():
        actual = delta(*inputs, replay)
        actual_grads = torch.autograd.grad((actual * weights).sum(), inputs)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-12, rtol=2e-12)
    record_property("delta_fp64_output_max_abs", (actual - expected).abs().max().item())
    record_property(
        "delta_fp64_gradient_max_abs",
        max(((a - b).abs().max().item() for a, b in zip(actual_grads, expected_grads))),
    )


@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
@pytest.mark.parametrize("segment_size", [1, 8])
def test_delta_long_causal_fp64_outputs_and_gradients(segment_size, read_mode, record_property):
    torch.manual_seed(7)
    n, heads, dk, dv = 512, 2, 16, 12
    trace = Trace(
        batches=[ForwardBatch(0, "prefill", (ForwardRequest(tuple(range(n)), 0, 0, ((0, 0),)),))]
    )
    order = torch.randperm(n)
    replay = GDNReplay.from_trace(trace, order, segment_size=segment_size, read_mode=read_mode)
    inputs = tuple(
        x.requires_grad_()
        for x in (
            torch.randn(n, heads, dk, dtype=torch.float64),
            torch.randn(n, heads, dk, dtype=torch.float64),
            torch.randn(n, heads, dv, dtype=torch.float64),
            0.995 + 0.005 * torch.rand(n, heads, dtype=torch.float64),
            0.01 + 0.04 * torch.rand(n, heads, dtype=torch.float64),
        )
    )
    query, key = (x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in inputs[:2])
    value, alpha, beta = inputs[2:]
    state = value.new_zeros(heads, dk, dv)
    outputs = [None] * n
    for row in order.argsort().tolist():
        state = state * alpha[row, :, None, None]
        residual = value[row] - torch.einsum("hkv,hk->hv", state, key[row])
        state = state + torch.einsum("hk,hv->hkv", key[row], beta[row, :, None] * residual)
        outputs[row] = torch.einsum("hkv,hk->hv", state, query[row] / dk**0.5)
    expected = torch.stack(outputs)
    weights = torch.randn_like(expected)
    expected_grads = torch.autograd.grad((expected * weights).sum(), inputs)
    with Float64DeltaMath():
        actual = delta(*inputs, replay)
        actual_grads = torch.autograd.grad((actual * weights).sum(), inputs)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-12, rtol=2e-12)
    record_property("long_causal_fp64_output_max_abs", (actual - expected).abs().max().item())
    record_property(
        "long_causal_fp64_gradient_max_abs",
        max((a - b).abs().max().item() for a, b in zip(actual_grads, expected_grads)),
    )


@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
@pytest.mark.parametrize("segment_size", [1, 2, 8])
@pytest.mark.parametrize("alpha,beta", [(0, 0), (0, 1), (1, 0), (1, 1), (1e-12, 1 - 1e-12)])
def test_delta_gate_boundaries(alpha, beta, segment_size, read_mode):
    torch.manual_seed(7)
    trace = mutation_trace()
    n = len(trace.tokens)
    order = torch.randperm(n)
    replay = GDNReplay.from_trace(trace, order, segment_size=segment_size, read_mode=read_mode)
    inputs = tuple(
        x.requires_grad_()
        for x in (
            torch.randn(n, 2, 3, dtype=torch.float64),
            torch.randn(n, 2, 3, dtype=torch.float64),
            torch.randn(n, 2, 4, dtype=torch.float64),
            torch.full((n, 2), alpha, dtype=torch.float64),
            torch.full((n, 2), beta, dtype=torch.float64),
        )
    )
    expected = sequential_delta(*inputs, trace, order)
    weights = torch.randn_like(expected)
    expected_grads = torch.autograd.grad((expected * weights).sum(), inputs)
    with Float64DeltaMath():
        actual = delta(*inputs, replay)
        actual_grads = torch.autograd.grad((actual * weights).sum(), inputs)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    if beta == 0:
        assert not actual.any()
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert actual_grad.isfinite().all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
@pytest.mark.parametrize("segment_size", [1, 2, 8])
def test_delta_copy_clear_gradcheck(segment_size, read_mode):
    trace = Trace(
        batches=[
            ForwardBatch(0, "prefill", (ForwardRequest((1, 2), 0, 0, ((0, 0),)),)),
            ForwardBatch(
                1,
                "decode",
                (
                    ForwardRequest((3,), 0, 0, ((1, 4), (0, 0))),
                    ForwardRequest((4,), 1, 4, ((0, 0), (1, 4))),
                ),
            ),
        ],
        updates=[
            BlockUpdate(1, 1, refs=((0, 2),)),
            BlockUpdate(1, 1, refs=((1, 2), (1, 2))),
            BlockUpdate(1, 0, refs=()),
        ],
    )
    replay = GDNReplay.from_trace(
        trace, torch.tensor([2, 0, 3, 1]), segment_size=segment_size, read_mode=read_mode
    )
    torch.manual_seed(7)
    inputs = tuple(
        x.requires_grad_()
        for x in (
            torch.randn(4, 1, 2, dtype=torch.float64),
            torch.randn(4, 1, 2, dtype=torch.float64),
            torch.randn(4, 1, 3, dtype=torch.float64),
            torch.rand(4, 1, dtype=torch.float64) * 0.6 + 0.2,
            torch.rand(4, 1, dtype=torch.float64) * 0.6 + 0.2,
        )
    )
    with Float64DeltaMath():
        assert torch.autograd.gradcheck(
            lambda *values: delta(*values, replay), inputs, eps=1e-6, atol=1e-7, rtol=1e-5
        )


@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
@pytest.mark.parametrize("segment_size", [1, 2, 8, 16])
@pytest.mark.parametrize("zero_gates", [False, True])
@pytest.mark.parametrize("cache_reused", [False, True])
def test_delta_compact_dense_compositions_and_shared_sources(
    segment_size, zero_gates, read_mode, cache_reused
):
    torch.manual_seed(7)
    n = 12
    levels = [
        torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]),
        torch.tensor([[13, 14], [15, 16], [17, 18]]),
        torch.tensor([[19, 20], [21, 13]]),
        torch.tensor([[22, 19], [20, 22], [13, 22], [22, 13], [13, 13]]),
        torch.tensor([[24, 21], [21, 25], [24, 25], [28, 22], [22, 28]]),
    ]
    reads = torch.tensor(
        [
            [31, 1, 0],
            [29, 13, 2],
            [30, 23, 3],
            [28, 22, 4],
            [32, 33, 5],
            [26, 27, 6],
            [19, 20, 7],
            [21, 22, 8],
            [13, 14, 9],
            [23, 23, 10],
            [1, 2, 11],
            [28, 28, 12],
        ]
    )
    members = [()] + [(i,) for i in range(n)]
    for level in levels:
        members.extend(members[left] + members[right] for left, right in level.tolist())
    inputs = tuple(
        x.requires_grad_()
        for x in (
            torch.randn(n, 2, 3, dtype=torch.float64),
            torch.randn(n, 2, 3, dtype=torch.float64),
            torch.randn(n, 2, 5, dtype=torch.float64),
            torch.rand(n, 2, dtype=torch.float64) * 0.3 + 0.7,
            torch.rand(n, 2, dtype=torch.float64),
        )
    )
    if zero_gates:
        with torch.no_grad():
            inputs[-2][::3] = 0
            inputs[-1][1::3] = 0
    query, key = (x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in inputs[:2])
    value, alpha, beta = inputs[2:]
    outputs = []
    for row, path in enumerate(reads.tolist()):
        state = value.new_zeros(2, 3, 5)
        for node in reversed(path):
            for source in members[node]:
                state = state * alpha[source, :, None, None]
                residual = value[source] - torch.einsum("hkv,hk->hv", state, key[source])
                state = state + torch.einsum(
                    "hk,hv->hkv", key[source], beta[source, :, None] * residual
                )
        outputs.append(torch.einsum("hkv,hk->hv", state, query[row] / 3**0.5))
    expected = torch.stack(outputs)
    weights = torch.randn_like(expected)
    expected_grads = torch.autograd.grad((expected * weights).sum(), inputs)
    replay = GDNReplay(
        torch.empty(n, 0, dtype=torch.long), levels, reads, segment_size, read_mode, cache_reused
    )
    if segment_size == 8 and cache_reused:
        assert {(0, 0), (0, 4), (4, 0), (4, 4)} <= {
            (left, right) for left, right, *_ in replay.partitions[0]
        }
    with Float64DeltaMath():
        actual = delta(*inputs, replay)
        actual_grads = torch.autograd.grad((actual * weights).sum(), inputs)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert actual_grad.isfinite().all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("gates", ["random", "zero", "one", "mixed"])
def test_delta_leaf_only_reads_match_recurrence(gates):
    torch.manual_seed(7)
    reads = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 0, 1, 0], [3, 2, 3, 1, 0], [4, 2, 0, 2, 4]])
    inputs = tuple(
        x.requires_grad_()
        for x in (
            torch.randn(4, 2, 3, dtype=torch.float64),
            torch.randn(4, 2, 3, dtype=torch.float64),
            torch.randn(4, 2, 5, dtype=torch.float64),
            torch.rand(4, 2, dtype=torch.float64),
            torch.rand(4, 2, dtype=torch.float64),
        )
    )
    with torch.no_grad():
        if gates in {"zero", "one"}:
            for gate in inputs[-2:]:
                gate.fill_(int(gates == "one"))
        elif gates == "mixed":
            inputs[-2].copy_(torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]]))
            inputs[-1].copy_(torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]]))
    query, key = (x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in inputs[:2])
    value, alpha, beta = inputs[2:]
    outputs = []
    for row, path in enumerate(reads.tolist()):
        state = value.new_zeros(2, 3, 5)
        for source in reversed(path):
            if source:
                source -= 1
                state = state * alpha[source, :, None, None]
                residual = value[source] - torch.einsum("hkv,hk->hv", state, key[source])
                state = state + torch.einsum(
                    "hk,hv->hkv", key[source], beta[source, :, None] * residual
                )
        outputs.append(torch.einsum("hkv,hk->hv", state, query[row] / 3**0.5))
    expected = torch.stack(outputs)
    weights = torch.randn_like(expected)
    expected_grads = torch.autograd.grad((expected * weights).sum(), inputs)
    replay = GDNReplay(torch.empty(4, 0, dtype=torch.long), [], reads)
    with Float64DeltaMath():
        actual = delta(*inputs, replay)
        actual_grads = torch.autograd.grad((actual * weights).sum(), inputs)
    assert not actual[0].any()
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert actual_grad.isfinite().all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("depth", [0, 3])
def test_delta_identity_only_reads_have_zero_outputs_and_gradients(depth):
    torch.manual_seed(7)
    n = 5
    replay = GDNReplay(
        torch.empty(n, 0, dtype=torch.long), [], torch.zeros(n, depth, dtype=torch.long)
    )
    inputs = tuple(
        x.requires_grad_()
        for x in (
            torch.randn(n, 2, 3),
            torch.randn(n, 2, 3),
            torch.randn(n, 2, 4),
            torch.rand(n, 2),
            torch.rand(n, 2),
        )
    )
    output = delta(*inputs, replay)
    assert not output.any()
    gradients = torch.autograd.grad(output.sum(), inputs)
    assert all(not gradient.any() for gradient in gradients)


def test_delta_leaf_saved_storage_is_linear_in_head_width(record_property):
    n, heads, depth = 16, 2, 6
    reads = (torch.arange(n)[:, None] + torch.arange(depth)[None]) % (n + 1)
    replay = GDNReplay(torch.empty(n, 0, dtype=torch.long), [], reads)
    sizes = []
    for dk in (16, 32, 64):
        torch.manual_seed(7)
        inputs = tuple(
            x.requires_grad_()
            for x in (
                torch.randn(n, heads, dk),
                torch.randn(n, heads, dk),
                torch.randn(n, heads, 3 * dk // 4),
                torch.rand(n, heads),
                torch.rand(n, heads),
            )
        )
        storages = {}

        def pack(tensor):
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = delta(*inputs, replay)
        size = sum(storage.nbytes() for storage in storages.values())
        output.square().mean().backward()
        assert all(tensor.grad.isfinite().all() for tensor in inputs)
        sizes.append(size)
        record_property(f"leaf_saved_bytes_width_{dk}", size)
    assert 0 < sizes[2] - sizes[1] <= 2.05 * (sizes[1] - sizes[0])


@pytest.mark.parametrize("segment_size", [1, 2, 8])
def test_delta_reused_leaf_merges_survive_clear_and_reuse(segment_size):

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                source, target, empty = [await session.create_block() for _ in range(3)]

                async def prefill(block, token):
                    await session.prefill_block([token], write_to=block, cache_view=[])

                await prefill(source, 1)
                await prefill(target, 2)
                for _ in range(8):
                    await session.append_block(target, source)
                copied = await session.merge_blocks(empty, target)
                await session.free_block(source)
                await prefill(source, 3)
                await prefill(copied, 4)
                order = torch.tensor([2, 0, 3, 1])
                replay = GDNReplay.from_trace(session.get_trace(), order, segment_size=segment_size)
                torch.manual_seed(7)
                inputs = tuple(
                    (
                        x.requires_grad_()
                        for x in (
                            torch.randn(4, 2, 3, dtype=torch.float64),
                            torch.randn(4, 2, 3, dtype=torch.float64),
                            torch.randn(4, 2, 5, dtype=torch.float64),
                            torch.rand(4, 2, dtype=torch.float64) * 0.6 + 0.2,
                            torch.rand(4, 2, dtype=torch.float64) * 0.6 + 0.2,
                        )
                    )
                )
                expected = sequential_delta(*inputs, session.get_trace(), order)
                inverse = order.argsort()
                weights = torch.randn_like(expected[-1])
                expected_grads = torch.autograd.grad((expected[inverse[3]] * weights).sum(), inputs)
                with Float64DeltaMath():
                    actual = delta(*inputs, replay)
                    actual_grads = torch.autograd.grad((actual[inverse[3]] * weights).sum(), inputs)
                torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-12, rtol=2e-12)
                    assert not actual_grad[inverse[2]].any()
                assert (actual_grads[2][inverse[:2]].abs().sum((1, 2)) > 0).all()
        finally:
            await llm.close()

    return asyncio.run(run())


def test_odd_history_self_copies_stay_compact_and_preserve_gradient_counts():

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                block = await session.create_block()
                await session.prefill_block([1, 2, 3], write_to=block, cache_view=[])
                for _ in range(40):
                    await session.append_block(block, block)
                assert block.num_tokens == 3 * 2**40
                await session.forward(cache_view=AsyncContext([block], block, 4))
                replay = GDNReplay.from_trace(session.get_trace(), torch.arange(4))
                assert (
                    sum((len(level) for level in replay.levels))
                    <= 2 * len(session.get_trace().updates) + 3
                )
                assert replay.reads.numel() <= 4 * 4
                assert replay.conv_sources.shape == (4, 4)
                inputs = tuple(
                    (
                        x.requires_grad_()
                        for x in (
                            torch.ones(4, 1, 1, dtype=torch.float64),
                            torch.ones(4, 1, 1, dtype=torch.float64),
                            torch.ones(4, 1, 1, dtype=torch.float64),
                            torch.ones(4, 1, dtype=torch.float64),
                            torch.zeros(4, 1, dtype=torch.float64),
                        )
                    )
                )
                with Float64DeltaMath():
                    output = delta(*inputs, replay)
                    gradients = torch.autograd.grad(output[-1].sum(), inputs)
                assert not output.any()
                for gradient in gradients:
                    assert gradient.isfinite().all()
                expected = torch.tensor([2**40, 2**40, 2**40, 1], dtype=torch.float64) / (1 + 1e-06)
                torch.testing.assert_close(gradients[-1][:, 0], expected, atol=2e-12, rtol=2e-12)
                for gradient in gradients[:-1]:
                    assert not gradient.any()
        finally:
            await llm.close()

    return asyncio.run(run())


@pytest.mark.parametrize("grad_indices", [tuple(range(5)), (0,), (1,), (2,), (3,), (4,), ()])
def test_delta_selective_input_gradients_match_recurrence(grad_indices):
    torch.manual_seed(7)
    trace = mutation_trace()
    n = len(trace.tokens)
    order = torch.randperm(n)
    replay = GDNReplay.from_trace(trace, order)
    inputs = tuple(
        tensor.requires_grad_(i in grad_indices)
        for i, tensor in enumerate(
            (
                torch.randn(n, 2, 3),
                torch.randn(n, 2, 3),
                torch.randn(n, 2, 4),
                torch.rand(n, 2),
                torch.rand(n, 2),
            )
        )
    )
    actual = delta(*inputs, replay)
    expected = sequential_delta(*inputs, trace, order)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    if grad_indices:
        weights = torch.randn_like(actual)
        parameters = tuple(inputs[i] for i in grad_indices)
        expected_grads = torch.autograd.grad((expected * weights).sum(), parameters)
        actual_grads = torch.autograd.grad((actual * weights).sum(), parameters)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=2e-5)
    else:
        assert not actual.requires_grad


def test_delta_read_saved_storage_grows_with_vectors_not_matrices(record_property):
    n, heads, dk, dv = 16, 2, 32, 24

    def saved_bytes(depth):
        torch.manual_seed(7)
        inputs = tuple(
            tensor.requires_grad_()
            for tensor in (
                torch.randn(n, heads, dk),
                torch.randn(n, heads, dk),
                torch.randn(n, heads, dv),
                torch.rand(n, heads),
                torch.rand(n, heads),
            )
        )
        reads = (torch.arange(n)[:, None] + torch.arange(depth)[None]) % n + 1
        replay = GDNReplay(torch.empty(n, 0, dtype=torch.long), [], reads)
        storages = {}

        def pack(tensor):
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = delta(*inputs, replay)
        total = sum(storage.nbytes() for storage in storages.values())
        output.square().mean().backward()
        assert all(tensor.grad.isfinite().all() for tensor in inputs)
        return total

    depths = (2, 18)
    sizes = [saved_bytes(depth) for depth in depths]
    vector_budget = (depths[1] - depths[0]) * n * (2 * heads * dk * 4 + 8)
    assert 0 < sizes[1] - sizes[0] <= vector_budget
    assert sizes[1] <= 2 * depths[1] * n * (heads * dk * 4 + 8)
    for depth, size in zip(depths, sizes):
        record_property(f"read_saved_bytes_{depth}", size)


@pytest.mark.parametrize("linear_loss", [True, False])
def test_delta_rejects_creating_second_derivative_graph(linear_loss):
    torch.manual_seed(7)
    trace = mutation_trace()
    n = len(trace.tokens)
    replay = GDNReplay.from_trace(trace, torch.arange(n))
    query = torch.randn(n, 1, 2, requires_grad=True)
    key = torch.randn(n, 1, 2, requires_grad=True)
    value = torch.randn(n, 1, 3, requires_grad=True)
    output = delta(query, key, value, torch.full((n, 1), 0.9), torch.full((n, 1), 0.1), replay)
    loss = output.sum() if linear_loss else output.square().sum()
    with pytest.raises(RuntimeError, match="first.order|once_differentiable|create_graph"):
        torch.autograd.grad(loss, (query, key, value), create_graph=True)


@pytest.mark.parametrize("segment_size", [1, 8])
def test_delta_summary_copy_and_storage_do_not_grow_with_tree_depth(segment_size, record_property):
    class MatrixCopies(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.elements = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func in {torch.ops.aten.cat.default, torch.ops.aten.clone.default}:
                if result.ndim == 4:
                    self.elements += result.numel()
            return result

    torch.manual_seed(7)
    n, heads, dk, dv = 64, 2, 16, 12
    inputs = tuple(
        x.requires_grad_()
        for x in (
            torch.randn(n, heads, dk, dtype=torch.float64),
            torch.randn(n, heads, dk, dtype=torch.float64),
            torch.randn(n, heads, dv, dtype=torch.float64),
            torch.full((n, heads), 0.99, dtype=torch.float64),
            torch.full((n, heads), 0.05, dtype=torch.float64),
        )
    )
    query, key = (x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in inputs[:2])
    value, alpha, beta = inputs[2:]
    state = value.new_zeros(heads, dk, dv)
    for row in range(n):
        state = state * alpha[row, :, None, None]
        residual = value[row] - torch.einsum("hkv,hk->hv", state, key[row])
        state = state + torch.einsum("hk,hv->hkv", key[row], beta[row, :, None] * residual)
    expected = torch.einsum("hkv,nhk->nhv", state, query / dk**0.5)
    weights = torch.randn_like(expected)
    expected_grads = torch.autograd.grad((expected * weights).sum(), inputs)
    saved_sizes = []
    for shape in ("balanced", "chain"):
        nodes, levels, next_node = list(range(1, n + 1)), [], n + 1
        if shape == "balanced":
            while len(nodes) > 1:
                levels.append(torch.tensor(list(zip(nodes[::2], nodes[1::2]))))
                nodes = list(range(next_node, next_node + len(levels[-1])))
                next_node += len(nodes)
        else:
            root = 1
            for leaf in range(2, n + 1):
                levels.append(torch.tensor([[root, leaf]]))
                root, next_node = next_node, next_node + 1
            nodes = [root]
        assert sum(len(level) for level in levels) == n - 1
        replay = GDNReplay(
            torch.empty(n, 0, dtype=torch.long),
            levels,
            torch.full((n, 1), nodes[0]),
            segment_size=segment_size,
        )
        storages = {}

        def pack(tensor):
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage
            return tensor

        with (
            Float64DeltaMath(),
            MatrixCopies() as copies,
            torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor),
        ):
            actual = delta(*inputs, replay)
        actual_grads = torch.autograd.grad((actual * weights).sum(), inputs)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-12, rtol=2e-12)
        table_elements = (n - 1) * heads * (dk * dk + dv * dk)
        assert copies.elements <= 4 * table_elements
        saved = sum(storage.nbytes() for storage in storages.values())
        vector_bytes = n * heads * (6 * dk + dv + 4) * 8
        metadata_bytes = (2 * n + 3 * (n - 1)) * 8
        assert saved <= table_elements * 8 + vector_bytes + metadata_bytes
        saved_sizes.append(saved)
        record_property(f"summary_matrix_copies_{shape}", copies.elements)
        record_property(f"summary_saved_bytes_{shape}", saved)
    if segment_size == 1:
        assert saved_sizes[0] == saved_sizes[1]
    else:
        assert saved_sizes[0] < saved_sizes[1] // 2


def test_delta_compact_segments_reduce_causal_saved_storage(record_property):
    torch.manual_seed(7)
    n, heads, dk, dv = 64, 2, 32, 24
    trace = Trace(
        batches=[ForwardBatch(0, "prefill", (ForwardRequest(tuple(range(n)), 0, 0, ((0, 0),)),))]
    )
    inputs = tuple(
        x.requires_grad_()
        for x in (
            torch.randn(n, heads, dk),
            torch.randn(n, heads, dk),
            torch.randn(n, heads, dv),
            torch.rand(n, heads),
            torch.rand(n, heads),
        )
    )
    sizes, outputs, gradients = [], [], []
    for segment_size in (1, 8):
        replay = GDNReplay.from_trace(
            trace, torch.arange(n), segment_size=segment_size, read_mode="factorized"
        )
        storages, matrices = {}, []

        def pack(tensor):
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage
            if tensor.ndim == 4 and tensor.shape[-2:] in {(dk, dk), (dv, dk)}:
                matrices.append(tensor.shape)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = delta(*inputs, replay)
        sizes.append(sum(storage.nbytes() for storage in storages.values()))
        assert len(matrices) == 2
        assert all(shape[0] <= n // segment_size - 1 for shape in matrices)
        outputs.append(output)
        gradients.append(torch.autograd.grad(output.square().sum(), inputs))
        record_property(f"causal_saved_bytes_segment_{segment_size}", sizes[-1])
    assert sizes[1] < sizes[0] // 2
    torch.testing.assert_close(outputs[1], outputs[0], atol=2e-7, rtol=2e-5)
    for actual, expected in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_delta_factorized_reads_preserve_vector_depth(record_property):
    torch.manual_seed(7)
    n, heads, dk, dv, depth = 8, 2, 16, 12, 3
    levels = [
        torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
        torch.tensor([[9, 10], [11, 12]]),
        torch.tensor([[13, 14]]),
    ]
    inputs = tuple(
        x.requires_grad_()
        for x in (
            torch.randn(n, heads, dk, dtype=torch.float64),
            torch.randn(n, heads, dk, dtype=torch.float64),
            torch.randn(n, heads, dv, dtype=torch.float64),
            torch.rand(n, heads, dtype=torch.float64),
            torch.rand(n, heads, dtype=torch.float64),
        )
    )
    sizes, outputs, gradients = [], [], []
    for read_mode in ("expanded", "factorized"):
        replay = GDNReplay(
            torch.empty(n, 0, dtype=torch.long),
            levels,
            torch.full((n, depth), 15),
            read_mode=read_mode,
        )
        storages, vectors = {}, []

        def pack(tensor):
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage
            if tensor.ndim == 4 and tensor.shape[1:] == (n, heads, dk):
                vectors.append(tensor.shape[0])
            return tensor

        with Float64DeltaMath():
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                output = delta(*inputs, replay)
            gradients.append(torch.autograd.grad(output.square().sum(), inputs))
        sizes.append(sum(storage.nbytes() for storage in storages.values()))
        outputs.append(output)
        assert vectors == [depth * (n if read_mode == "expanded" else 1)]
        record_property(f"shared_segment_saved_bytes_{read_mode}", sizes[-1])
    assert sizes[1] < sizes[0] // 2
    torch.testing.assert_close(outputs[1], outputs[0], atol=2e-12, rtol=2e-12)
    for actual, expected in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
