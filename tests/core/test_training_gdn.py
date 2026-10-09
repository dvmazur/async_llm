import inspect
import json
import re
from itertools import groupby
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from minisgl.training.gdn import GDNReplay, delta, gdn_forward
from minisgl.training.replay import Replay, forward
from minisgl.training.sft import supervised_loss
from minisgl.training.stress import workload
from minisgl.training.trace import BlockUpdate, RecordingSession, Token, Trace
from test_training import (
    Session,
    apply_updates,
    example_trace,
    mutation_trace,
    repeated_view_trace,
    single_block_trace,
)
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5


@pytest.fixture(autouse=True)
def native_torch_kernels(monkeypatch):
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    ):
        monkeypatch.setattr(modeling_qwen3_5, name, inspect.unwrap(getattr(modeling_qwen3_5, name)))


def tiny_hybrid():
    torch.manual_seed(7)
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=6,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
        },
    )
    config._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(config).eval()
    with torch.no_grad():
        model.model.layers[0].linear_attn.A_log.fill_(-1)
    return model


def sequential_gdn(module, hidden, trace):
    """Replay block contents and raw convolution windows, without affine summaries."""
    blocks, origins, outputs = {}, {}, {}
    h, dk, dv = module.num_v_heads, module.head_k_dim, module.head_v_dim
    width = module.conv_kernel_size
    zero_window = hidden.new_zeros(width, module.conv_dim)

    def advance(state, transform):
        key, value, alpha, beta = transform
        decayed = state * alpha[:, None, None]
        residual = value - torch.einsum("hkv,hk->hv", decayed, key)
        return decayed + torch.einsum("hk,hv->hkv", key, beta[:, None] * residual)

    for step, entries in groupby(enumerate(trace.tokens), key=lambda entry: entry[1].step):
        entries = list(entries)
        apply_updates(trace, step, blocks)
        snapshot = {block: list(rows) for block, rows in blocks.items()}
        writers = {token.block for _, token in entries}
        pending = []
        for row, token in entries:
            visible, tail = [], zero_window
            for block, length in token.view:
                if token.kind == "decode":
                    length -= int(block in writers)
                elif block == token.block:
                    length -= 1
                reads = blocks if token.kind == "prefill" and block == token.block else snapshot
                members = reads.get(block, [])[:length]
                assert len(members) == length
                visible.extend(members)
                if members:
                    tail = origins[members[-1]][1]
            x = hidden[row]
            raw = module.in_proj_qkv(x)
            window = torch.cat([tail[1:], raw[None]], dim=0)
            convolved = F.silu((window * module.conv1d.weight[:, 0].T).sum(0))
            query, key, value = convolved.split([module.key_dim, module.key_dim, module.value_dim])
            query, key = (t.reshape(module.num_k_heads, dk) for t in (query, key))
            query, key = (
                t * torch.rsqrt(t.square().sum(-1, keepdim=True) + 1e-6) for t in (query, key)
            )
            query, key = (t.repeat_interleave(h // module.num_k_heads, dim=0) for t in (query, key))
            value = value.reshape(h, dv)
            beta = module.in_proj_b(x).sigmoid()
            alpha = (
                -module.A_log.float().exp()
                * F.softplus(module.in_proj_a(x).float() + module.dt_bias.float())
            ).exp()
            transform = (key, value, alpha, beta)
            state = hidden.new_zeros(h, dk, dv)
            for source in visible:
                state = advance(state, origins[source][0])
            state = advance(state, transform)
            core = torch.einsum("hkv,hk->hv", state, query / dk**0.5)
            gate = module.in_proj_z(x).reshape(h, dv)
            outputs[row] = module.out_proj(module.norm(core, gate).flatten())
            origins[row] = (transform, window)
            if token.kind == "decode":
                pending.append((token.block, row))
            else:
                blocks.setdefault(token.block, []).append(row)
        for block, row in pending:
            blocks.setdefault(block, []).append(row)
    return torch.stack([outputs[row] for row in range(len(trace.tokens))])


def sequential_delta(query, key, value, alpha, beta, trace, order):
    query, key = (x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (query, key))
    inverse = order.argsort().tolist()
    blocks, outputs = {}, [None] * len(query)
    for step, entries in groupby(enumerate(trace.tokens), key=lambda entry: entry[1].step):
        entries = list(entries)
        apply_updates(trace, step, blocks)
        snapshot = {block: list(rows) for block, rows in blocks.items()}
        writers = {token.block for _, token in entries}
        pending = []
        for row, token in entries:
            visible = []
            for block, length in token.view:
                length -= (
                    int(block in writers) if token.kind == "decode" else int(block == token.block)
                )
                reads = blocks if token.kind == "prefill" and block == token.block else snapshot
                members = reads.get(block, [])[:length]
                assert len(members) == length
                visible.extend(members)
            state = value.new_zeros(key.shape[1], key.shape[2], value.shape[2])
            for source in [*visible, row]:
                source = inverse[source]
                state = state * alpha[source, :, None, None]
                residual = value[source] - torch.einsum("hkv,hk->hv", state, key[source])
                state = state + torch.einsum(
                    "hk,hv->hkv", key[source], beta[source, :, None] * residual
                )
            outputs[inverse[row]] = torch.einsum(
                "hkv,hk->hv", state, query[inverse[row]] / key.shape[2] ** 0.5
            )
            if token.kind == "decode":
                pending.append((token.block, row))
            else:
                blocks.setdefault(token.block, []).append(row)
        for block, row in pending:
            blocks.setdefault(block, []).append(row)
    return torch.stack(outputs)


class Float64DeltaMath(TorchFunctionMode):
    """Preserve double algebra without changing production's FP32 contract."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.Tensor.float and args[0].dtype == torch.float64:
            return args[0]
        if func is torch.eye:
            kwargs = {**(kwargs or {}), "dtype": torch.float64}
        return func(*args, **(kwargs or {}))


def uneven_history_trace():
    session = RecordingSession(Session())
    tip, large, small = [session.create_block() for _ in range(3)]

    def prefill(block, ids, context=()):
        session.prefill_batch(
            [NS(block=block, input_ids=torch.tensor(ids), context=context, pixel_values=None)]
        )

    prefill(tip, [1, 2, 3])
    prefill(large, [4, 5, 6, 7, 8])
    prefill(small, [9])
    for i in range(8):
        right = large if i % 2 == 0 else small
        copied = session.merge_blocks(tip, right)
        session.append_block(copied, right)
        prefill(tip, [10 + i], [right])
        session.free_block(tip)
        prefill(tip, [20 + i])
        session.decode_step(
            [
                NS(cache_view=[tip, copied], output_block=copied),
                NS(cache_view=[copied, tip], output_block=tip),
            ],
            torch.tensor([i + 1, i + 2]),
        )
        tip = copied
    return session.trace


@pytest.fixture
def cuda_delta_math():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


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


@pytest.mark.parametrize(
    "device,backend", [("cpu", "torch"), ("cuda", "torch"), ("cuda", "triton")]
)
@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_delta_autocast_preserves_fp32_core(request, device, backend, read_mode, dtype):
    if device == "cuda":
        request.getfixturevalue("cuda_delta_math")
    torch.manual_seed(7)
    n = 17
    trace = Trace([Token(i, 0, 0, i, ((0, i + 1),), "prefill") for i in range(n)])
    replay = GDNReplay.from_trace(
        trace, torch.arange(n), device=device, read_mode=read_mode, read_backend=backend
    )
    inputs = tuple(
        x.to(dtype).requires_grad_()
        for x in (
            torch.randn(n, 2, 5, device=device),
            torch.randn(n, 2, 5, device=device),
            torch.randn(n, 2, 7, device=device),
            torch.rand(n, 2, device=device),
            torch.rand(n, 2, device=device),
        )
    )
    weights = torch.randn(n, 2, 7, device=device)
    expected = delta(*inputs, replay)
    expected_grads = torch.autograd.grad((expected * weights).sum(), inputs)
    with torch.autocast(device, dtype=torch.bfloat16):
        actual = delta(*inputs, replay)
        actual_grads = torch.autograd.grad((actual * weights).sum(), inputs)
        assert torch.is_autocast_enabled(device)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)


@pytest.mark.parametrize(
    "device,backend", [("cpu", "torch"), ("cuda", "torch"), ("cuda", "triton")]
)
@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
def test_gdn_inference_metadata_supports_backward(request, device, backend, read_mode):
    if device == "cuda":
        request.getfixturevalue("cuda_delta_math")
    torch.manual_seed(7)
    trace = mutation_trace()
    n = len(trace.tokens)
    inputs = tuple(
        x.requires_grad_()
        for x in (
            torch.randn(n, 2, 5, device=device),
            torch.randn(n, 2, 5, device=device),
            torch.randn(n, 2, 7, device=device),
            torch.rand(n, 2, device=device),
            torch.rand(n, 2, device=device),
        )
    )
    options = {"device": device, "read_mode": read_mode, "read_backend": backend}
    reference = GDNReplay.from_trace(trace, torch.arange(n), **options)
    weights = torch.randn(n, 2, 7, device=device)
    expected = delta(*inputs, reference)
    expected_grads = torch.autograd.grad((expected * weights).sum(), inputs)
    with torch.inference_mode():
        replay = GDNReplay.from_trace(trace, torch.arange(n), **options)
        warmed = delta(*inputs, replay)
        metadata = (
            replay.conv_sources,
            replay.levels,
            replay.reads,
            replay.partitions,
            replay.read_paths,
        )
        assert torch.is_inference_mode_enabled() and warmed.is_inference()

    def check_normal_tensors(value):
        if isinstance(value, torch.Tensor):
            assert not value.is_inference()
        else:
            for item in value:
                if isinstance(item, (torch.Tensor, list, tuple)):
                    check_normal_tensors(item)

    check_normal_tensors(metadata)
    actual = delta(*inputs, replay)
    actual_grads = torch.autograd.grad((actual * weights).sum(), inputs)
    assert replay.partitions is metadata[3] and replay.read_paths is metadata[4]
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize(
    "trace_kind,segment_size,read_mode,dk,dv,boundaries",
    [
        ("mutation", 1, "expanded", 5, 7, False),
        ("mutation", 1, "factorized", 128, 128, False),
        ("mutation", 8, "expanded", 7, 5, True),
        ("mutation", 8, "factorized", 5, 7, True),
        ("stress", 8, "expanded", 16, 12, False),
        ("stress", 8, "factorized", 16, 12, False),
        ("uneven", 8, "factorized", 3, 5, True),
    ],
)
def test_triton_delta_outputs_and_gradients(
    cuda_delta_math, trace_kind, segment_size, read_mode, dk, dv, boundaries
):
    if trace_kind == "stress":
        session = RecordingSession(Session())
        workload(session, vocab_size=32, large_size=2)
        trace = session.trace
    else:
        trace = uneven_history_trace() if trace_kind == "uneven" else mutation_trace()
    torch.manual_seed(7)
    n, heads = len(trace.tokens), 2
    order = torch.randperm(n)
    alpha, beta = (torch.rand(n, heads * 2, device="cuda")[:, ::2] for _ in range(2))
    if boundaries:
        alpha[::3], alpha[1::3] = 0, 1
        beta[::4], beta[1::4] = 0, 1
    inputs = tuple(
        x.requires_grad_()
        for x in (
            torch.randn(n, heads, dk, device="cuda"),
            torch.randn(n, heads, dk, device="cuda"),
            torch.randn(n, heads, dv * 2, device="cuda")[..., ::2],
            alpha,
            beta,
        )
    )
    weights = torch.randn(n, heads, dv, device="cuda")
    outputs, gradients = [], []
    for backend in ("torch", "triton"):
        replay = GDNReplay.from_trace(
            trace,
            order,
            device="cuda",
            segment_size=segment_size,
            read_mode=read_mode,
            read_backend=backend,
        )
        output = delta(*inputs, replay)
        outputs.append(output)
        gradients.append(torch.autograd.grad((output * weights).sum(), inputs))
    torch.testing.assert_close(outputs[1], outputs[0], atol=3e-6, rtol=3e-5)
    for actual, expected in zip(gradients[1], gradients[0], strict=True):
        assert actual.isfinite().all()
        torch.testing.assert_close(actual, expected, atol=6e-6, rtol=3e-4)


@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
def test_triton_delta_no_grad_saves_no_tensors(cuda_delta_math, read_mode):
    trace = mutation_trace()
    n = len(trace.tokens)
    replay = GDNReplay.from_trace(
        trace, torch.arange(n), device="cuda", read_mode=read_mode, read_backend="triton"
    )
    inputs = tuple(
        x.requires_grad_()
        for x in (
            torch.randn(n, 2, 5, device="cuda"),
            torch.randn(n, 2, 5, device="cuda"),
            torch.randn(n, 2, 7, device="cuda"),
            torch.rand(n, 2, device="cuda"),
            torch.rand(n, 2, device="cuda"),
        )
    )
    expected = delta(*inputs, replay).detach()

    def unexpected_save(tensor):
        raise AssertionError("No-grad replay must not save tensors")

    with (
        torch.no_grad(),
        torch.autograd.graph.saved_tensors_hooks(unexpected_save, lambda tensor: tensor),
    ):
        actual = delta(*inputs, replay)
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


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


@pytest.mark.parametrize(
    "read_mode,use_reentrant", [("expanded", None), ("factorized", False), ("expanded", True)]
)
def test_triton_hybrid_parameter_gradients(cuda_delta_math, read_mode, use_reentrant):
    reference, model = tiny_hybrid().cuda().train(), tiny_hybrid().cuda().train()
    if use_reentrant is not None:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        )
    trace = mutation_trace()
    options = {"device": "cuda", "gdn_read_mode": read_mode}
    native = Replay.from_trace(trace, **options, gdn_read_backend="torch")
    replay = Replay.from_trace(trace, **options, gdn_read_backend="triton")
    expected = forward(reference, native)
    actual = forward(model, replay)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    targets = replay.input_ids.roll(-1)
    F.cross_entropy(expected, targets).backward()
    F.cross_entropy(actual, targets).backward()
    for (name, parameter), (_, native_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert parameter.grad is not None and native_parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad,
            native_parameter.grad,
            atol=2e-6,
            rtol=3e-4,
            msg=f"Gradient mismatch: {name}",
        )


@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
@pytest.mark.parametrize("segment_size", [1, 2, 8, 16])
@pytest.mark.parametrize("trace_kind", ["single", "mutation", "uneven", "stress", "repeated"])
def test_delta_fp64_outputs_and_gradients(trace_kind, segment_size, read_mode, record_property):
    if trace_kind == "stress":
        session = RecordingSession(Session())
        workload(session, vocab_size=32, large_size=2)
        trace = session.trace
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
        x.requires_grad_()
        for x in (
            torch.randn(n, 2, 3, dtype=torch.float64),
            torch.randn(n, 2, 3, dtype=torch.float64),
            torch.randn(n, 2, 4, dtype=torch.float64),
            torch.rand(n, 2, dtype=torch.float64),
            torch.rand(n, 2, dtype=torch.float64),
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
        max((a - b).abs().max().item() for a, b in zip(actual_grads, expected_grads)),
    )


@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
@pytest.mark.parametrize("segment_size", [1, 8])
def test_delta_long_causal_fp64_outputs_and_gradients(segment_size, read_mode, record_property):
    torch.manual_seed(7)
    n, heads, dk, dv = 512, 2, 16, 12
    trace = Trace([Token(i, 0, 0, i, ((0, i + 1),), "prefill") for i in range(n)])
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
        [
            Token(1, 0, 0, 0, ((0, 1),), "prefill"),
            Token(2, 0, 0, 1, ((0, 2),), "prefill"),
            Token(3, 1, 0, 0, ((1, 5), (0, 1)), "decode"),
            Token(4, 1, 1, 4, ((0, 1), (1, 5)), "decode"),
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
    session = RecordingSession(Session())
    source, target, empty = [session.create_block() for _ in range(3)]

    def prefill(block, token):
        session.prefill_batch(
            [NS(block=block, input_ids=torch.tensor([token]), context=[], pixel_values=None)]
        )

    prefill(source, 1)
    prefill(target, 2)
    for _ in range(8):
        session.append_block(target, source)
    copied = session.merge_blocks(empty, target)
    session.free_block(source)
    prefill(source, 3)
    prefill(copied, 4)
    order = torch.tensor([2, 0, 3, 1])
    replay = GDNReplay.from_trace(session.trace, order, segment_size=segment_size)
    torch.manual_seed(7)
    inputs = tuple(
        x.requires_grad_()
        for x in (
            torch.randn(4, 2, 3, dtype=torch.float64),
            torch.randn(4, 2, 3, dtype=torch.float64),
            torch.randn(4, 2, 5, dtype=torch.float64),
            torch.rand(4, 2, dtype=torch.float64) * 0.6 + 0.2,
            torch.rand(4, 2, dtype=torch.float64) * 0.6 + 0.2,
        )
    )
    expected = sequential_delta(*inputs, session.trace, order)
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


def test_odd_history_self_copies_stay_compact_and_preserve_gradient_counts():
    session = RecordingSession(Session())
    block = session.create_block()
    session.prefill_batch(
        [NS(block=block, input_ids=torch.tensor([1, 2, 3]), context=[], pixel_values=None)]
    )
    for _ in range(40):
        session.append_block(block, block)
    assert block.num_tokens == 3 * 2**40
    session.decode_step([NS(cache_view=[block], output_block=block)], torch.tensor([4]))
    replay = GDNReplay.from_trace(session.trace, torch.arange(4))
    assert sum(len(level) for level in replay.levels) <= 2 * len(session.trace.updates) + 3
    assert replay.reads.numel() <= 4 * 4
    assert replay.conv_sources.shape == (4, 4)
    inputs = tuple(
        x.requires_grad_()
        for x in (
            torch.ones(4, 1, 1, dtype=torch.float64),
            torch.ones(4, 1, 1, dtype=torch.float64),
            torch.ones(4, 1, 1, dtype=torch.float64),
            torch.ones(4, 1, dtype=torch.float64),
            torch.zeros(4, 1, dtype=torch.float64),
        )
    )
    with Float64DeltaMath():
        output = delta(*inputs, replay)
        gradients = torch.autograd.grad(output[-1].sum(), inputs)
    assert not output.any()
    for gradient in gradients:
        assert gradient.isfinite().all()
    expected = torch.tensor([2**40, 2**40, 2**40, 1], dtype=torch.float64) / (1 + 1e-6)
    torch.testing.assert_close(gradients[-1][:, 0], expected, atol=2e-12, rtol=2e-12)
    for gradient in gradients[:-1]:
        assert not gradient.any()


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


def test_gdn_train_and_eval_no_grad_save_no_tensors():
    module = tiny_hybrid().model.layers[0].linear_attn
    trace = mutation_trace()
    replay = GDNReplay.from_trace(trace, torch.arange(len(trace.tokens)))
    hidden = torch.randn(1, len(trace.tokens), module.hidden_size, requires_grad=True)
    expected = gdn_forward(module.train(), hidden, replay).detach()

    def unexpected_save(tensor):
        raise AssertionError("Inference must not save tensors for backward")

    with (
        torch.no_grad(),
        torch.autograd.graph.saved_tensors_hooks(unexpected_save, lambda tensor: tensor),
    ):
        for training in (True, False):
            actual = gdn_forward(module.train(training), hidden, replay)
            assert not actual.requires_grad
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


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
    trace = Trace([Token(i, 0, 0, i, ((0, i + 1),), "prefill") for i in range(n)])
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


@pytest.mark.parametrize(
    "trace_factory", [example_trace, mutation_trace, uneven_history_trace, repeated_view_trace]
)
@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
def test_gdn_parallel_outputs_and_gradients(trace_factory, read_mode):
    module = tiny_hybrid().model.layers[0].linear_attn
    trace = trace_factory()
    order = Replay.from_trace(trace).order
    replay = GDNReplay.from_trace(
        trace, order, conv_kernel_size=module.conv_kernel_size, read_mode=read_mode
    )
    hidden = torch.randn(len(trace.tokens), module.hidden_size, requires_grad=True)
    expected = sequential_gdn(module, hidden, trace)
    actual = gdn_forward(module, hidden[order][None], replay)[0][order.argsort()]
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    parameters = (hidden, *module.parameters())
    weights = torch.randn_like(expected)
    expected_grads = torch.autograd.grad((expected * weights).sum(), parameters)
    actual_grads = torch.autograd.grad((actual * weights).sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=3e-4)


def test_gdn_synthetic_stress_workload_outputs_and_gradients():
    session = RecordingSession(Session())
    workload(session, vocab_size=32, large_size=2)
    trace = session.trace
    module = tiny_hybrid().model.layers[0].linear_attn
    order = Replay.from_trace(trace).order
    replay = GDNReplay.from_trace(trace, order, conv_kernel_size=module.conv_kernel_size)
    hidden = torch.randn(len(trace.tokens), module.hidden_size, requires_grad=True)
    expected = sequential_gdn(module, hidden, trace)
    actual = gdn_forward(module, hidden[order][None], replay)[0][order.argsort()]
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    weights = torch.randn_like(expected)
    parameters = (hidden, *module.parameters())
    expected_grads = torch.autograd.grad((expected * weights).sum(), parameters)
    actual_grads = torch.autograd.grad((actual * weights).sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=3e-4)


def test_gdn_decode_reads_peer_before_batch():
    trace = Trace(
        [
            Token(1, 0, 0, 0, ((0, 1),), "prefill"),
            Token(2, 1, 1, 0, ((1, 1),), "prefill"),
            Token(3, 2, 0, 1, ((1, 2), (0, 2)), "decode"),
            Token(4, 2, 1, 1, ((0, 2), (1, 2)), "decode"),
        ]
    )
    module = tiny_hybrid().model.layers[0].linear_attn
    order = torch.arange(4)
    replay = GDNReplay.from_trace(trace, order)
    hidden = torch.randn(1, 4, module.hidden_size, requires_grad=True)
    outputs = gdn_forward(module, hidden, replay)
    torch.testing.assert_close(outputs[0], sequential_gdn(module, hidden[0], trace))
    (gradient,) = torch.autograd.grad(outputs[0, 2].square().sum(), hidden)
    assert not gradient[0, 3].any()
    assert gradient[0, 1].abs().sum() > 0
    changed = hidden.detach().clone()
    changed[0, 3] += 10
    torch.testing.assert_close(gdn_forward(module, changed, replay)[0, 2], outputs[0, 2])


def test_gdn_copied_history_survives_clear_and_isolates_reuse():
    session = RecordingSession(Session())
    source, child, empty = [session.create_block() for _ in range(3)]

    def prefill(block, ids, context=()):
        session.prefill_batch(
            [NS(block=block, input_ids=torch.tensor(ids), context=context, pixel_values=None)]
        )

    prefill(source, [1, 2])
    prefill(child, [3], [source])
    copied = session.merge_blocks(empty, child)
    session.free_block(source)
    session.free_block(child)
    prefill(source, [4])
    prefill(copied, [5])
    trace = session.trace
    module = tiny_hybrid().model.layers[0].linear_attn
    hidden = torch.randn(1, len(trace.tokens), module.hidden_size, requires_grad=True)
    replay = GDNReplay.from_trace(trace, torch.arange(len(trace.tokens)))
    actual = gdn_forward(module, hidden, replay)[0]
    torch.testing.assert_close(actual, sequential_gdn(module, hidden[0], trace))
    (copied_grad,) = torch.autograd.grad(actual[4].square().sum(), hidden, retain_graph=True)
    (reused_grad,) = torch.autograd.grad(actual[3].square().sum(), hidden)
    assert (copied_grad[0, :3].abs().sum(-1) > 0).all()
    assert not copied_grad[0, 3].any()
    assert not reused_grad[0, :3].any()


def test_hybrid_gdn_options_survive_transfer_and_separate_compiled_cache():
    model = tiny_hybrid()
    replay = Replay.from_trace(mutation_trace())
    expected = None
    for i, (segment_size, read_mode, cache_reused) in enumerate(
        (
            (8, "factorized", True),
            (1, "factorized", True),
            (8, "expanded", True),
            (8, "expanded", False),
        ),
        1,
    ):
        replay.gdn_segment_size, replay.gdn_read_mode = segment_size, read_mode
        replay.gdn_cache_reused = cache_reused
        with torch.no_grad():
            actual = forward(model, replay)
        if expected is None:
            expected = actual
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        assert len(replay._gdn) == i
        compiled = list(replay._gdn.values())[-1]
        assert (compiled.segment_size, compiled.read_mode, compiled.cache_reused) == (
            segment_size,
            read_mode,
            cache_reused,
        )
        transferred = replay.to("cpu")
        assert (transferred.gdn_segment_size, transferred.gdn_read_mode) == (
            segment_size,
            read_mode,
        )
        assert transferred.gdn_cache_reused == cache_reused
        assert transferred is replay
        assert transferred._gdn is replay._gdn
    with torch.no_grad():
        forward(model, replay)
    assert len(replay._gdn) == 4 and list(replay._gdn.values())[-1] is compiled


@pytest.mark.parametrize("mixed_steps", [False, True])
def test_hybrid_causal_forward_and_backward_match_huggingface(mixed_steps):
    reference, model = tiny_hybrid().train(), tiny_hybrid().train()
    if mixed_steps:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    replay = Replay.from_trace(single_block_trace(mixed_steps), query_chunk_size=3)
    ids = replay.input_ids[None]
    expected = reference(ids, use_cache=False).logits[0]
    actual = forward(model, replay)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    targets = replay.input_ids.roll(-1)
    F.cross_entropy(expected, targets).backward()
    F.cross_entropy(actual, targets).backward()
    for (name, parameter), (_, native) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert parameter.grad is not None and native.grad is not None
        torch.testing.assert_close(
            parameter.grad, native.grad, atol=1e-6, rtol=2e-4, msg=f"Gradient mismatch: {name}"
        )
    with torch.no_grad():
        torch.testing.assert_close(model(ids, use_cache=False).logits[0], expected)


@pytest.mark.parametrize(
    "device,backend,use_reentrant",
    [
        ("cpu", "torch", None),
        ("cpu", "torch", True),
        ("cuda", "torch", False),
        ("cuda", "triton", True),
    ],
)
def test_hybrid_autocast_parameter_gradients(request, device, backend, use_reentrant):
    if device == "cuda":
        request.getfixturevalue("cuda_delta_math")
    model = tiny_hybrid().to(device).train()
    if use_reentrant is not None:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        )
    trace = Trace([Token(i, 0, 0, i, ((0, i + 1),), "prefill") for i in range(17)])
    replay = Replay.from_trace(
        trace, device=device, gdn_read_mode="factorized", gdn_read_backend=backend
    )
    with torch.autocast(device, dtype=torch.bfloat16):
        logits = forward(model, replay)
        loss = F.cross_entropy(logits, replay.input_ids.roll(-1))
        loss.backward()
    assert logits.dtype == torch.bfloat16
    assert loss.isfinite()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.dtype == torch.float32, name
        assert parameter.grad.isfinite().all(), name


@pytest.mark.parametrize(
    "device,backend,use_reentrant",
    [
        ("cpu", "torch", None),
        ("cpu", "torch", False),
        ("cpu", "torch", True),
        ("cuda", "torch", False),
        ("cuda", "triton", None),
        ("cuda", "triton", True),
    ],
)
@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
def test_hybrid_inference_warmup_then_training(request, device, backend, use_reentrant, read_mode):
    if device == "cuda":
        request.getfixturevalue("cuda_delta_math")
    reference, model = tiny_hybrid().to(device).train(), tiny_hybrid().to(device)
    if use_reentrant is not None:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        )
    trace = mutation_trace()
    options = {"device": device, "gdn_read_mode": read_mode, "gdn_read_backend": backend}
    native = Replay.from_trace(trace, **options)
    expected = forward(reference, native)
    expected.square().mean().backward()
    with torch.inference_mode():
        replay = Replay.from_trace(trace, **options)
        warmed = forward(model.eval(), replay)
        compiled = next(iter(replay._gdn.values()))
        assert warmed.is_inference()
    actual = forward(model.train(), replay)
    actual.square().mean().backward()
    assert next(iter(replay._gdn.values())) is compiled
    torch.testing.assert_close(actual, expected)
    for parameter, original in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, original.grad, atol=1e-6, rtol=2e-4)


def test_hybrid_replay_checkpoint_reloads_as_native_huggingface(tmp_path):
    model = tiny_hybrid()
    ids = torch.tensor([[1, 2, 3]])
    original_keys = set(model.state_dict())
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        forward(model, Replay.from_trace(mutation_trace()))
    model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert not any("attn_implementation" in key for key in config)
    restored = Qwen3_5ForCausalLM.from_pretrained(tmp_path, attn_implementation="eager").eval()
    assert set(restored.state_dict()) == original_keys
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], parameter, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(restored(ids, use_cache=False).logits, expected)


def test_hybrid_replay_preserves_native_sdpa_masks_and_cache():
    model = tiny_hybrid().eval()
    model.set_attn_implementation("sdpa")
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]])
    with (
        torch.no_grad(),
        patch.object(
            F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention
        ) as sdpa,
    ):
        expected = model(ids, attention_mask=mask, use_cache=False).logits
        cache = model(ids[:1, :-1], use_cache=True).past_key_values
        assert sdpa.call_count == 2
        forward(model, Replay.from_trace(mutation_trace()))
        assert sdpa.call_count == 2 and cache.get_seq_length() == ids.shape[1] - 1
        assert model.config._attn_implementation == "sdpa"
        assert model.model.layers[1].self_attn.config._attn_implementation == "sdpa"
        actual = model(ids, attention_mask=mask, use_cache=False).logits
        assert sdpa.call_count == 3
        torch.testing.assert_close(actual, expected)
        continued = model(ids[:1, -1:], past_key_values=cache, use_cache=True).logits
        assert sdpa.call_count == 4
        torch.testing.assert_close(continued, expected[:1, -1:], atol=2e-6, rtol=2e-5)
        changed = ids.clone()
        changed[:, -1], changed[1, :2] = 31, 30
        perturbed = model(changed, attention_mask=mask, use_cache=False).logits
        torch.testing.assert_close(perturbed[0, :-1], expected[0, :-1])
        torch.testing.assert_close(perturbed[1, 2:-1], expected[1, 2:-1])


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_hybrid_checkpointed_replay_gradients(use_reentrant):
    reference, checkpointed = tiny_hybrid().train(), tiny_hybrid().train()
    reference.set_attn_implementation("sdpa")
    checkpointed.set_attn_implementation("sdpa")
    checkpointed.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
    )
    replay = Replay.from_trace(mutation_trace())
    with patch.object(
        F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention
    ) as sdpa:
        expected = forward(reference, replay)
        actual = forward(checkpointed, replay)
        torch.testing.assert_close(actual, expected)
        assert sdpa.call_count == 0
        with torch.no_grad():
            checkpointed(torch.tensor([[1, 2, 3]]), use_cache=False)
        assert sdpa.call_count == 1
        expected.square().mean().backward()
        actual.square().mean().backward()
        assert sdpa.call_count == 1
    for actual_parameter, expected_parameter in zip(
        checkpointed.parameters(), reference.parameters()
    ):
        assert actual_parameter.grad is not None and expected_parameter.grad is not None
        torch.testing.assert_close(
            actual_parameter.grad, expected_parameter.grad, atol=1e-6, rtol=2e-4
        )


def test_hybrid_checkpointed_chunked_supervised_loss():
    reference, checkpointed = tiny_hybrid().train(), tiny_hybrid().train()
    checkpointed.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    replay = Replay.from_trace(mutation_trace(), query_chunk_size=3)
    targets = torch.arange(len(replay.input_ids)) % 32
    targets[::3] = -100
    expected = F.cross_entropy(forward(reference, replay), targets, reduction="sum")
    actual = supervised_loss(checkpointed, replay, targets, chunk_size=3)
    torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad(expected, tuple(reference.parameters()))
    actual_grads = torch.autograd.grad(actual, tuple(checkpointed.parameters()))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=2e-4)


def test_hybrid_overfits_recorded_sequence_targets():
    model = tiny_hybrid().train()
    trace = example_trace()
    replay = Replay.from_trace(trace)
    targets = torch.full((len(trace.tokens),), -100, dtype=torch.long)
    latest = {}
    for row, token in enumerate(trace.tokens):
        if token.block in latest:
            targets[latest[token.block]] = token.token_id
        latest[token.block] = row
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = F.cross_entropy(forward(model, replay), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.1
