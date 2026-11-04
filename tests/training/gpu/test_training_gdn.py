import asyncio
import json
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from minisgl.training import RecordedAsyncLLM, Trace, replay_trace
from minisgl.training.gdn import GDNReplay, delta, gdn_forward
from minisgl.training.trace import ForwardBatch, ForwardRequest
from minisgl_examples.training.loss import compute_supervised_loss
from transformers import Qwen3_5ForCausalLM

from tests.training.fixtures import cuda_delta_math as cuda_delta_math
from tests.training.fixtures import native_torch_kernels as native_torch_kernels
from tests.training.helpers import Session, generate_stress_example, make_llm
from tests.training.models import tiny_hybrid, torch_hybrid
from tests.training.references import (
    sequential_gdn,
)
from tests.training.workloads import (
    example_trace,
    mutation_trace,
    repeated_view_trace,
    single_block_trace,
    uneven_history_trace,
)


@pytest.mark.parametrize("device,backend", [("cuda", "torch"), ("cuda", "triton")])
@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_delta_autocast_preserves_fp32_core(request, device, backend, read_mode, dtype):
    if device == "cuda":
        request.getfixturevalue("cuda_delta_math")
    torch.manual_seed(7)
    n = 17
    trace = Trace(
        batches=[ForwardBatch(0, "prefill", (ForwardRequest(tuple(range(n)), 0, 0, ((0, 0),)),))]
    )
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
        atol = 1e-6 * expected_grad.abs().max().item()
        torch.testing.assert_close(actual_grad, expected_grad, atol=atol, rtol=0)


@pytest.mark.parametrize("device,backend", [("cuda", "torch"), ("cuda", "triton")])
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
        data = generate_stress_example()
        trace = Trace.from_dict(data["trace"])
    else:
        trace = uneven_history_trace() if trace_kind == "uneven" else mutation_trace()
    torch.manual_seed(7)
    n, heads = (len(trace.tokens), 2)
    order = torch.randperm(n)
    alpha, beta = (torch.rand(n, heads * 2, device="cuda")[:, ::2] for _ in range(2))
    if boundaries:
        alpha[::3], alpha[1::3] = (0, 1)
        beta[::4], beta[1::4] = (0, 1)
    inputs = tuple(
        (
            x.requires_grad_()
            for x in (
                torch.randn(n, heads, dk, device="cuda"),
                torch.randn(n, heads, dk, device="cuda"),
                torch.randn(n, heads, dv * 2, device="cuda")[..., ::2],
                alpha,
                beta,
            )
        )
    )
    weights = torch.randn(n, heads, dv, device="cuda")
    outputs, gradients = ([], [])
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
    torch.testing.assert_close(outputs[1], outputs[0], atol=3e-06, rtol=3e-05)
    for actual, expected in zip(gradients[1], gradients[0], strict=True):
        assert actual.isfinite().all()
        torch.testing.assert_close(actual, expected, atol=6e-06, rtol=0.0003)


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
    native = trace.to(**options, gdn_read_backend="torch")
    plan = trace.to(**options, gdn_read_backend="triton")
    expected = replay_trace(reference, native, hidden_only=False)
    actual = replay_trace(model, plan, hidden_only=False)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    targets = plan.input_ids.roll(-1)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gdn_train_and_eval_no_grad_save_no_tensors():
    module = tiny_hybrid().cuda().model.layers[0].linear_attn
    trace = mutation_trace()
    replay = GDNReplay.from_trace(trace, torch.arange(len(trace.tokens)), device="cuda")
    hidden = torch.randn(
        1, len(trace.tokens), module.hidden_size, device="cuda", requires_grad=True
    )
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


@pytest.mark.parametrize(
    "trace_factory", [example_trace, mutation_trace, uneven_history_trace, repeated_view_trace]
)
@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gdn_parallel_outputs_and_gradients(trace_factory, read_mode):
    module = tiny_hybrid().cuda().model.layers[0].linear_attn
    trace = trace_factory()
    order = trace.to().order
    replay = GDNReplay.from_trace(
        trace, order, device="cuda", conv_kernel_size=module.conv_kernel_size, read_mode=read_mode
    )
    hidden = torch.randn(len(trace.tokens), module.hidden_size, device="cuda", requires_grad=True)
    expected = sequential_gdn(module, hidden, trace)
    actual = gdn_forward(module, hidden[order][None], replay)[0][order.argsort()]
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    parameters = (hidden, *module.parameters())
    weights = torch.randn_like(expected)
    expected_grads = torch.autograd.grad((expected * weights).sum(), parameters)
    actual_grads = torch.autograd.grad((actual * weights).sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gdn_synthetic_stress_workload_outputs_and_gradients():
    data = generate_stress_example()
    trace = Trace.from_dict(data["trace"])
    module = tiny_hybrid().cuda().model.layers[0].linear_attn
    order = trace.to().order
    replay = GDNReplay.from_trace(
        trace, order, device="cuda", conv_kernel_size=module.conv_kernel_size
    )
    hidden = torch.randn(len(trace.tokens), module.hidden_size, device="cuda", requires_grad=True)
    expected = sequential_gdn(module, hidden, trace)
    actual = gdn_forward(module, hidden[order][None], replay)[0][order.argsort()]
    torch.testing.assert_close(actual, expected, atol=2e-07, rtol=2e-05)
    weights = torch.randn_like(expected)
    parameters = (hidden, *module.parameters())
    expected_grads = torch.autograd.grad((expected * weights).sum(), parameters)
    actual_grads = torch.autograd.grad((actual * weights).sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-06, rtol=0.0003)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gdn_decode_reads_peer_before_batch():
    trace = Trace(
        batches=[
            ForwardBatch(0, "prefill", (ForwardRequest((1,), 0, 0, ((0, 0),)),)),
            ForwardBatch(1, "prefill", (ForwardRequest((2,), 1, 0, ((1, 0),)),)),
            ForwardBatch(
                2,
                "decode",
                (
                    ForwardRequest((3,), 0, 1, ((1, 1), (0, 1))),
                    ForwardRequest((4,), 1, 1, ((0, 1), (1, 1))),
                ),
            ),
        ]
    )
    module = tiny_hybrid().cuda().model.layers[0].linear_attn
    order = torch.arange(4)
    replay = GDNReplay.from_trace(trace, order, device="cuda")
    hidden = torch.randn(1, 4, module.hidden_size, device="cuda", requires_grad=True)
    outputs = gdn_forward(module, hidden, replay)
    torch.testing.assert_close(outputs[0], sequential_gdn(module, hidden[0], trace))
    (gradient,) = torch.autograd.grad(outputs[0, 2].square().sum(), hidden)
    assert not gradient[0, 3].any()
    assert gradient[0, 1].abs().sum() > 0
    changed = hidden.detach().clone()
    changed[0, 3] += 10
    torch.testing.assert_close(gdn_forward(module, changed, replay)[0, 2], outputs[0, 2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gdn_copied_history_survives_clear_and_isolates_reuse():

    async def run():
        llm, raw = make_llm(Session())
        try:
            with RecordedAsyncLLM(llm) as session:
                source, child, empty = [await session.create_block() for _ in range(3)]

                async def prefill(block, ids, context=()):
                    await session.prefill_block(ids, write_to=block, cache_view=context)

                await prefill(source, [1, 2])
                await prefill(child, [3], [source])
                copied = await session.merge_blocks(empty, child)
                await session.free_block(source)
                await session.free_block(child)
                await prefill(source, [4])
                await prefill(copied, [5])
                trace = session.get_trace()
                module = tiny_hybrid().cuda().model.layers[0].linear_attn
                hidden = torch.randn(
                    1, len(trace.tokens), module.hidden_size, device="cuda", requires_grad=True
                )
                replay = GDNReplay.from_trace(trace, torch.arange(len(trace.tokens)), device="cuda")
                actual = gdn_forward(module, hidden, replay)[0]
                torch.testing.assert_close(actual, sequential_gdn(module, hidden[0], trace))
                (copied_grad,) = torch.autograd.grad(
                    actual[4].square().sum(), hidden, retain_graph=True
                )
                (reused_grad,) = torch.autograd.grad(actual[3].square().sum(), hidden)
                assert (copied_grad[0, :3].abs().sum(-1) > 0).all()
                assert not copied_grad[0, 3].any()
                assert not reused_grad[0, :3].any()
        finally:
            await llm.close()

    return asyncio.run(run())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_hybrid_gdn_options_survive_transfer_and_separate_compiled_cache():
    model = tiny_hybrid().cuda()
    plan = mutation_trace().to(device="cuda")
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
        plan.gdn_segment_size, plan.gdn_read_mode = segment_size, read_mode
        plan.gdn_cache_reused = cache_reused
        with torch.no_grad():
            actual = replay_trace(model, plan, hidden_only=False)
        if expected is None:
            expected = actual
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        assert len(plan._gdn) == i
        compiled = list(plan._gdn.values())[-1]
        assert (compiled.segment_size, compiled.read_mode, compiled.cache_reused) == (
            segment_size,
            read_mode,
            cache_reused,
        )
        transferred = plan.to("cuda")
        assert (transferred.gdn_segment_size, transferred.gdn_read_mode) == (
            segment_size,
            read_mode,
        )
        assert transferred.gdn_cache_reused == cache_reused
        assert transferred is plan
        assert transferred._gdn is plan._gdn
    with torch.no_grad():
        replay_trace(model, plan, hidden_only=False)
    assert len(plan._gdn) == 4 and list(plan._gdn.values())[-1] is compiled


@pytest.mark.parametrize("mixed_steps", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_hybrid_causal_forward_and_backward_match_huggingface(mixed_steps):
    reference, model = torch_hybrid().cuda().train(), tiny_hybrid().cuda().train()
    if mixed_steps:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    plan = single_block_trace(mixed_steps).to(device="cuda", query_chunk_size=3)
    ids = plan.input_ids[None]
    expected = reference(ids, use_cache=False).logits[0]
    with torch.no_grad():
        native = model(ids, use_cache=False).logits[0]
    actual = replay_trace(model, plan, hidden_only=False)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    targets = plan.input_ids.roll(-1)
    F.cross_entropy(expected, targets).backward()
    F.cross_entropy(actual, targets).backward()
    for (name, parameter), (_, original) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert parameter.grad is not None and original.grad is not None
        torch.testing.assert_close(
            parameter.grad, original.grad, atol=1e-6, rtol=2e-4, msg=f"Gradient mismatch: {name}"
        )
    with torch.no_grad():
        torch.testing.assert_close(model(ids, use_cache=False).logits[0], native)


@pytest.mark.parametrize(
    "backend,use_reentrant",
    [("torch", None), ("torch", False), ("torch", True), ("triton", True)],
)
def test_hybrid_autocast_parameter_gradients(cuda_delta_math, backend, use_reentrant):
    model = tiny_hybrid().cuda().train()
    if use_reentrant is not None:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        )
    trace = Trace(
        batches=[ForwardBatch(0, "prefill", (ForwardRequest(tuple(range(17)), 0, 0, ((0, 0),)),))]
    )
    plan = trace.to(device="cuda", gdn_read_mode="factorized", gdn_read_backend=backend)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = replay_trace(model, plan, hidden_only=False)
        loss = F.cross_entropy(logits, plan.input_ids.roll(-1))
        loss.backward()
    assert logits.dtype == torch.bfloat16
    assert loss.isfinite()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.dtype == torch.float32, name
        assert parameter.grad.isfinite().all(), name


@pytest.mark.parametrize(
    "backend,use_reentrant",
    [
        ("torch", None),
        ("torch", False),
        ("torch", True),
        ("triton", None),
        ("triton", True),
    ],
)
@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
def test_hybrid_inference_warmup_then_training(cuda_delta_math, backend, use_reentrant, read_mode):
    reference, model = tiny_hybrid().cuda().train(), tiny_hybrid().cuda()
    if use_reentrant is not None:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        )
    trace = mutation_trace()
    options = {"device": "cuda", "gdn_read_mode": read_mode, "gdn_read_backend": backend}
    native = trace.to(**options)
    expected = replay_trace(reference, native, hidden_only=False)
    expected.square().mean().backward()
    with torch.inference_mode():
        plan = trace.to(**options)
        warmed = replay_trace(model.eval(), plan, hidden_only=False)
        compiled = next(iter(plan._gdn.values()))
        assert warmed.is_inference()
    actual = replay_trace(model.train(), plan, hidden_only=False)
    actual.square().mean().backward()
    assert next(iter(plan._gdn.values())) is compiled
    torch.testing.assert_close(actual, expected)
    for parameter, original in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, original.grad, atol=1e-6, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_hybrid_replay_checkpoint_reloads_as_native_huggingface(tmp_path):
    model = tiny_hybrid().cuda()
    ids = torch.tensor([[1, 2, 3]], device="cuda")
    original_keys = set(model.state_dict())
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        replay_trace(model, mutation_trace().to(device="cuda"), hidden_only=False)
    model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert not any("attn_implementation" in key for key in config)
    restored = Qwen3_5ForCausalLM.from_pretrained(tmp_path, attn_implementation="eager")
    restored = restored.cuda().eval()
    assert set(restored.state_dict()) == original_keys
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], parameter, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(restored(ids, use_cache=False).logits, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_hybrid_replay_preserves_native_sdpa_masks_and_cache():
    model = torch_hybrid().cuda().eval()
    model.set_attn_implementation("sdpa")
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], device="cuda")
    mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]], device="cuda")
    with (
        torch.no_grad(),
        patch.object(
            F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention
        ) as sdpa,
    ):
        expected = model(ids, attention_mask=mask, use_cache=False).logits
        cache = model(ids[:1, :-1], use_cache=True).past_key_values
        assert sdpa.call_count == 2
        replay_trace(model, mutation_trace().to(device="cuda"), hidden_only=False)
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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_hybrid_checkpointed_replay_gradients(use_reentrant):
    reference, checkpointed = tiny_hybrid().cuda().train(), tiny_hybrid().cuda().train()
    reference.set_attn_implementation("sdpa")
    checkpointed.set_attn_implementation("sdpa")
    checkpointed.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
    )
    plan = mutation_trace().to(device="cuda")
    with patch.object(
        F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention
    ) as sdpa:
        expected = replay_trace(reference, plan, hidden_only=False)
        actual = replay_trace(checkpointed, plan, hidden_only=False)
        torch.testing.assert_close(actual, expected)
        assert sdpa.call_count == 0
        with torch.no_grad():
            checkpointed(torch.tensor([[1, 2, 3]], device="cuda"), use_cache=False)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_hybrid_checkpointed_chunked_supervised_loss():
    reference, checkpointed = tiny_hybrid().cuda().train(), tiny_hybrid().cuda().train()
    checkpointed.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    plan = mutation_trace().to(device="cuda", query_chunk_size=3)
    targets = torch.arange(len(plan.input_ids), device="cuda") % 32
    targets[::3] = -100
    expected = F.cross_entropy(
        replay_trace(reference, plan, hidden_only=False), targets, reduction="sum"
    )
    actual = compute_supervised_loss(checkpointed, plan, targets, chunk_size=3)
    torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad(expected, tuple(reference.parameters()))
    actual_grads = torch.autograd.grad(actual, tuple(checkpointed.parameters()))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_hybrid_overfits_recorded_sequence_targets():
    model = tiny_hybrid().cuda().train()
    trace = example_trace()
    plan = trace.to(device="cuda")
    targets = torch.full((len(trace.tokens),), -100, dtype=torch.long, device="cuda")
    latest = {}
    for row, token in enumerate(trace.tokens):
        if token.block in latest:
            targets[latest[token.block]] = token.token_id
        latest[token.block] = row
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = F.cross_entropy(replay_trace(model, plan, hidden_only=False), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.1
