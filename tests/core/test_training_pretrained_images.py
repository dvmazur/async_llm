import gc
import hashlib
import json
import os
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F
from minisgl.training.replay import Replay, forward
from minisgl.training.sft import supervised_loss
from minisgl.training.trace import ForwardBatch, ForwardRequest, TensorData, Trace
from PIL import Image
from test_training_logits import torch_reference_kernels as torch_reference_kernels
from transformers import AutoModelForImageTextToText, AutoProcessor


@pytest.fixture
def pretrained_source(torch_reference_kernels, monkeypatch):
    source = os.environ.get("MINISGL_TRAINING_LOGITS_MODEL")
    if not source:
        pytest.skip("Set MINISGL_TRAINING_LOGITS_MODEL to a local pretrained Qwen3.5 model")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.cuda.reset_peak_memory_stats()
    yield source
    gc.collect()
    torch.cuda.empty_cache()


def load_model(source):
    model = AutoModelForImageTextToText.from_pretrained(
        source,
        dtype=torch.float32,
        device_map="cuda",
        attn_implementation="eager",
        local_files_only=True,
    )
    assert model.config.model_type == "qwen3_5"
    assert all(parameter.requires_grad for parameter in model.parameters())
    return model


def processed_inputs(processor):
    images = [
        Image.frombytes(
            "RGB",
            (width, height),
            bytes(
                (17 * x + 29 * y + 73 * c) % 256
                for y in range(height)
                for x in range(width)
                for c in range(3)
            ),
        )
        for width, height in ((128, 128), (192, 128))
    ]
    text = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "Compare the colors and patterns in these images."},
                ],
            },
            {
                "role": "assistant",
                "content": "Both images contain repeated colored stripes with different widths.",
            },
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    inputs = processor(text=[text], images=images, do_resize=False, return_tensors="pt")
    assert inputs["image_grid_thw"].shape == (2, 3)
    assert inputs["attention_mask"].all()
    return text, dict(inputs)


def saved_trace(model, inputs, path):
    positions, _ = model.model.get_rope_index(
        inputs["input_ids"],
        inputs["mm_token_type_ids"],
        inputs["image_grid_thw"],
        attention_mask=inputs["attention_mask"],
    )
    request = ForwardRequest(
        input_ids=tuple(inputs["input_ids"][0].tolist()),
        write_to=0,
        write_start=0,
        cache_view=((0, 0),),
        pixel_values=TensorData.capture(inputs["pixel_values"]),
        image_grid_thw=TensorData.capture(inputs["image_grid_thw"]),
        mm_token_type_ids=TensorData.capture(inputs["mm_token_type_ids"][0]),
        mrope_positions=TensorData.capture(positions[:, 0]),
        write_span=0,
        cache_spans=(0,),
    )
    trace = Trace(steps=1, batches=[ForwardBatch(0, "prefill", (request,))])
    path.write_text(json.dumps(asdict(trace)))
    restored = Trace.from_dict(json.loads(path.read_text()))
    assert asdict(restored) == asdict(trace)
    return restored


def logit_errors(actual, expected):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    error = actual.double() - expected.double()
    return {
        "max_abs": error.abs().max().item(),
        "rms": error.square().mean().sqrt().item(),
        "argmax_disagreements": (actual.argmax(-1) != expected.argmax(-1)).sum().item(),
    }


def assert_logits(metrics):
    assert metrics["max_abs"] < 2e-4, metrics
    assert metrics["rms"] < 2e-5, metrics
    assert metrics["argmax_disagreements"] == 0, metrics


def weight_hashes(model):
    return {
        name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in model.state_dict().items()
    }


@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
def test_pretrained_image_training_checkpoint(pretrained_source, read_mode, tmp_path):
    parity_failures = []
    report = {
        "read_mode": read_mode,
        "dtype": "float32",
        "status": "running",
        "optimizer_state_resume_tested": False,
        "parity_failures": parity_failures,
        "tolerances": {
            "logits_max_abs": 2e-4,
            "logits_rms": 2e-5,
            "gradient_atol": 3e-6,
            "gradient_rtol": 4e-4,
            "loss_atol": 3e-6,
            "loss_rtol": 3e-5,
        },
    }

    def progress(stage, **metrics):
        report.update(stage=stage, **metrics)
        (tmp_path / "report.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps({"stage": stage, **{k: v for k, v in metrics.items() if k != "gradients"}}),
            flush=True,
        )

    def check_logits(label, metrics):
        try:
            assert_logits(metrics)
        except AssertionError:
            parity_failures.append(label)

    def check_loss(label, actual, expected):
        try:
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        except AssertionError:
            parity_failures.append(label)

    progress("load")
    processor = AutoProcessor.from_pretrained(pretrained_source, local_files_only=True)
    model = load_model(pretrained_source).train()
    text, inputs = processed_inputs(processor)
    trace = saved_trace(model, inputs, tmp_path / "trace.json")
    plan = Replay.from_trace(
        trace,
        device="cuda",
        query_chunk_size=32,
        gdn_segment_size=8,
        gdn_read_mode=read_mode,
        gdn_read_backend="torch",
    )
    targets = torch.full_like(inputs["input_ids"][0], -100)
    targets[:-1] = inputs["input_ids"][0, 1:]
    targets[:-1][inputs["mm_token_type_ids"][0, 1:] != 0] = -100
    count = int((targets != -100).sum())
    assert count > 1
    cuda_inputs = {name: value.cuda() for name, value in inputs.items()}
    cuda_targets = targets.cuda()
    progress(
        "native_forward",
        tokens=trace.num_tokens,
        supervised_tokens=count,
        image_grids=inputs["image_grid_thw"].tolist(),
    )
    expected = model(**cuda_inputs, use_cache=False).logits[0]
    expected_loss = F.cross_entropy(expected, cuda_targets)
    expected_logits = expected.detach().cpu()
    expected_loss_value = expected_loss.detach().cpu()
    progress("native_backward", native_loss=expected_loss.item())
    expected_loss.backward()
    native_gradients = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        native_gradients[name] = parameter.grad.detach().cpu()
    model.zero_grad(set_to_none=True)
    del expected, expected_loss

    progress("replay_forward")
    with torch.no_grad():
        actual = forward(model, plan).cpu()
    metrics = logit_errors(actual, expected_logits)
    progress("replay_logits", forward=metrics)
    check_logits("initial_logits", metrics)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert model.is_gradient_checkpointing
    loss = supervised_loss(model, plan, targets, chunk_size=16, checkpointing=True) / count
    progress("replay_backward", replay_loss=loss.item())
    check_loss("initial_loss", loss.detach().cpu(), expected_loss_value)
    loss.backward()
    gradients, failures = {}, []
    norms = {"vision": 0.0, "decoder": 0.0}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        actual_gradient = parameter.grad.detach().cpu()
        expected_gradient = native_gradients.pop(name)
        assert torch.isfinite(actual_gradient).all(), name
        difference = actual_gradient - expected_gradient
        matches = torch.isclose(actual_gradient, expected_gradient, atol=3e-6, rtol=4e-4)
        gradients[name] = {
            "max_abs": difference.abs().max().item(),
            "error_rms": difference.double().square().mean().sqrt().item(),
            "reference_max_abs": expected_gradient.abs().max().item(),
            "reference_rms": expected_gradient.double().square().mean().sqrt().item(),
            "max_tolerance_excess": (difference.abs() - (3e-6 + 4e-4 * expected_gradient.abs()))
            .max()
            .clamp_min(0)
            .item(),
            "elements": parameter.numel(),
            "mismatches": int((~matches).sum()),
        }
        if not matches.all():
            failures.append(name)
        group = "vision" if name.startswith("model.visual.") else "decoder"
        norms[group] += actual_gradient.double().square().sum().item()
    if failures:
        parity_failures.append("gradients")
    progress(
        "gradients",
        gradients=gradients,
        parameters_compared=len(gradients),
        gradient_failures=failures,
        gradient_norms_squared=norms,
        gradient_elements_compared=sum(item["elements"] for item in gradients.values()),
        gradient_elements_mismatched=sum(item["mismatches"] for item in gradients.values()),
    )
    assert all(value > 0 for value in norms.values())
    del native_gradients, actual_gradient, expected_gradient, difference, matches, loss

    samples = {
        "vision": model.model.visual.patch_embed.proj.weight,
        "decoder": model.model.language_model.layers[0].mlp.down_proj.weight,
    }
    before = {name: parameter.detach().cpu().clone() for name, parameter in samples.items()}
    progress("optimizer_start")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
    optimizer.step()
    updates = {
        name: (parameter.detach().cpu() - before[name]).abs().max().item()
        for name, parameter in samples.items()
    }
    assert all(value > 0 for value in updates.values()), updates
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    optimizer.zero_grad(set_to_none=True)
    model.gradient_checkpointing_disable()
    model.eval()
    with torch.no_grad():
        updated_logits = model(**cuda_inputs, use_cache=False).logits[0].cpu()
        updated_loss = F.cross_entropy(updated_logits, targets)
        updated_replay = forward(model, plan).cpu()
    metrics = logit_errors(updated_replay, updated_logits)
    progress(
        "optimizer_step", updates=updates, updated_loss=updated_loss.item(), updated_forward=metrics
    )
    check_logits("updated_logits", metrics)
    assert not torch.equal(updated_logits, expected_logits)

    checkpoint = tmp_path / "checkpoint"
    progress("checkpoint_save_start")
    hashes = weight_hashes(model)
    model.save_pretrained(checkpoint)
    processor.save_pretrained(checkpoint)
    progress("checkpoint_saved", checkpoint=str(checkpoint), state_tensors=len(hashes))
    del optimizer, model, samples, parameter
    gc.collect()
    torch.cuda.empty_cache()

    progress("checkpoint_reload_start")
    reloaded_processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    restored_text, restored_inputs = processed_inputs(reloaded_processor)
    assert text == restored_text and inputs.keys() == restored_inputs.keys()
    for name in inputs:
        assert torch.equal(inputs[name], restored_inputs[name]), name
    reloaded = load_model(checkpoint).eval()
    assert hashes == weight_hashes(reloaded)
    with torch.no_grad():
        reloaded_logits = reloaded(**cuda_inputs, use_cache=False).logits[0].cpu()
        reloaded_replay = forward(reloaded, plan).cpu()
        reloaded_loss = supervised_loss(reloaded, plan, targets, chunk_size=16) / count
    reload_metrics = logit_errors(reloaded_logits, updated_logits)
    replay_metrics = logit_errors(reloaded_replay, updated_logits)
    progress(
        "reload",
        reload=reload_metrics,
        reloaded_replay=replay_metrics,
        reloaded_loss=reloaded_loss.item(),
    )
    check_logits("reload_logits", reload_metrics)
    check_logits("reloaded_replay_logits", replay_metrics)
    check_loss("reload_loss", reloaded_loss.cpu(), updated_loss)
    del reloaded
    progress(
        "complete",
        status="failed" if parity_failures else "passed",
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
    )
    assert (
        not parity_failures
    ), f"Numerical parity failures: {parity_failures}; see {tmp_path / 'report.json'}"
