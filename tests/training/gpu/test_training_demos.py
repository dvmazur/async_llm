"""Demo traces, CUDA training, optimizer precision, and checkpoint reload."""

import pytest
import torch
import torch.nn.functional as F
from minisgl.models import ModelConfig
from minisgl.models.register import _MODEL_REGISTRY
from minisgl.training import Trace, replay_trace
from minisgl_examples.training import student
from minisgl_examples.training.distillation import demo as distillation
from minisgl_examples.training.loss import compute_supervised_loss
from minisgl_examples.training.parallel_thinking import demo
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PreTrainedTokenizerFast,
)

from tests.scripts.training_helpers import collect_example
from tests.scripts.training_helpers import exported_example as exported_example
from tests.scripts.training_helpers import source as source
from tests.training.fixtures import torch_reference_kernels as torch_reference_kernels
from tests.training.models import tiny_hybrid, tiny_image_model, tiny_model


@pytest.mark.parametrize(
    "master_dtype,optimizer_device",
    [(torch.float32, "cuda"), (torch.bfloat16, "cpu")],
    ids=["fp32-gpu-masters", "bf16-cpu-masters"],
)
def test_bf16_checkpointed_training_matches_memory_and_disk_inputs(
    source, tmp_path, monkeypatch, master_dtype, optimizer_device
):
    model_path, trace_path = source
    original_load = student.load_student
    captured, dtypes = {}, []

    def load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        if "model" not in captured:
            model = captured["model"] = result[0]
            model.model.layers[0].self_attn.q_proj.register_forward_pre_hook(
                lambda layer, args: dtypes.append(layer.weight.dtype) if model.training else None
            )
            model.model.embed_tokens.weight.register_hook(
                lambda gradient: captured.update(embedding_gradient=gradient.detach().cpu().clone())
            )
            captured["before"] = model.model.embed_tokens.weight.detach().cpu().clone()
        return result

    original_adamw = torch.optim.AdamW

    def adamw(*args, **kwargs):
        optimizer = captured["optimizer"] = original_adamw(*args, **kwargs)
        original_step = optimizer.step

        def step(*args, **kwargs):
            captured["optimizer_devices"] = {
                parameter.device.type for parameter in optimizer.param_groups[0]["params"]
            }
            return original_step(*args, **kwargs)

        optimizer.step = step
        return optimizer

    monkeypatch.setattr(student, "load_student", load)
    monkeypatch.setattr(torch.optim, "AdamW", adamw)
    options = {
        "steps": 1,
        "lr": 1e-3,
        "device": "cuda",
        "dtype": torch.bfloat16,
        "master_dtype": master_dtype,
        "optimizer_device": optimizer_device,
    }
    memory = student.load_example(trace_path)
    metrics = student.train_student(memory, str(model_path), tmp_path / "memory", **options)
    model = captured["model"]
    assert len(dtypes) >= 2 and set(dtypes) == {torch.bfloat16}
    parameters = captured["optimizer"].param_groups[0]["params"]
    assert all(parameter.dtype == master_dtype for parameter in parameters)
    assert captured["optimizer_devices"] == {optimizer_device}
    assert all(
        parameter.grad.dtype == master_dtype and torch.isfinite(parameter.grad).all()
        for parameter in parameters
        if parameter.grad is not None
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in parameters
    )
    assert torch.isfinite(captured["embedding_gradient"]).all()
    assert captured["embedding_gradient"].abs().sum() > 0
    assert not torch.equal(model.model.embed_tokens.weight.cpu(), captured["before"])
    for state in captured["optimizer"].state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == master_dtype
        assert state["exp_avg"].device.type == optimizer_device
    assert metrics["completed_steps"] == 1 and metrics["reload_exact"]
    reloaded = student.verify_checkpoint(
        trace_path, metrics["checkpoint"], device="cuda", master_dtype=master_dtype
    )
    assert reloaded == metrics["final"]
    disk = student.train_student(trace_path, str(model_path), tmp_path / "disk", **options)
    assert disk["initial"] == metrics["initial"] and disk["final"] == metrics["final"]
    assert disk["losses"] == metrics["losses"] and disk["reload_exact"]
    memory_weights = AutoModelForCausalLM.from_pretrained(metrics["checkpoint"]).state_dict()
    disk_weights = AutoModelForCausalLM.from_pretrained(disk["checkpoint"]).state_dict()
    assert memory_weights.keys() == disk_weights.keys()
    for key in memory_weights:
        torch.testing.assert_close(memory_weights[key], disk_weights[key], rtol=0, atol=0)


def test_distillation_wrapper_trains_loaded_trace(source, tmp_path):
    model_path, trace_path = source
    report = distillation.train_student(
        str(model_path),
        student.load_example(trace_path),
        tmp_path / "trained",
        steps=1,
        learning_rate=1e-3,
    )
    assert report["completed_steps"] == 1 and report["reload_exact"]


@pytest.mark.parametrize("synchronized", [True, False])
def test_recorded_parallel_schedules_support_forward_and_backward(synchronized):
    _, example = collect_example(synchronized)
    trace, targets = example["trace"], example["targets"]
    rows = torch.tensor([i for i, target in enumerate(targets) if target != -100], device="cuda")
    model = tiny_model().cuda().train()
    logits = replay_trace(model, trace, rows=rows, hidden_only=False)
    F.cross_entropy(logits, torch.tensor(targets, device="cuda")[rows]).backward()
    assert torch.isfinite(model.model.embed_tokens.weight.grad).all()
    assert model.model.embed_tokens.weight.grad.abs().sum() > 0


def test_training_step_checkpoint_reload(tmp_path):
    _, example = collect_example(True)
    source, output = tmp_path / "source", tmp_path / "trained"
    reference = tiny_model().cpu()
    reference.save_pretrained(source)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token="0"))
    )
    tokenizer.save_pretrained(source)
    report = demo.train_student(
        str(source), example, output, steps=1, chunk_size=4, device="cuda", dtype=torch.float32
    )
    assert report["reload_exact"] and len(report["losses"]) == 1
    assert report["initial"]["examples"][0]["tokens"] == 10
    updated = AutoModelForCausalLM.from_pretrained(output, device_map="cpu")
    assert not torch.equal(updated.model.embed_tokens.weight, reference.model.embed_tokens.weight)


def test_qwen35_training_preserves_native_checkpoint_architecture(tmp_path, monkeypatch):
    _, example = collect_example(True)
    source, output = tmp_path / "source", tmp_path / "trained"
    reference = tiny_image_model().cpu().eval()
    reference.save_pretrained(source)
    saved = []
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token="0"))
    )
    tokenizer.save_pretrained(source)

    class Processor:
        def __init__(self):
            self.tokenizer = tokenizer

        def save_pretrained(self, path):
            saved.append(path)
            self.tokenizer.save_pretrained(path)

    monkeypatch.setattr(student.AutoProcessor, "from_pretrained", lambda path: Processor())
    report = demo.train_student(
        str(source), example, output, steps=1, chunk_size=4, device="cuda", dtype=torch.float32
    )
    config = AutoConfig.from_pretrained(output)
    native = ModelConfig.from_hf(config)
    assert native.architectures == ["Qwen3_5ForConditionalGeneration"]
    assert native.architectures[0] in _MODEL_REGISTRY and native.is_multimodal
    assert report["reload_exact"] and saved == [output]
    path = student.save_example(example, tmp_path / "recorded.json")
    verified = demo.verify_checkpoint(
        path, output, device="cuda", dtype=torch.float32, chunk_size=4
    )
    assert verified == report["final"]
    updated = AutoModelForImageTextToText.from_pretrained(output, device_map="cpu")
    for name, parameter in reference.model.visual.named_parameters():
        torch.testing.assert_close(
            updated.model.visual.get_parameter(name), parameter, rtol=0, atol=0
        )
    assert not torch.equal(
        updated.model.language_model.embed_tokens.weight,
        reference.model.language_model.embed_tokens.weight,
    )


@pytest.mark.parametrize("read_mode", ["expanded", "factorized"])
def test_export_reloads_for_hybrid_forward_and_backward(exported_example, read_mode):
    (example,) = exported_example["data"]["examples"]
    trace = Trace.from_dict(example["trace"])
    plan = trace.to("cuda", gdn_segment_size=2, gdn_read_mode=read_mode)
    targets = torch.tensor(example["targets"], device="cuda")
    rows = (targets != -100).nonzero().flatten()
    assert not torch.equal(plan.order, torch.arange(trace.num_tokens, device="cuda"))
    model = tiny_hybrid().cuda().train()
    with torch.no_grad():
        logits = replay_trace(model, plan, hidden_only=False)
        expected = F.cross_entropy(logits[rows], targets[rows], reduction="sum")
    loss = compute_supervised_loss(model, plan, targets, chunk_size=2)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for parameter in (
        model.model.embed_tokens.weight,
        model.model.layers[0].linear_attn.in_proj_qkv.weight,
        model.lm_head.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
