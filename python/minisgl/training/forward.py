"""Differentiable execution of a recorded shared-cache trajectory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy, deepcopy
from types import MethodType
from typing import TYPE_CHECKING

import torch

from .attention import replay_self_attention
from .gdn import replay_gdn
from .trace import Trace

if TYPE_CHECKING:
    from minisgl.engine import Engine
    from minisgl.llm import AsyncLLM


def _rows(
    rows: Sequence[int] | torch.Tensor | None, count: int, device: torch.device
) -> torch.Tensor | None:
    if rows is None:
        return None
    if not isinstance(rows, (torch.Tensor, Sequence)) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a one-dimensional integer sequence or tensor")
    if not isinstance(rows, torch.Tensor):
        if any(isinstance(row, bool) or not isinstance(row, int) for row in rows):
            raise TypeError("rows must contain integers")
        rows = torch.tensor(rows, dtype=torch.long, device=device)
    if rows.ndim != 1 or rows.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("rows must be a one-dimensional integer sequence or tensor")
    rows = rows.to(device=device, dtype=torch.long)
    if len(rows) and ((rows < 0).any() or (rows >= count).any()):
        raise IndexError("rows must index tokens in the original trace order")
    return rows


def _forward(
    model: torch.nn.Module, trace: Trace, rows: torch.Tensor | None, hidden_only: bool
) -> torch.Tensor:
    if model.config.model_type not in {"qwen3", "qwen3_5_text", "qwen3_5"}:
        raise ValueError("Only dense text Qwen3 and Qwen3.5 are supported")
    backbone = model.model
    if model.config.model_type == "qwen3_5":
        backbone = backbone.language_model
    config, rotary = backbone.config, backbone.rotary_emb
    if getattr(rotary, "rope_type", "default") != "default":
        raise NotImplementedError("Only default RoPE is supported")
    if getattr(backbone, "has_sliding_layers", False) or config.attention_dropout:
        raise NotImplementedError("Replay requires full attention and zero attention dropout")
    kwargs = {}
    if "linear_attention" in config.layer_types:
        from .gdn import GDNReplay

        if config.hidden_act not in {"silu", "swish"}:
            raise NotImplementedError("GDN replay requires SiLU convolution")
        key = (
            config.linear_conv_kernel_dim,
            trace.input_ids.device,
            trace.gdn_segment_size,
            trace.gdn_read_mode,
            trace.gdn_cache_reused,
            trace.gdn_read_backend,
        )
        if key not in trace._gdn:
            trace._gdn[key] = GDNReplay.from_trace(
                trace,
                trace.order,
                device=key[1],
                conv_kernel_size=key[0],
                segment_size=key[2],
                read_mode=key[3],
                cache_reused=key[4],
                read_backend=key[5],
            )
        kwargs["gdn_replay"] = trace._gdn[key]
    for layer in backbone.layers:
        for name, wrapper in (("self_attn", replay_self_attention), ("linear_attn", replay_gdn)):
            module = getattr(layer, name, None)
            if module is not None and not hasattr(module, "_replay_native_forward"):
                module._replay_native_forward = module.forward
                module.forward = MethodType(wrapper, module)
    indices = trace.order.argsort()
    if rows is not None:
        indices = indices[rows]
    if not hidden_only:
        kwargs["logits_to_keep"] = indices
    if trace.multimodal:
        from .images import compute_input_embeddings

        kwargs["inputs_embeds"] = compute_input_embeddings(model, trace)[None]
    else:
        kwargs["input_ids"] = trace.input_ids[None]
    output = (backbone if hidden_only else model)(
        # The replay attention backend applies the recorded block-relative rotations.
        position_ids=torch.zeros_like(trace.input_ids)[None],
        attention_mask={"full_attention": None, "linear_attention": None},
        use_cache=False,
        replay=trace,
        inv_freq=rotary.inv_freq,
        mrope_section=getattr(rotary, "mrope_section", None),
        return_dict=True,
        **kwargs,
    )
    return output.last_hidden_state[0, indices] if hidden_only else output.logits[0]


class _LiveModel(torch.nn.Module):
    """HF operations with meta parameters, bound to live engine weights per call."""

    def __init__(self, engine: Engine, weights: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        config = deepcopy(engine.config.hf_config)
        if config.model_type not in {"qwen3", "qwen3_5_text", "qwen3_5"}:
            raise ValueError("Only dense text Qwen3 and Qwen3.5 are supported")
        first = next(iter(weights.values()))
        factory = (
            AutoModelForImageTextToText if config.model_type == "qwen3_5" else AutoModelForCausalLM
        )
        with torch.inference_mode(False), torch.device("meta"):
            self.model = factory.from_config(config, dtype=first.dtype, attn_implementation="eager")
        self.model.eval()
        # Only rotary buffers need materialization; model parameters stay on meta.
        with torch.inference_mode(False):
            for module in list(self.model.modules()):
                rotary = getattr(module, "rotary_emb", None)
                if rotary is not None:
                    module.rotary_emb = type(rotary)(rotary.config, device=first.device)
                rotary = getattr(module, "rotary_pos_emb", None)
                if rotary is not None:
                    with torch.device(first.device):
                        module.rotary_pos_emb = type(rotary)(rotary.dim, rotary.theta)
        if any(buffer.is_meta for buffer in self.model.buffers()):
            raise NotImplementedError("Unsupported model buffer in AsyncLLM training adapter")
        shapes = {
            name: tuple(value.shape)
            for name, value in self.model.named_parameters(remove_duplicate=False)
        }
        self.bindings = {}
        for name, shape in shapes.items():
            source = name.replace("model.language_model.", "model.", 1)
            offset = None
            if source == "lm_head.weight" and config.get_text_config().tie_word_embeddings:
                source = "model.embed_tokens.weight"
            for projections, fused in ((("q", "k", "v"), "qkv"), (("gate", "up"), "gate_up")):
                for index, projection in enumerate(projections):
                    part = f".{projection}_proj."
                    if part not in name:
                        continue
                    source = source.replace(part, f".{fused}_proj.")
                    offset = sum(
                        shapes[name.replace(part, f".{prior}_proj.")][0]
                        for prior in projections[:index]
                    )
            if source not in weights:
                raise ValueError(f"AsyncLLM is missing training weight {source}")
            tensor = (
                weights[source] if offset is None else weights[source].narrow(0, offset, shape[0])
            )
            if tuple(tensor.shape) != shape:
                raise ValueError(f"AsyncLLM weight shape mismatch for {source}")
            self.bindings[name] = (source, offset, shape)
        for weight in weights.values():
            weight.requires_grad_(True)
        self._source_weights = dict(weights)

    def forward(self, trace: Trace, rows: torch.Tensor | None, hidden_only: bool) -> torch.Tensor:
        return _forward(self.model, trace, rows, hidden_only)

    def run(
        self,
        weights: Mapping[str, torch.Tensor],
        trace: Trace,
        rows: torch.Tensor | None,
        hidden_only: bool,
    ) -> torch.Tensor:
        for name, weight in weights.items():
            if self._source_weights.get(name) is not weight:
                weight.requires_grad_(True)
        self._source_weights = dict(weights)
        parameters = {}
        for name, (source, offset, shape) in self.bindings.items():
            value = weights[source]
            if offset is not None:
                value = value.narrow(0, offset, shape[0])
            if tuple(value.shape) != shape:
                raise ValueError(f"AsyncLLM weight shape mismatch for {source}")
            parameters[f"model.{name}"] = value
        parameters.update(self.named_buffers())
        return torch.func.functional_call(
            self,
            parameters,
            (trace, rows, hidden_only),
            strict=True,
        )


def _live_model(llm: AsyncLLM) -> tuple[_LiveModel, dict[str, torch.Tensor], torch.device]:
    if getattr(llm.async_engine, "has_work", False):
        raise RuntimeError("Finish pending AsyncLLM requests before replay_trace")
    engine = llm.engine
    if engine is None:
        engine = getattr(llm.async_engine.session, "engine", None)
    if engine is None:
        raise ValueError("AsyncLLM replay requires its live model engine")
    if engine.config.tp_info.size != 1 or engine.config.quantization is not None:
        raise NotImplementedError("AsyncLLM replay requires single-rank, unquantized weights")
    weights = engine.model.state_dict()
    if not weights or any(
        not weight.is_floating_point() or weight.is_meta for weight in weights.values()
    ):
        raise ValueError("AsyncLLM replay requires materialized floating-point weights")
    if any(weight.is_inference() or not weight.is_leaf for weight in weights.values()):
        raise ValueError(
            "AsyncLLM weights must be ordinary leaf tensors; construct it outside inference_mode"
        )
    devices = {weight.device for weight in weights.values()}
    if len(devices) != 1:
        raise NotImplementedError("AsyncLLM replay requires weights on one device")
    shell = getattr(engine, "_training_model", None)
    if shell is None:
        shell = engine._training_model = _LiveModel(engine, weights)
    return shell, weights, next(iter(devices))


def replay_trace(
    model: AsyncLLM | torch.nn.Module,
    trace: Trace,
    rows: Sequence[int] | torch.Tensor | None = None,
    *,
    hidden_only: bool = True,
) -> torch.Tensor:
    """Recompute hidden states or logits in original trace order, with gradients.

    ``rows`` selects original token rows, allowing repeats and an empty selection.
    AsyncLLM uses its live, single-rank unquantized weights, enabling their gradients
    on the first call. Optimizers can use ``model.engine.model.state_dict().values()``;
    free inference blocks before resuming generation after a weight update.
    Hugging Face Qwen3 and Qwen3.5 models are also accepted.
    """
    if not isinstance(trace, Trace):
        raise TypeError("trace must be a Trace")
    live = None
    if isinstance(model, torch.nn.Module):
        device = next(model.parameters()).device
    elif hasattr(model, "async_engine"):
        live, weights, device = _live_model(model)
    else:
        raise TypeError("model must be an AsyncLLM or Hugging Face model")
    # A forward/backward pair keeps its visibility even if recording continues.
    trace = trace.to(device)
    rows = _rows(rows, trace.num_tokens, device)
    if not trace.cache_plans:
        trace._gdn.clear()
        trace._attention.clear()
    trace = copy(trace)
    trace.batches, trace.updates = list(trace.batches), list(trace.updates)
    trace._replay_snapshot = True
    if not trace.cache_plans:
        trace._gdn, trace._attention = {}, {}
    return (
        _forward(model, trace, rows, hidden_only)
        if live is None
        else live.run(weights, trace, rows, hidden_only)
    )
