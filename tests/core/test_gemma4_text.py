"""Small GPU counterparts of Qwen decoder, cache and native/software FP8 checks.

Reference is the unpatched Transformers Gemma4ForCausalLM. Tests use the real
mini-sglang model, KV pool and attention adapter; no replacement model outputs.
"""
from types import SimpleNamespace

import pytest
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM as HFModel

from minisgl.core import Batch, Context, set_global_ctx
from minisgl.distributed import DistributedInfo
from minisgl.models.config import ModelConfig
from minisgl.models.gemma4 import Gemma4ForCausalLM
from minisgl.kvcache import create_kvcache_pool
from minisgl.attention.gemma4 import Gemma4AttentionBackend
from minisgl.utils import torch_dtype
from test_gemma4_config import tiny_config
from tests.gemma_tolerances import GEMMA_SMALL_LOGIT_ATOL

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def models(monkeypatch):
    import minisgl.core as core
    import minisgl.distributed.info as info
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 1))
    torch.manual_seed(123)
    config = tiny_config()
    config.sliding_window = 8
    config._attn_implementation = "sdpa"
    # Match from_pretrained(dtype=BF16): parameters initialize in BF16, while
    # the reference's explicitly-FP32 RoPE buffers must not be downcast by .to().
    with torch.device("cuda"), torch_dtype(torch.bfloat16):
        reference = HFModel(config).eval()
    common = ModelConfig.from_hf(config)
    with torch.device("meta"):
        actual = Gemma4ForCausalLM(common)
    actual.load_state_dict({k: v.clone() for k, v in reference.state_dict().items()})
    context = Context(page_size=4)
    context.kv_cache = create_kvcache_pool(common, 64, 4, torch.bfloat16, torch.device("cuda"))
    context.page_table = torch.arange(256, device="cuda", dtype=torch.int32).reshape(4, 64)
    monkeypatch.setattr(core, "_GLOBAL_CTX", context)
    context.attn_backend = Gemma4AttentionBackend(common)
    yield reference, actual, context


def run(actual, context, ids, cached=0, table=0):
    req = SimpleNamespace(device_len=cached + len(ids), cached_len=cached, table_idx=table)
    batch = Batch(reqs=[req], phase="decode" if len(ids) == 1 and cached else "prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = torch.tensor(ids, device="cuda", dtype=torch.int32)
    batch.positions = torch.arange(cached, cached + len(ids), device="cuda")
    batch.out_loc = context.page_table[table, cached:cached + len(ids)]
    context.attn_backend.prepare_metadata(batch)
    with context.forward_batch(batch), torch.inference_mode():
        return actual.forward().clone()


@pytest.mark.parametrize("length", [1, 7, 8, 9, 17])
def test_prefill_matches_transformers_across_window(models, length):
    reference, actual, context = models
    ids = list(range(3, 3 + length))
    with torch.inference_mode():
        expected = reference(torch.tensor([ids], device="cuda"), use_cache=False).logits[:, -1]
    got = run(actual, context, ids)
    torch.testing.assert_close(got, expected, atol=GEMMA_SMALL_LOGIT_ATOL, rtol=0)


def test_decode_and_extension_match_one_shot(models):
    reference, actual, context = models
    ids = list(range(3, 24))
    run(actual, context, ids[:5])
    for start, stop in [(5, 6), (6, 13), (13, 14), (14, 21)]:
        got = run(actual, context, ids[start:stop], cached=start)
        with torch.inference_mode():
            expected = reference(torch.tensor([ids[:stop]], device="cuda"), use_cache=False).logits[:, -1]
        torch.testing.assert_close(got, expected, atol=GEMMA_SMALL_LOGIT_ATOL, rtol=0)


def test_paged_storage_uses_each_layers_shape(models):
    _, _, ctx = models
    assert ctx.kv_cache.k_cache(0).shape == (64, 4, 2, 128)
    assert ctx.kv_cache.k_cache(1).shape == (64, 4, 1, 256)
    assert ctx.kv_cache.dtype == torch.bfloat16


@pytest.mark.parametrize("emulated", [False, True])
def test_fp8_decoder_keeps_fp8_storage(models, monkeypatch, emulated):
    reference, actual, context = models
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP
    assert type(actual.model.layers[0].mlp) is Gemma4TextMLP
    monkeypatch.setenv("MINISGL_FP8_EMULATE", "1" if emulated else "0")
    state = {k: v.clone() for k, v in reference.state_dict().items()}
    count = 0
    for name, module in list(reference.named_modules()):
        if not isinstance(module, torch.nn.Linear) or name == "lm_head":
            continue
        weight = state[name + ".weight"].float()
        n, k = weight.shape
        tiles = weight.reshape(n // 128, 128, k // 128, 128).permute(0, 2, 1, 3)
        scales = tiles.abs().amax((-2, -1)).clamp_min(1e-10) / 448
        quant = (tiles / scales[..., None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        state[name + ".weight"] = quant.permute(0, 2, 1, 3).reshape(n, k).contiguous()
        state[name + ".weight_scale_inv"] = scales.contiguous()
        # Independent Torch implementation of the SAME serialized W8A8 scheme.
        # Comparing a random quantized network to unquantized random weights is
        # not a kernel-correctness oracle; learned-model quality has separate E2E gates.
        parent_name, _, leaf = name.rpartition(".")
        setattr(reference.get_submodule(parent_name), leaf,
                TorchBlockFP8(state[name + ".weight"], scales))
        count += 1
    actual.load_state_dict(state)
    assert sum(t.dtype == torch.float8_e4m3fn for t in actual.state_dict().values()) == count
    ids = list(range(3, 16))
    got = run(actual, context, ids)
    with torch.inference_mode():
        expected = reference(torch.tensor([ids], device="cuda"), use_cache=False).logits[:, -1]
    tv = (got.float().softmax(-1) - expected.float().softmax(-1)).abs().sum() / 2
    assert tv < 0.015, float(tv)


class TorchBlockFP8(torch.nn.Module):
    """Independent small-matrix oracle; never installed in production."""
    def __init__(self, weight, scales):
        super().__init__()
        self.weight, self.scales = weight, scales

    def forward(self, x):
        shape, k = x.shape[:-1], x.shape[-1]
        groups = x.float().reshape(-1, k // 128, 128)
        xs = groups.abs().amax(-1).clamp_min(1e-10) / 448
        values = (groups / xs[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        result = torch.zeros((values.shape[0], self.weight.shape[0]), device=x.device)
        for g in range(k // 128):
            product = values[:, g] @ self.weight[:, g*128:(g+1)*128].float().T
            ws = self.scales[:, g].repeat_interleave(128)
            result += product * (xs[:, g, None] * ws[None])
        return result.to(x.dtype).reshape(*shape, self.weight.shape[0])
