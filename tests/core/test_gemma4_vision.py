"""Gemma counterparts of the Qwen vision-tower and multimodal decoder tests."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Gemma4Config, Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration as HFModel

from minisgl.core import Batch, Context
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
def models(monkeypatch, request):
    import minisgl.core as core
    import minisgl.distributed.info as info
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 1))
    torch.manual_seed(37)
    text = tiny_config()
    text.sliding_window, dtype = getattr(request, "param", (text.sliding_window, torch.bfloat16))
    text.use_bidirectional_attention = "vision"
    vision = Gemma4VisionConfig(hidden_size=128, intermediate_size=256,
                               num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                               head_dim=32, global_head_dim=32, patch_size=2,
                               position_embedding_size=64, pooling_kernel_size=3,
                               default_output_length=4, max_position_embeddings=256)
    config = Gemma4Config(text_config=text.to_dict(), vision_config=vision.to_dict(),
                         image_token_id=99, vision_soft_tokens_per_image=4)
    config.architectures = ["Gemma4ForConditionalGeneration"]
    config._attn_implementation = "sdpa"
    with torch.device("cuda"), torch_dtype(dtype):
        reference = HFModel(config).eval()
    common = ModelConfig.from_hf(config)
    with torch.device("meta"):
        actual = Gemma4ForCausalLM(common)
    actual.load_state_dict({k: v.clone() for k, v in reference.state_dict().items()})
    context = Context(page_size=4)
    context.kv_cache = create_kvcache_pool(common, 64, 4, dtype, torch.device("cuda"))
    context.page_table = torch.arange(256, device="cuda", dtype=torch.int32).reshape(4, 64)
    monkeypatch.setattr(core, "_GLOBAL_CTX", context)
    context.attn_backend = Gemma4AttentionBackend(common)
    return reference, actual, context


def inputs(num_images, height=6, width=6):
    xy = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij"), -1)
    patches = height * width
    positions = xy.reshape(1, patches, 2).repeat(num_images, 1, 1).cuda()
    pixels = torch.randn(num_images, patches, 12, device="cuda", dtype=torch.bfloat16)
    ids = [3] + [t for _ in range(num_images) for t in [4, *([99] * (patches // 9)), 5]] + [6]
    ids = torch.tensor([ids], device="cuda")
    return dict(input_ids=ids, pixel_values=pixels, image_position_ids=positions,
                mm_token_type_ids=(ids == 99).long())


@pytest.mark.parametrize("num_images", [1, 2])
@pytest.mark.parametrize("grid", [(6, 6), (6, 9), (9, 6)])
@pytest.mark.parametrize("padded", [False, True])
def test_vision_tower_matches_transformers(models, num_images, grid, padded):
    reference, actual, _ = models
    data = inputs(num_images, *grid)
    if padded:
        data["pixel_values"] = torch.nn.functional.pad(data["pixel_values"], (0, 0, 0, 18))
        data["image_position_ids"] = torch.nn.functional.pad(
            data["image_position_ids"], (0, 0, 0, 18), value=-1)
    with torch.inference_mode():
        expected = torch.cat(reference.get_image_features(
            data["pixel_values"], data["image_position_ids"], return_dict=True).pooler_output)
        got = actual.encode_images(data["pixel_values"], data["image_position_ids"])
    assert got.shape == (grid[0] * grid[1] // 9 * num_images, 256)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_images", [1, 2])
@pytest.mark.parametrize("precomputed", [False, True])
def test_image_prefill_matches_transformers(models, num_images, precomputed):
    reference, actual, context = models
    data = inputs(num_images)
    n = data["input_ids"].numel()
    req = SimpleNamespace(device_len=n, cached_len=0, table_idx=0)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = data["input_ids"][0].int()
    batch.positions = torch.arange(n, device="cuda")
    batch.out_loc = context.page_table[0, :n]
    batch.mm_token_type_ids = data["mm_token_type_ids"][0]
    if precomputed:
        batch.image_embeds = actual.encode_images(data["pixel_values"], data["image_position_ids"])
    else:
        batch.pixel_values, batch.image_position_ids = data["pixel_values"], data["image_position_ids"]
    context.attn_backend.prepare_metadata(batch)
    with torch.inference_mode():
        expected = reference(**data, use_cache=False).logits[:, -1]
        with context.forward_batch(batch):
            got = actual.forward()
    torch.testing.assert_close(got, expected, atol=GEMMA_SMALL_LOGIT_ATOL, rtol=0)
