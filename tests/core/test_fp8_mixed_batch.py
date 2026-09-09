"""FP8 mixed unit tests: bookkeeping (CPU) and a random GDN+MoE block (GPU).

No model checkpoints. All fresh full-model tests live in tests/e2e/fp8.
"""

from dataclasses import replace
import json

import pytest
import torch

from minisgl.core import Batch
from minisgl.scheduler.utils import mix_batches
from test_mixed_batch import _hybrid_config, _make_req
from tests.e2e.fp8 import comparison as metrics, serving as mixed
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _teacher_cases():
    return [dict(name="a", prompt_ids=[1, 2], teacher_tokens=[3, 4, 5]),
            dict(name="b", prompt_ids=[2, 1, 2], teacher_tokens=[5, 6])]


def test_recorder_routes_changing_row_order_and_preserves_logits():
    recorder = mixed.TeacherRecorder(_teacher_cases())
    req0 = _make_req(0, input_len=2, cached_len=0)
    recorder.before_forward(Batch(reqs=[req0], phase="prefill"))
    first = torch.arange(8).float().view(1, 8)
    assert recorder.sample(first, None).tolist() == [3]
    req0.complete_one()
    req1 = _make_req(1, input_len=3, cached_len=0)
    batch = mix_batches(Batch(reqs=[req1], phase="prefill"), Batch(reqs=[req0], phase="decode"))
    recorder.before_forward(batch)
    logits = torch.arange(16).float().view(2, 8)
    original = logits.clone()
    assert recorder.sample(logits, None).tolist() == [5, 4]  # UID order is now 1, 0
    torch.testing.assert_close(logits, original, rtol=0, atol=0)
    logits.zero_()  # Saved data must also survive reuse of the source buffer.
    req0.complete_one()
    req1.complete_one()
    recorder.before_forward(Batch(reqs=[req0, req1], phase="decode"))
    assert recorder.sample(first.expand(2, -1), None).tolist() == [5, 6]
    a, b = recorder.finish()
    torch.testing.assert_close(a[1], original[1])
    torch.testing.assert_close(b[0], original[0])
    coverage = mixed.check_schedule(recorder.forwards, "mixed")
    assert coverage["mixed_batches"] == 1
    assert coverage["mixed_prefill_rows"] == coverage["mixed_decode_rows"] == 1
    with pytest.raises(AssertionError):
        mixed.check_schedule(recorder.forwards, "sequential")


def test_recorder_rejects_missing_rows_nonfinite_and_missing_mixing():
    recorder = mixed.TeacherRecorder(_teacher_cases())
    with pytest.raises(AssertionError, match="incomplete"):
        recorder.finish()
    req = _make_req(0, input_len=2, cached_len=0)
    recorder.before_forward(Batch(reqs=[req], phase="prefill"))
    with pytest.raises(AssertionError):
        recorder.sample(torch.zeros(2, 8), None)
    with pytest.raises(AssertionError, match="non-finite"):
        recorder.sample(torch.full((1, 8), float("nan")), None)
    recorder.sample(torch.zeros(1, 8), None)
    with pytest.raises(AssertionError, match="no mixed"):
        mixed.check_schedule(recorder.forwards, "mixed")
    with pytest.raises(AssertionError, match="missing/duplicated"):
        recorder.before_forward(Batch(reqs=[req], phase="prefill"))


def test_external_comparison_requires_evidence_of_mixed_execution(tmp_path):
    roots = [tmp_path / name for name in ("mini", "sglang", "transformers")]
    for root in roots:
        root.mkdir()
        manifest = dict(fixtures_sha256="same-fixtures", cases=["text_0"],
                        arguments=dict(model="same-model", tokens=2))
        if root == roots[0]:
            manifest["arguments"]["mini_scheduling"] = "mixed"
        (root / "complete.json").write_text(json.dumps(manifest))
        logits = torch.tensor([[1., 0., -1.], [2., 0., -1.]])
        torch.save(logits, root / "text_0_decode.pt")
        torch.save(logits[0], root / "text_0_cold0.pt")
    schedule = dict(mode="mixed", forwards=[
        dict(label="decode", mixed=False, rows=[dict(phase="decode")])])
    path = roots[0] / "schedule.json"
    path.write_text(json.dumps(schedule))
    with pytest.raises(AssertionError, match="no mixed"):
        metrics.compare(*roots)
    schedule["forwards"].append(dict(label="decode", mixed=True,
                                     rows=[dict(phase="prefill"), dict(phase="decode")]))
    path.write_text(json.dumps(schedule))
    result = metrics.compare(*roots)
    assert result["passed"] and result["mini_scheduling"] == "mixed"
    assert result["mixed_schedule"] == schedule
    assert result["metrics"]["mini"]["prefill"]["positions"] == 2  # actual + cold
    changed = logits.clone()
    changed[0] = changed[0].flip(0)
    torch.save(changed, roots[0] / "text_0_decode.pt")
    assert not metrics.compare(*roots)["passed"]  # Cannot hide a broken mixed-prefill row.


@pytest.fixture
def fp8_block(monkeypatch):
    import minisgl.core as core
    import minisgl.distributed.info as dist
    from minisgl.distributed import DistributedInfo
    from minisgl.kvcache import GDNStatePool
    from minisgl.models.qwen3_5_moe import Qwen3_5MoeDecoderLayer
    from minisgl.moe import create_moe_backend
    from minisgl.utils import torch_dtype

    monkeypatch.setattr(dist, "_TP_INFO", DistributedInfo(0, 1))
    ctx = core.Context(page_size=1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    # Keep the upstream tiny GDN config, but align all quantized dimensions to
    # the checkpoint's 128x128 FP8 blocks and include routed + shared experts.
    config = replace(_hybrid_config(), hidden_size=256, linear_key_head_dim=32,
                     linear_value_head_dim=32, num_experts=8, num_experts_per_tok=2,
                     moe_intermediate_size=128, shared_expert_intermediate_size=128,
                     norm_topk_prob=True, model_type="qwen3_5_moe")
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        layer = Qwen3_5MoeDecoderLayer(config, 0, 0, 0)
    quantized = {
        "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
        "linear_attn.out_proj.weight", "mlp.experts.gate_up_proj", "mlp.experts.down_proj",
        "mlp.shared_expert.gate_up_proj.weight", "mlp.shared_expert.down_proj.weight",
    }
    torch.manual_seed(271)
    state = {}
    for name, template in layer.state_dict().items():
        if name in quantized:
            shape = template.shape
            state[name] = (torch.randn(shape, device="cuda") * 20).to(torch.float8_e4m3fn)
            state[name + "_scale_inv"] = torch.rand(
                (*shape[:-2], shape[-2] // 128, shape[-1] // 128), device="cuda") * .001 + .001
        else:
            state[name] = (torch.randn(template.shape, device="cuda") * .05).to(template.dtype)
    # Exercise the production FP8 state-dict loading, not manual attribute dispatch.
    layer.load_state_dict(state)
    assert not state
    assert {k for k, v in layer.state_dict().items() if v.dtype == torch.float8_e4m3fn} == quantized
    ctx.moe_backend = create_moe_backend("fused")
    ctx.gdn_state = GDNStatePool(config, 16, torch.device("cuda"), torch.bfloat16)
    return layer, ctx


def _close_block_output(actual, expected):
    # Different GEMM row tilings can round differently in BF16. This is a
    # same-FP8-weight comparison, so allow <1% global L2, not FP8-vs-BF16 error.
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    relative = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    assert relative < .01, relative.item()
    torch.testing.assert_close(actual, expected, rtol=.03, atol=.003)


@requires_cuda
@pytest.mark.parametrize("prefill_lens,num_decode", [([1], 1), ([5, 3], 3), ([65, 3, 1], 2)])
@pytest.mark.parametrize("use_fla", [False, True], ids=["torch-fallback", "fla"])
@torch.inference_mode()
def test_fp8_gdn_moe_mixed_matches_separate(fp8_block, monkeypatch, prefill_lens, num_decode, use_fla):
    import minisgl.models.qwen3_5_delta as gdn_module

    if use_fla:
        if gdn_module._fla_chunk is None or gdn_module._fla_recurrent is None:
            pytest.skip("FLA is not installed; Torch fallback is tested separately")
    else:
        monkeypatch.setattr(gdn_module, "_fla_chunk", None)
        monkeypatch.setattr(gdn_module, "_fla_recurrent", None)
    layer, ctx = fp8_block
    pool = ctx.gdn_state
    # Noncontiguous slots and unequal lengths catch indexing/split errors.
    slots = [7, 2, 9, 0, 5, 11]
    prefills = [_make_req(slots[i], input_len=n, cached_len=0) for i, n in enumerate(prefill_lens)]
    decodes = [_make_req(slots[len(prefills) + i], input_len=19 + i, cached_len=18 + i)
               for i in range(num_decode)]
    split = sum(prefill_lens)
    pool.conv_state.normal_(std=.1)
    pool.recurrent_state.normal_(std=.1)
    before = pool.conv_state.clone(), pool.recurrent_state.clone()
    x = torch.randn(split + num_decode, 256, dtype=torch.bfloat16, device="cuda")

    def run(batch, rows):
        with ctx.forward_batch(batch):
            # Some fused MoE paths reuse input storage.
            return tuple(v.clone() for v in layer.forward(rows.clone()))

    batch = mix_batches(Batch(reqs=prefills, phase="prefill"), Batch(reqs=decodes, phase="decode"))
    assert batch.is_mixed
    got = run(batch, x)
    after = pool.conv_state.clone(), pool.recurrent_state.clone()
    pool.conv_state.copy_(before[0])
    pool.recurrent_state.copy_(before[1])
    left = run(Batch(reqs=prefills, phase="prefill"), x[:split])
    right = run(Batch(reqs=decodes, phase="decode"), x[split:])
    for actual, a, b in zip(got, left, right):
        _close_block_output(actual, torch.cat([a, b]))
    # GDN runs the same two segments in both cases: state should be exact.
    torch.testing.assert_close(after[0], pool.conv_state, rtol=0, atol=0)
    torch.testing.assert_close(after[1], pool.recurrent_state, rtol=0, atol=0)
    for req in decodes:
        torch.testing.assert_close(after[0][0, req.table_idx, :, :-1],
                                   before[0][0, req.table_idx, :, 1:], rtol=0, atol=0)
    used = {r.table_idx for r in prefills + decodes}
    for slot in set(range(16)) - used:
        assert torch.equal(after[0][0, slot], before[0][0, slot])
        assert torch.equal(after[1][0, slot], before[1][0, slot])
    # Follow with more decode steps: catch damaged state hidden by one output.
    for _ in range(3):
        rows = torch.randn(num_decode, 256, device="cuda", dtype=torch.bfloat16)
        want = run(Batch(reqs=decodes, phase="decode"), rows)
        reference = pool.conv_state.clone(), pool.recurrent_state.clone()
        pool.conv_state.copy_(after[0])
        pool.recurrent_state.copy_(after[1])
        actual = run(Batch(reqs=decodes, phase="decode"), rows)
        for a, b in zip(actual, want):
            _close_block_output(a, b)
        after = pool.conv_state.clone(), pool.recurrent_state.clone()
        torch.testing.assert_close(after[0], reference[0], rtol=0, atol=0)
        torch.testing.assert_close(after[1], reference[1], rtol=0, atol=0)
