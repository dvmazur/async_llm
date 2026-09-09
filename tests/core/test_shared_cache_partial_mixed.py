"""Partial mixed batch vs two complete, ordered model passes.

GPU cases use a tiny randomly initialized *real* hybrid/MoE model, real
FlashInfer attention, FLA, pointer compose/capture and the real paged session.
No checkpoint is needed. CPU cases exercise scheduler behavior independently.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import minisgl.core as core
import minisgl.distributed.info as dist_info
import minisgl.models.qwen3_5_delta as delta
from minisgl.core import Context, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.scheduler.async_engine import AsyncCacheEngine
from minisgl.shared_cache import AsyncContext, CacheBlock, PrefillJob, SharedCacheSession, WorkerGroup
from test_async_cache_engine import StubSession


class MixedStub(StubSession):
    max_prefill_rows = None
    can_mix = SharedCacheSession.can_mix

    def __init__(self):
        super().__init__()
        self.mixed_calls = 0

    def mixed_step(self, jobs, group, ids):
        self.mixed_calls += 1
        return self.prefill_batch(jobs), self.decode_step(group, ids)


def test_scheduler_mixes_and_preserves_sampling_and_raw_logits():
    from minisgl.engine.sample import Sampler
    session = MixedStub()
    engine = AsyncCacheEngine(session=session, sampler=Sampler(torch.device("cpu"), 32),
                              enable_mixed_batch=True)
    common, pf, a, b = [session.create_block() for _ in range(4)]
    f = engine.submit_prefill(torch.tensor([3, 4]), write_to=pf, cache_view=[common, a],
                             return_logits=True)
    da = engine.submit_decode(AsyncContext([common, pf, b, a]), 5,
                              forbid_ids=[7], return_logits=True)
    db = engine.submit_decode(AsyncContext([common, a, b]), 8)
    duplicate = engine.submit_decode(AsyncContext([common, a]), 9)
    assert engine.tick() == "mixed"
    assert session.mixed_calls == 1
    assert f.result().argmax() == 6
    token, raw = da.result()
    assert token == 6 and raw[7] == 20
    assert db.result() == 10
    assert isinstance(duplicate.exception(), ValueError)
    assert engine.tick() is None
    assert [a.num_tokens, b.num_tokens, pf.num_tokens] == [1, 1, 2]


@pytest.mark.parametrize("reason", ["same_writer", "chunked", "pending_prefill", "disabled"])
def test_scheduler_keeps_unsupported_groups_separate(reason):
    from minisgl.engine.sample import Sampler
    session = MixedStub()
    engine = AsyncCacheEngine(session=session, sampler=Sampler(torch.device("cpu"), 32),
                              enable_mixed_batch=reason != "disabled")
    pf, tail = session.create_block(), session.create_block()
    engine.submit_prefill(torch.tensor([3, 4]), write_to=pf)
    if reason == "pending_prefill":
        engine.submit_prefill(torch.tensor([6, 7]), write_to=pf)
    if reason == "chunked":
        session.max_prefill_rows = 1
    if reason == "same_writer":
        tail = pf
    future = engine.submit_decode(AsyncContext([tail]), 5)
    assert engine.tick() == "prefill"
    assert not future.done()
    while engine.has_work:
        engine.tick()
    assert future.done()
    if reason != "pending_prefill":
        assert session.mixed_calls == 0


def test_mixed_failure_resolves_both_groups():
    from minisgl.engine.sample import Sampler
    session = MixedStub()
    engine = AsyncCacheEngine(session=session, sampler=Sampler(torch.device("cpu"), 32),
                              enable_mixed_batch=True)
    pf, tail = session.create_block(), session.create_block()
    f = engine.submit_prefill(torch.tensor([3, 4]), write_to=pf)
    d = engine.submit_decode(AsyncContext([tail]), 5)
    session.fail_next = RuntimeError("injected")
    with pytest.raises(RuntimeError, match="injected"):
        engine.tick()
    assert isinstance(f.exception(), RuntimeError)
    assert isinstance(d.exception(), RuntimeError)
    assert not engine.has_work


def test_admission_failure_does_not_orphan_prefill_future(monkeypatch):
    from minisgl.engine.sample import Sampler
    session = MixedStub()
    engine = AsyncCacheEngine(session=session, sampler=Sampler(torch.device("cpu"), 32),
                              enable_mixed_batch=True)
    pf, tail = session.create_block(), session.create_block()
    f = engine.submit_prefill(torch.tensor([3, 4]), write_to=pf)
    d = engine.submit_decode(AsyncContext([tail]), 5)
    def fail(*args):
        raise ValueError("bad admission")
    monkeypatch.setattr(session, "can_mix", fail)
    with pytest.raises(ValueError, match="bad admission"):
        engine.tick()
    assert isinstance(f.exception(), ValueError)
    assert not d.done()
    assert engine.tick() == "decode" and d.result() == 7


@pytest.fixture
def tiny_session(monkeypatch, request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for real attention/model integration")
    from minisgl.kvcache import GDNStatePool, PageAllocator, create_kvcache_pool
    from minisgl.models.config import ModelConfig, RotaryConfig
    from minisgl.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
    from minisgl.moe import create_moe_backend
    from minisgl.utils import torch_dtype
    from test_qwen3_5_moe import _tiny_hf_config

    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
    config = replace(
        ModelConfig.from_hf(_tiny_hf_config()),
        hidden_size=128, num_layers=4, head_dim=64, num_qo_heads=4, num_kv_heads=1,
        vocab_size=128, layer_types=("linear_attention", "full_attention") * 2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        rotary_config=RotaryConfig(64, 32, 1024, 10000., None),
        mrope_section=(6, 5, 5), vision_config=None,
        moe_intermediate_size=64, shared_expert_intermediate_size=64,
    )
    device = torch.device("cuda:0")
    page_size = getattr(request, "param", 1)
    ctx = Context(page_size)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = Qwen3_5MoeForCausalLM(config)
    gen = torch.Generator(device=device).manual_seed(812)
    state = {}
    for name, param in model.state_dict().items():
        value = torch.randn(param.shape, device=device, dtype=param.dtype, generator=gen) * .04
        if name.endswith("A_log"):
            value.zero_()
        if "norm" in name and name.endswith("weight"):
            value.fill_(1.)
        state[name] = value
    model.load_state_dict(state)
    ctx.kv_cache = create_kvcache_pool(config, 2048, page_size, device=device, dtype=torch.bfloat16)
    ctx.gdn_state = GDNStatePool(config, 9, device=device, dtype=torch.bfloat16)
    ctx.moe_backend = create_moe_backend("fused")
    ctx.page_table = torch.zeros(9, 1024, device=device, dtype=torch.int32)
    engine = SimpleNamespace(
        model=model, device=device, ctx=ctx, kv_cache=ctx.kv_cache,
        page_table=ctx.page_table, attn_backend=None, max_seq_len=1024,
        config=SimpleNamespace(model_config=config, max_prefill_rows=None,
                               get_default_sampling_params=lambda: SamplingParams()),
        page_allocator=PageAllocator(2048, page_size, device),
    )
    session = SharedCacheSession(engine)
    yield session
    torch.cuda.synchronize()


def _ids(start, length):
    return torch.arange(start, start + length, dtype=torch.int32) % 128


def _scenario(session, lengths, dependent, nonempty, mrope, implicit_self=False):
    common, p, q, a, b = [session.create_block() for _ in range(5)]
    seed = [PrefillJob(common, _ids(1, 5))]
    if nonempty:
        seed.extend(PrefillJob(block, _ids(12 + i * 4, 3)) for i, block in enumerate((p, q, a, b)))
    session._prefill_batch_fused(seed)
    if mrope:
        # Real 3-axis positions plus a compressed block span; no textual overlay.
        rel = torch.arange(lengths[0]).expand(3, -1).clone()
        rel[0] = 0
        rel[1] //= 2
        rel[2] %= 2
    else:
        rel = None
    jobs = [
        PrefillJob(p, _ids(30, lengths[0]), [common, a] if dependent else [common], mrope_rel=rel),
    ]
    if len(lengths) > 1:
        jobs.append(PrefillJob(q, _ids(50, lengths[1]), [common]))
    group = WorkerGroup(
        cache_structure=[
            [common, p, b, a] if dependent else [common, b, a],
            [common, a] if implicit_self else [common, a, b],
        ], write_to=[a, b],
    )
    return [common, p, q, a, b], jobs, group


def _snapshot(session, blocks):
    values = []
    for block in blocks:
        slots = block.token_slots_tensor().long()
        kv = []
        for layer in range(2):
            for getter in (session.kv_cache.k_cache, session.kv_cache.v_cache):
                tensor = getter(layer)
                kv.append(tensor.flatten(0, 1)[slots].clone())
        values.append({
            "tokens": list(block.token_ids), "length": block.num_tokens,
            "span": block.mrope_span, "pages": block.num_pages,
            "affine": {k: tuple(t.clone() for t in v) for k, v in block.linear_affine.items()},
            "conv": {k: v.clone() for k, v in block.linear_conv_state.items()}, "kv": kv,
        })
    return values


@pytest.mark.parametrize("tiny_session", [1, 4], indirect=True)
@pytest.mark.parametrize("fallback", [False, True], ids=["fla", "no_fla"])
@pytest.mark.parametrize("cache_ratio", [0., 1.6], ids=["cache_off", "cache_on"])
@pytest.mark.parametrize("case", ["ragged", "dependent", "fresh_dependency", "empty", "one_token", "single_prefill", "mrope", "implicit_self"])
def test_complete_mixed_model_matches_two_passes(tiny_session, monkeypatch, fallback, cache_ratio, case):
    session = tiny_session
    session.sc_gdn.configure_compose_cache(cache_ratio)
    if fallback:
        monkeypatch.setattr(delta, "_fla_chunk", None)
        monkeypatch.setattr(delta, "_fla_recurrent", None)
    lengths = [1, 1] if case == "one_token" else [7, 3]
    if case == "single_prefill":
        lengths = [7]
    all_outputs, all_states = [], []
    for mixed in (False, True):
        blocks, jobs, group = _scenario(
            session, lengths, case in ("dependent", "fresh_dependency", "mrope", "implicit_self"),
            case not in ("empty", "fresh_dependency"), case == "mrope", case == "implicit_self",
        )
        calls = []
        projection_rows = {}
        restore_ops = []
        for i, layer in enumerate(session.engine.model.model.layers.op_list):
            ops = [(f"{i}.moe", layer.mlp)]
            if layer._is_linear:
                ops.extend((f"{i}.{name}", getattr(layer.linear_attn, name)) for name in (
                    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"
                ))
            else:
                ops.extend((f"{i}.{name}", getattr(layer.self_attn, name))
                           for name in ("qkv_proj", "o_proj"))
            for name, op in ops:
                projection_rows[name] = []
                forward = op.forward
                def count_rows(x, forward=forward, name=name):
                    projection_rows[name].append(x.shape[0])
                    return forward(x)
                restore_ops.append((op, forward))
                monkeypatch.setattr(op, "forward", count_rows)
        original = session.engine.model.forward
        def counted():
            calls.append(session.engine.ctx.batch.is_mixed)
            return original()
        monkeypatch.setattr(session.engine.model, "forward", counted)
        with torch.inference_mode():
            if mixed:
                pf, dec = session.mixed_step(jobs, group, _ids(80, 2))
            else:
                pf = session._prefill_batch_fused(jobs)
                dec = session.decode_step(group, _ids(80, 2))
            assert calls == ([True] if mixed else [False, False])
            expected_rows = [sum(lengths) + 2] if mixed else [sum(lengths), 2]
            assert all(rows == expected_rows for rows in projection_rows.values()), projection_rows
            out = torch.cat([*pf, dec]).clone()
            states = _snapshot(session, blocks)
            # A mixed write must remain usable on later steps with cache hits.
            follow = [session.decode_step(group, _ids(83 + i, 2)).clone() for i in range(3)]
        all_outputs.append([out, *follow])
        all_states.append(states)
        monkeypatch.setattr(session.engine.model, "forward", original)
        for op, forward in restore_ops:
            monkeypatch.setattr(op, "forward", forward)
        for block in blocks:
            session.free_block(block)
    for a, b in zip(*all_outputs):
        torch.testing.assert_close(a, b, rtol=.04, atol=.02)
    for a, b in zip(*all_states):
        for name in ("tokens", "length", "span", "pages"):
            assert a[name] == b[name]
        for name in ("affine", "conv"):
            torch.testing.assert_close(a[name], b[name], rtol=.04, atol=.004)
        torch.testing.assert_close(a["kv"], b["kv"], rtol=.04, atol=.02)
    assert session.page_allocator.num_free_pages == 2048


def test_context_views_share_cache_but_not_metadata(tiny_session):
    gdn = tiny_session.sc_gdn
    a, b = tiny_session.create_block(), tiny_session.create_block()
    pf = gdn.context_view([[a], [b]], [a, b], [7, 3])
    dec = gdn.context_view([[a, b]], [b])
    assert pf.compose_state_cache is dec.compose_state_cache is gdn.compose_state_cache
    assert pf.prefill_cu_seqlens_cpu.tolist() == [0, 7, 10]
    assert dec.prefill_segments is None and dec.prefill_cu_seqlens is None
    assert pf.cache_structure == [[a], [b]]


def test_failed_mixed_forward_reclaims_uncommitted_pages(tiny_session, monkeypatch):
    session = tiny_session
    blocks, jobs, group = _scenario(session, [7, 3], True, True, False)
    before = session.page_allocator.num_free_pages
    lengths = [b.num_tokens for b in blocks]
    def fail(*args, **kwargs):
        raise RuntimeError("injected before model")
    monkeypatch.setattr(session, "_forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        session.mixed_step(jobs, group, _ids(80, 2))
    assert session.page_allocator.num_free_pages == before
    assert [b.num_tokens for b in blocks] == lengths
    for b in blocks:
        session.free_block(b)


def test_post_prefill_snapshot_does_not_mutate_live_block():
    from minisgl.shared_cache.session import _PreparedPrefill
    device = torch.device("cpu")
    p, tail = CacheBlock(device, 4), CacheBlock(device, 4)
    p.grow_pages(torch.tensor([0]), 3)
    p.token_ids = [1, 2, 3]
    group = WorkerGroup(cache_structure=[[p, tail]], write_to=[tail])
    job = PrefillJob(p, _ids(4, 4))
    plan = _PreparedPrefill(None, [job], [[p]], [job.input_ids], [torch.tensor([4])], [5])
    planned_group = SharedCacheSession._post_prefill_group(group, plan)
    snapshot = planned_group.cache_structure[0][0]
    assert snapshot is not p
    assert snapshot.num_tokens == 7 and snapshot.mrope_span == 5
    assert snapshot.page_starts == [0, 4]
    assert p.num_tokens == 3 and p.mrope_span == 3 and p.page_starts == [0]
    assert p.token_ids == [1, 2, 3]
    assert planned_group.write_to == [tail]
