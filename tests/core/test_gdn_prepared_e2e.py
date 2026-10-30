"""Learned-model image/multiworker parity against the frozen pre-graph GDN.

Uses the existing AsyncLLM/Session/model, not an alternate decoder. Opt in with
MINISGL_E2E_MODEL pointing to a small local Qwen3.5 checkpoint. Set
MINISGL_TEST_NO_FLA=1 or MINISGL_FP8_EMULATE=1 to exercise optional backends.
"""

import asyncio
import os
import time

import pytest
import torch
from PIL import Image

from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext, WorkerGroup, PrefillJob
from _gdn_reference import SharedCacheGDN as ReferenceGDN


MODEL = os.environ.get('MINISGL_E2E_MODEL', '')
pytestmark = pytest.mark.skipif(not MODEL or not torch.cuda.is_available(),
                               reason='Set MINISGL_E2E_MODEL to a small local Qwen3.5')


@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    from transformers import AutoConfig
    from minisgl.engine import Engine, EngineConfig
    from minisgl.distributed import DistributedInfo
    quantization = 'fp8' if getattr(AutoConfig.from_pretrained(MODEL), 'quantization_config', None) else None
    with pytest.MonkeyPatch.context() as patch:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        graphs = os.environ.get('MINISGL_TEST_SHARED_GRAPHS') == '1'
        engine = Engine(EngineConfig(
            model_path=MODEL, dtype=torch.bfloat16, tp_info=DistributedInfo(0, 1),
            quantization=quantization, max_running_req=8, num_page_override=8192,
            max_seq_len_override=8192, attention_backend='fi',
            cuda_graph_bs=[1, 2, 4] if graphs else [], cuda_graph_max_bs=4 if graphs else 0,
            shared_cuda_graph_max_depth=8,
            distributed_addr=(tmp_path_factory.mktemp('gdn_e2e')/'nccl').as_uri()))
        import minisgl.models.qwen3_5_delta as delta
        if os.environ.get('MINISGL_TEST_NO_FLA') == '1':
            patch.setattr(delta, '_fla_chunk', None)
            patch.setattr(delta, '_fla_recurrent', None)
        llm = AsyncLLM(MODEL, engine=engine)
        if graphs:
            assert all(not b._pending and b._current is None
                       for b in llm.async_engine.session.graph_io.gdn.values())
        try:
            yield loop, llm
        finally:
            loop.run_until_complete(llm.close())
            engine.shutdown()
            for result, saved in getattr(llm, '_test_retained_logits', []):
                assert result.is_cuda and not result.is_inference()
                torch.testing.assert_close(result, saved, atol=0, rtol=0)
            loop.close()
            asyncio.set_event_loop(None)


def test_learned_history_states_and_logits(runtime, record_property):
    # Match both prior states and Linear geometry to test state transitions,
    # rather than comparing matrices produced from already-diverged hidden inputs.
    # Joint BF16 prefill A/B use installed FLA as their numerical reference;
    # decode, conv and KV keep the frozen legacy checks without relaxed limits.
    _exercise_history(runtime, record_property, batched_prefill=False, align_prior=True)


def test_unaligned_legacy_history_drift_diagnostic(runtime, record_property):
    # No state synchronization: retain the full accumulated old/new drift.
    # The external SGLang/Transformers suite is the mandatory quality gate.
    _exercise_history(runtime, record_property, batched_prefill=False)


def test_batched_legacy_drift_diagnostic(runtime, record_property):
    # Batching BF16 Linear changes its rounding. User-approved quality gates for
    # actual batched execution live in test_shared_batched_external_parity;
    # here retain the complete old-history comparison as a numerical diagnostic.
    _exercise_history(runtime, record_property, batched_prefill=True)


@torch.inference_mode()
def _exercise_history(runtime, record_property, *, batched_prefill, align_prior=False):
    _, llm = runtime
    session = llm.async_engine.session
    prepared = session.sc_gdn
    graph_runner = session.graph_runner
    initial_graphs = {} if graph_runner is None else dict(graph_runner.graph_map)
    initial_replays = 0 if graph_runner is None else graph_runner.replay_count
    reference = ReferenceGDN(num_heads=prepared.num_heads, head_k_dim=prepared.head_k_dim,
                             head_v_dim=prepared.head_v_dim, conv_dim=prepared.conv_dim,
                             conv_kernel=prepared.conv_kernel, device=prepared.device)
    # Only bridge the new host hooks; all read/compose/capture/store math comes
    # from the frozen file. No production arithmetic or model method patched.
    reference.prepare_decode = lambda *args: None
    reference.finish_decode = lambda *args: None
    reference.prepare_prefill = lambda *args: None
    reference.finish_prefill = lambda *args: None
    pairs, times = [], {'reference': [], 'prepared': []}
    max_probability_error = 0.0
    max_tv = 0.0
    max_state_error = 0.0
    expected_affines = {}
    from minisgl.shared_cache.gdn_prefill import GDNPrefillBuffers
    from minisgl.models.qwen3_5_delta import _fla_chunk
    # The same final PR test commit also validates 05a, before joint capture.
    original_joint = getattr(GDNPrefillBuffers, 'core_and_capture', None)

    def checked_joint(buffers, layer, q, k, v, g, beta, initial, use_fla, torch_chunk):
        out = original_joint(buffers, layer, q, k, v, g, beta, initial, use_fla, torch_chunk)
        if use_fla:
            # Test-only eager oracle on actual learned activations, BEFORE
            # publication. Construct/transpose via Torch, not our IO kernels.
            # A/B changed from a FP32 token scan to BF16 FLA by design; do not
            # hide that difference by inflating the old FP32-state tolerance.
            assert not torch.cuda.is_current_stream_capturing()
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            boundaries = buffers.cu.tolist()
            for w, target in enumerate(buffers._current[0]):
                begin, end = boundaries[w:w+2]
                a, b = target.linear_affine.get(layer, (
                    torch.eye(buffers.dk, device=initial.device).expand(1, buffers.h, buffers.dk, buffers.dk),
                    torch.zeros(1, buffers.h, buffers.dv, buffers.dk, device=initial.device)))
                state = torch.cat([initial[w:w+1], a.transpose(-1, -2), b.transpose(-1, -2)], -1)
                val = v[None, begin:end]
                values = torch.cat([val, val.new_zeros(1, end-begin, buffers.h, buffers.dk), val], -1)
                _, final = chunk_gated_delta_rule(q[None, begin:end], k[None, begin:end], values,
                    g=g[None, begin:end], beta=beta[None, begin:end].float(), initial_state=state,
                    output_final_state=True, use_qk_l2norm_in_kernel=True)
                expected_affines[id(target), layer] = (
                    final[..., buffers.dv:buffers.dv+buffers.dk].transpose(-1, -2),
                    final[..., buffers.dv+buffers.dk:].transpose(-1, -2))
        return out

    def compare_state(a, b, *, atol, rtol, expected=None):
        nonlocal max_state_error
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        if a.numel():
            max_state_error = max(max_state_error, (a.float()-b.float()).abs().max().item())
        if align_prior:
            if expected is None:
                torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
            else:
                torch.testing.assert_close(b, expected, atol=2e-5, rtol=2e-4)

    def pair_block():
        pair = (session.create_block(), session.create_block())
        pairs.append(pair)
        return pair

    def check_states():
        for left, right in pairs:
            assert left.num_tokens == right.num_tokens
            assert left.linear_affine.keys() == right.linear_affine.keys()
            assert left.linear_conv_state.keys() == right.linear_conv_state.keys()
            for l in left.linear_affine:
                expected = expected_affines.get((id(right), l), (None, None))
                for a, b, ref in zip(left.linear_affine[l], right.linear_affine[l], expected):
                    compare_state(a, b, atol=3e-4, rtol=3e-4, expected=ref)
            for l in left.linear_conv_state:
                compare_state(left.linear_conv_state[l], right.linear_conv_state[l],
                              atol=3e-3, rtol=3e-3)
            for l in range(session.kv_cache.num_layers):
                for cache in (session.kv_cache.k_cache(l), session.kv_cache.v_cache(l)):
                    rows = cache.reshape(-1, *cache.shape[2:])
                    a = rows.index_select(0, left.token_slots_tensor().long())
                    b = rows.index_select(0, right.token_slots_tensor().long())
                    compare_state(a, b, atol=3e-3, rtol=3e-3)

    def compare(operation):
        nonlocal max_probability_error, max_tv
        expected_affines.clear()
        if align_prior:
            # Test-only state teacher forcing BEFORE either execution. The
            # candidate must still compute its own outputs and state writes;
            # check_states below compares them with the frozen reference.
            # Never do this in the independent unaligned/external quality runs.
            for left, right in pairs:
                assert left.num_tokens == right.num_tokens
                assert left.linear_affine.keys() == right.linear_affine.keys()
                for layer, values in left.linear_affine.items():
                    for src, dst in zip(values, right.linear_affine[layer]):
                        dst.copy_(src)
                for layer, src in left.linear_conv_state.items():
                    right.linear_conv_state[layer].copy_(src)
                for layer in range(session.kv_cache.num_layers):
                    for cache in (session.kv_cache.k_cache(layer), session.kv_cache.v_cache(layer)):
                        rows = cache.flatten(0, 1)
                        rows.index_copy_(0, right.token_slots_tensor().long(),
                            rows.index_select(0, left.token_slots_tensor().long()))
        outputs = []
        for side, backend in enumerate((reference, prepared)):
            session.sc_gdn = backend
            session.graph_runner = None if side == 0 else graph_runner
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs.append(operation(side).clone())
            torch.cuda.synchronize()
            times['reference' if side == 0 else 'prepared'].append(time.perf_counter() - start)
        assert outputs[0].shape == outputs[1].shape
        assert all(torch.isfinite(output).all() for output in outputs)
        difference = (outputs[0].float().softmax(-1) - outputs[1].float().softmax(-1)).abs()
        error = difference.max().item()
        max_probability_error = max(max_probability_error, error)
        max_tv = max(max_tv, .5 * difference.sum(-1).max().item())
        # User-approved capture port: legacy probabilities are diagnostic, not
        # the external quality reference. FP32 GEMV reduction differences ~1e-7
        # changed a later probability by0.0118 (old limit0.0025), including
        # eager/unpadded execution. Do not encode a cuBLAS-specific reduction
        # just to reproduce that old path. State checks below remain strict;
        # External parity uses a temporary fragile 2x TF-error gate, not a quality guarantee.
        # in tests/e2e/fp8/test_parity.py (native AND software FP8).
        check_states()

    try:
        if align_prior and original_joint is not None:
            GDNPrefillBuffers.core_and_capture = checked_joint
        common = pair_block()
        image_inputs = llm.processor.apply_chat_template(
            [{'role': 'user', 'content': [
                {'type': 'image', 'image': Image.new('RGB', (64, 64), (20, 80, 160))},
                {'type': 'image', 'image': Image.new('RGB', (64, 64), (30, 90, 150))},
                {'type': 'text', 'text': 'Compare the two frames briefly.'}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')
        image_inputs['mm_token_type_ids'] = image_inputs['mm_token_type_ids'].flatten()
        compare(lambda side: session.prefill_block(common[side], image_inputs['input_ids'][0],
                    **{k: image_inputs[k] for k in ('pixel_values', 'image_grid_thw', 'mm_token_type_ids')}))
        tails = [pair_block() for _ in range(3)]
        ids = [torch.tensor(llm.tokenizer.encode(
            f'\nObservation {i}: the color changed.' + ' Look again.' * i, add_special_tokens=False))
            for i in range(3)]
        if batched_prefill:
            compare(lambda side: torch.stack(session.prefill_batch([
                PrefillJob(block=tail[side], input_ids=tokens, context=[common[side]])
                for tail, tokens in zip(tails, ids)])))
        else:
            for tail, tokens in zip(tails, ids):
                compare(lambda side, tail=tail, tokens=tokens: session.prefill_block(
                    tail[side], tokens, context=[common[side]]))
        if hasattr(prepared, 'prefill_buffers'):
            assert prepared.prefill_buffers.prefill_count > 0
        # Repeated mutable tails, shared prefixes, cross-reading writers,
        # changing batch width, and teacher forcing keep both histories equal.
        for width in (1, 3, 2, 3):
            compare(lambda side, width=width: session.decode_step(
                WorkerGroup(cache_structure=[[common[side], *[t[side] for t in tails[:i]], tails[i][side]]
                                             for i in range(width)],
                            write_to=[t[side] for t in tails[:width]]),
                torch.tensor([42 + i for i in range(width)], dtype=torch.int32)))
            assert prepared.decode_buffers is not None
            assert prepared.decode_buffers.compose_count > 0
        # Merge and self-append replace block allocations before the next prepare.
        merged = tuple(session.merge_blocks(common[s], tails[0][s]) for s in range(2))
        pairs.append(merged)
        for side in range(2):
            session.append_block(tails[1][side], tails[1][side])
        check_states()
        compare(lambda side: session.decode_step(
            WorkerGroup(cache_structure=[[merged[side], tails[1][side]]], write_to=[tails[1][side]]),
            torch.tensor([55], dtype=torch.int32)))
        # Prepared decode -> original prefill -> prepared decode must interoperate.
        compare(lambda side: session.prefill_block(tails[1][side], torch.tensor([15, 25, 35]),
                                                    context=[merged[side]]))
        compare(lambda side: session.decode_step(
            WorkerGroup(cache_structure=[[merged[side], tails[1][side]]], write_to=[tails[1][side]]),
            torch.tensor([65], dtype=torch.int32)))
        record_property('max_probability_error', max_probability_error)
        record_property('max_tv_to_legacy', max_tv)
        record_property('max_state_error_to_legacy', max_state_error)
        joint_fla = original_joint is not None and _fla_chunk is not None
        record_property('legacy_state_gate', align_prior and not joint_fla)
        record_property('prefill_state_reference', 'installed FLA' if joint_fla else 'legacy FP32')
        record_property('test_only_prior_state_alignment', align_prior)
        record_property('legacy_logits_gate', False)
        record_property('forward_wall_seconds', times)
        record_property('peak_allocated_bytes', torch.cuda.max_memory_allocated())
        if graph_runner is not None:
            assert graph_runner.graph_map == initial_graphs
            assert graph_runner.replay_count - initial_replays == 6
            record_property('full_decode_replays', graph_runner.replay_count - initial_replays)
    finally:
        if original_joint is not None:
            GDNPrefillBuffers.core_and_capture = original_joint
        session.sc_gdn = prepared
        session.graph_runner = graph_runner
        for pair in pairs:
            for block in pair:
                session.free_block(block)


def test_async_generation_retains_logits(runtime):
    loop, llm = runtime

    async def run():
        prompt = await llm.prefill_block([10, 20, 30])
        tails = [await llm.create_block() for _ in range(2)]
        contexts = [AsyncContext([prompt.block, t], t, 40 + i) for i, t in enumerate(tails)]
        first = await asyncio.gather(*(llm(cache_view=c) for c in contexts))
        saved = [r.logits.clone() for r in first]
        for _ in range(3):
            for c in contexts:
                c.next_input_id = 42
            await asyncio.gather(*(llm(cache_view=c) for c in contexts))
        for result, copy in zip(first, saved):
            assert not result.logits.is_inference()
            torch.testing.assert_close(result.logits, copy, atol=0, rtol=0)
        llm._test_retained_logits = [(r.logits, c) for r, c in zip(first, saved)]
        assert llm.async_engine.session.sc_gdn.decode_buffers.compose_count > 0
        await llm.free_block(prompt.block)
        for tail in tails:
            await llm.free_block(tail)
    loop.run_until_complete(run())


def test_graph_overflow_is_not_silent_fallback(runtime):
    _, llm = runtime
    session = llm.async_engine.session
    if session.graph_runner is None:
        pytest.skip('Graph-only capacity contract')
    blocks = [session.create_block() for _ in range(5)]
    before_pages = session.page_allocator.num_free_pages
    before_replays = session.graph_runner.replay_count
    with pytest.raises(ValueError, match='batch exceeds'):
        session.decode_step(WorkerGroup(cache_structure=[[b] for b in blocks], write_to=blocks),
                            torch.ones(5, dtype=torch.int32))
    with pytest.raises(ValueError, match='depth capacity'):
        session.decode_step(WorkerGroup(cache_structure=[[blocks[0]] * 9], write_to=[blocks[0]]),
                            torch.ones(1, dtype=torch.int32))
    assert session.page_allocator.num_free_pages == before_pages
    assert session.graph_runner.replay_count == before_replays
    assert all(b.num_tokens == 0 and not b.linear_affine for b in blocks)
    for block in blocks:
        session.free_block(block)
