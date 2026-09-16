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
from minisgl.shared_cache import AsyncContext, WorkerGroup
from _gdn_reference import SharedCacheGDN as ReferenceGDN


MODEL = os.environ.get('MINISGL_E2E_MODEL', '')
pytestmark = pytest.mark.skipif(not MODEL or not torch.cuda.is_available(),
                               reason='Set MINISGL_E2E_MODEL to a small local Qwen3.5')


@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    from transformers import AutoConfig
    quantization = 'fp8' if getattr(AutoConfig.from_pretrained(MODEL), 'quantization_config', None) else None
    with pytest.MonkeyPatch.context() as patch:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        llm = AsyncLLM(MODEL, quantization=quantization, max_running_req=8, num_page_override=8192,
                       max_seq_len_override=8192, attention_backend='fi',
                       distributed_addr=(tmp_path_factory.mktemp('gdn_e2e')/'nccl').as_uri())
        import minisgl.models.qwen3_5_delta as delta
        if os.environ.get('MINISGL_TEST_NO_FLA') == '1':
            patch.setattr(delta, '_fla_chunk', None)
            patch.setattr(delta, '_fla_recurrent', None)
        try:
            yield loop, llm
        finally:
            loop.run_until_complete(llm.close())
            loop.close()
            asyncio.set_event_loop(None)


@torch.inference_mode()
def test_learned_history_states_and_logits(runtime, record_property):
    _, llm = runtime
    session = llm.async_engine.session
    prepared = session.sc_gdn
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
                for a, b in zip(left.linear_affine[l], right.linear_affine[l]):
                    torch.testing.assert_close(a, b, atol=3e-4, rtol=3e-4)
            for l in left.linear_conv_state:
                torch.testing.assert_close(left.linear_conv_state[l], right.linear_conv_state[l],
                                           atol=3e-3, rtol=3e-3)
            for l in range(session.kv_cache.num_layers):
                for cache in (session.kv_cache.k_cache(l), session.kv_cache.v_cache(l)):
                    rows = cache.reshape(-1, *cache.shape[2:])
                    a = rows.index_select(0, left.token_slots_tensor().long())
                    b = rows.index_select(0, right.token_slots_tensor().long())
                    torch.testing.assert_close(a, b, atol=3e-3, rtol=3e-3)

    def compare(operation):
        nonlocal max_probability_error
        outputs = []
        for side, backend in enumerate((reference, prepared)):
            session.sc_gdn = backend
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs.append(operation(side).clone())
            torch.cuda.synchronize()
            times['reference' if side == 0 else 'prepared'].append(time.perf_counter() - start)
        error = (outputs[0].float().softmax(-1) - outputs[1].float().softmax(-1)).abs().max().item()
        max_probability_error = max(max_probability_error, error)
        torch.testing.assert_close(outputs[0].float().softmax(-1), outputs[1].float().softmax(-1),
                                   atol=2.5e-3, rtol=0)
        check_states()

    try:
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
        for i, tail in enumerate(tails):
            ids = llm.tokenizer.encode(f'\nObservation {i}: the color changed.', add_special_tokens=False)
            compare(lambda side, tail=tail, ids=ids: session.prefill_block(
                tail[side], torch.tensor(ids), context=[common[side]]))
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
        record_property('forward_wall_seconds', times)
        record_property('peak_allocated_bytes', torch.cuda.max_memory_allocated())
    finally:
        session.sc_gdn = prepared
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
        assert llm.async_engine.session.sc_gdn.decode_buffers.compose_count > 0
        await llm.free_block(prompt.block)
        for tail in tails:
            await llm.free_block(tail)
    loop.run_until_complete(run())
