"""Full learned-model prefill/decode graphs, not just layer captures."""
import asyncio
import os
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext, PrefillJob, WorkerGroup

MODEL = os.environ.get('MINISGL_E2E_MODEL', '')
pytestmark = pytest.mark.skipif(not MODEL or not torch.cuda.is_available(),
                               reason='Set MINISGL_E2E_MODEL to a small local Qwen3.5 checkpoint')


@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    from minisgl.engine import Engine, EngineConfig
    from minisgl.distributed import DistributedInfo
    from transformers import AutoConfig
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    quant = 'fp8' if getattr(AutoConfig.from_pretrained(MODEL), 'quantization_config', None) else None
    engine = Engine(EngineConfig(model_path=MODEL, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        quantization=quant, num_page_override=2048, page_size=16, max_seq_len_override=2048,
        max_running_req=8, attention_backend='fi', cuda_graph_bs=[1, 2, 4],
        cuda_graph_max_bs=0 if os.environ.get('MINISGL_TEST_GRAPHS_OFF') == '1' else 4,
        shared_cuda_graph_max_depth=8, shared_cuda_graph_prefill_rows=[64, 256, 512],
        distributed_addr=(tmp_path_factory.mktemp('prefill_graph')/'nccl').as_uri()))
    with pytest.MonkeyPatch.context() as monkeypatch:
        import minisgl.models.qwen3_5_delta as delta
        if os.environ.get('MINISGL_TEST_NO_FLA') == '1':
            monkeypatch.setattr(delta, '_fla_chunk', None)
            monkeypatch.setattr(delta, '_fla_recurrent', None)
        llm = None
        retained = []
        try:
            llm = AsyncLLM(MODEL, engine=engine)
            yield loop, llm, retained
        finally:
            if llm is not None:
                loop.run_until_complete(llm.close())
            engine.shutdown()
            for result, saved in retained:
                torch.testing.assert_close(result, saved, atol=0, rtol=0)
            loop.close()
            asyncio.set_event_loop(None)


@torch.inference_mode()
def test_full_replay_changes_images_lengths_and_block_topology(runtime, record_property):
    _, llm, retained = runtime
    session = llm.async_engine.session
    runner = session.graph_runner
    graphs = dict(runner.graph_map)
    before_pre, before_dec = runner.prefill_replay_count, runner.replay_count
    assert set(graphs) == {1, 2, 4, ('prefill', 64), ('prefill', 256), ('prefill', 512)}
    blocks = [session.create_block() for _ in range(5)]
    visual = llm.async_engine.session.engine.model.model.visual
    visual_forward = visual.forward
    image_calls = []

    def counted_visual(*args):
        assert not torch.cuda.is_current_stream_capturing()
        image_calls.append(1)
        return visual_forward(*args)

    try:
        with patch.object(visual, 'forward', counted_visual), patch.object(
                session.engine.model, 'forward', side_effect=AssertionError('model.forward ran during replay')):
            for step in range(2):
                data = llm.processor.apply_chat_template([{'role':'user', 'content':[
                    {'type':'image', 'image':Image.new('RGB', (64, 64), (10+step*20, 80, 160))},
                    {'type':'image', 'image':Image.new('RGB', (64, 64), (30, 90+step*20, 150))},
                    {'type':'text', 'text':'Compare the frames briefly.'}]}], add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors='pt')
                output = session.prefill_block(blocks[step], data['input_ids'][0],
                    **{k:data[k].flatten() if k=='mm_token_type_ids' else data[k]
                       for k in ('pixel_values', 'image_grid_thw', 'mm_token_type_ids')})
                retained.append((output, output.clone()))
            for lengths in ([10, 13, 16], [65, 20], [1], [32, 32, 32]):
                group = blocks[2:2+len(lengths)]
                outputs = session.prefill_batch([PrefillJob(block,
                    torch.arange(10, 10+length, dtype=torch.int32), context=blocks[:2])
                    for block, length in zip(group, lengths)])
                for output in outputs:
                    assert torch.isfinite(output).all()
                    retained.append((output, output.clone()))
                result = session.decode_step(WorkerGroup(
                    cache_structure=[[*blocks[:2], block] for block in group], write_to=group),
                    torch.full((len(group),), 42, dtype=torch.int32))
                assert torch.isfinite(result).all()
            merged = session.merge_blocks(blocks[0], blocks[1])
            blocks.append(merged)
            session.append_block(blocks[2], blocks[2])
            result = session.prefill_block(blocks[2], torch.tensor([20, 30, 40]), context=[merged])
            retained.append((result, result.clone()))
        assert len(image_calls) == 2
        assert runner.prefill_replay_count-before_pre == 7
        assert runner.replay_count-before_dec == 4
        assert runner.graph_map == graphs
        record_property('full_prefill_replays', runner.prefill_replay_count-before_pre)
        record_property('full_decode_replays', runner.replay_count-before_dec)
        for result, saved in retained:
            torch.testing.assert_close(result, saved, atol=0, rtol=0)
    finally:
        for block in blocks:
            session.free_block(block)


@torch.inference_mode()
def test_prefill_overflow_precedes_allocations(runtime):
    _, llm, _ = runtime
    session = llm.async_engine.session
    blocks = [session.create_block() for _ in range(5)]
    before = session.page_allocator.num_free_pages
    replays = session.graph_runner.prefill_replay_count
    try:
        with pytest.raises(ValueError, match='capacity'):
            session.prefill_block(blocks[0], torch.ones(513, dtype=torch.int32))
        with pytest.raises(ValueError, match='capacity'):
            session.prefill_batch([PrefillJob(b, torch.tensor([10, 20])) for b in blocks])
        assert session.page_allocator.num_free_pages == before
        assert session.graph_runner.prefill_replay_count == replays
        assert all(not b.page_starts and not b.linear_affine and b.num_tokens==0 for b in blocks)
    finally:
        for block in blocks:
            session.free_block(block)


@torch.inference_mode()
def test_same_prepared_full_forward_matches_eager_before_publication(runtime):
    """A separate observer, NOT used by the external graph-only quality run."""
    _, llm, _ = runtime
    session = llm.async_engine.session
    runner, io, ctx = session.graph_runner, session.graph_io, session.engine.ctx
    replay = runner.replay
    checked = []
    blocks = [session.create_block() for _ in range(4)]

    def checked_replay(batch):
        result = replay(batch)
        if batch.is_prefill:
            saved = result.clone()
            rows = batch.attn_metadata.graph_buffers.rows
            buffers = io.prefill_gdn[rows]
            states = [tensor.clone() for tensor in buffers._current[1:4]]
            slots = batch.out_loc.long()
            kv = [cache(layer).flatten(0, 1).index_select(0, slots).clone()
                  for layer in range(session.kv_cache.num_layers)
                  for cache in (session.kv_cache.k_cache, session.kv_cache.v_cache)]
            # READ tables still point to pre-step states; no publication has
            # happened. The second execution rewrites only the same outputs.
            with patch.object(ctx, '_batch', io.prefill_batches[rows]):
                expected = session.engine.model.forward()
            torch.testing.assert_close(saved, expected[:batch.size].float(), atol=0, rtol=0)
            for actual, reference in zip(buffers._current[1:4], states):
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            after = [cache(layer).flatten(0, 1).index_select(0, slots)
                     for layer in range(session.kv_cache.num_layers)
                     for cache in (session.kv_cache.k_cache, session.kv_cache.v_cache)]
            for actual, reference in zip(after, kv):
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            checked.append(rows)
        return result

    try:
        with patch.object(runner, 'replay', checked_replay):
            session.prefill_block(blocks[0], torch.arange(10, 90, dtype=torch.int32))
            for lengths in ([3, 7, 13], [65, 20], [1]):
                session.prefill_batch([PrefillJob(block, torch.arange(10, 10+length, dtype=torch.int32),
                                                  context=[blocks[0]])
                                       for block, length in zip(blocks[1:], lengths)])
        assert checked == [256, 64, 256, 64]
    finally:
        for block in blocks:
            session.free_block(block)


def test_async_prefill_and_decode_results_are_owned(runtime):
    loop, llm, retained = runtime
    async def run():
        prompt = await llm.prefill_block([10, 20, 30], return_logits=True)
        tail = await llm.create_block()
        context = AsyncContext([prompt.block, tail], tail, 42)
        result = await llm(cache_view=context)
        for logits in (prompt.logits, result.logits):
            assert logits.is_cuda and not logits.is_inference()
            retained.append((logits, logits.clone()))
        for _ in range(3):
            context.next_input_id = 43
            await llm(cache_view=context)
        await llm.free_block(prompt.block)
        await llm.free_block(tail)
        for logits, saved in retained:
            torch.testing.assert_close(logits, saved, atol=0, rtol=0)
    loop.run_until_complete(run())


@torch.inference_mode()
def test_depth_and_page_capacity_fail_before_allocating(runtime):
    _, llm, _ = runtime
    session = llm.async_engine.session
    context, target = session.create_block(), session.create_block()
    try:
        # A long READ-only prefix, built with legal graph-sized appends.
        for _ in range(4):
            session.prefill_block(context, torch.arange(10, 522, dtype=torch.int32))
        before = session.page_allocator.num_free_pages
        replays = session.graph_runner.prefill_replay_count
        for repeats, message in ((8, 'depth capacity'), (5, 'page capacity')):
            with pytest.raises(ValueError, match=message):
                session.prefill_block(target, torch.tensor([10]), context=[context]*repeats)
            assert session.page_allocator.num_free_pages == before
            assert session.graph_runner.prefill_replay_count == replays
            assert target.num_tokens == 0 and not target.page_starts and not target.linear_affine
    finally:
        session.free_block(context)
        session.free_block(target)


@pytest.mark.skipif(os.environ.get('MINISGL_TEST_GRAPHS_OFF') != '1', reason='Explicit graph-OFF run')
@torch.inference_mode()
def test_global_graph_off_overrides_prefill_catalog(runtime):
    _, llm, _ = runtime
    session = llm.async_engine.session
    assert session.graph_runner is None and session.graph_io is None
    block = session.create_block()
    try:
        with patch.object(session.engine.model, 'forward', wraps=session.engine.model.forward) as forward:
            a = session.prefill_block(block, torch.tensor([10, 20, 30]))
            b = session.decode_step(WorkerGroup(cache_structure=[[block]], write_to=[block]),
                                    torch.tensor([42], dtype=torch.int32))
            assert forward.call_count == 2
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
    finally:
        session.free_block(block)


@torch.inference_mode()
def test_eager_and_full_graph_prefills_interoperate_on_existing_blocks(runtime):
    _, llm, _ = runtime
    session = llm.async_engine.session
    runner, io = session.graph_runner, session.graph_io
    prefix, tail = session.create_block(), session.create_block()
    before = runner.prefill_replay_count
    try:
        # Keep the same Engine/checkpoint/pages; explicitly select the old
        # eager entry in this test, then restore the already-captured profiles.
        with patch.object(io, 'prefill_rows', []), patch.object(runner, 'prefill_graph_rows', []):
            session.prefill_block(prefix, torch.tensor([10, 20, 30]))
        first = session.prefill_block(tail, torch.tensor([40, 50]), context=[prefix])
        with patch.object(io, 'prefill_rows', []), patch.object(runner, 'prefill_graph_rows', []):
            session.prefill_block(tail, torch.tensor([60, 70, 80]), context=[prefix])
        second = session.prefill_block(tail, torch.tensor([90]), context=[prefix])
        assert torch.isfinite(first).all() and torch.isfinite(second).all()
        assert prefix.num_tokens == 3 and tail.num_tokens == 6
        assert runner.prefill_replay_count-before == 2
    finally:
        session.free_block(prefix)
        session.free_block(tail)
