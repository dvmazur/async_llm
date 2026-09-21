"""A real learned-model decode catalogue exceeding the old 128MiB arena."""
import os

import pytest
import torch


@pytest.mark.skipif(not os.environ.get('MINISGL_E2E_MODEL') or not torch.cuda.is_available(),
                    reason='Set MINISGL_E2E_MODEL to the small Qwen3.5 checkpoint')
@torch.inference_mode()
def test_learned_64_worker_16_segment_decode_workspace(tmp_path):
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig
    from minisgl.shared_cache import SharedCacheSession, WorkerGroup

    engine = Engine(EngineConfig(model_path=os.environ['MINISGL_E2E_MODEL'],
        tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16, attention_backend='fi',
        max_running_req=64, num_page_override=256, page_size=16, max_seq_len_override=256,
        cuda_graph_bs=[1, 64], cuda_graph_max_bs=64, shared_cuda_graph_max_depth=16,
        distributed_addr=(tmp_path/'distributed').as_uri()))
    try:
        session = SharedCacheSession(engine)
        runner, io = session.graph_runner, session.graph_io
        workspace = session.sc_attn._workspace
        required = max(r[1] for b in io.attention.values() for r in b.workspace_requirements())
        assert required > 128*1024*1024, 'checkpoint must exercise the original overflow'
        assert workspace.numel() >= required
        groups = [[session.create_block() for _ in range(64)] for _ in range(2)]
        ids = torch.arange(10, 74, dtype=torch.int32)
        def step(blocks):
            return session.decode_step(WorkerGroup(cache_structure=[[b] for b in blocks],
                                                  write_to=blocks), ids).clone()
        before = runner.replay_count
        actual = step(groups[0])
        assert runner.replay_count == before + 1
        session.graph_runner, session.graph_io = None, None
        try:
            expected = step(groups[1])
        finally:
            session.graph_runner, session.graph_io = runner, io
        # Same learned workload/width; eager Attention has only the active slots.
        torch.testing.assert_close(actual.float().softmax(-1), expected.float().softmax(-1),
                                   atol=2.5e-3, rtol=0)
        assert session.sc_attn._workspace is workspace
        for blocks in groups:
            assert [b.num_tokens for b in blocks] == [1]*64
            for block in blocks:
                session.free_block(block)
    finally:
        engine.shutdown()
