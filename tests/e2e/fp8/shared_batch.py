"""Learned shared-cache batched-prefill parity through the real Session API.

Teacher forcing only; no model, projection, kernel or rounding overrides.
Prefill executes prepared GDN eagerly; the entire decode model is captured.
"""
from pathlib import Path
import sys

import torch

from .common import ROOT, save_json


def check_coverage(coverage):
    forwards = coverage['forwards']
    prefill = [f for f in forwards if f['phase'] == 'prefill']
    decode = [f for f in forwards if f['phase'] == 'decode']
    assert prefill and decode, 'both model phases must actually execute'
    assert any(f['workers'] > 1 for f in prefill), 'missing batched prefill'
    assert coverage['prefill_layer_calls'] == len(prefill) * coverage['linear_layers'] > 0
    assert coverage['decode_replays'] == len(decode), 'decode fell back to eager'
    assert coverage['capture_profiles'] == [1, 2, 4], 'unbounded/unexpected capture profiles'


@torch.inference_mode()
def mini_shared_batch(args, cases):
    source = args.mini_repo or ROOT
    sys.path.insert(0, str(source/'python'))
    from minisgl.engine import Engine, EngineConfig
    from minisgl.distributed import DistributedInfo
    from minisgl.shared_cache import SharedCacheSession, WorkerGroup, PrefillJob
    from transformers import GenerationConfig
    from .worker import image_inputs

    assert args.quantization == 'fp8'
    assert Path(sys.modules[Engine.__module__].__file__).resolve().is_relative_to(source)
    engine = Engine(EngineConfig(model_path=args.model, tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16, quantization=args.quantization, max_running_req=8,
        num_page_override=8192, max_seq_len_override=2048, attention_backend='fi',
        use_pynccl=False, cuda_graph_bs=[1, 2, 4], cuda_graph_max_bs=4,
        shared_cuda_graph_max_depth=8, distributed_addr=f'tcp://127.0.0.1:{args.port}',
        generation_config=GenerationConfig(do_sample=False, temperature=None, top_k=None, top_p=None)))
    import minisgl.models.qwen3_5_delta as delta
    if args.disable_fla:
        delta._fla_chunk = delta._fla_recurrent = None
    session = SharedCacheSession(engine)
    from minisgl.shared_cache.gdn_prefill import GDNPrefillBuffers
    original_core, original_forward = GDNPrefillBuffers.core, session._forward
    layer_calls, forwards = [], []

    def observed_core(self, *a, **kw):
        layer_calls.append(1)
        return original_core(self, *a, **kw)

    def observed_forward(batch, *a, **kw):
        result = original_forward(batch, *a, **kw)
        forwards.append(dict(phase=batch.phase, workers=batch.size, rows=batch.input_ids.numel()))
        return result

    GDNPrefillBuffers.core, session._forward = observed_core, observed_forward
    save_json(args.output/'storage.json', dict(source=str(source),
        fp8_tensors=sum(w.dtype == torch.float8_e4m3fn for w in engine.model.state_dict().values()),
        fla_recurrent=delta._fla_recurrent is not None))

    def jobs(group, blocks, step=0):
        return [PrefillJob(block, torch.tensor(c['prompt_ids']+c['teacher_tokens'][:step],
                                              dtype=torch.int32),
                          **image_inputs(c, step, flatten=True))
                for c, block in zip(group, blocks)]

    live = []
    try:
        for offset in range(0, len(cases), 4):
            group = cases[offset:offset+4]
            live = [session.create_block() for _ in group]
            logits = torch.cat(session.prefill_batch(jobs(group, live)))
            cache = WorkerGroup(cache_structure=[[b] for b in live], write_to=live)
            histories = [[] for _ in group]
            for step in range(args.tokens):
                for history, row in zip(histories, logits):
                    history.append(row.cpu())
                if step+1 < args.tokens:
                    logits = session.decode_step(cache, torch.tensor(
                        [c['teacher_tokens'][step] for c in group], device='cuda', dtype=torch.int32))
            for case, history in zip(group, histories):
                torch.save(torch.stack(history), args.output/f"{case['name']}_decode.pt")
            for block in live:
                session.free_block(block)
            live = []
            for step in sorted({s for c in group for s in c['cold_steps']}):
                selected = [c for c in group if step in c['cold_steps']]
                live = [session.create_block() for _ in selected]
                values = session.prefill_batch(jobs(selected, live, step))
                for case, value in zip(selected, values):
                    torch.save(value[0].cpu(), args.output/f"{case['name']}_cold{step}.pt")
                for block in live:
                    session.free_block(block)
                live = []
            print('DONE batch', [c['name'] for c in group], flush=True)
        coverage = dict(forwards=forwards, prefill_layer_calls=len(layer_calls),
                        linear_layers=session._model_config.num_linear_layers,
                        decode_replays=session.graph_runner.replay_count,
                        capture_profiles=sorted(session.graph_runner.graph_map))
        check_coverage(coverage)
        save_json(args.output/'batch_coverage.json', coverage)
    finally:
        GDNPrefillBuffers.core = original_core
        for block in live:
            session.free_block(block)
        engine.shutdown()
