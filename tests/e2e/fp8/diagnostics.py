"""Explicit research-only substitutions and layer dumps; never an acceptance worker.

Run with python -m tests.e2e.fp8.diagnostics and the usual worker arguments plus
--audit-linear / --audit-moe / --reference-quant-outputs / --capture-prefill /
--sglang-native-mrope. Reuses the former research functions without changing
kernel mathematics. Production validation rejects substituted outputs/references.
"""
import sys
from pathlib import Path

import torch

from . import worker
from .common import ROOT, save_json
from .worker import image_inputs

AUDIT={'active':False}


def install_sglang_observer():
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.managers.scheduler import Scheduler
    if '--sglang-native-mrope' in sys.argv:
        # Explicit second reference, not a silent patch to the default baseline.
        # Installed SGLang's fused CUDA caller feeds [3,T] positions into a
        # [T]-only kernel. Reuse its existing native mRoPE preparation instead.
        from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer
        fused_prepare = Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused
        def prepare(self, positions, hidden_states):
            if positions.ndim == 2:
                return self.forward_prepare_native(positions, hidden_states)
            return fused_prepare(self, positions, hidden_states)
        Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused = prepare
    original=ModelRunner.forward
    def observed(self,batch,*args,**kwargs):
        capture = AUDIT['active'] and AUDIT.get('trace_path') and not AUDIT['logits']
        if capture:
            if not hasattr(self,'_boundary_trace'):
                from .tracing import BoundaryTrace
                self._boundary_trace=BoundaryTrace()
                self._boundary_trace.attach_torch(self.model)
            self._boundary_trace.begin()
        result=original(self,batch,*args,**kwargs)
        if capture:
            self._boundary_trace.finish(Path(AUDIT['trace_path']))
        if AUDIT['active']:
            logits=result.logits_output.next_token_logits
            assert batch.batch_size==1
            i=len(AUDIT['logits'])
            if i<len(AUDIT['teacher']):
                AUDIT['logits'].append(logits[0].detach().float().cpu())
                AUDIT['modes'].append('decode' if batch.forward_mode.is_decode() else 'prefill')
                logits.fill_(-float('inf'))
                logits[0,AUDIT['teacher'][i]]=0
        return result
    def control(self,teacher=None,path=None,trace_path=None):
        if path is None:
            AUDIT.update(active=True,teacher=teacher,logits=[],modes=[],trace_path=trace_path)
        else:
            AUDIT['active']=False
            torch.save(torch.stack(AUDIT['logits']),path)
            save_json(Path(path).with_suffix('.json'),{'modes':AUDIT['modes']})
            AUDIT['logits']=[]
    ModelRunner.forward=observed
    Scheduler.upstream_fp8_parity_control=control


@torch.inference_mode()
def mini(args,cases):
    repo=args.mini_repo or ROOT
    sys.path.insert(0,str(repo/'python'))
    from minisgl.engine import Engine,EngineConfig
    from minisgl.distributed import DistributedInfo
    from minisgl.shared_cache import SharedCacheSession,WorkerGroup
    from transformers import GenerationConfig
    assert Path(sys.modules[Engine.__module__].__file__).resolve().is_relative_to(repo)
    extra={'quantization':args.quantization} if args.quantization else {}
    engine=Engine(EngineConfig(model_path=args.model,tp_info=DistributedInfo(0,1),
        dtype=torch.bfloat16,max_running_req=8,num_page_override=8192,
        max_seq_len_override=2048,attention_backend='fi',use_pynccl=False,
        distributed_addr=f'tcp://127.0.0.1:{args.port}',
        generation_config=GenerationConfig(do_sample=False,temperature=None,top_k=None,top_p=None),**extra))
    import minisgl.models.qwen3_5_delta as gdn
    assert Path(gdn.__file__).resolve().is_relative_to(repo)
    if args.disable_fla:
        # Test optional dependency absence without altering the installed env.
        gdn._fla_chunk = gdn._fla_recurrent = None
    session=SharedCacheSession(engine)
    assert not hasattr(session,'mixed_step')
    assert not hasattr(session.sc_gdn,'compose_state_cache')
    weights=engine.model.state_dict()
    if args.capture_prefill:
        from .tracing import BoundaryTrace
        boundary_trace=BoundaryTrace()
        boundary_trace.attach_mini(engine.model)
    save_json(args.output/'storage.json',dict(
        fp8_tensors=sum(v.dtype==torch.float8_e4m3fn for v in weights.values()),
        bytes=sum(v.numel()*v.element_size() for v in weights.values()),
        source=str(repo),fla_recurrent=gdn._fla_recurrent is not None))
    linear_audit=[]
    moe_audit=[]
    if args.audit_moe:
        assert args.quantization=='fp8'
        import minisgl.moe.fused as mini_moe
        from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler
        from sglang.srt.distributed import init_distributed_environment,initialize_model_parallel,destroy_model_parallel
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl as reference_moe
        set_global_server_args_for_scheduler(ServerArgs(model_path=args.model))
        init_distributed_environment(world_size=1,rank=0,local_rank=0,
                                     distributed_init_method=f'tcp://127.0.0.1:{args.port+1}')
        initialize_model_parallel(tensor_model_parallel_size=1)
        original_moe=mini_moe.fused_experts_impl
        names={v.data_ptr():k for k,v in weights.items() if v.dtype==torch.float8_e4m3fn}
        def observed_moe(x,w1,w2,scores,ids,activation='silu',apply_router_weight_on_input=False,
                         w1_scale=None,w2_scale=None):
            reference=reference_moe(x.clone(),w1,w2,scores,ids,activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,use_fp8_w8a8=True,
                w1_scale=w1_scale,w2_scale=w2_scale,block_shape=[128,128])
            actual=original_moe(x,w1,w2,scores,ids,activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,w1_scale=w1_scale,w2_scale=w2_scale)
            a,r=actual.float(),reference.float()
            moe_audit.append(dict(name=names[w1.data_ptr()],input_shape=list(x.shape),
                relative_l2=((a-r).norm()/r.norm().clamp_min(1e-20)).item(),
                max_abs=(a-r).abs().max().item(),exact_fraction=(a==r).float().mean().item()))
            return reference if args.reference_quant_outputs else actual
        mini_moe.fused_experts_impl=observed_moe
    if args.audit_linear:
        assert args.quantization=='fp8'
        import minisgl.kernel.fp8 as fp8
        from sglang.srt.layers.quantization.fp8_utils import cutlass_w8a8_block_fp8_linear_with_fallback
        original_linear=fp8.block_fp8_linear
        names={v.data_ptr():k for k,v in weights.items() if v.dtype==torch.float8_e4m3fn}
        def observed_linear(x,w,s,bias=None):
            actual=original_linear(x,w,s,bias)
            reference=cutlass_w8a8_block_fp8_linear_with_fallback(x.contiguous(),w,[128,128],s,bias=bias)
            a,r=actual.float(),reference.float()
            linear_audit.append(dict(name=names[w.data_ptr()],input_shape=list(x.shape),
                relative_l2=((a-r).norm()/r.norm().clamp_min(1e-20)).item(),
                max_abs=(a-r).abs().max().item(),exact_fraction=(a==r).float().mean().item()))
            return reference if args.reference_quant_outputs else actual
        fp8.block_fp8_linear=observed_linear
    try:
        for case in cases:
            block=session.create_block()
            group=WorkerGroup(cache_structure=[[block]],write_to=[block])
            if args.capture_prefill:boundary_trace.begin()
            logits=session.prefill_block(block,torch.tensor(case['prompt_ids'],dtype=torch.int32),
                                        **image_inputs(case,flatten=True))
            if args.capture_prefill:boundary_trace.finish(args.output/'layers'/f"{case['name']}.pt")
            values=[]
            for i,token in enumerate(case['teacher_tokens']):
                # Keep the model's original dtype on disk, losslessly. The
                # comparison upcasts all engines' logits to FP32 for metrics.
                values.append(logits[0].cpu())
                if i+1<len(case['teacher_tokens']):
                    logits=session.decode_step(group,torch.tensor([token],dtype=torch.int32,device='cuda'))
            torch.save(torch.stack(values),args.output/f"{case['name']}_decode.pt")
            session.free_block(block)
            for step in case['cold_steps']:
                block=session.create_block()
                ids=case['prompt_ids']+case['teacher_tokens'][:step]
                logits=session.prefill_block(block,torch.tensor(ids,dtype=torch.int32),
                    **image_inputs(case,step,flatten=True))
                torch.save(logits[0].cpu(),args.output/f"{case['name']}_cold{step}.pt")
                session.free_block(block)
            print('DONE',case['name'],flush=True)
    finally:
        if args.audit_linear:
            fp8.block_fp8_linear=original_linear
            save_json(args.output/'linear_audit.json',linear_audit)
        if args.audit_moe:
            mini_moe.fused_experts_impl=original_moe
            save_json(args.output/'moe_audit.json',moe_audit)
            destroy_model_parallel()
        engine.shutdown()


def sglang_run(args,cases):
    import sglang
    engine=sglang.Engine(model_path=args.model,dtype='bfloat16',context_length=2048,
        max_running_requests=8,max_total_tokens=8192,max_mamba_cache_size=64,
        mem_fraction_static=args.mem_fraction_static,disable_cuda_graph=True,disable_prefill_cuda_graph=True,
        chunked_prefill_size=2048,random_seed=0,log_level='warning')
    try:
        for c in cases:
            jobs=[('decode',0,c['teacher_tokens'])]+[(f'cold{s}',s,[c['teacher_tokens'][s]]) for s in c['cold_steps']]
            for label,offset,teacher in jobs:
                engine.flush_cache()
                engine.collective_rpc('upstream_fp8_parity_control',teacher=teacher,
                    trace_path=str(args.output/'layers'/f"{c['name']}.pt") if args.capture_prefill and label=='decode' else None)
                mm=image_inputs(c,offset)
                out=engine.generate(input_ids=c['prompt_ids']+c['teacher_tokens'][:offset],
                    sampling_params=dict(temperature=0,max_new_tokens=len(teacher),ignore_eos=True),
                    **({'image_data':[dict(format='processor_output',**mm)]} if mm else {}))
                engine.collective_rpc('upstream_fp8_parity_control',path=str(args.output/f"{c['name']}_{label}.pt"))
                assert out['output_ids']==teacher
            print('DONE',c['name'],flush=True)
    finally:engine.shutdown()


def main():
    parser=worker.make_parser()
    parser.add_argument('--capture-prefill', action='store_true')
    parser.add_argument('--sglang-native-mrope', action='store_true')
    parser.add_argument('--audit-linear', action='store_true')
    parser.add_argument('--audit-moe', action='store_true')
    parser.add_argument('--reference-quant-outputs', action='store_true')
    args=parser.parse_args()
    assert args.mini_scheduling == 'shared-cache', 'diagnostics use the explicit shared-cache path'
    if args.sglang_native_mrope:
        assert args.engine == 'sglang'
    if args.audit_linear or args.audit_moe:
        assert args.engine == 'mini' and args.quantization == 'fp8'
    if args.reference_quant_outputs:
        assert args.audit_linear and args.audit_moe
    worker.mini=mini
    worker.sglang_run=sglang_run
    worker.main(args)


if __name__ in ('__main__', '__mp_main__') and '--engine' in sys.argv:
    if sys.argv[sys.argv.index('--engine')+1]=='sglang':
        install_sglang_observer()

if __name__ == '__main__':
    main()
