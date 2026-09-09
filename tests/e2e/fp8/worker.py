"""Isolated model worker; launched by the FP8 pytest fixtures, not a benchmark.

No production math patches. SGLang observer records logits before teacher-forced
sampling only. Mini defaults to the upstream shared-cache/session path, without
mixed. ``--mini-scheduling mixed`` instead tests ordinary serving on text fixtures;
``sequential`` is its same-engine, same-history reference (see serving.py).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

import torch

from .common import ROOT, save_json, source_hashes

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


# SGLang reimports the main script in its spawned scheduler process.
if '--engine' in sys.argv and sys.argv[sys.argv.index('--engine')+1]=='sglang':
    install_sglang_observer()


def fixtures(args):
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(args.model)
    tokenizer=processor.tokenizer
    cases=[]
    questions=[
        'Explain why repeating an unsuccessful action may not help.',
        'Calculate 17 + 28 and explain the addition.',
        'Describe two differences between a hypothesis and an observation.',
        'Ответь кратко: как проверить, изменилось ли положение камеры?',
        'Summarize: a box blocks a corridor, and a side passage is visible.',
        'Write a short Python function that returns the maximum of two numbers.',
        'What information can and cannot be inferred from two successive images?',
        'Explain how to approach a staircase safely without assuming hidden geometry.',
    ]
    answer=('The observation provides evidence about the current state. A proposed explanation '
            'should be checked against a new observation. Repeating an action without progress '
            'does not establish that it will work next time. The visible changes should guide '
            'the next step while uncertainty about hidden information remains explicit. ')*3
    teacher=tokenizer.encode(answer,add_special_tokens=False)[:args.tokens]
    assert len(teacher)==args.tokens
    for i,question in enumerate(questions):
        question+=(' Previous observation: a wall and a side opening are visible.'*(i*3))
        prompt=tokenizer.apply_chat_template([dict(role='user',content=question)],
            add_generation_prompt=True,enable_thinking=False,tokenize=False)
        ids=tokenizer.encode(prompt,add_special_tokens=False)
        assert isinstance(ids,list) and all(isinstance(token,int) for token in ids)
        cases.append(dict(name=f'text_{i}',prompt_ids=ids,teacher_tokens=teacher,cold_steps=[0,7,args.tokens-1]))
        rng=np.random.default_rng(21800+i)
        pixels=rng.integers(0,256,(64+32*(i%3),96+32*(i%2),3),dtype=np.uint8)
        images=[Image.fromarray(pixels),Image.fromarray(np.roll(pixels,5+i,axis=1))]
        inputs=processor.apply_chat_template([dict(role='user',content=[
            *[dict(type='image',image=image) for image in images],
            dict(type='text',text=questions[i])])],add_generation_prompt=True,
            enable_thinking=False,tokenize=True,return_dict=True,return_tensors='pt')
        data={k:inputs[k].cpu() for k in ('pixel_values','image_grid_thw','mm_token_type_ids')}
        path=args.output/f'pixels_{i}.pt'
        torch.save(data,path)
        cases.append(dict(name=f'image_{i}',prompt_ids=inputs['input_ids'][0].tolist(),
            teacher_tokens=teacher,cold_steps=[0,7,args.tokens-1],image_tensors=str(path),
            image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save_json(args.output/'fixtures.json',dict(model=args.model,cases=cases))


def image_inputs(case,extra=0,flatten=False,device='cpu'):
    if 'image_tensors' not in case:return {}
    path=Path(case['image_tensors'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==case['image_sha256']
    data=torch.load(path,map_location='cpu',weights_only=True)
    types=data['mm_token_type_ids']
    data['mm_token_type_ids']=torch.cat([types,types.new_zeros((1,extra))],dim=1)
    if flatten:data['mm_token_type_ids']=data['mm_token_type_ids'].flatten()
    return {k:v.to(device) for k,v in data.items()}


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


@torch.inference_mode()
def transformers_run(args,cases):
    from transformers import AutoModelForImageTextToText,AutoConfig
    config=AutoConfig.from_pretrained(args.model)
    quant=getattr(config,'quantization_config',None)
    if quant:
        # Community checkpoint has a tied BF16 lm_head, no output-head scale.
        # HF names that module "lm_head", not the config's "model.lm_head".
        quant['modules_to_not_convert']=list(quant.get('modules_to_not_convert',[]))+['lm_head']
        save_json(args.output/'effective_reference_quant_config.json',quant)
    model,info=AutoModelForImageTextToText.from_pretrained(args.model,config=config,
        dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
    save_json(args.output/'loading_info.json',info)
    assert not info.get('missing_keys') and not info.get('mismatched_keys'),info
    model.eval()
    for c in cases:
        output=model(input_ids=torch.tensor([c['prompt_ids']],device='cuda'),use_cache=True,
                     logits_to_keep=1,**image_inputs(c,device='cuda'))
        values=[]
        for i,token in enumerate(c['teacher_tokens']):
            values.append(output.logits[0,-1].float().cpu())
            if i+1<len(c['teacher_tokens']):
                output=model(input_ids=torch.tensor([[token]],device='cuda'),
                    past_key_values=output.past_key_values,use_cache=True,logits_to_keep=1)
        torch.save(torch.stack(values),args.output/f"{c['name']}_decode.pt")
        del output
        for step in c['cold_steps']:
            ids=c['prompt_ids']+c['teacher_tokens'][:step]
            output=model(input_ids=torch.tensor([ids],device='cuda'),use_cache=False,
                         logits_to_keep=1,**image_inputs(c,step,device='cuda'))
            torch.save(output.logits[0,-1].float().cpu(),args.output/f"{c['name']}_cold{step}.pt")
        print('DONE',c['name'],flush=True)


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
    p=argparse.ArgumentParser()
    p.add_argument('--engine',choices=['fixtures','mini','sglang','transformers'],required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixtures',type=Path)
    p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--limit',type=int)
    p.add_argument('--cases',nargs='+',help='Select named fixture cases without changing token/image data')
    p.add_argument('--capture-prefill',action='store_true',help='Test-only module boundary snapshots')
    p.add_argument('--disable-fla',action='store_true',help='Mini-only test of the unchanged Torch scan fallback')
    p.add_argument('--sglang-native-mrope',action='store_true',
        help='Explicit diagnostic reference: existing SGLang native mRoPE for 3D positions, not default SGLang')
    p.add_argument('--mini-repo',type=Path)
    p.add_argument('--mini-scheduling',choices=['shared-cache','mixed','sequential'],default='shared-cache',
        help='Mini adapter: shared-cache (existing parity) or ordinary text-only serving with/without mixing')
    p.add_argument('--quantization',choices=['fp8'])
    p.add_argument('--audit-linear',action='store_true',help='Test-only SGLang GEMM comparison on identical model inputs')
    p.add_argument('--audit-moe',action='store_true',help='Test-only SGLang MoE comparison on identical model inputs and routes')
    p.add_argument('--reference-quant-outputs',action='store_true',
        help='Explicit diagnostic ablation: use SGLang quantized outputs; never a production validation')
    p.add_argument('--mem-fraction-static',type=float,default=.15,help='SGLang reference memory limit; use e.g. 0.6 for 35B on GB10')
    p.add_argument('--port',type=int,default=25876)
    args=p.parse_args()
    if args.disable_fla:assert args.engine=='mini'
    if args.mini_scheduling!='shared-cache':assert args.engine=='mini'
    if args.sglang_native_mrope:assert args.engine=='sglang'
    if args.reference_quant_outputs:
        assert args.engine=='mini' and args.audit_linear and args.audit_moe
    args.output.mkdir(parents=True,exist_ok=False)
    if args.engine=='fixtures':fixtures(args);return
    cases=json.loads(args.fixtures.read_text())['cases']
    if args.cases:
        cases=[c for c in cases if c['name'] in args.cases]
        assert set(args.cases)=={c['name'] for c in cases}
    if args.limit:cases=cases[:args.limit]
    for case in cases:
        assert isinstance(case['prompt_ids'],list) and all(isinstance(t,int) for t in case['prompt_ids'])
        if 'image_tensors' in case and not Path(case['image_tensors']).is_file():
            # Fixture directories can be copied to another host unchanged;
            # image_inputs still checks the original serialized tensor hash.
            case['image_tensors']=str(args.fixtures.parent/Path(case['image_tensors']).name)
        case['teacher_tokens']=case['teacher_tokens'][:args.tokens]
        case['cold_steps']=sorted(set(min(s,len(case['teacher_tokens'])-1) for s in case['cold_steps']))
    hashes={}
    if args.engine=='mini':
        hashes=source_hashes(args.mini_repo or ROOT)
        if args.mini_scheduling=='shared-cache':mini(args,cases)
        else:
            from .serving import mini_serving
            mini_serving(args,cases)
    elif args.engine=='sglang':sglang_run(args,cases)
    else:transformers_run(args,cases)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in hashes.items())
    save_json(args.output/'complete.json',dict(arguments=vars(args),torch=torch.__version__,source_hashes=hashes,
        cases=[c['name'] for c in cases],fixtures_sha256=hashlib.sha256(args.fixtures.read_bytes()).hexdigest()))


if __name__=='__main__':main()
