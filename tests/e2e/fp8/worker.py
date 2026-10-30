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
    original=ModelRunner.forward
    def observed(self,batch,*args,**kwargs):
        result=original(self,batch,*args,**kwargs)
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
    def control(self,teacher=None,path=None):
        if path is None:
            AUDIT.update(active=True,teacher=teacher,logits=[],modes=[])
        else:
            AUDIT['active']=False
            torch.save(torch.stack(AUDIT['logits']),path)
            save_json(Path(path).with_suffix('.json'),{'modes':AUDIT['modes']})
            AUDIT['logits']=[]
    ModelRunner.forward=observed
    Scheduler.upstream_fp8_parity_control=control



def fixtures(args):
    if args.fixture_set == 'chains':
        from transformers import AutoTokenizer
        from .chains import fixtures as chain_fixtures
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        save_json(args.output/'fixtures.json', dict(model=args.model, cases=chain_fixtures(tokenizer, args.tokens)))
        return
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
    save_json(args.output/'storage.json',dict(
        fp8_tensors=sum(v.dtype==torch.float8_e4m3fn for v in weights.values()),
        bytes=sum(v.numel()*v.element_size() for v in weights.values()),
        source=str(repo),fla_recurrent=gdn._fla_recurrent is not None))
    try:
        for case in cases:
            block=session.create_block()
            group=WorkerGroup(cache_structure=[[block]],write_to=[block])
            logits=session.prefill_block(block,torch.tensor(case['prompt_ids'],dtype=torch.int32),
                                        **image_inputs(case,flatten=True))
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
                engine.collective_rpc('upstream_fp8_parity_control',teacher=teacher)
                mm=image_inputs(c,offset)
                out=engine.generate(input_ids=c['prompt_ids']+c['teacher_tokens'][:offset],
                    sampling_params=dict(temperature=0,max_new_tokens=len(teacher),ignore_eos=True),
                    **({'image_data':[dict(format='processor_output',**mm)]} if mm else {}))
                engine.collective_rpc('upstream_fp8_parity_control',path=str(args.output/f"{c['name']}_{label}.pt"))
                assert out['output_ids']==teacher
            print('DONE',c['name'],flush=True)
    finally:engine.shutdown()


def make_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--engine',choices=['fixtures','mini','sglang','transformers'],required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixtures',type=Path)
    p.add_argument('--fixture-set',choices=['standard','chains'],default='standard')
    p.add_argument('--bf16-control',action='store_true')
    p.add_argument('--require-moe',action='store_true',help='Reject dense checkpoints; chain worker verifies actual FP8 expert calls')
    p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--limit',type=int)
    p.add_argument('--cases',nargs='+',help='Select named fixture cases without changing token/image data')
    p.add_argument('--disable-fla',action='store_true',help='Mini-only test of the unchanged Torch scan fallback')
    p.add_argument('--mini-repo',type=Path)
    p.add_argument('--mini-scheduling',choices=['shared-cache','mixed','sequential','shared-chains',
        'shared-batched','shared-full-graph','shared-full-graph-software'],default='shared-cache',
        help='Mini adapter: shared-cache (existing parity) or ordinary text-only serving with/without mixing')
    p.add_argument('--quantization',choices=['fp8'])
    p.add_argument('--mem-fraction-static',type=float,default=.15,help='SGLang reference memory limit; use e.g. 0.6 for 35B on GB10')
    p.add_argument('--port',type=int,default=25876)
    return p


def main(args=None):
    if args is None:
        args=make_parser().parse_args()
    if args.bf16_control:
        assert args.engine == 'fixtures' and args.fixture_set == 'chains'
    if args.bf16_control or args.require_moe:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model)
        if args.bf16_control:
            assert not getattr(config, 'quantization_config', None), 'BF16 control requires an unquantized checkpoint'
        if args.require_moe:
            assert (getattr(config, 'quantization_config', None) or {}).get('quant_method') == 'fp8', '--require-moe requires an FP8 checkpoint'
            text_config = getattr(config, 'text_config', config)
            experts = getattr(text_config, 'num_experts', 0) or getattr(text_config, 'num_local_experts', 0)
            assert experts > 0, '--require-moe requires a MoE checkpoint, not the dense control'
    if args.disable_fla:assert args.engine=='mini'
    if args.mini_scheduling!='shared-cache':assert args.engine=='mini'
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
        elif args.mini_scheduling in ('shared-batched','shared-full-graph','shared-full-graph-software'):
            from .shared_batch import mini_shared_batch
            mini_shared_batch(args,cases)
        elif args.mini_scheduling == 'shared-chains':
            from .chains import mini_chains
            mini_chains(args, cases)
        else:
            from .serving import mini_serving
            mini_serving(args,cases)
    elif args.engine=='sglang':sglang_run(args,cases)
    else:transformers_run(args,cases)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in hashes.items())
    manifest = dict(arguments=vars(args),torch=torch.__version__,source_hashes=hashes,
        cases=[c['name'] for c in cases],fixtures_sha256=hashlib.sha256(args.fixtures.read_bytes()).hexdigest())
    save_json(args.output/'complete.json',manifest)
    if args.engine == 'mini' and args.mini_scheduling == 'shared-chains':
        from .chains import LAYOUTS
        for layout in LAYOUTS:
            save_json(args.output.parent/layout/'complete.json',dict(manifest,
                arguments=dict(vars(args),mini_scheduling=layout)))


# SGLang reimports the entry module in spawned scheduler processes. Merely
# importing this helper (e.g. from diagnostics) must not install a second observer.
if __name__ in ('__main__', '__mp_main__') and '--engine' in sys.argv:
    if sys.argv[sys.argv.index('--engine')+1]=='sglang':
        install_sglang_observer()

if __name__=='__main__':main()
