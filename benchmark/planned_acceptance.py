"""Three-way checkpoint parity: natural old, same-M old, planned full graph.

Independent states evolve on each path. Same-M control isolates implementation
error from BF16 GEMM padding/routing drift; natural-path probability TV remains
an independent gate. No thresholds are enlarged for a failed case.
"""
import json
from pathlib import Path
import torch
import pytest


@torch.inference_mode()
def teacher(new,natural,processor,output,image_path="/workspace/results/speleo_live.png",trace_case=None):
    from reference_legacy_session import legacy_session
    from test_decoder import capacity_linear_reference
    from minisgl.planned.runner import ProgramRunner
    from minisgl.shared_cache.session import PrefillJob
    from minisgl.shared_cache.worker_group import WorkerGroup
    from PIL import Image
    import minisgl.core as core
    from minisgl.layers.base import BaseOP
    from minisgl.layers.linear import _LinearTPImpl
    def linears(op):
        if isinstance(op,_LinearTPImpl):yield op
        elif isinstance(op,BaseOP):
            for value in vars(op).values():
                if isinstance(value,BaseOP):yield from linears(value)
                elif isinstance(value,list):
                    for item in value:
                        if isinstance(item,BaseOP):yield from linears(item)
    vision_linears=list(linears(getattr(new.model.model,"visual",None)))
    matched=legacy_session(new.model,page_size=new.page_size,pages=new.store.page_capacity,max_seq_len=8192)
    blocks={"new":[new.create_block() for _ in range(4)],"natural":[natural.create_block() for _ in range(4)],
            "matched":[matched.create_block() for _ in range(4)]}
    report=dict(status="running",cases=[],protocol="strict parity vs independent same-M old; natural-old drift calibrated by old-only padding control",
                limits=dict(matched_logits_relative_l2=.025,matched_probability_tv=.006,
                            extra_tv_beyond_old_padding_control=.006),
                note="Natural vs same-M legacy itself may exceed .006 TV due to BF16 shape/routing. Report it; do not charge that baseline drift to changed kernels.")
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    def save():path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    def ids(text):return processor.tokenizer(text,return_tensors="pt",add_special_tokens=False)["input_ids"].flatten()
    plans=[]
    with pytest.MonkeyPatch.context() as patch:
        original=ProgramRunner.prepare_execution
        def preparation(r,f,a,inp,features=None):
            plans.append(f);return original(r,f,a,inp,features)
        patch.setattr(ProgramRunner,"prepare_execution",preparation)
        def compare(label,operation):
            from planned_trace_body import instrument_old,instrument_new
            nt,ot={},{}
            with patch.context() as context:
                if label==trace_case:instrument_new(new,context,nt)
                result=operation(new,blocks["new"])
            with patch.context() as ctx:
                ctx.setattr(core,"_GLOBAL_CTX",natural.engine.ctx)
                old=operation(natural,blocks["natural"])
            with patch.context() as ctx,capacity_linear_reference(plans[-1],exclude_layers=vision_linears):
                ctx.setattr(core,"_GLOBAL_CTX",matched.engine.ctx)
                if label==trace_case:instrument_old(matched.engine.model,ctx,ot)
                shape_control=operation(matched,blocks["matched"])
            if label==trace_case:
                stages=[]
                for key in ot:
                    a,b=nt[key].float(),ot[key].float()
                    stages.append(dict(layer=key[0],stage=key[1],relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-10)),
                                       max_abs=float((a-b).abs().max())))
                report["traced_stages"]=stages
                globals()["TRACE_OLD"],globals()["TRACE_NEW"]=ot,nt
            def metrics(a,b):
                a,b=a.float(),b.float()
                assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
                return dict(relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-10)),
                    max_tv=float(((a.softmax(-1)-b.softmax(-1)).abs().sum(-1)/2).max()),
                    top1_agreement=float((a.argmax(-1)==b.argmax(-1)).float().mean()))
            row=dict(label=label,new_vs_natural=metrics(result,old),new_vs_matched=metrics(result,shape_control),
                     old_padding_control=metrics(shape_control,old),planned=new.last_forward)
            report["cases"].append(row);save();print("PARITY",json.dumps(row),flush=True)
            assert row["new_vs_matched"]["relative_l2"]<.025,row
            assert row["new_vs_matched"]["max_tv"]<.006,row
            assert row["new_vs_natural"]["max_tv"]<=row["old_padding_control"]["max_tv"]+.006,row
            for a,b,c in zip(blocks["new"],blocks["natural"],blocks["matched"]):
                assert a.token_ids==b.token_ids==c.token_ids and a.mrope_span==b.mrope_span==c.mrope_span
        def prefill(s,jobs):
            return torch.cat(s.prefill_batch(jobs) if s is new else s._prefill_batch_fused(jobs))
        def group(bs):
            c,a,b,image=bs
            return WorkerGroup(cache_structure=[[c,image,b,a],[c,image,a,b]],write_to=[a,b])
        common=ids("<|im_start|>system\nBe concise. Describe only what you observe, then choose a useful next action.<|im_end|>\n<|im_start|>user\nFind a route down through the cave.\n")
        try:
            compare("common",lambda s,b:prefill(s,[PrefillJob(b[0],common)]))
            compare("branches",lambda s,b:prefill(s,[
                PrefillJob(b[1],ids("<|im_start|>assistant\n<think>\nInspect the cave and identify the next safe step."),[b[0]]),
                PrefillJob(b[2],ids("<|im_start|>assistant\nThe visible scene shows"),[b[0]])]))
            forced=ids(" a stone wall with an opening below. Move forward and inspect the lower passage.")
            for step in range(8):
                tokens=torch.tensor([int(forced[step%len(forced)]),int(forced[(step+3)%len(forced)])])
                compare(f"decode{step}",lambda s,b:s.decode_step(group(b),tokens))
            image=Image.open(image_path).convert("RGB")
            batch=processor.apply_chat_template([{"role":"user","content":[{"type":"image","image":image},
                {"type":"image","image":image}]}],add_generation_prompt=False,tokenize=True,return_dict=True,return_tensors="pt")
            def image_call(s,b):
                jobs=[PrefillJob(b[3],batch["input_ids"].flatten(),[b[0]],pixel_values=batch["pixel_values"],
                      image_grid_thw=batch["image_grid_thw"],mm_token_type_ids=batch["mm_token_type_ids"].flatten())]
                pf,dec=s.mixed_step(jobs,group(b),torch.tensor([int(forced[0]),int(forced[1])]))
                return torch.cat([*pf,dec])
            compare("two_real_images_mixed",image_call)
            for step in range(4):
                tokens=torch.tensor([int(forced[step]),int(forced[step+2])])
                compare(f"post_image_decode{step}",lambda s,b:s.decode_step(group(b),tokens))
            report["status"]="passed";report["memory"]=new.memory_report();save()
        except BaseException as exc:
            report["status"]="failed";report["failure"]=dict(type=type(exc).__name__,message=str(exc));save();raise
        finally:
            for s,key in ((new,"new"),(natural,"natural"),(matched,"matched")):
                for block in blocks[key]:s.free_block(block)
    return report
