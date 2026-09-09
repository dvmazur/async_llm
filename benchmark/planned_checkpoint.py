"""Real-checkpoint acceptance: one load, paired teacher forcing, actual vision.

This is a correctness/integration harness, not a benchmark of rental-GPU speed.
Reports compact scalar metrics rather than retaining multi-GiB tensor dumps.
"""
import argparse
from pathlib import Path
import json
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"python"),str(ROOT/"tests"/"planned")]


def save(path,value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--image",type=Path,default=Path("/workspace/results/speleo_live.png"))
    p.add_argument("--live",action="store_true")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import torch
    from transformers import AutoProcessor
    from PIL import Image
    from minisgl.planned.loading import load_model
    from minisgl.planned.session import PlannedSession
    from minisgl.planned.catalogue import RuntimeProfile
    from minisgl.planned.forward_plan import PlanCapacity
    from minisgl.planned.attention_plan import AttentionCapacity
    from reference_legacy_session import legacy_session
    from minisgl.shared_cache.session import PrefillJob
    from minisgl.shared_cache.worker_group import WorkerGroup
    import minisgl.core as core
    torch.manual_seed(409)
    started=time.perf_counter()
    report=dict(model=str(args.model.resolve()),torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),stages=[],status="loading")
    save(args.output/"checkpoint.json",report)
    print("Loading real checkpoint once",flush=True)
    with torch.inference_mode():
        net=load_model(args.model)
        torch.cuda.synchronize()
        report["load_seconds"]=time.perf_counter()-started
        att=AttentionCapacity(16,4096,16,4096)
        profiles=[RuntimeProfile(PlanCapacity(r,p,d,16,64),att)
                  for r,p,d in ((0,0,4),(0,0,16),(8,128,16),(8,512,16),(8,2048,16),(8,4096,16))]
        new=PlannedSession(net,profiles)
        old=legacy_session(net,page_size=16,pages=4096,max_seq_len=8192,workers=16)
        # With `python -i`, preserve weights after a failed gate for focused
        # diagnostics without another 70-GB load. Normal CLI exits as usual.
        globals().update(active_model=net,active_session=new,active_reference=old)
        core.set_global_ctx(old.engine.ctx)
        processor=AutoProcessor.from_pretrained(args.model,local_files_only=True)
        globals().update(active_processor=processor)
        report["status"]="teacher_forcing";save(args.output/"checkpoint.json",report)
        print("MODEL_READY",json.dumps(dict(load_seconds=report["load_seconds"],memory=new.memory_report())),flush=True)
        from planned_acceptance import teacher
        acceptance=teacher(new,old,processor,args.output/"teacher.json",image_path=str(args.image))
        report["teacher"]=dict(path=str(args.output/"teacher.json"),status=acceptance["status"],
            cases=len(acceptance["cases"]),
            max_matched_tv=max(row["new_vs_matched"]["max_tv"] for row in acceptance["cases"]),
            max_matched_logits_l2=max(row["new_vs_matched"]["relative_l2"] for row in acceptance["cases"]),
            max_old_padding_tv=max(row["old_padding_control"]["max_tv"] for row in acceptance["cases"]))
        report["memory"]=new.memory_report();report["status"]="teacher_forcing_passed"
        save(args.output/"checkpoint.json",report)
        if args.live:
            from planned_live import run_live
            report["live"]=run_live(new,args.model,args.output/"live")
            report["status"]="passed"
            save(args.output/"checkpoint.json",report)
    print("DONE",report["status"],flush=True)


if __name__=="__main__":
    try:main()
    except BaseException as exc:
        # Preserve the first failing stage/metrics even if the process exits.
        if "--output" in sys.argv:
            target=Path(sys.argv[sys.argv.index("--output")+1])/"checkpoint.json"
            if target.exists():
                report=json.loads(target.read_text());report["status"]="failed"
                report["failure"]=dict(type=type(exc).__name__,message=str(exc))
                save(target,report)
        raise
