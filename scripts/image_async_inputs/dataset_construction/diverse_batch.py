"""Bounded concurrent screening/building; each build has an isolated workspace.

Requires the locked DATASET_BUDGET_LEDGER. Errors are recorded, never retried
automatically. This script does not accept samples; visual review is separate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from build_dataset import build, model_job, parsed
from model_assist import PROMPT
from pipeline import records, read, save, require


def screen(root, out, workers):
    require(os.environ.get('DATASET_BUDGET_LEDGER'), 'Budget ledger required')
    out.mkdir(parents=True,exist_ok=True)
    def job(record):
        sid=record['id']; source=record['source']
        prompt=PROMPT+'\nSource data:\n'+json.dumps({'question':source['question'],'source_answer':source['answer']})
        row={'id':sid,'dataset':source['dataset'],'source_id':source['source_id'],'category':source['category']}
        try:
            response=model_job(out/f'{sid}.response.json',prompt,[root/source['image']])
            suggestion=parsed(response)
            row['suggestion']=suggestion
            row['status']='pending_review' if suggestion.get('eligible') is True else 'model_rejected'
            if row['status']=='pending_review':
                save(out/f'{sid}.proposal.json',suggestion['proposal'])
        except Exception as exc:
            row.update(status='error',error=str(exc))
        save(out/f'{sid}.screen.json',row)
        print(json.dumps({k:v for k,v in row.items() if k!='suggestion'}),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(job,records(root)))
    save(out/'summary.json',{'items':results})


def build_many(plan_path,out,workers):
    plan=read(plan_path)
    out.mkdir(parents=True,exist_ok=True)
    def job(item):
        name=item['key']
        root=out/name; root.mkdir(exist_ok=True)
        single={'version':2,'budget_usd':plan.get('budget_usd',10),'semantic_answer_grading':True,
                'items':[{k:v for k,v in item.items() if k!='key'}]}
        pp=root/'job_plan.json'
        if pp.exists():
            require(read(pp)==single,'Immutable build plan changed')
        else:
            save(pp,single)
        try:
            build(pp,root)
            row={'key':name,'status':'checked'}
        except Exception as exc:
            row={'key':name,'status':'blocked','error':str(exc)}
        save(root/'batch_status.json',row)
        print(json.dumps(row),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(job,plan['items']))
    save(out/(plan_path.stem+'.results.json'),{'items':results})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['screen','build'])
    p.add_argument('--workspace',type=Path)
    p.add_argument('--plan',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    require(1<=a.workers<=4,'Use 1-4 workers')
    if a.stage=='screen': screen(a.workspace.resolve(),a.output.resolve(),a.workers)
    else: build_many(a.plan.resolve(),a.output.resolve(),a.workers)
