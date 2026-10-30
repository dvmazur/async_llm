"""Pinned, CPU-only MathVision/CharXiv source acquisition (evaluation use only)."""
import argparse
import hashlib
import io
import json
import urllib.request
from pathlib import Path

from PIL import Image
import pyarrow.parquet as pq
from pipeline import digest, import_sources, require, save

SPECS = {
    'mathvision': {'repo':'MathLLMs/MathVision','revision':'2837ddb3f13abaf6b3997c12d80753e5470bd46a',
        'file':'data/test-00000-of-00001-3532b8d3f1b4047a.parquet',
        'sha256':'bcb3078a77bd6ee0e4e22135d53b9a48604fcd1b62dec69e279e812ba4dd7d37','split':'test',
        'license':'Dataset card MIT; retain original competition/image rights and attribution; local evaluation use'},
    'charxiv': {'repo':'princeton-nlp/CharXiv','revision':'f441eb632fc62f6f777830a0f47619e6e86459b0',
        'file':'val.parquet','sha256':'ed0613ac7d5c045ac30a63d78529217bf267218948170a2b26ec37eaeea1605d',
        'split':'validation','license':'Questions CC-BY-SA-4.0; charts copyright original arXiv authors; evaluation only, no training'}
}

def acquire(name, cache):
    spec = SPECS[name]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / Path(spec['file']).name
    if not path.exists():
        url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['file']}"
        tmp = path.with_suffix('.partial')
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open('wb') as out:
            while chunk := response.read(1024*1024):
                out.write(chunk)
        require(digest(tmp)==spec['sha256'], 'Source checksum mismatch')
        tmp.replace(path)
    require(digest(path)==spec['sha256'], 'Cached checksum mismatch')
    save(cache/'source_lock.json',spec)
    return pq.read_table(path)

def select(name, table, root, limit, offset, seed):
    require(not root.exists(), 'Use a fresh source workspace')
    spec=SPECS[name]
    image_key='decoded_image' if name=='mathvision' else 'image'
    rows=table.drop([image_key]).to_pylist()
    eligible=[]
    for i,row in enumerate(rows):
        sid=str(row['id']) if name=='mathvision' else Path(row['figure_path']).stem
        if name=='charxiv' and (sid=='0' or not row['reasoning_a']):
            continue  # official erratum: image 0 has an incorrect reasoning label
        if name=='mathvision' and not row['answer']:
            continue
        # Prefer manageable diagrams / limited chart panels without changing questions.
        rank=(row['level']>3) if name=='mathvision' else (row['num_subplots']>4)
        eligible.append((rank,hashlib.sha256(f'{seed}:{sid}'.encode()).hexdigest(),i,sid,row))
    eligible.sort()
    (root/'records').mkdir(parents=True)
    staging=root/'sources'; staging.mkdir()
    sources=[]
    for _,__,i,sid,row in eligible[offset:offset+limit]:
        raw=table[image_key][i].as_py()['bytes']
        im=Image.open(io.BytesIO(raw)); im.load()
        ext={'PNG':'.png','JPEG':'.jpg','WEBP':'.webp'}[im.format]
        target=staging/(sid+ext); target.write_bytes(raw)
        if name=='mathvision':
            choices=row['options'] or []
            question=row['question'].replace('<image1>','').strip()
            answer=row['answer'].strip()
            if len(answer)==1 and 'A'<=answer<='Z' and ord(answer)-65<len(choices):
                answer=choices[ord(answer)-65]
            if choices:
                question+='\nChoices:\n'+'\n'.join(f'({chr(65+j)}) {v}' for j,v in enumerate(choices))
            original={**row,'choices':choices}
            category=row['subject']
        else:
            question=row['reasoning_q']; answer=row['reasoning_a']
            original=row; category='scientific_chart_'+row['category']
        sources.append({'dataset':spec['repo'],'source_id':sid,'source_split':spec['split'],
            'category':category,'question':question,'answer':str(answer),'image':target.name,
            'source_url':f"https://huggingface.co/datasets/{spec['repo']}/blob/{spec['revision']}/README.md",
            'license':spec['license'],'source_revision':spec['revision'],'original':original,
            'selection_seed':seed,'selection_offset':offset})
    manifest=staging/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in sources))
    import_sources(root,manifest)
    save(root/'selection.json',{'source':spec,'seed':seed,'offset':offset,'count':len(sources)})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=SPECS)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--workspace',type=Path)
    p.add_argument('--limit',type=int,default=60)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    table=acquire(args.dataset,args.cache)
    if args.workspace:
        select(args.dataset,table,args.workspace.resolve(),args.limit,args.offset,args.seed)
    else:
        print(table.schema)
        row=table.slice(0,1).to_pylist()[0]
        print({k: ('<image bytes>' if isinstance(v,dict) and 'bytes' in v else v) for k,v in row.items()})
