"""Freeze source-balanced, category-interleaved sample IDs without reading labels."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path


def select_ids(rows, count=50, seed=42):
    groups = defaultdict(lambda: defaultdict(list))
    def rank(value):
        return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()
    for row in rows:
        groups[row['dataset']][row['category']].append(row['id'])
    sources = {}
    for dataset, categories in sorted(groups.items()):
        queues = [sorted(categories[c],key=rank) for c in sorted(categories,key=rank)]
        ordered = []
        while any(queues):
            for queue in queues:
                if queue:
                    ordered.append(queue.pop(0))
        sources[dataset] = ordered
    result = []
    while len(result) < count and any(sources.values()):
        for queue in sources.values():
            if queue and len(result) < count:
                result.append(queue.pop(0))
    if len(result) != count or len(set(result)) != count:
        raise ValueError('Not enough unique samples')
    return result


def apply_manifest(rows, manifest, dataset_sha256):
    if manifest['dataset_sha256'] != dataset_sha256:
        raise ValueError('Subset dataset checksum mismatch')
    ids = manifest['sample_ids']
    if len(ids) != manifest['count'] or len(ids) != len(set(ids)):
        raise ValueError('Invalid subset count or duplicate IDs')
    indexed = {r['id']:r for r in rows}
    if not set(ids) <= indexed.keys():
        raise ValueError('Unknown subset sample ID')
    return [indexed[sid] for sid in ids]


def main():
    import pyarrow.parquet as pq
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=50)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    rows=pq.read_table(a.dataset,columns=['id','dataset','category']).to_pylist()
    ids=select_ids(rows,a.count,a.seed)
    by_id={r['id']:r for r in rows}
    result={'version':1,'dataset_sha256':hashlib.sha256(a.dataset.read_bytes()).hexdigest(),
            'seed':a.seed,'count':a.count,'strategy':'source_round_robin_category_interleaved_sha256',
            'source_counts':dict(Counter(by_id[s]['dataset'] for s in ids)), 'sample_ids':ids}
    if a.output.exists():
        if json.loads(a.output.read_text()) != result:
            raise ValueError('Refusing to change existing fixed subset')
    else:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'count':a.count,'source_counts':result['source_counts'],'path':str(a.output)}))


if __name__=='__main__':
    main()
