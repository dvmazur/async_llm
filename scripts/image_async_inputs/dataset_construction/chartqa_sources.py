"""Pinned original ChartQA images; CPU-only, no model calls."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request

from PIL import Image
import pyarrow.parquet as pq
from pipeline import digest, import_sources, require, save

SPEC = {"repo": "ahmed-masry/ChartQA", "revision": "af8b6f5c08c95085271561c2a3f9d15f2b5a9031",
        "file": "data/test-00000-of-00001.parquet",
        "sha256": "84889bae30aaf6e24ff899e1604221606c0b65e658b6c70a7d02c79f50fc87df",
        "split": "test"}


def acquire(cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / Path(SPEC['file']).name
    base = f"https://huggingface.co/datasets/{SPEC['repo']}"
    if not path.exists():
        temporary = path.with_suffix('.partial')
        with urllib.request.urlopen(f"{base}/resolve/{SPEC['revision']}/{SPEC['file']}", timeout=120) as response:
            with temporary.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        require(digest(temporary) == SPEC['sha256'], 'ChartQA download checksum mismatch')
        temporary.replace(path)
    require(digest(path) == SPEC['sha256'], 'ChartQA cache checksum mismatch')
    save(cache / 'source_lock.json', SPEC)
    return pq.read_table(path)


def select(table, root, limit, offset=0):
    require(not root.exists(), 'Use a fresh workspace')
    rows = table.drop(['image']).to_pylist()
    ranked = sorted(enumerate(rows), key=lambda pair: (
        pair[1]['type'] != 'human',
        hashlib.sha256(f"42:{pair[1]['imgname']}:{pair[1]['query']}".encode()).hexdigest()))
    unique, seen = [], set()
    for i, row in ranked:
        if row['imgname'] not in seen:
            unique.append((i, row)); seen.add(row['imgname'])
    sources = []
    staging = root / 'sources'; staging.mkdir(parents=True)
    (root / 'records').mkdir()
    for i, row in unique[offset:offset+limit]:
        raw = table['image'][i].as_py()
        if isinstance(raw, dict):
            raw = raw['bytes']
        im = Image.open(io.BytesIO(raw)); im.load()
        ext = {'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp'}[im.format]
        sid = hashlib.sha256(f"{row['imgname']}:{row['query']}".encode()).hexdigest()[:16]
        target = staging / (sid + ext); target.write_bytes(raw)
        sources.append({'dataset': SPEC['repo'], 'source_id': sid, 'source_split': SPEC['split'],
            'category': 'chart_' + row['type'], 'question': row['query'], 'answer': str(row['label']),
            'image': target.name, 'source_revision': SPEC['revision'], 'original': row,
            'source_url': f"https://huggingface.co/datasets/{SPEC['repo']}/blob/{SPEC['revision']}/README.md",
            'license': 'Dataset card GPL-3.0; retain original chart rights and attribution; local evaluation only',
            'selection_seed': 42, 'selection_offset': offset})
    manifest = staging / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in sources))
    import_sources(root, manifest)
    save(root/'selection.json', {'source': SPEC, 'count': len(sources), 'offset': offset})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('work/chartqa_cache'))
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()
    table = acquire(args.cache)
    if args.workspace:
        select(table, args.workspace.resolve(), args.limit, args.offset)
    else:
        print(table.schema)
        print(table.drop(['image']).slice(0, 2).to_pylist())
