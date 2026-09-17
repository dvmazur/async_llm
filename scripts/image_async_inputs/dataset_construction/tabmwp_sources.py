"""Acquire original TabMWP table PNGs from a pinned source commit, CPU only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

from pipeline import digest, import_sources, require, save

REVISION = '5a1a52214521590b075f545b76a4f5ce666345e3'
BASE = f'https://raw.githubusercontent.com/lupantech/PromptPG/{REVISION}/'
QUESTION_SHA = 'c47607dad36477725087cb4e5317fe1e89a69dfed404eca430673bda36fda45c'


def fetch(relative, path):
    if not path.exists():
        with urllib.request.urlopen(BASE + relative, timeout=60) as response:
            raw = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.partial')
        temporary.write_bytes(raw)
        temporary.replace(path)
    return path


def prepare(cache, root, limit, offset):
    require(not root.exists(), 'Use a fresh workspace')
    questions = fetch('data/tabmwp/problems_test.json', cache / 'problems_test.json')
    require(digest(questions) == QUESTION_SHA, 'Question checksum mismatch')
    rows = json.loads(questions.read_text())
    eligible = [(sid, r) for sid, r in rows.items() if 3 <= r['row_num'] <= 15
                and r['column_num'] <= 6 and r['grade'] >= 5]
    eligible.sort(key=lambda item: hashlib.sha256(f'42:{item[0]}'.encode()).hexdigest())
    selected = eligible[offset:offset+limit]
    (root/'records').mkdir(parents=True)
    staging = root/'sources'; staging.mkdir()
    def one(item):
        sid, row = item
        image = fetch(f'data/tabmwp/tables/{sid}.png', cache/'tables'/f'{sid}.png')
        question = row['question']
        if row.get('choices'):
            question += '\nChoices:\n' + '\n'.join(f'({chr(65+i)}) {v}' for i,v in enumerate(row['choices']))
        if row.get('unit'):
            question += '\nAnswer unit: ' + row['unit'] + '.'
        return {'dataset':'lupantech/TabMWP', 'source_id':sid, 'source_split':'test',
                'category':f"table_math_grade_{row['grade']}", 'question':question,
                'answer':str(row['answer']), 'image':str(image.resolve()),
                'source_url':f'https://github.com/lupantech/PromptPG/tree/{REVISION}',
                'license':'TabMWP dataset CC-BY-NC-SA-4.0; noncommercial evaluation only; retain original content rights',
                'source_revision':REVISION, 'original':row,
                'selection_seed':42, 'selection_offset':offset}
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = list(pool.map(one, selected))
    manifest = staging/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in sources))
    import_sources(root, manifest)
    save(root/'selection.json', {'revision':REVISION, 'question_sha256':QUESTION_SHA,
        'count':len(sources), 'offset':offset,
        'image_sha256':{r['source_id']:digest(Path(r['image'])) for r in sources}})


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('work/tabmwp_cache'))
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--limit',type=int,default=40)
    p.add_argument('--offset',type=int,default=0)
    args=p.parse_args()
    prepare(args.cache,args.workspace.resolve(),args.limit,args.offset)
