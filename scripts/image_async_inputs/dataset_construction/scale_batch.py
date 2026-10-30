"""Offline plan creation for model-proposed candidates; never accepts samples.

At scale, proposal screening feeds tentative edits; final normalized-image visual
review is still mandatory. Already attempted IDs and exact shared-source images
are excluded, including the existing accepted export. Plans are immutable.
"""
import argparse
from pathlib import Path

from pipeline import read, records, require, save


def make_plan(workspace, proposals, builds, output, limit):
    require(not output.exists(), 'Use a fresh plan filename')
    used_ids, used_images = set(), set()
    for path in output.parent.glob('v3_*build*.json'):
        for item in read(path).get('items',[]):
            used_ids.add(item['key'])
    for path in builds.glob('*/records/*.json'):
        r = read(path); used_ids.add(r['id']); used_images.add(r['after_sha256'])
    import json
    existing = Path('datasets/diverse_corrections_v2/annotations.jsonl')
    for line in existing.read_text().splitlines():
        r = json.loads(line); used_ids.add(r['id']); used_images.add(r['after_sha256'])
    items = []
    for r in records(workspace):
        if r['id'] in used_ids or r['after_sha256'] in used_images:
            continue
        path = proposals / (r['id'] + '.screen.json')
        if not path.exists() or read(path).get('status') != 'pending_review':
            continue
        proposal_path = proposals / (r['id'] + '.proposal.json')
        p = read(proposal_path)
        required = ('edit_type','changed_fact_before','changed_fact_after','expected_answer_before',
                    'solution_before','solution_after','editing_method','reasoning_change')
        if not all(isinstance(p.get(k),str) and p[k].strip() for k in required):
            continue
        if p['expected_answer_before'].strip() == r['source']['answer'].strip():
            continue
        used_images.add(r['after_sha256'])
        items.append({'key':r['id'],'source_id':r['source']['source_id'],
            'workspace':str(workspace.resolve()), 'proposal':str(proposal_path.resolve()),
            'difficulty_group':'reasoning',
            'notes':'Tentative model-screened edit, not pre-approved ground truth. Requires normalized blind solves, pair audit and explicit final visual review.'})
        if len(items) >= limit:
            break
    save(output, {'version':3,'budget_usd':150,'items':items})
    print(f'Planned {len(items)} tentative builds; no acceptance decisions or API calls.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--proposals',type=Path,required=True)
    p.add_argument('--builds',type=Path,default=Path('work/v3_builds'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=40)
    args=p.parse_args()
    require(args.limit > 0, 'Positive limit required')
    make_plan(args.workspace,args.proposals,args.builds,args.output,args.limit)
