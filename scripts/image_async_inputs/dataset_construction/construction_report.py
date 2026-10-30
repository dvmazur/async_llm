"""Offline report of screening, build outcomes and reviewer overrides."""
from collections import Counter
from pathlib import Path
from pipeline import read,save,records
from image_edit_test import parse_solve

def report():
    reviews=read(Path('work/v2_visual_reviews.json'))
    screens=[]
    for name in ['mathvista','mathvision','charxiv','mathvision_b']:
        root=Path('work')/('v2_'+name); out=Path(str(root)+'_proposals')
        for record in records(root):
            sid=record['id']; response_path=out/(sid+'.response.json')
            row={'id':sid,'dataset':record['source']['dataset'],'source_id':record['source']['source_id'],
                 'question':record['source']['question'],'status':'not_completed'}
            if response_path.exists():
                try:
                    response=read(response_path)
                    if response['choices'][0]['finish_reason']!='stop': raise ValueError('incomplete')
                    suggestion=parse_solve(response['choices'][0]['message']['content'])[0]
                    row['status']='proposed' if suggestion.get('eligible') is True else 'model_rejected'
                    row['rejection_reason']=suggestion.get('rejection_reason')
                except (ValueError,KeyError,IndexError,TypeError): row['status']='invalid_response'
            screens.append(row)
    builds=[]
    for root in sorted(Path('work/v2_builds').iterdir()):
        if not root.is_dir(): continue
        for record in records(root):
            path=root/'evidence'/record['id']/'checks.json'
            checks=read(path) if path.exists() else {}
            builds.append({'key':root.name,'id':record['id'],'dataset':record['source']['dataset'],
                'record_status':record['status'],'model_checks_pass':checks.get('passes_model_checks'),
                'review':reviews.get(root.name,{'decision':'not_accepted','notes':'Model/normalization checks incomplete or failed; excluded.'}),
                'checks_path':str(path) if path.exists() else None})
    ledger=read(Path('work/budget_diverse_v2.json'))
    from budget import charged
    save(Path('work/v2_construction_report.json'),{
        'screening_counts':dict(Counter(r['status'] for r in screens)), 'screening':screens,
        'build_count':len(builds),'builds':builds,
        'review_counts':dict(Counter(r['review']['decision'] for r in builds)),
        'reported_cost_usd':sum(r.get('cost',0) for r in ledger['requests']),
        'accounted_cost_usd':charged(ledger),
        'limitations':['Same-model proposals and model checks; Codex review is not independent human annotation.',
            'Model passed but reviewer rejected cx2053: unintended horizontal point movement.',
            'Per-sample raw aspect allowances recorded; target processor parity still pending.',
            'Earlier six-pair pilot reused; prior API spend excluded.',
            'Conservative reservations retained for interrupted requests without reported cost.']})

if __name__=='__main__': report()
