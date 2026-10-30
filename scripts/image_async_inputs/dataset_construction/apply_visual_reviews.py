"""Apply explicit reviewer decisions only, with image-checksum-bound evidence."""
import argparse
from pathlib import Path
from pipeline import read, records, require, save, transition, record_path


def apply(builds, reviews_path):
    reviews=read(reviews_path)
    done=[]
    for root in sorted(p for p in builds.iterdir() if p.is_dir()):
        for r in records(root):
            sid=r['id']
            if sid not in reviews or r['status'] in ('accepted','rejected'):
                continue
            review=reviews[sid]
            if review.get('difficulty_group'):
                require(review['difficulty_group'] in ('reasoning','calibration','reasoning_reveal'), 'Invalid group')
                r['source']['difficulty_group']=review['difficulty_group']
                save(record_path(root,sid),r)
            if review['decision']=='accept':
                checks=read(root/'evidence'/sid/'checks.json')
                require(checks['passes_model_checks'],f'Model checks fail: {sid}')
                for state in ('before','after'):
                    require(checks[state]['image_sha256']==r['normalization']['images'][state]['sha256'],
                            'Checked images differ from reviewed assets')
            payload={**review,'reviewer':'Codex visual/semantic review, not independent human annotation',
                'answer_before':r['proposal']['expected_answer_before'],'answer_after':r['source']['answer'],
                **{k:True for k in ('isolated_edit','readable','both_solvable','visual_required',
                                   'answers_distinct','source_terms_checked','same_dimensions')
                   if review['decision']=='accept'}}
            transition(root,sid,'review',payload)
            done.append({'id':sid,'decision':review['decision']})
    print(done)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--builds',type=Path,default=Path('work/v3_builds'))
    p.add_argument('--reviews',type=Path,default=Path('work/v3_visual_reviews.json'))
    a=p.parse_args(); apply(a.builds,a.reviews)
