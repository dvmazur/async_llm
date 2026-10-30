"""Validate all fixed-50 conditions and write the report before full evaluation."""
import argparse
import json
from pathlib import Path

KS = [-1, 0, 16, 32, 64, 128, 256, 512]


def report(root, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    ids = manifest['sample_ids']
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError('Expected exactly 50 fixed unique samples')
    results = []
    reference_config = None
    reference_seeds = None
    for k in KS:
        folder = root / f'k_{k}'
        summary = json.loads((folder/'summary.json').read_text())
        config = json.loads((folder/'config.json').read_text())
        if not summary.get('complete') or summary['completed'] != 50 or summary['requested'] != 50:
            raise ValueError(f'Incomplete condition k={k}; will not launch full sweep')
        if config['k_steps'] != k or config['sample_manifest'] != manifest:
            raise ValueError(f'Configuration or subset mismatch at k={k}')
        if config['dataset_sha256'] != manifest['dataset_sha256']:
            raise ValueError('Dataset changed')
        common = {key:value for key,value in config.items()
                  if key not in ('output','k_steps','cuda_visible_devices','distributed_port')}
        if reference_config is None:
            reference_config = common
        elif common != reference_config:
            raise ValueError('Conditions used different configuration/code')
        rows = [json.loads((folder/(sid+'.json')).read_text()) for sid in ids]
        if [r['id'] for r in rows] != ids:
            raise ValueError('Result identity mismatch')
        seeds = [r['sample_seed'] for r in rows]
        if reference_seeds is None:
            reference_seeds = seeds
        elif reference_seeds != seeds:
            raise ValueError('Sampling seeds differ across k')
        correct = sum(r['correct_after'] for r in rows)
        if summary['correct_after'] != correct:
            raise ValueError('Summary disagrees with sample results')
        groups = {}
        for source in sorted({r['dataset'] for r in rows}):
            selected = [r for r in rows if r['dataset']==source]
            groups[source] = {'count':len(selected),'correct_after':sum(r['correct_after'] for r in selected)}
        results.append({'k':k,'count':50,'correct_after':correct,'accuracy_after':correct/50,
                        'matches_before':sum(r['matches_before'] for r in rows),
                        'image_replacements':sum(r['image_replaced'] for r in rows),
                        'missing_boxed_answers':sum(r['predicted_answer'] is None for r in rows),
                        'elapsed_seconds':sum(r['elapsed_seconds'] for r in rows),'by_source':groups})
    output = {'complete':True,'dataset_sha256':manifest['dataset_sha256'],
              'sample_count':50,'conditions':results,'config':reference_config}
    (root/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    lines = ['# Fixed-50 image async-thoughts results\n\n',
             'Same 50 samples and sampling seeds for every condition. Automated scoring; no semantic API judge.\n\n',
             '| k | Correct after | Accuracy | Matches old answer | Image replacements | Missing boxed answer |\n',
             '|---:|---:|---:|---:|---:|---:|\n']
    for r in results:
        lines.append(f"| {r['k']} | {r['correct_after']}/50 | {r['accuracy_after']:.1%} | {r['matches_before']} | {r['image_replacements']} | {r['missing_boxed_answers']} |\n")
    lines.append('\nAt k=-1 the old image is intentionally retained; at k=0 the corrected image is present from the start. Neither baseline replaces an image mid-generation. Positive-k cases can finish before injection. Interpret replacement counts and source breakdowns alongside accuracy.\n\n')
    lines.append('## Per-source corrected-answer accuracy\n\n| Source | n | '+ ' | '.join(str(k) for k in KS)+' |\n')
    lines.append('|---|---:|'+ '|'.join(['---:']*len(KS))+'|\n')
    for source, group in results[0]['by_source'].items():
        lines.append(f"| {source} | {group['count']} | "+' | '.join(
            f"{r['by_source'][source]['correct_after']}/{group['count']}" for r in results)+' |\n')
    (root/'RESULTS.md').write_text(''.join(lines))
    print(''.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args(); report(a.root,a.manifest)
