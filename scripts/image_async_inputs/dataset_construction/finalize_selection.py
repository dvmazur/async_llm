"""Freeze accepted inputs with explicit final exclusions; never make paid calls."""
from pathlib import Path
from pipeline import read, records, save, require


def select():
    exclusions = read(Path('work/v3_final_exclusions.json'))
    roots = [Path('work/v2_assembled'), *sorted(Path('work/v3_builds').iterdir())]
    samples = []
    excluded = set()
    hashes = set()
    for root in roots:
        if not root.is_dir():
            continue
        for r in records(root):
            if r['status'] != 'accepted':
                continue
            if r['id'] in exclusions:
                excluded.add(r['id'])
                continue
            sha = r['normalization']['images']['after']['sha256']
            require(sha not in hashes, 'Unresolved duplicate source image')
            hashes.add(sha)
            samples.append({'workspace': str(root.resolve()), 'id': r['id']})
    require(excluded == set(exclusions), 'Exclusion does not identify an accepted sample')
    require(500 <= len(samples) <= 600, 'Requested release range not met')
    config = {'samples': samples, 'final_exclusions': exclusions,
              'supporting_evidence': ['work/v3_visual_reviews.json',
                  'work/v3_final_exclusions.json', 'work/v3_duplicate_audit/candidates.json',
                  'RUN_V3.md']}
    path = Path('work/v3_assembly.json')
    if path.exists():
        require(read(path) == config, 'Selection changed; use a new release version')
    else:
        save(path, config)
    print({'selected': len(samples), 'excluded': len(excluded)})


if __name__ == '__main__':
    select()
