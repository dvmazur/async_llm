"""Exit 0 for verified complete output, 10 to run, or 1 for incompatible output."""
import hashlib
import json
from pathlib import Path
import sys

from image_async_thoughts_eval import arguments
import image_async_thoughts_eval as evaluator


def main():
    a=arguments()
    path=a.output/'summary.json'
    if not path.exists() or not json.loads(path.read_text()).get('complete'):
        return 10
    actual=json.loads((a.output/'config.json').read_text())
    expected={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    expected['sample_manifest']=json.loads(a.sample_manifest.read_text()) if a.sample_manifest else None
    # Hardware assignment and rendezvous port may differ; experiment settings may not.
    for key,value in expected.items():
        if key!='distributed_port' and actual.get(key)!=value:
            raise ValueError(f'Completed condition configuration differs: {key}')
    if actual['dataset_sha256']!=hashlib.sha256(a.dataset.read_bytes()).hexdigest():
        raise ValueError('Completed condition dataset changed')
    if actual['evaluator_sha256']!=hashlib.sha256(Path(evaluator.__file__).read_bytes()).hexdigest():
        raise ValueError('Completed condition evaluator changed')
    summary=json.loads(path.read_text())
    if summary['completed']!=summary['requested']:
        raise ValueError('Completed condition count mismatch')
    print(f'Skipping verified complete k={a.k_steps}: {a.output}',flush=True)
    return 0


if __name__=='__main__': sys.exit(main())
