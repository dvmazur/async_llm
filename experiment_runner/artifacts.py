"""Pack logs only, never model/venv/compiler caches or arbitrary neighbouring files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def pack_results(directory, archive, *, allow_partial=False):
    directory, archive = Path(directory).resolve(), Path(archive).resolve()
    status = json.loads((directory/'status.json').read_text())
    if status['status'] != 'completed' and not allow_partial:
        raise ValueError('partial results require allow_partial=True')
    if archive.exists():
        raise FileExistsError(archive)
    allowed = {'.json', '.jsonl', '.log', '.png', '.gif', '.csv'}
    files = sorted(p for p in directory.rglob('*') if p.is_file())
    if any(p.is_symlink() or p.suffix not in allowed for p in files):
        raise ValueError('results contain symlinks or unexpected file types; inspect before packing')
    if status['status'] == 'completed':
        spec = json.loads((directory/'run.json').read_text())
        expected = len(spec['gpus']) * spec['concurrency'] * spec['repeats']
        if len(list(directory.glob('gpu-*/slot-*/repeat-*/completion.json'))) != expected:
            raise ValueError('missing episode completion records')
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as out:
        for path in files:
            name = str(path.relative_to(directory))
            manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            out.write(path, directory.name+'/'+name)
        out.writestr(directory.name+'/checksums.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as check:
        if check.testzip() is not None:
            raise RuntimeError('archive CRC failed')
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(archive.suffix+'.sha256').write_text(f'{checksum}  {archive.name}\n')
    return checksum


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory')
    p.add_argument('archive')
    p.add_argument('--allow-partial', action='store_true')
    a = p.parse_args()
    print(pack_results(a.directory, a.archive, allow_partial=a.allow_partial))


if __name__ == '__main__':
    main()
