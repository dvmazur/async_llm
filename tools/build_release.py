"""Package runner sources. Engine checkout, venv and weights are supplied separately."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def source_files(root):
    files = [root/name for name in ('pyproject.toml', 'README.md', 'DOCUMENTATION.md')]
    for name in ('experiment_runner', 'environment', 'pipelines', 'tools', 'tests'):
        files.extend(sorted((root/name).rglob('*.py')))
    # Experiments may keep JSON/TOML settings beside their Python entry point.
    # Do not recursively sweep output logs into the source release.
    scripts = [p for p in (root/'experiments').rglob('*.py')
               if not {'results', '__pycache__'}.intersection(p.relative_to(root).parts)
               and not p.name.startswith('rtx_')]  # machine-local controls, not portable examples
    files.extend(scripts)
    for directory in {p.parent for p in scripts}:
        files.extend(directory.glob('*.json'))
        files.extend(directory.glob('*.toml'))
    return sorted(set(files))


def build(output):
    root = Path(__file__).resolve().parent.parent
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = source_files(root)
    sums = {}
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = str(path.relative_to(root))
            sums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, 'speleo-runner/'+name)
        archive.writestr('speleo-runner/tools/release.json', json.dumps(dict(
            sha256=sums, includes_engine=False, includes_weights=False, includes_venv=False), indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip():
            raise RuntimeError('ZIP CRC check failed')
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix+'.sha256').write_text(f'{checksum}  {output.name}\n')
    return checksum


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    print(build(args.output))
