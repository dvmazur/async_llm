"""Build a self-contained source ZIP and an exact engine git bundle. No venv/weights."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile


def source_files(root):
    files = [root/name for name in ('pyproject.toml', 'README.md')]
    if (root/'QUICKSTART.md').is_file():
        files.append(root/'QUICKSTART.md')
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


def build(engine, output):
    root = Path(__file__).resolve().parent.parent
    engine, output = Path(engine).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if subprocess.check_output(['git', '-C', str(engine), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('bundle must describe a clean engine checkout')
    revision = subprocess.check_output(['git', '-C', str(engine), 'rev-parse', 'HEAD'], text=True).strip()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = source_files(root)
    with tempfile.TemporaryDirectory(prefix='speleo-release-') as temp:
        bundle = Path(temp)/'engine.bundle'
        subprocess.run(['git', '-C', str(engine), 'bundle', 'create', str(bundle), 'HEAD'], check=True)
        subprocess.run(['git', '-C', str(engine), 'bundle', 'verify', str(bundle)], check=True)
        sums = {}
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                name = str(path.relative_to(root))
                sums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                archive.write(path, 'speleo-runner/'+name)
            sums['engine.bundle'] = hashlib.sha256(bundle.read_bytes()).hexdigest()
            archive.write(bundle, 'speleo-runner/engine.bundle')
            archive.writestr('speleo-runner/tools/release.json', json.dumps(dict(engine_revision=revision,
                sha256=sums, includes_weights=False, includes_venv=False), indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip():
            raise RuntimeError('ZIP CRC check failed')
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix+'.sha256').write_text(f'{checksum}  {output.name}\n')
    return checksum


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--engine', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    print(build(args.engine, args.output))
