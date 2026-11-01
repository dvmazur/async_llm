#!/usr/bin/env bash
# Linux x86_64 / Python 3.12. Uses the repository lockfile without changing it.
set -euo pipefail
cd "$(dirname "$0")/../.."
UV="${UV:-$HOME/.local/bin/uv}"
if [[ ! -x .venv/bin/python ]]; then
    "$UV" venv --python 3.12 .venv
fi
requirements="$(mktemp)"
trap 'rm -f "$requirements"' EXIT
"$UV" export --frozen --no-hashes --no-emit-project --format requirements-txt --output-file "$requirements" >/dev/null
.venv/bin/python - "$requirements" <<'PY'
import sys
import tomllib
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
for package in tomllib.loads(Path("uv.lock").read_text())["package"]:
    if package["name"] not in ("torch", "torchvision", "sglang-kernel"):
        continue
    urls = [w["url"] for w in package["wheels"] if "x86_64" in w["url"]
            and ("cp312" in w["url"] or "cp310-abi3" in w["url"])]
    if len(urls) != 1:
        raise RuntimeError(f"Cannot select an unambiguous wheel for {package['name']}")
    # Same official PyTorch wheel; imladris cannot reach the R2 mirror.
    url = urls[0].replace("download-r2.pytorch.org", "download.pytorch.org")
    text = text.replace(f"{package['name']}=={package['version']}", f"{package['name']} @ {url}")
path.write_text(text)
PY
"$UV" pip install --python .venv/bin/python --no-deps -r "$requirements"
"$UV" pip install --python .venv/bin/python --no-deps -e .
"$UV" pip install --python .venv/bin/python transformers==5.14.1 jupyterlab ipykernel nbclient ninja cloudpickle==3.1.2
.venv/bin/python -m ipykernel install --user --name async-llm --display-name 'AsyncLLM (uv)'
