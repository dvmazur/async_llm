#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
UV="${UV:-uv}"
"$UV" sync --frozen
"$UV" pip install --python .venv/bin/python -e scripts jupyterlab ipykernel
.venv/bin/python -m ipykernel install --user --name async-llm --display-name 'AsyncLLM (uv)'
