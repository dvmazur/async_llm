"""Resolve script-style imports for pytest, unittest discovery and direct runs."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "python", REPO_ROOT / "scripts" / "self_evolving_agent"):
    sys.path.insert(0, str(path))
