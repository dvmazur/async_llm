"""Artifact utilities shared by the pytest suite and isolated model workers."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def source_hashes(root=ROOT):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / "python").rglob('*')
            if path.is_file() and '__pycache__' not in path.parts}


def validate_current_mini(directory, mode, root=ROOT, *, quantization="fp8"):
    manifest = json.loads((directory / "complete.json").read_text())
    storage = json.loads((directory / "storage.json").read_text())
    assert quantization in (None, "fp8")
    if quantization == "fp8":
        assert storage["fp8_tensors"] > 0, "expected serialized FP8 weights"
    else:
        assert storage["fp8_tensors"] == 0, "BF16 control accidentally used FP8"
        assert storage["bf16_tensors"] > 0, "expected BF16 weights"
    assert manifest["arguments"].get("quantization") == quantization
    assert manifest["arguments"].get("mini_scheduling", "shared-cache") == mode
    assert not manifest["arguments"].get("reference_quant_outputs", False)
    recorded = {str(Path(p).relative_to(storage["source"])): sha
                for p, sha in manifest["source_hashes"].items()}
    current = {str(Path(p).relative_to(root)): sha for p, sha in source_hashes(root).items()}
    assert current == recorded, "Saved logits must come from CURRENT production code"
    return manifest
