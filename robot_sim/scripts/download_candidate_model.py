"""Download the pinned Qwen 27B checkpoint into the ignored project runtime."""

import argparse
import json
import os
from pathlib import Path

MODEL = "Qwen/Qwen3.8-27B-FP8"
REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    target = root / "robot_sim/.runtime/models/Qwen3.8-27B-FP8"
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_HOME"] = str(root / "robot_sim/.runtime/hf")
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL,
        revision=args.revision,
        local_dir=target,
        max_workers=args.workers,
        ignore_patterns=["mtp.safetensors", "*.md", "crc32.txt", "safetensors-md5sum.txt"],
    )
    (target / "download_revision.json").write_text(
        json.dumps({"repo": MODEL, "revision": args.revision}, indent=2)
    )
    print(target)


if __name__ == "__main__":
    main()
