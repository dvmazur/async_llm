"""Restore mutable/engine.py and mutable/prompt.py to their pristine seeds,
so the next run_persistent.py starts a clean evolution run."""
import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = HERE / "seeds"
ENGINE_SEED = SEEDS / "engine_seed.py"

PROMPT_SEEDS = {
    "detailed": SEEDS / "prompt_seed.py",
    "minimal": SEEDS / "prompt_seed_minimal.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt", choices=sorted(PROMPT_SEEDS), default="detailed",
        help="which prompt seed variant to install as mutable/prompt.py (default: detailed)")
    parser.add_argument(
        "--mutable-dir", default="mutable",
        help="directory (relative to this script) to restore engine.py/prompt.py into -- "
             "point separate concurrent runs at separate dirs, e.g. via SEA_MUTABLE_DIR "
             "(default: mutable)")
    args = parser.parse_args()

    mutable = HERE / args.mutable_dir
    mutable.mkdir(exist_ok=True)

    engine_path = mutable / "engine.py"
    shutil.copyfile(ENGINE_SEED, engine_path)
    print(f"restored {engine_path} from {ENGINE_SEED}")

    prompt_seed = PROMPT_SEEDS[args.prompt]
    prompt_path = mutable / "prompt.py"
    shutil.copyfile(prompt_seed, prompt_path)
    print(f"restored {prompt_path} from {prompt_seed} (variant={args.prompt})")


if __name__ == "__main__":
    main()
