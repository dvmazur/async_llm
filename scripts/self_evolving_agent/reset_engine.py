"""Restore mutable/engine.py and mutable/prompt.py to their pristine seeds,
so the next run_persistent.py starts a clean evolution run."""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
MUTABLE = HERE / "mutable"

SEEDS = HERE / "seeds"
ENGINE_SEED = SEEDS / "engine_seed.py"
ENGINE_PATH = MUTABLE / "engine.py"

PROMPT_SEED = SEEDS / "prompt_seed.py"
PROMPT_PATH = MUTABLE / "prompt.py"


def main() -> None:
    MUTABLE.mkdir(exist_ok=True)

    shutil.copyfile(ENGINE_SEED, ENGINE_PATH)
    print(f"restored {ENGINE_PATH} from {ENGINE_SEED}")

    shutil.copyfile(PROMPT_SEED, PROMPT_PATH)
    print(f"restored {PROMPT_PATH} from {PROMPT_SEED}")


if __name__ == "__main__":
    main()
