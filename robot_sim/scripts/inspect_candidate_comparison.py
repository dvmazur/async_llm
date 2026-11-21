"""Read-only compact progress view for a long paired experiment."""

import argparse
import json
from collections import deque
from pathlib import Path


def last_record(file):
    if not file.exists():
        return {}
    with file.open() as stream:
        for line in reversed(deque(stream, maxlen=3)):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / "experiment.json").read_text())
    completed, active = [], []
    for trial in manifest["trials"]:
        folder = args.root / trial["episode"]
        summary = folder / "summary.json"
        if summary.exists():
            result = json.loads(summary.read_text())
            completed.append(
                {
                    "episode": trial["episode"],
                    "passed": result["passed"],
                    "reason": result["reason"],
                    "path_m": round(result["truth_path_m"], 2),
                }
            )
        elif folder.exists():
            frame = last_record(folder / "frames.jsonl")
            truth = last_record(folder / "evaluation_trajectory.jsonl")
            active.append(
                {
                    "episode": trial["episode"],
                    "state": frame.get("state", "startup"),
                    "sim_time": frame.get("sim_time"),
                    "path_m": round(frame.get("path_m", 0), 2),
                    "decisions": frame.get("decisions", 0),
                    "distance_m": round(truth.get("distance_to_target_m", 0), 2),
                    "last_event": last_record(folder / "events.jsonl"),
                }
            )
    print(
        json.dumps(
            {
                "completed": len(completed),
                "planned": len(manifest["trials"]),
                "latest_results": completed[-4:],
                "active": active,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
