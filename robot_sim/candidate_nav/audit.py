"""Verify recorded model inputs, selected goals, concurrency and video artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def records(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def audit_episode(directory, server_log=None):
    failures = []
    decisions = records(directory / "decisions.jsonl")
    events = records(directory / "events.jsonl")
    metadata = json.loads((directory / "model_metadata.json").read_text())
    policy = json.loads((directory / "policy_inputs.json").read_text())
    direct = policy.get("control_mode") == "direct"
    if "Qwen3.8-27B" not in metadata.get("model", "") or metadata.get("quantization") != "fp8":
        failures.append("Unexpected model/checkpoint")
    if metadata.get("tp_size") != 2 or not metadata.get("shared_cache"):
        failures.append("Shared-cache TP2 backend not recorded")
    snapshots = {}
    ages = []
    early = 0
    for folder in sorted((directory / "observations").iterdir()):
        if not folder.is_dir():
            continue
        observation = json.loads((folder / "observation.json").read_text())
        snapshots[observation["snapshot_id"]] = (folder, observation)
        age = observation["sim_time"] - observation["camera_stamp"]
        ages.append(age)
        if not -0.01 <= age < 1.01:
            failures.append(f"{folder.name}: stale/future RGB ({age:.3f}s)")
        if any(
            key in observation
            for key in ("target_xy", "ground_truth", "evaluation", "world", "spawn")
        ):
            failures.append(f"{folder.name}: privileged input field")
        pixels = np.asarray(Image.open(folder / "map.png").convert("RGB"))
        if np.any(np.all(pixels == [255, 65, 88], axis=-1)):
            failures.append(f"{folder.name}: evaluator target color in policy map")
    for decision in decisions:
        identifier = decision["snapshot_id"]
        if identifier not in snapshots:
            failures.append(f"Missing snapshot {identifier}")
            continue
        folder, observation = snapshots[identifier]
        if decision["map_version"] != observation["map_version"]:
            failures.append(f"{identifier}: map version mismatch")
        for key, file in (("camera", "camera.png"), ("map", "map.png")):
            digest = hashlib.sha256(Image.open(folder / file).convert("RGB").tobytes()).hexdigest()
            if digest != decision["image_sha256"][key]:
                failures.append(f"{identifier}: {key} image hash mismatch")
        allowed = [candidate["id"] for candidate in observation["candidates"]] + ["W", "Z"]
        if direct:
            from .direct_control import validate_response

            allowed = ["F", "L", "R", "S", "Z"]
            try:
                validate_response(decision, observation)
            except ValueError as error:
                failures.append(f"{identifier}: {error}")
            if (
                observation.get("candidates")
                or observation.get("protocol") != "direct_navigation_v1"
            ):
                failures.append(f"{identifier}: invalid direct input/candidates")
        if decision["goal_id"] not in allowed:
            failures.append(f"{identifier}: decision outside allowlist")
        early += int(decision.get("analysis_in_progress", False))
    sent = 0
    commands = 0
    for event in events:
        if direct and event["event"] == "command_started":
            commands += 1
            matches = [d for d in decisions if d["snapshot_id"] == event["source_snapshot"]]
            if (
                len(matches) != 1
                or matches[0].get("action") != event["action"]
                or matches[0].get("task_complete")
            ):
                failures.append("Executed command differs from model action")
        if event["event"] != "goal_sent":
            continue
        sent += 1
        identifier = event["source_snapshot"]
        candidate = event["candidate"]
        _, observation = snapshots[identifier]
        matches = [item for item in observation["candidates"] if item["id"] == candidate["id"]]
        if not matches or any(
            abs(matches[0][key] - candidate[key]) > 0.002 for key in ("x", "y", "yaw")
        ):
            failures.append(f"{identifier}: executed point differs from model candidate")
    if not decisions:
        failures.append("No recorded model decisions")
    elif not early:
        failures.append("No selection published while analysis streams were running")
    video = directory / "replay.mp4"
    capture = cv2.VideoCapture(str(video))
    valid_frame, frame = capture.read()
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not valid_frame or frame.shape[:2] != (720, 1280) or frame_count < 10:
        failures.append("Missing/undecodable video")
    previews = []
    if valid_frame:
        for label, position in (
            ("start", 0),
            ("middle", frame_count // 2),
            ("end", max(0, frame_count - 20)),
        ):
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if ok:
                file = directory / f"preview_{label}.jpg"
                cv2.imwrite(str(file), frame)
                previews.append(file.name)
    capture.release()
    completed_audits = []
    generations = []
    generation_root = directory / "generations"
    for folder in sorted(generation_root.iterdir()) if generation_root.exists() else []:
        observation = json.loads((folder / "observation.json").read_text())
        if observation.get("candidates"):
            failures.append(f"Generation {folder.name}: precomputed candidates supplied")
        if any(
            k in observation for k in ("target_xy", "ground_truth", "evaluation", "world", "spawn")
        ):
            failures.append(f"Generation {folder.name}: privileged input")
        if not (folder / "response.json").exists():
            continue  # A request may be interrupted by the declared simulation budget.
        response = json.loads((folder / "response.json").read_text())
        generations.append(response)
        for key, file in (("camera", "camera.png"), ("map", "map.png")):
            digest = hashlib.sha256(Image.open(folder / file).convert("RGB").tobytes()).hexdigest()
            if digest != response["image_sha256"][key]:
                failures.append(f"Generation {folder.name}: {key} hash mismatch")
        pixels = np.asarray(Image.open(folder / "map.png").convert("RGB"))
        if np.any(np.all(pixels == [255, 65, 88], axis=-1)):
            failures.append(f"Generation {folder.name}: target color in policy map")
    if direct:
        processes = json.loads((directory / "processes.json").read_text())
        if (
            sent
            or any(p["name"] == "nav2" for p in processes)
            or policy.get("nav2_execution") is not False
        ):
            failures.append("Nav2/waypoint execution in direct arm")
    if policy.get("candidate_mode") == "qwen":
        for _, observation in snapshots.values():
            generation = generation_root / f"{observation['candidate_generation_snapshot']:04d}"
            checks = json.loads((generation / "validation.json").read_text())
            accepted = [item["candidate"] for item in checks if item["accepted"]]
            for candidate in observation["candidates"]:
                if not any(
                    candidate["id"] == c["id"]
                    and all(abs(candidate[k] - c[k]) < 0.002 for k in ("x", "y", "yaw"))
                    for c in accepted
                ):
                    failures.append("Selector candidate has no validated Qwen proposal")
    if server_log is not None:
        keys = {
            (d["snapshot_id"], d["image_sha256"]["camera"], d["image_sha256"]["map"])
            for d in decisions
        }
        for line in server_log.read_text(errors="replace").splitlines():
            if not line.startswith("INFERENCE_AUDIT "):
                continue
            record = json.loads(line[len("INFERENCE_AUDIT ") :])
            hashes = record.get("image_sha256", {})
            key = (record.get("snapshot_id"), hashes.get("camera"), hashes.get("map"))
            if record.get("episode_id") == directory.name and key in keys:
                completed_audits.append(record)
        (directory / "inference_audit.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in completed_audits)
        )
        if len(completed_audits) != len(decisions):
            failures.append("Missing or ambiguous completed inference audit records")
        generation_keys = {(r["snapshot_id"], r["image_sha256"]["map"]) for r in generations}
        found = []
        for line in server_log.read_text(errors="replace").splitlines():
            if line.startswith("GENERATION_AUDIT "):
                record = json.loads(line[len("GENERATION_AUDIT ") :])
                if (
                    record.get("episode_id") == directory.name
                    and (record["snapshot_id"], record["image_sha256"]["map"]) in generation_keys
                ):
                    found.append(record)
        if len(found) != len(generations):
            failures.append("Missing or ambiguous generation inference audits")
        (directory / "generation_audit.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in found)
        )
    result = {
        "passed": not failures,
        "failures": failures,
        "snapshots": len(snapshots),
        "decisions": len(decisions),
        "goals_sent": sent,
        "direct_commands_sent": commands,
        "early_replies_during_analysis": early,
        "max_camera_age_sim_s": max(ages) if ages else None,
        "video_frames": frame_count,
        "video_fps": fps,
        "previews": previews,
        "completed_inference_audits": len(completed_audits),
        "completed_generations": len(generations),
        "input_images_match_model_hashes": not any("hash mismatch" in error for error in failures),
        "evaluator_target_excluded_from_policy_maps": not any(
            "target color" in error for error in failures
        ),
    }
    (directory / "artifact_audit.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--server-log", type=Path)
    args = p.parse_args()
    result = audit_episode(args.directory, args.server_log)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
