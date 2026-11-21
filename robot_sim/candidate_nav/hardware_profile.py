"""GPU discovery and live-model profiles; deliberately no torch/CUDA imports."""

import json
from pathlib import Path
import subprocess

PROFILES = {
    "legacy": dict(tp_size=2, min_free_mib=22000, prefill_rows=256, cache_tokens=16384),
    "h100": dict(tp_size=1, min_free_mib=60000, prefill_rows=1024, cache_tokens=32768),
}


def settings(root):
    file = Path(root) / "robot_sim/.runtime/live_model.json"
    data = json.loads(file.read_text()) if file.exists() else {}
    data.setdefault("profile", "legacy")
    data.setdefault("gpu", "auto")
    data.setdefault("fp8_backend", "triton")
    for key in ("shared_gpu_ids", "shared_27b_gpu_ids", "shared_122b_gpu_ids"):
        indices = data.setdefault(key, [])
        if not isinstance(indices, list) or any(type(i) is not int or i not in (0, 1) for i in indices):
            raise ValueError(f"{key} must list explicitly authorized GPU indices")
    if data["profile"] not in PROFILES:
        raise ValueError("Unknown hardware profile")
    if data["gpu"] not in ("auto", "0", "1"):
        raise ValueError("GPU must be auto, 0 or 1")
    if data["fp8_backend"] not in ("triton", "auto_sm90"):
        raise ValueError("Unknown FP8 backend")
    data["session"] = "rover-live-qwen-h100" if data["profile"] == "h100" else "rover-live-qwen"
    return data


def discover():
    def query(args):
        return subprocess.run(
            ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    rows = query(["--query-gpu=index,uuid,name,memory.free,memory.total,compute_cap"])
    jobs = query(["--query-compute-apps=gpu_uuid,pid"])
    busy = {line.split(",")[0].strip() for line in jobs.splitlines() if line.strip()}
    devices = []
    for line in rows.splitlines():
        index, uuid, name, free, total, cap = [x.strip() for x in line.split(",")]
        devices.append(
            dict(
                index=int(index),
                uuid=uuid,
                name=name,
                free_mib=int(free),
                total_mib=int(total),
                capability=float(cap),
                busy=uuid in busy,
            )
        )
    return devices


def choose_devices(profile, request, devices, *, shared_gpu_ids=()):
    if any(type(i) is not int or i not in (0,1) for i in shared_gpu_ids):
        raise ValueError("Invalid shared GPU authorization")
    if profile not in PROFILES or request not in ("auto", "0", "1"):
        raise ValueError("Invalid GPU profile/request")
    if profile == "legacy":
        if request != "auto":
            raise ValueError("Legacy profile requires both GPUs 0,1")
        selected = [next((d for d in devices if d["index"] == i), None) for i in (0, 1)]
        if any(d is None or d["free_mib"] < 22000 for d in selected):
            raise RuntimeError("Legacy needs 22000 MiB free on GPUs 0 and 1; no jobs touched")
        return selected
    candidates = []
    rejected = []
    for device in devices:
        index = device["index"]
        if request != "auto" and index != int(request):
            continue
        sharing = index in shared_gpu_ids
        minimum = 45056 if sharing else PROFILES["h100"]["min_free_mib"]
        free = device["free_mib"]
        if device["capability"] != 9.0 or device["total_mib"] < 75000:
            rejected.append(f"GPU {index}: requires H100 80GB")
            continue
        reasons = []
        if free < minimum:
            reasons.append(f"requires >= {minimum} MiB free")
        if device["busy"] and not sharing:
            reasons.append("running compute jobs and sharing is not enabled")
        if reasons:
            rejected.append(f"GPU {index}: {free} MiB free, " + ", ".join(reasons))
        else:
            candidates.append(device)
    if not candidates:
        detail = " / ".join(rejected) or f"GPU selection {request!r} has no visible devices"
        raise RuntimeError(f"No eligible H100: {detail}")
    return [max(candidates, key=lambda d: (d["free_mib"], -d["index"]))]
