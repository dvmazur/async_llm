"""Check physical GPU occupancy before a model stage; never stop processes."""

from __future__ import annotations

import os
import subprocess


def gpu_state():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    owners = {}
    for line in processes.splitlines():
        if not line.strip():
            continue
        uuid, pid = [part.strip() for part in line.split(",")]
        owners.setdefault(uuid, []).append(int(pid))
    devices = []
    for line in output.splitlines():
        index, uuid, memory, remaining, total, utilization = [
            part.strip() for part in line.split(",")
        ]
        pids = owners.get(uuid, [])
        foreign = [pid for pid in pids if pid != os.getpid()]
        mine = os.getpid() in pids
        free = not foreign and (mine or (int(memory) <= 64 and int(utilization) == 0))
        shared = (
            os.environ.get("DOOM_ALLOW_SHARED_GPU") == "1"
            and int(remaining) >= (memory_limit_gib() + 16) * 1024
        )
        devices.append(
            {
                "index": int(index),
                "uuid": uuid,
                "memory_mib": int(memory),
                "free_mib": int(remaining),
                "total_mib": int(total),
                "utilization": int(utilization),
                "pids": pids,
                "available": free or shared,
            }
        )
    return devices


def memory_limit_gib():
    return float(os.environ.get("DOOM_GPU_LIMIT_GIB", "32"))


def apply_memory_limit():
    """Bound our allocator, independently of other CUDA processes.

    Call after native Engine construction: Engine requires an uninitialized CUDA
    runtime on entry. Inference model/KV allocation is already explicitly bounded.
    HF replay/training can call this before loading their model.
    """
    import torch

    budget = memory_limit_gib() * 1024**3
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(budget / total, 1.0), 0)


def require_free_gpu():
    devices = gpu_state()
    requested = os.environ.get("CUDA_VISIBLE_DEVICES")
    if requested is not None:
        candidates = [d for d in devices if requested in (str(d["index"]), d["uuid"])]
        if len(candidates) != 1:
            raise RuntimeError(
                "Set CUDA_VISIBLE_DEVICES to exactly one physical GPU; restart a notebook kernel after changing it"
            )
    else:
        candidates = [d for d in devices if d["available"]][:1]
    if not candidates or not candidates[0]["available"]:
        raise RuntimeError(
            f"No free GPU or sufficient authorized shared VRAM for this stage: {devices}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(candidates[0]["index"])
    return candidates[0]
