"""Start only on eligible devices; never stop or signal GPU jobs."""

import argparse, json, os, shlex, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from robot_sim.candidate_nav.hardware_profile import settings, discover, choose_devices, PROFILES
from robot_sim.candidate_nav.model_catalog import model_key, model_path, validate_model, MODELS, model_tp, select_devices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", choices=("auto", "0", "1"))
    p.add_argument("--model", type=model_key, default="27b")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    config = settings(ROOT)
    request = args.gpu or os.environ.get("ROBOT_LLM_GPU_REQUEST") or config["gpu"]
    validate_model(ROOT,args.model)
    devices = discover()
    try:
        selected = select_devices(config,args.model,request,devices)
    except RuntimeError as error:
        if args.check_only:
            print(
                json.dumps(
                    dict(ready=False, reason=str(error), devices=devices, config=config), indent=2
                )
            )
            return
        raise SystemExit(str(error))
    if args.check_only:
        print(json.dumps(dict(ready=True, selected=selected, config=config), indent=2))
        return
    if subprocess.run(
        ["ss", "-H", "-lnt", "( sport = :18766 )"], capture_output=True, text=True, check=True
    ).stdout.strip():
        raise SystemExit("Port 18766 is occupied; no process stopped")
    profile = dict(PROFILES[config["profile"]],tp_size=model_tp(config["profile"],args.model))
    env = dict(os.environ)
    env.update(
        ROBOT_LLM_GPU=",".join(str(d["index"]) for d in selected),
        ROBOT_LLM_TP_SIZE=str(profile["tp_size"]),
        MINISGL_FP8_BACKEND="triton",
        ROBOT_LLM_HARDWARE_PROFILE=config["profile"],
    )
    env.pop("MINISGL_FP8_EMULATE", None)
    env["ROBOT_FP8_BACKEND_REQUEST"] = config["fp8_backend"]
    env["MODEL"] = str(model_path(ROOT,args.model))
    if config["profile"] == "h100":
        # Same serialized FP8 bytes; only the disk reader changes.
        env.setdefault("MINISGL_WEIGHT_BACKEND", "pread")
        env["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        env["MAX_JOBS"] = "4"
        for key, folder in (
            ("TRITON_CACHE_DIR", "triton-h100"),
            ("TORCH_EXTENSIONS_DIR", "torch_extensions-h100"),
            ("CUDA_CACHE_PATH", "cuda_cache-h100"),
            ("TVM_FFI_CACHE_DIR", "tvm_ffi-h100"),
        ):
            env[key] = str(ROOT / "robot_sim/.runtime" / folder)
        env["CUDA_HOME"] = "/usr/local/cuda-13.0"
        env["PATH"] = str(Path(config["python"]).parent) + ":" + env["CUDA_HOME"] + "/bin:" + env["PATH"]
        if config["fp8_backend"] == "auto_sm90":
            python = config["python"]
            test_env = dict(
                env,
                CUDA_VISIBLE_DEVICES=env["ROBOT_LLM_GPU"],
                PYTHONPATH=str(ROOT / "python") + ":" + str(ROOT),
            )
            print("Checking Hopper FP8 numerical parity before model startup…", flush=True)
            result = subprocess.run(
                [python, str(ROOT / "robot_sim/scripts/check_h100_fp8.py")], env=test_env
            )
            if result.returncode == 0:
                env["MINISGL_FP8_BACKEND"] = "flashinfer_sm90"
            else:
                print(
                    "Hopper GEMM validation did not pass; using native-FP8 Triton backend",
                    flush=True,
                )
    else:
        env["ROBOT_SIM_CPU"] = env.get("ROBOT_SIM_CPU", "1")
    # Recheck after the parity subprocess and size the allocator from this sample.
    # Keep the chosen H100 fixed so the model uses the device tested above.
    launch_request = str(selected[0]['index']) if config['profile'] == 'h100' else request
    selected = select_devices(config, args.model, launch_request, discover())
    if config['profile'] == 'h100':
        cap = 45056
        if args.model == '122b':
            cap = 73728
        elif args.model == '35b' and not selected[0]['busy']:
            cap = 57344
        free_mib = min(device['free_mib'] for device in selected)
        limit_mib = min(cap, free_mib - 8192)
        env['ROBOT_LLM_MEMORY_LIMIT_MIB'] = str(limit_mib)
        print(f"Smallest free GPU memory: {free_mib} MiB. PyTorch allocator limit: {limit_mib} MiB per GPU.",
              flush=True)
    cmd = [
        "bash",
        str(ROOT / "robot_sim/scripts/run_candidate_llm.sh"),
        "--analysis-tokens",
        "24",
        "--context-window",
        "12288",
        "--trajectory-policy",
        "evidence",
        "--hardware-profile",
        config["profile"],
    ]
    print(
        f"Starting {MODELS[args.model]}: {config['profile']} GPU={env['ROBOT_LLM_GPU']} TP={profile['tp_size']} FP8={env['MINISGL_FP8_BACKEND']}",
        flush=True,
    )
    os.chdir(ROOT)
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
