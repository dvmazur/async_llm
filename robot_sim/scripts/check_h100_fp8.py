"""Small parity gate, invoked only after an idle H100 is selected. No full model."""

import json, os, sys, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
from robot_sim.candidate_nav.hardware_profile import discover, choose_devices

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if visible not in ("0", "1"):
    raise SystemExit("Select exactly one idle physical GPU in CUDA_VISIBLE_DEVICES")
choose_devices("h100", visible, discover())  # Must precede every CUDA import/allocation.
import torch
from minisgl.kernel.fp8 import block_fp8_linear, quantize_fp8_groups


def run():
    if torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("SM90 required")
    torch.manual_seed(107)
    records = []
    for rows, n, k in (
        (1, 1024, 2048),
        (5, 1024, 3584),
        (64, 2048, 1024),
        (137, 1024, 2048),
        (1024, 1024, 2048),
    ):
        x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        w = (torch.randn(n, k, device="cuda") * 20).to(torch.float8_e4m3fn)
        scales = torch.rand(n // 128, k // 128, device="cuda") * 0.002 + 0.001
        os.environ["MINISGL_FP8_BACKEND"] = "triton"
        reference = block_fp8_linear(x, w, scales)
        os.environ["MINISGL_FP8_BACKEND"] = "flashinfer_sm90"
        actual = block_fp8_linear(x, w, scales)
        relative = float((actual.float() - reference.float()).norm() / reference.float().norm())
        equal = float((actual == reference).float().mean())
        if not torch.isfinite(actual).all() or relative >= 2e-4 or equal <= 0.999:
            raise AssertionError(f"Parity failed: rows={rows} relative={relative} equal={equal}")
        records.append(dict(rows=rows, n=n, k=k, relative_error=relative, equal_fraction=equal))
    x = torch.zeros(1, 256, device="cuda", dtype=torch.bfloat16)
    x[0, 0] = 1.5
    x[0, 128] = 0.625
    w = torch.zeros(128, 256, device="cuda").to(torch.float8_e4m3fn)
    w[:, 0] = 1
    w[:, 128] = 1
    scales = torch.tensor([[0.4874778985977173, 0.436303049325943]], device="cuda")
    torch.testing.assert_close(
        block_fp8_linear(x, w, scales),
        torch.ones(1, 128, device="cuda", dtype=x.dtype),
        rtol=0,
        atol=0,
    )
    torch.cuda.synchronize()
    result = dict(
        passed=True,
        gpu=visible,
        torch=torch.__version__,
        cases=records,
        kernel_sha256=hashlib.sha256(
            (ROOT / "python/minisgl/kernel/fp8.py").read_bytes()
        ).hexdigest(),
    )
    file = ROOT / "robot_sim/.runtime/live_rover/hopper-parity.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(result, indent=2))
    print("Hopper FP8 parity passed", flush=True)


if __name__ == "__main__":
    run()
