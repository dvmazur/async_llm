#!/usr/bin/env bash
# Run the traced MATH-500 evaluator or its full-sweep coordinator.
set -euo pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"

site_packages="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export CUDA_HOME="${CUDA_HOME:-$site_packages/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include:$CUDA_HOME/include/cccl${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MAX_JOBS="${MAX_JOBS:-8}"
export PYTHONPATH="$PWD/scripts:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export HOGWILD_EVALUATOR_LAUNCHER="$PWD/run_hogwild.sh"
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF

package=minisgl_examples.training.parallel_thinking
export HOGWILD_EVAL_MODULE="$package.eval"
if [[ "${1:-}" == --full-math500 ]]; then
    shift
    export CUDA_VISIBLE_DEVICES=""
    exec uv run --no-sync python -u -m "$package.sweep" "$@"
fi
if [[ "${1:-}" == --budget-sweep ]]; then
    shift
    if [[ "${1:-}" != --worker ]]; then
        export CUDA_VISIBLE_DEVICES=""
        exec uv run --no-sync python -u -m "$package.budget_curve_sweep" "$@"
    fi
    shift
    export HOGWILD_EVAL_MODULE="$package.budget_curve_eval"
fi
if [[ " $* " == *' --help '* || " $* " == *' -h '* ]]; then
    export CUDA_VISIBLE_DEVICES=""
    exec uv run --no-sync python -m "$HOGWILD_EVAL_MODULE" "$@"
fi
if [[ ! "${HOGWILD_GPU:-}" =~ ^[0-9]+$ ]]; then
    echo 'HOGWILD_GPU must select one physical GPU for an evaluator.' >&2
    exit 2
fi
export HOGWILD_GPU CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$HOGWILD_GPU" --query-gpu=uuid --format=csv,noheader)"

# Stage pageable CPU tensors to avoid safetensors 0.8's large pinned allocations.
exec uv run --no-sync python -u - "$@" <<'PYTHON'
def _install_cpu_weight_reader(weight_module):
    from types import SimpleNamespace

    original = weight_module.safetensors

    class CPUWeightReader:
        def __init__(self, filename, framework, device="cpu", **kwargs):
            if framework != "pt":
                raise ValueError("The checkpoint adapter supports only PyTorch weights")
            self.device = device
            self.context = original.safe_open(filename, framework=framework, device="cpu", **kwargs)

        def __enter__(self):
            self.reader = self.context.__enter__()
            return self

        def __exit__(self, *exception):
            return self.context.__exit__(*exception)

        def keys(self):
            return self.reader.keys()

        def get_tensor(self, name):
            return self.reader.get_tensor(name).to(device=self.device, non_blocking=False)

    weight_module.safetensors = SimpleNamespace(safe_open=CPUWeightReader)


if __name__ == "__main__":
    import os
    import runpy
    from minisgl.models import weight

    _install_cpu_weight_reader(weight)
    print("WEIGHT_IO=cpu-mmap-blocking-copy-v1", flush=True)
    runpy.run_module(os.environ["HOGWILD_EVAL_MODULE"], run_name="__main__", alter_sys=True)
PYTHON
