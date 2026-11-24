from __future__ import annotations

import functools
from typing import Tuple


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    return torch.cuda.get_device_capability()


def is_arch_supported(major: int, minor: int = 0) -> bool:
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    return arch >= (major, minor)


def get_device_capability(device=None) -> Tuple[int, int] | None:
    """Return the capability of *device* without assuming the default GPU."""
    import torch

    if device is not None and not isinstance(device, int) and torch.device(device).type != "cuda":
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(device)


def is_turing(device=None) -> bool:
    """Whether *device* is an NVIDIA Turing GPU (SM75).

    Volta and Pascal are intentionally not included: mini-sglang does not
    claim support for those architectures.
    """
    return get_device_capability(device) == (7, 5)


def is_sm90_supported() -> bool:
    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    return is_arch_supported(10, 0)
