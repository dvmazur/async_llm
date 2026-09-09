"""Small host-built metadata uploads without draining the CUDA stream."""

import torch


def device_metadata(values, *, device, dtype=torch.int64):
    """Copy an immutable, pinned CPU table on the current CUDA stream.

    PyTorch's pinned caching allocator records the asynchronous copy's event;
    dropping ``host`` here cannot recycle its memory before the copy completes.
    Never mutate/reuse the CPU table after enqueueing the copy.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return torch.tensor(values, dtype=dtype, device=device)
    host = torch.tensor(values, dtype=dtype, device="cpu", pin_memory=True)
    return host.to(device=device, non_blocking=True)
