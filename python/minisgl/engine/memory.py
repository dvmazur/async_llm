"""Conservative allocator default for dynamic, large GDN state batches."""
from __future__ import annotations

import re

import torch


def prefer_expandable_segments(enabled: bool) -> bool:
    """Enable native expandable segments only if the caller has no explicit choice.

    This must run before Engine initializes CUDA. Preserve the complete existing
    allocator settings string (including programmatic settings), not just an
    environment variable. Neither the allocator backend nor os.environ changes.
    Returns whether this call changed the configuration.
    """
    if not enabled or torch.cuda.get_allocator_backend() != "native":
        return False
    if torch.cuda.is_initialized():
        raise RuntimeError("Configure GDN allocator before CUDA initialization")
    settings = torch.cuda.memory._snapshot().get("allocator_settings", {})
    raw = settings.get("PYTORCH_CUDA_ALLOC_CONF")
    setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    # Older Torch without settings introspection retains the inherited policy.
    if raw is None or setter is None:
        return False
    if re.search(r"(?:^|,)\s*expandable_segments\s*:", raw):
        return False
    combined = raw.rstrip(", ")
    if combined:
        combined += ","
    setter(combined + "expandable_segments:True")
    return True
