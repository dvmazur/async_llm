"""One-token recurrent consumer of deferred FP32 [1,H,Dv,Dk] state rows."""
import torch

from .metadata import device_metadata


def recurrent_gdn_pointer(query, key, value, g, beta, rows):
    import triton
    from .triton.gdn_recurrent import gdn_recurrent_pointer_kernel

    if not query.is_cuda or query.ndim != 4 or query.shape[1] != 1:
        raise ValueError("pointer recurrence requires CUDA one-token [B,1,H,K] queries")
    b, _, h, dk = query.shape
    if key.shape != query.shape or value.ndim != 4 or value.shape[:2] != (b, 1):
        raise ValueError("invalid one-token query/key/value shapes")
    hv, dv = value.shape[2:]
    if min(b, h, hv, dk, dv) < 1:
        raise ValueError("recurrent batch and head dimensions must be positive")
    if hv % h or g.shape != (b, 1, hv) or beta.shape != g.shape or len(rows) != b:
        raise ValueError("invalid recurrent heads, gates, or initial-state row count")
    for row in rows:
        if (row.shape != (1, hv, dv, dk) or row.dtype != torch.float32
                or row.device != query.device or not row.is_contiguous()):
            raise ValueError("state rows must be contiguous FP32 [1,Hv,Dv,Dk] on the query device")
    inputs = (query, key, value, g, beta)
    if any(x.device != query.device for x in inputs):
        raise ValueError("recurrent inputs must share a device")
    if any(x.requires_grad for x in (*inputs, *rows)) and torch.is_grad_enabled():
        raise ValueError("pointer recurrence is inference-only")
    # FLA's input_guard does the same normalization for its dense entry point.
    query, key, value, g, beta = (x.contiguous() for x in inputs)
    pointers = device_metadata([row.data_ptr() for row in rows], device=query.device)
    output = torch.empty_like(value)
    final_state = torch.empty((b, hv, dv, dk), device=query.device, dtype=torch.float32)
    bv = min(8, triton.next_power_of_2(dv))
    gdn_recurrent_pointer_kernel[(triton.cdiv(dv, bv), b * hv)](
        query, key, value, g, beta, pointers, output, final_state,
        scale=dk ** -0.5, H=h, HV=hv, K=dk, V=dv,
        BK=triton.next_power_of_2(dk), BV=bv, num_warps=1, num_stages=3,
    )
    # Strong row references stay alive through launch; normal current-stream
    # allocator semantics protect them until the queued read completes.
    return output, final_state
