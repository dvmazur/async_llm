"""Historical key ranges for the packed FA4 queries."""

from functools import lru_cache


@lru_cache(maxsize=1)
def historical_mask():
    # Ordinary Torch replay and CPU planning do not need the optional CuTe stack.
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils

    @cute.jit
    def mask(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
        # Callback indices are local to this varlen sequence. A partial tile can
        # invoke the callback on padded rows before FA applies its boundary mask;
        # keep those metadata loads inside the sequence, including the last one.
        row = seqlen_info.offset_q + cutlass.min(q_idx[0], seqlen_info.seqlen_q - 1)
        left = utils.scalar_to_ssa(aux_tensors[0][row], cutlass.Int32)
        right = utils.scalar_to_ssa(aux_tensors[1][row], cutlass.Int32)
        return (kv_idx >= left) & (kv_idx < right)

    return mask
