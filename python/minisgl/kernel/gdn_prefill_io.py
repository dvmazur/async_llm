"""Only packing/transposition for joint FLA prefill; no recurrence math."""

import triton as tr
import triton.language as tl


@tr.jit
def _pack_initial(Read, Initial, Out, Cu, H: tl.constexpr, DK: tl.constexpr,
                  DV: tl.constexpr, BLOCK: tl.constexpr):
    worker, head = tl.program_id(0), tl.program_id(1)
    if Cu is not None:
        if tl.load(Cu + worker) == tl.load(Cu + worker + 1):
            return
    offsets = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    width = DV + DK + DV
    key, value = offsets // width, offsets % width
    valid = key < DK
    is_s = value < DV
    is_a = (value >= DV) & (value < DV + DK)
    local = tl.where(is_a, value - DV, value - DV - DK)
    address = tl.load(Read + worker * 3 + tl.where(is_a, 0, 1))
    ptr = address.to(tl.pointer_type(tl.float32))
    # Block A/B are [value,key], whereas FLA state is [key,value].
    source = (head * tl.where(is_a, DK, DV) + local) * DK + key
    affine = tl.load(ptr + source, valid & ~is_s & (address != 0), other=0.)
    affine = tl.where(is_a & (address == 0), (local == key).to(tl.float32), affine)
    state = tl.load(Initial + (worker * H + head) * DK * DV + key * DV + value,
                    valid & is_s, other=0.)
    tl.store(Out + (worker * H + head) * DK * width + offsets,
             tl.where(is_s, state, affine), valid)


@tr.jit
def _store_affines(Write, Final, H: tl.constexpr, DK: tl.constexpr,
                   DV: tl.constexpr, BLOCK: tl.constexpr):
    worker, head = tl.program_id(0), tl.program_id(1)
    offsets = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    row, key = offsets // DK, offsets % DK
    is_a = row < DK
    local = tl.where(is_a, row, row - DK)
    address = tl.load(Write + worker * 3 + tl.where(is_a, 0, 1))
    ptr = address.to(tl.pointer_type(tl.float32))
    valid = (row < DK + DV) & (address != 0)
    source = (worker * H + head) * DK * (DV + DK + DV) + key * (DV + DK + DV) + DV + row
    value = tl.load(Final + source, valid, other=0.)
    tl.store(ptr + (head * tl.where(is_a, DK, DV) + local) * DK + key, value, valid)


def pack_initial(read, initial, out, cu=None):
    workers, heads, dk, dv = initial.shape
    _pack_initial[(workers, heads, tr.cdiv(dk * (dv + dk + dv), 256))](
        read, initial, out, cu, heads, dk, dv, 256)


def store_affines(write, final, dv):
    workers, heads, dk, _ = final.shape
    _store_affines[(workers, heads, tr.cdiv(dk * (dk + dv), 256))](
        write, final, heads, dk, dv, 256)
