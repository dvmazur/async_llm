"""Small indirect copies for block-owned GDN tensors; no GDN arithmetic here."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _gather_rows(Pointers, Out, N: tl.constexpr, STRIDE: tl.constexpr, IDENTITY: tl.constexpr,
                 BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    address = tl.load(Pointers + row * STRIDE)
    ptr = address.to(tl.pointer_type(Out.dtype.element_ty))
    values = tl.load(ptr + offsets, (address != 0) & (offsets < N), other=0)
    if IDENTITY:
        diagonal = (offsets // IDENTITY) % IDENTITY == offsets % IDENTITY
        values = tl.where(address == 0, diagonal.to(values.dtype), values)
    tl.store(Out + row * N + offsets, values, offsets < N)


@tr.jit
def _scatter_rows(Pointers, Source, N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    address = tl.load(Pointers + row * STRIDE)
    ptr = address.to(tl.pointer_type(Source.dtype.element_ty))
    values = tl.load(Source + row * N + offsets, offsets < N, other=0)
    tl.store(ptr + offsets, values, (address != 0) & (offsets < N))


@tr.jit
def _collect_states(States, Terminals, Out, H: tl.constexpr, DK: tl.constexpr,
                    DV: tl.constexpr, BLOCK: tl.constexpr):
    """Only terminal nodes write; transpose block [dv,dk] to recurrent [dk,dv]."""
    worker = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    node = tl.load(Terminals + worker)
    v = offsets % DV
    k = (offsets // DV) % DK
    h = offsets // (DV * DK)
    source = ((node * H + h) * DV + v) * DK + k
    valid = (node >= 0) & (offsets < H * DK * DV)
    values = tl.load(States + source, valid, other=0)
    tl.store(Out + worker * H * DK * DV + offsets, values, valid)


def gather_rows(pointers: torch.Tensor, out: torch.Tensor, identity: int = 0):
    assert out.is_contiguous()
    n = out.numel() // out.shape[0]
    _gather_rows[(out.shape[0], tr.cdiv(n, 512))](pointers, out, n, pointers.stride(0), identity, 512)
    return out


def scatter_rows(pointers: torch.Tensor, source: torch.Tensor):
    source = source.contiguous()
    n = source.numel() // source.shape[0]
    _scatter_rows[(source.shape[0], tr.cdiv(n, 512))](pointers, source, n, pointers.stride(0), 512)


def collect_states(states: torch.Tensor, terminals: torch.Tensor, out: torch.Tensor):
    _, h, dk, dv = out.shape
    _collect_states[(out.shape[0], tr.cdiv(h * dk * dv, 512))](
        states, terminals, out, h, dk, dv, 512)
