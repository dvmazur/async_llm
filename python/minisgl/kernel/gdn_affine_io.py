"""Pack/split block-owned A/B for one auxiliary FLA final-state pass."""
import triton as tr
import triton.language as tl


@tr.jit
def _affine_state_io(Read, Write, State, H: tl.constexpr, DK: tl.constexpr,
                     DV: tl.constexpr, PACK: tl.constexpr, BLOCK: tl.constexpr):
    worker = tl.program_id(0)
    wa, wb = tl.load(Write + worker*3), tl.load(Write + worker*3+1)
    active = (wa != 0) | (wb != 0)
    if not PACK and not active:
        return
    width = DK + DV
    off = tl.program_id(1)*BLOCK + tl.arange(0, BLOCK)
    row = off % width
    key = (off // width) % DK
    head = off // (width*DK)
    is_a = row < DK
    local_row = tl.where(is_a, row, row-DK)
    nrows = tl.where(is_a, DK, DV)
    block_off = (head*nrows + local_row)*DK + key
    valid = off < H*DK*width
    state_off = worker*H*DK*width + off
    if PACK:
        ra, rb = tl.load(Read + worker*3), tl.load(Read + worker*3+1)
        address = tl.where(is_a, ra, rb)
        pointer = address.to(tl.pointer_type(tl.float32))
        value = tl.load(pointer + block_off, valid & active & (address != 0), other=0.)
        identity = is_a & (local_row == key)
        value = tl.where(address == 0, identity.to(tl.float32), value)
        tl.store(State + state_off, tl.where(active, value, 0.), valid)
    else:
        address = tl.where(is_a, wa, wb)
        pointer = address.to(tl.pointer_type(tl.float32))
        value = tl.load(State + state_off, valid, other=0.)
        tl.store(pointer + block_off, value, valid & (address != 0))


def pack_affine_initial(read, write, initial, dk, dv):
    # FLA layout [W,H,Dk,Dk+Dv] holds [A^T | B^T]. Empty targets use I/0.
    workers, heads = initial.shape[:2]
    _affine_state_io[(workers, tr.cdiv(heads*dk*(dk+dv), 512))](
        read, write, initial, heads, dk, dv, True, 512)


def store_affine_final(write, final, dk, dv):
    # Scatter into owned outputs, never publish views retaining FLA scratch.
    workers, heads = final.shape[:2]
    _affine_state_io[(workers, tr.cdiv(heads*dk*(dk+dv), 512))](
        write, write, final, heads, dk, dv, False, 512)
