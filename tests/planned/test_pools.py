import pytest
import torch

from minisgl.planned.pools import GDNPool, PoolShape


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_one_all_layer_allocation_and_byte_accounting(dtype):
    shape = PoolShape(4, 7, 2, 5, 16, 4)
    pool = GDNPool(shape, device="cpu", activation_dtype=dtype)
    assert pool.affine.shape == (4,7,2,2,5,5)
    expected = 4*7*2*2*5*5*4 + 4*7*16*4*(4 if dtype == torch.float32 else 2)
    assert pool.reserved_bytes == expected
    assert pool.bytes_per_slot * 7 == expected
    for layer in range(4):
        for slot in range(7):
            A,B,C = pool.internal_views(layer, slot)
            assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous()
            assert A.untyped_storage().data_ptr() == pool.affine.untyped_storage().data_ptr()
            assert B.untyped_storage().data_ptr() == pool.affine.untyped_storage().data_ptr()
            assert C.untyped_storage().data_ptr() == pool.conv.untyped_storage().data_ptr()
            A.fill_(layer*10+slot); B.fill_(-1); C.fill_(slot)
    assert pool.affine[3,6,0,0,0,0] == 36
    assert pool.affine[0,0,0,0,0,0] == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pool_base_addresses_do_not_change_on_reusing_slots(device):
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA required")
    from minisgl.planned.slots import SlotRegistry
    from minisgl.planned.forward_plan import PlanCapacity, DecodeRequest
    from test_slots import Completion
    pool = GDNPool(PoolShape(2,1,2,16,16,4), device=device)
    reg = SlotRegistry(1)
    pointers = pool.affine.data_ptr(), pool.conv.data_ptr()
    for _ in range(8):
        b = reg.create()
        tx = reg.begin(capacity=PlanCapacity(0,0,1,2,1), decode=[DecodeRequest((), b)])
        reg.mark_submitted(tx)
        slot = tx.plan.writes[0].slot
        for layer in range(2):
            A,B,C = pool.internal_views(layer, slot)
            A.copy_(torch.eye(16, device=device)); B.zero_(); C.zero_()
        event = Completion(True)
        if device == "cuda":
            event = torch.cuda.Event(); event.record(); event.synchronize()
        assert reg.finish(tx, event)
        reg.release_handle(b)
        assert pointers == (pool.affine.data_ptr(), pool.conv.data_ptr())
