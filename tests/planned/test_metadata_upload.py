from types import SimpleNamespace
import pytest
import torch

from minisgl.planned.gdn_device import _DeviceTables
from minisgl.planned.shared_attention import _SubTables,_AttentionTables


def test_properties_evaluated_once_and_validation_is_before_any_copy():
    class Plan:
        count=0
        @property
        def values(self):self.count+=1;return (1,2,3)
    class Tables(_DeviceTables):fields=('values',)
    plan=Plan();tables=Tables(plan,'cpu');plan.count=0
    tables.upload(plan)
    assert plan.count==1
    before=tables.values.clone()
    with pytest.raises(ValueError):tables.upload(SimpleNamespace(values=(2,3)))
    torch.testing.assert_close(tables.values,before)


def test_position_axes_upload_without_flattening_and_clear_stale_padding():
    class Plan:
        source_rows=(0,1,-1,-1)
        positions=((1,2,0,0),(3,4,0,0),(5,6,0,0))
        @property
        def positions_flat(self):raise AssertionError('Python flatten materialized')
    p=Plan();t=_SubTables(p,'cpu');address=t.positions_flat.data_ptr()
    torch.testing.assert_close(t.positions_flat,torch.tensor([1,2,0,0,3,4,0,0,5,6,0,0]))
    p.positions=((0,0,0,0),)*3;p.source_rows=(-1,)*4;t.upload(p)
    assert not t.positions_flat.any() and t.positions_flat.data_ptr()==address
    p.positions=((0,0),(0,0),(0,0))
    with pytest.raises(ValueError):t.upload(p)
