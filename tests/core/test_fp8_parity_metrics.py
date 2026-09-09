import pytest
import torch

from tests.e2e.fp8 import comparison as metrics


def test_metrics_identical_and_offset():
    x=torch.tensor([[1.,2.,3.],[-1.,0.,1.]])
    tv,l2,agree=metrics.distances(x+10,x)
    torch.testing.assert_close(tv,torch.zeros(2),rtol=0,atol=0)
    torch.testing.assert_close(l2,torch.zeros(2),rtol=0,atol=0)
    assert agree.all()


def test_metrics_detect_change_and_invalid():
    x=torch.tensor([[1.,2.,3.]])
    tv,l2,agree=metrics.distances(x.flip(-1),x)
    assert tv.item()>.5 and l2.item()>1 and not agree.item()
    with pytest.raises(AssertionError):metrics.distances(x*float('nan'),x)
