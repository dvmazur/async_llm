from dataclasses import replace

import pytest
import torch
from minisgl.training import replay_trace
from minisgl.training.trace import TensorData

from tests.training.fixtures import torch_reference_kernels as torch_reference_kernels
from tests.training.models import tiny_image_model
from tests.training.workloads import single_image_trace


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("mrope_positions", torch.zeros(2, 9, dtype=torch.long), "positions"),
        ("mrope_positions", None, "positions"),
        ("write_span", 1, "write span"),
        ("cache_spans", (1,), "cache spans"),
        ("image_grid_thw", None, "image_grid_thw"),
        ("image_grid_thw", torch.tensor([[1, 4, 5]]), "merge-aligned"),
        ("pixel_values", torch.zeros(23, 12), "patch shape"),
        ("mm_token_type_ids", torch.full((9,), 2), "0/1"),
        ("mm_token_type_ids", torch.zeros(9, dtype=torch.long), "disagree"),
    ],
)
def test_malformed_image_metadata_fails_explicitly(field, value, message):
    trace = single_image_trace()
    batch = trace.batches[0]
    value = TensorData.capture(value) if isinstance(value, torch.Tensor) else value
    trace.batches[0] = replace(batch, requests=(replace(batch.requests[0], **{field: value}),))
    with pytest.raises(ValueError, match=message):
        replay_trace(tiny_image_model(), trace.to(), hidden_only=False)


def test_image_token_ids_require_payload_without_token_types():
    trace = single_image_trace()
    batch = trace.batches[0]
    request = replace(
        batch.requests[0], pixel_values=None, image_grid_thw=None, mm_token_type_ids=None
    )
    trace.batches[0] = replace(batch, requests=(request,))
    with pytest.raises(ValueError, match="require recorded pixels"):
        replay_trace(tiny_image_model(), trace.to(), hidden_only=False)
