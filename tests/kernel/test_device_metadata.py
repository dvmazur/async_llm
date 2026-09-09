"""Pinned metadata must survive CPU release and outstanding CUDA work."""
import gc

import pytest
import torch
from minisgl.kernel.metadata import device_metadata


def test_cpu_metadata_preserves_values_and_dtype():
    value = device_metadata([[1, 2], [3, 4]], device="cpu")
    assert value.dtype == torch.int64 and value.tolist() == [[1, 2], [3, 4]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_metadata_is_not_reused_before_copy_finishes():
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for i in range(256):
        with torch.cuda.stream(streams[i % 2]):
            if i < 2:
                torch.cuda._sleep(10_000_000)
            metadata = device_metadata([[i, i + 1, i + 2]] * 19, device="cuda")
            outputs.append((i, metadata.clone()))
        if i % 16 == 0:
            gc.collect()
    for stream in streams:
        stream.synchronize()
    for i, output in outputs:
        assert output.cpu().tolist() == [[i, i + 1, i + 2]] * 19


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pinned_upload_has_no_stream_synchronize():
    # Warm host allocation/event bookkeeping before checking the hot path.
    for _ in range(32):
        device_metadata([[1, 2, 3]] * 9, device="cuda")
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        outputs = [device_metadata([[i, i + 1, i + 2]] * 9, device="cuda") for i in range(32)]
        torch.cuda.synchronize()
    assert not any(e.key == "cudaStreamSynchronize" and e.count for e in prof.key_averages())
    assert outputs[-1][0].cpu().tolist() == [31, 32, 33]
