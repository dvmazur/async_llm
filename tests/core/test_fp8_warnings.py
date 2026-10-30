"""The software FP8 diagnostic must not freeze backend selection."""
import warnings

import pytest

from minisgl.kernel import fp8_format


@pytest.fixture(autouse=True)
def clear_fp8_warning_cache():
    fp8_format._warn_emulated_fp8.cache_clear()
    yield
    fp8_format._warn_emulated_fp8.cache_clear()


def test_software_fp8_warns_once_without_caching_environment_selection(monkeypatch):
    monkeypatch.setattr(fp8_format, '_native_e4m3', lambda _: True)
    monkeypatch.setenv('MINISGL_FP8_EMULATE', '1')
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        for _ in range(3):assert fp8_format.emulate_fp8('cuda:0')
    assert len(records) == 1 and 'MINISGL_FP8_EMULATE=1' in str(records[0].message)
    monkeypatch.delenv('MINISGL_FP8_EMULATE')
    assert not fp8_format.emulate_fp8('cuda:0')
    monkeypatch.setattr(fp8_format, '_native_e4m3', lambda _: False)
    with pytest.warns(RuntimeWarning, match='lacks native E4M3'):
        assert fp8_format.emulate_fp8('cuda:0')
