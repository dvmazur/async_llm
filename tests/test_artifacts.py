import json
import zipfile
import pytest

from experiment_runner.artifacts import pack_results
from experiment_runner.logs import atomic
from tools.plot_height import aggregate


def test_pack_checks_completeness_and_does_not_delete(tmp_path):
    source = tmp_path/'run'
    source.mkdir()
    atomic(source/'status.json', {'status': 'completed'})
    atomic(source/'run.json', {'gpus': [0], 'concurrency': 1, 'repeats': 1})
    archive = tmp_path/'result.zip'
    with pytest.raises(ValueError, match='missing episode'):
        pack_results(source, archive)
    atomic(source/'gpu-000/slot-000/repeat-000/completion.json', {'status': 'completed'})
    digest = pack_results(source, archive)
    assert len(digest) == 64
    assert (source/'run.json').exists()
    with zipfile.ZipFile(archive) as z:
        assert 'run/checksums.json' in z.namelist()
        assert z.testzip() is None
    with pytest.raises(FileExistsError):
        pack_results(source, archive)


def test_partial_and_unknown_files_require_attention(tmp_path):
    source = tmp_path/'run'
    source.mkdir()
    atomic(source/'status.json', {'status': 'failed'})
    with pytest.raises(ValueError):
        pack_results(source, tmp_path/'a.zip')
    (source/'key.pem').write_text('do not pack')
    with pytest.raises(ValueError, match='unexpected'):
        pack_results(source, tmp_path/'a.zip', allow_partial=True)


def test_height_does_not_fill_missing_with_zero():
    episodes = [dict(status='completed', heights=[dict(step=0, height=10), dict(step=1, height=8)]),
                dict(status='completed', heights=[dict(step=0, height=4)])]
    assert aggregate(episodes) == [dict(step=0, mean=7, count=2), dict(step=1, mean=8, count=1)]
    assert aggregate(episodes, descent=True)[1] == dict(step=1, mean=2, count=1)
