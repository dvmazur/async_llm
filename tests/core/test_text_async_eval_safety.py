"""Portable CPU coverage for exact-config, fail-fast text evaluation."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip('datasets')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/text_async_inputs'))
import math500_async_thoughts_eval as ev


def arguments(directory, start=0, end=2):
    with patch.object(sys, 'argv', ['evaluator', '--dataset_path', str(directory),
            '--path-to-results', str(directory / 'results'), '--k-steps', '16',
            '--start', str(start), '--end', str(end), '--budget', '16384',
            '--defer-writer-reminder']):
        return ev.arguments()


def test_resource_options_preserve_sampling_defaults(tmp_path):
    a = arguments(tmp_path)
    assert (a.budget, a.max_seq_len, a.kv_tokens, a.max_prefill_rows) == (16384, 65536, 98304, 256)
    assert (a.temperature, a.top_p, a.top_k, a.seed, a.probe_period) == (.6, .95, 20, 42, 30)
    assert a.defer_writer_reminder


def test_save_replaces_atomically_without_leaving_temporary_file(tmp_path):
    target = tmp_path / 'result.json'
    ev.save(target, {'version': 1})
    ev.save(target, {'version': 2})
    assert ev.json.loads(target.read_text()) == {'version': 2}
    assert not target.with_suffix('.json.tmp').exists()


@pytest.mark.parametrize('saved_error', [False, True])
def test_bad_resume_is_rejected_before_model_initialization(tmp_path, saved_error):
    a = arguments(tmp_path)
    folder = Path(a.path_to_results) / 'k_16'
    folder.mkdir(parents=True)
    ev.save(folder / 'config.json', {} if saved_error else {'different': True})
    if saved_error:
        ev.save(folder / 'sample_0.error.json', {'error': 'previous failure'})
    with patch.object(ev, 'load_from_disk', return_value=[{}, {}]), \
         patch.object(ev, 'run_config', return_value={}), patch.object(ev, 'AsyncLLM') as llm:
        with pytest.raises(ValueError, match='Saved sample errors' if saved_error else 'Run configuration changed'):
            asyncio.run(ev.run(a))
        llm.assert_not_called()


def test_failure_stops_at_first_sample_and_closes_model(tmp_path):
    a = arguments(tmp_path)
    llm = SimpleNamespace(close=AsyncMock())
    generate = AsyncMock(side_effect=RuntimeError('deliberate failure'))
    data = [{'problem_shards': ['first', 'second'], 'answer': '2'}] * 2
    with patch.object(ev, 'load_from_disk', return_value=data), \
         patch.object(ev, 'run_config', return_value={}), \
         patch.object(ev.AutoTokenizer, 'from_pretrained', return_value=object()), \
         patch.object(ev, 'AsyncLLM', return_value=llm), patch.object(ev, 'generate', generate):
        with pytest.raises(RuntimeError, match='deliberate failure'):
            asyncio.run(ev.run(a))
    assert generate.await_count == 1
    llm.close.assert_awaited_once()
    folder = Path(a.path_to_results) / 'k_16'
    assert (folder / 'sample_0.error.json').exists()
    assert not (folder / 'summary.json').exists()


def test_range_uses_original_row_indices_and_seeds(tmp_path):
    a = arguments(tmp_path, start=2, end=4)
    data = [{'problem_shards': [f'first{i}', f'second{i}'], 'answer': '2'} for i in range(4)]
    llm = SimpleNamespace(close=AsyncMock())
    generate = AsyncMock(return_value=('\\boxed{2}', 'thoughts', True))
    with patch.object(ev, 'load_from_disk', return_value=data), \
         patch.object(ev, 'run_config', return_value={}), \
         patch.object(ev.AutoTokenizer, 'from_pretrained', return_value=object()), \
         patch.object(ev, 'AsyncLLM', return_value=llm), patch.object(ev, 'generate', generate), \
         patch.object(ev, 'check_equality', return_value=True), \
         patch.object(ev.torch, 'manual_seed') as seed, patch.object(ev.torch.cuda, 'manual_seed_all'):
        asyncio.run(ev.run(a))
    assert [call.args for call in seed.call_args_list] == [(44,), (45,)]
    assert [call.args[2:4] for call in generate.call_args_list] == [('first2', 'second2'), ('first3', 'second3')]
    folder = Path(a.path_to_results) / 'k_16'
    assert ev.json.loads((folder / 'summary.json').read_text())['total'] == 2
    assert ev.json.loads((folder / 'sample_2.json').read_text())['sampling']['seed'] == 44
