import argparse
import hashlib
import json
from pathlib import Path

import pytest

from minisgl_examples.training.parallel_thinking import budget_curve_sweep as original
from minisgl_examples.training.parallel_thinking import budget_short_sweep as short


def args(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'short'
    source.mkdir()
    output.mkdir()
    return argparse.Namespace(source_root=source, output=output, gpus=[0, 2], indices=list(range(500)),
                              budgets=list(short.BUDGETS), conditions=list(original.CONDITIONS),
                              wait_for_source=False, max_prefill_rows=256, memory_ratio=0.75)


def case(key):
    mode, workers, index, budget = key
    trace = 'a' * 64
    context = hashlib.sha256(str(budget).encode()).hexdigest()
    return dict(mode=mode, workers=workers, index=index, budget=budget, unique_id=f'id-{index}', gold='3',
                prediction='3', answer_token_ids=[1], answer_text='3}', stop_reason='answer', correct=True,
                grading_status='correct', generated_tokens=11, answer_tokens=1, reasoning_tokens=10,
                effective_rounds=10, prompt_tokens=4, context_tokens=16, context_sha256=context,
                view_sha256='c' * 64, cue_token_ids=[1, 2], publication_count=0, source_trace_sha256=trace,
                source_trace_path=f'/original/{mode}-{workers}-{index}', wall_seconds=0.1,
                reused_readout=False, reused_from_budget=None)


def serialized(rows):
    return ''.join(json.dumps(row) + '\n' for row in rows)


def catalog_for(rows):
    return {(r['mode'], r['workers'], r['index']):
            (r['unique_id'], r['gold'], r['source_trace_sha256'], r['source_trace_path']) for r in rows}


def test_runtime_identity_and_command(tmp_path, monkeypatch):
    opts = args(tmp_path)
    snapshot = opts.source_root / 'report/tasks.csv'
    snapshot.parent.mkdir()
    snapshot.write_text('pinned')
    monkeypatch.setattr(original, 'identity', lambda _: {'code_sha256': {}, 'source_root': str(opts.source_root),
                        'active_source_manifest_sha256': 'a' * 64, 'source_manifest_sha256': 'b' * 64,
                        'source_tasks_sha256': short.sha256(snapshot)})
    config = short.configuration(opts)
    assert config['format'] == short.FORMAT and config['max_context'] == 136192
    assert sorted(Path(key).name for key in config['code_sha256']) == ['budget_short_report.py', 'budget_short_sweep.py']
    for filename, digest in config['code_sha256'].items():
        assert short.sha256(original.PROJECT / filename) == digest
    cmd = short.worker_command(opts, {'modulo': 2, 'remainder': 0})
    assert cmd[:2] == [str(original.LAUNCHER), '--worker']
    assert cmd[cmd.index('--budgets') + 1:cmd.index('--conditions')] == ['256', '512', '1024']
    assert cmd[-2:] == ['--max-context', '136192']
    snapshot.write_text('changed')
    assert short.configuration(opts) != config


def test_old_plan_rejected_before_any_write(tmp_path):
    opts = args(tmp_path)
    plan = opts.output / 'plan.json'
    plan.write_text(json.dumps({'config': {'format': 'budget-readout-fixed-lanes-v1'}}))
    for output in (opts.output, opts.output / 'nested', opts.source_root / 'nested'):
        with pytest.raises(SystemExit):
            short.main(['--source-root', str(opts.source_root), '--output', str(output), '--gpus', '2'])
    assert list(opts.output.iterdir()) == [plan] and not list(opts.source_root.iterdir())


@pytest.mark.parametrize('change,match', [({'gold': '4'}, 'source case'), ({'context_tokens': 20}, 'metrics'),
                         ({'budget': 1000}, 'short budget'), ({'reused_readout': True}, 'reuse'),
                         ({'reasoning_tokens': None}, 'reasoning_tokens')])
def test_rejects_incompatible_rows(tmp_path, change, match):
    key = ('native', 1, 0, 256)
    row = case(key)
    catalog = catalog_for([row]); row.update(change)
    p = tmp_path / 'rows.jsonl'; p.write_text(serialized([row]))
    with pytest.raises(ValueError, match=match):
        short.read_rows(p, catalog)


def test_schema_and_semantic_reuse(tmp_path):
    row = case(('native', 1, 0, 256)); catalog = catalog_for([row]); p = tmp_path / 'rows.jsonl'
    bad = row.copy(); del bad['unique_id']; p.write_text(serialized([bad]))
    with pytest.raises(ValueError, match='fields'):
        short.read_rows(p, catalog)
    reused = dict(row, budget=512, reused_readout=True, reused_from_budget=256)
    p.write_text(serialized([row, reused]) + '{"incomplete":')
    assert len(short.read_rows(p, catalog)[0]) == 2
    reused['answer_token_ids'] = [2]; p.write_text(serialized([row, reused]))
    with pytest.raises(ValueError, match='reuse'):
        short.read_rows(p, catalog)


def test_all_10500_rows_complete_without_gpu_and_frozen_namespace_is_untouched(tmp_path, monkeypatch):
    opts = args(tmp_path)
    config = vars(opts) | {'source_root': str(opts.source_root), 'output': str(opts.output), 'format': short.FORMAT}
    group = [(mode, workers, index, budget) for mode, workers in original.CONDITIONS.values()
             for index in opts.indices for budget in short.BUDGETS]
    catalog = catalog_for(case(key) for key in group)
    runner = short.coordinator(catalog)
    assert runner.report.BUDGETS == short.BUDGETS and original.report.BUDGETS == tuple(range(1000, 8001, 1000))
    monkeypatch.setattr(runner, 'configuration', lambda _: config)
    state = lambda index: dict(gpu=index, uuid=f'GPU-{index}', free=index == 2)
    monkeypatch.setattr(runner, 'gpu_state', state)
    plan = runner.load_plan(opts.output, config, opts.gpus, [state(gpu) for gpu in opts.gpus])
    assert [lane['gpu'] for lane in plan['lanes']] == [2]
    lane = runner.lane_path(opts.output, plan['lanes'][0]); lane.mkdir(parents=True)
    (lane / 'results.jsonl').write_text(serialized(case(key) for key in group))
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **kw: pytest.fail('A completed resume cannot launch a worker'))
    assert runner.run(opts) == 0
    final = json.loads((opts.output / 'complete.json').read_text())
    assert final['completed'] == final['target'] == 10500
    summary = json.loads((opts.output / 'report/summary.json').read_text())
    assert len(summary['cells']) == 21 and all(cell['tasks'] == 500 for cell in summary['cells'])
