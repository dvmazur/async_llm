import json
import pytest
from experiment_runner.summary import Summary, batch_stats
from experiment_runner.logs import atomic, JsonlWriter


def test_batch_means_are_not_padded_or_token_weighted():
    rows = [dict(phase='decode', decode_requests=b, prefill_requests=0, prefill_rows=0, padded_capacity=64)
            for b in (2, 6)]
    rows += [dict(phase='prefill', decode_requests=0, prefill_requests=b, prefill_rows=t, padded_capacity=4096)
             for b, t in ((1, 20), (3, 180))]
    result = batch_stats(rows)
    assert result['decode']['mean_decode_requests'] == 4
    assert result['prefill']['mean_prefill_requests'] == 2
    assert result['prefill']['mean_prefill_rows'] == 100
    assert result['mixed']['mean_decode_requests'] is None


@pytest.mark.parametrize('legacy_readout_events', [False, True])
@pytest.mark.parametrize('counter_name', ['action_readouts', 'restricted_readouts'])
def test_overlapping_episodes_global_denominator(tmp_path, legacy_readout_events, counter_name):
    gpu = tmp_path/'gpu-000'
    gpu.mkdir()
    atomic(gpu/'status.json', {'status': 'completed'})
    generation = JsonlWriter(gpu/'generation.jsonl')
    for i, (start, end, tokens) in enumerate(((10., 20., 100), (15., 30., 200))):
        ep = gpu/f'slot-{i:03}/repeat-000'
        ep.mkdir(parents=True)
        atomic(ep/'context.json', dict(episode_id=str(i), model_seed=i, world_seed=i))
        atomic(ep/'completion.json', dict(status='completed', workload_start=start, workload_end=end))
        events, steps = JsonlWriter(ep/'events.jsonl'), JsonlWriter(ep/'steps.jsonl')
        events.emit(kind='stream', sampled_tokens=tokens)
        events.emit(kind='decision', action='wait')
        if legacy_readout_events:
            generation.emit(kind='action_readout', episode_id=str(i), action='wait')
        steps.emit(kind='observation', step=0, height=2.)
        steps.emit(kind='action', step=1, height=-3., reward=1.)
        events.close(); steps.close()
        for _ in range(tokens):
            generation.emit(kind='sample', episode_id=str(i))
    generation.close()
    forwards = JsonlWriter(gpu/'forwards.jsonl')
    for time in (12., 18., 25.):
        forwards.emit(monotonic=time, phase='decode', decode_requests=2, prefill_requests=0, prefill_rows=0)
    forwards.close()
    atomic(gpu/'engine-totals.json', {counter_name: 2})
    s = Summary(tmp_path).write()
    assert s['generated_tps'] == 15.  # 300 / (30-10), not 300/(10+15)
    assert s['tokens_per_action'] == 150.
    assert s['batches']['decode']['mean_decode_requests'] == 2.
    assert s['episodes'][0]['heights'][0] == dict(step=0, height=2.)
    assert s['workers'][0]['action_readouts'] == 2
    assert s['workers'][0]['output_tokens_including_readouts_tps'] == 302/20
    assert all(e['action_readouts'] == 1 for e in s['episodes'])
    assert all(e['total_reward'] == 1. for e in s['episodes'])
    assert Summary(tmp_path).compute() == s
    atomic(gpu/'engine-totals.json', {counter_name: 999})
    with pytest.raises(ValueError, match='action readout accounting mismatch'):
        Summary(tmp_path).compute()


def test_generated_action_is_not_counted_again_as_a_readout(tmp_path):
    gpu = tmp_path/'gpu-000'
    ep = gpu/'slot-000/repeat-000'
    ep.mkdir(parents=True)
    atomic(gpu/'status.json', dict(status='completed'))
    atomic(gpu/'engine-totals.json', dict(restricted_readouts=0, sampled_tokens=3))
    atomic(ep/'context.json', dict(episode_id='one', model_seed=0, world_seed=0))
    atomic(ep/'completion.json', dict(status='completed', workload_start=0., workload_end=2.))
    events = JsonlWriter(ep/'events.jsonl')
    events.emit(kind='assessment', output_tokens=3)
    events.emit(kind='decision', mode='generated', action='dig')
    events.close()
    steps = JsonlWriter(ep/'steps.jsonl')
    steps.emit(kind='action', step=1, reward=1.)
    steps.close()
    generation = JsonlWriter(gpu/'generation.jsonl')
    for _ in range(3): generation.emit(kind='sample')
    generation.close()
    report = Summary(tmp_path).compute()
    assert report['generated_tokens'] == 3 and report['tokens_per_action'] == 3
    assert report['workers'][0]['action_readouts'] == 0
    assert report['workers'][0]['output_tokens_including_readouts_tps'] == 1.5
