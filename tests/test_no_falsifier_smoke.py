"""Two-action CPU smoke of the real policy with an in-memory engine and World."""
import asyncio
import json

from experiment_runner.logs import read_jsonl
from test_policy import Engine, World, pipeline


def test_two_actions_without_falsifier(tmp_path):
    engine, world = Engine(), World()
    p = pipeline(tmp_path, engine, world, max_actions=2)
    asyncio.run(p.run())

    events = list(read_jsonl(tmp_path/'events.jsonl'))
    streams = [event for event in events if event['kind'] == 'stream']
    assert [(s['observation'], s['role'], s['sampled_tokens']) for s in streams] == [
        (0, 'observer', 18), (0, 'planner', 60), (0, 'executor', 16),
        (1, 'observer', 18), (1, 'executor', 16),
    ]
    assert {e['role'] for e in events if e['kind'] == 'role_parameters'} == {
        'observer', 'planner', 'executor'}
    prompts = [engine.common.raw.data[0][1]]
    prompts += [c[1] for c in engine.calls if c[0] == 'text']
    assert all('falsifier' not in text.lower() and 'objection' not in text.lower()
               for text in prompts)
    decisions = [e for e in events if e['kind'] == 'decision']
    assert len(decisions) == 2 and all('critique' not in e for e in decisions)
    assert [e['published_plan_based_on'] for e in decisions] == [0, 0]
    assert world.i == 2 and world.closed
    assert p.history is p.common is None
    assert list(engine.live.values()) == [engine.common.raw]
    assert json.loads((tmp_path/'completion.json').read_text())['status'] == 'completed'
