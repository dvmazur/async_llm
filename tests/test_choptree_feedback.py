import asyncio
import json

from experiment_runner import EpisodeContext, Recorder
from experiment_runner.logs import read_jsonl
from pipelines.choptree import ACTION_NAMES, FEEDBACK_PROMPT, ChopTreeFeedback, ChopTreeTextFeedback
from pipelines.probe import ProbePipeline, messages_for
from test_choptree import Engine, World


def test_bounded_feedback_and_counter_resets():
    f = ChopTreeFeedback()
    assert json.loads(f.text())['recent'] == []
    for _ in range(1000):
        f.observe('dig', 0)
    x = json.loads(f.text())
    assert len(x['recent']) == 6 and x['consecutive_digs_without_reward'] == 1000
    assert x['actions_since_reward'] == 1000 and x['total_wood_reward'] == 0
    f.observe('dig', 1)
    x = json.loads(f.text())
    assert x['consecutive_digs_without_reward'] == x['actions_since_reward'] == 0
    assert x['total_wood_reward'] == 1
    f.observe('dig', 0)
    f.observe('right', 0)
    x = json.loads(f.text())
    assert x['consecutive_digs_without_reward'] == 0 and x['actions_since_reward'] == 2
    assert len(f.text()) < 500


def test_feedback_is_explicit_text_not_hidden_state():
    messages = messages_for('old', 'new', FEEDBACK_PROMPT, last_action='dig', feedback='test')
    assert len(messages) == 2
    assert messages[1]['content'][-2] == dict(type='text', text='Recent interaction feedback: test')


def test_text_feedback_uses_public_outcomes_and_detects_loops():
    f = ChopTreeTextFeedback()
    assert 'Last actions, oldest first: none.' in f.text()
    for _ in range(4): f.observe('dig',0)
    assert 'STALLED: 4' in f.text()
    f.observe('dig',1)
    assert 'SUCCESS:' in f.text() and 'STALLED:' not in f.text()
    for a in ('right','left','right','left','right','left'): f.observe(a,0)
    assert 'TURNING LOOP:' in f.text()
    f.observe('forward',0)
    assert 'TURNING LOOP:' not in f.text() and 'SUCCESS:' not in f.text()


def test_only_completed_world_steps_update_feedback(tmp_path):
    seen = []
    class FeedbackEngine(Engine):
        async def prefill_action(self, messages, block):
            assert messages[0]['content'] == FEEDBACK_PROMPT
            text = next(x['text'] for x in messages[1]['content']
                        if x['type']=='text' and x['text'].startswith('Recent interaction'))
            seen.append(json.loads(text.split(': ', 1)[1]))
            assert len(self.live)==1 and block.raw.num_tokens == 0
            return object(), 650
    engine = FeedbackEngine()
    world = World(engine)
    feedback = ChopTreeFeedback()
    pipeline = ProbePipeline(world, Recorder(tmp_path), engine,
        context=EpisodeContext('feedback',0,0,tmp_path,0,0,0), max_actions=10,
        action_delay=0, prompt=FEEDBACK_PROMPT, action_names=ACTION_NAMES,
        include_last_action=True, feedback=feedback)
    asyncio.run(pipeline.run())
    assert [x['step'] for x in seen] == [0,1]
    assert seen[1]['recent'] == [dict(action='dig',reward=1.)]
    assert feedback.steps == 2 and feedback.total_reward == 2
    assert not engine.live and world.closed
    policy = next(x for x in read_jsonl(tmp_path/'events.jsonl') if x['kind']=='policy')
    assert policy['feedback_input'] is True
