import base64
import io
import tempfile
import unittest

from PIL import Image

from robot_sim.candidate_nav import target_vision as tracker
from robot_sim.candidate_nav.long_term_memory import EpisodeMemory
from robot_sim.tests.test_long_term_memory import motion


class TargetVisionTests(unittest.TestCase):
    instruction = 'Get to the iron trash can, avoiding obstacles visible on camera.'

    def frame(self, index):
        return dict(frame_id=index, stamp=float(index), pose=[0, 0, 0], plan_id=1)

    def test_target_survives_obstacle_sightings_and_stale_analyses(self):
        state = tracker.initial_state(self.instruction)
        visible = tracker.parse_report(
            'TARGET SEEN|ribbed metal cylinder|33,0,66,35\nSCENE Low platform obstructs approach.')
        tracker.apply_report(state, visible, self.frame(2))
        reference_frame = state['target']['last_seen'].copy()
        tracker.apply_report(state, tracker.parse_report('TARGET LOST\nSCENE Low platform.'), self.frame(3))
        for index in range(34, 39):
            report = tracker.parse_report('TARGET LOST\nSCENE A foot and leg in the right part of the image.')
            tracker.apply_report(state, report, self.frame(index))
        tracker.apply_report(state, visible, self.frame(1))
        self.assertEqual(state['target']['status'], 'not_visible')
        self.assertEqual(state['target']['last_seen'], reference_frame)
        self.assertEqual(state['latest_frame']['frame_id'], 38)
        self.assertIn('foot and leg', state['scene'])

    def test_objectless_task_records_scene_without_inventing_target(self):
        state = tracker.initial_state('Leave this maze.')
        report = tracker.parse_report('TARGET NONE\nSCENE An opening is visible on the left.')
        tracker.apply_report(state, report, self.frame(1))
        self.assertEqual(state['target'], dict(status='none', last_seen=None))
        self.assertIn('opening', tracker.describe(state))
        report = tracker.parse_report('TARGET NONE\nSCENE Floor on the left.\nPATH O U B')
        tracker.apply_report(state, report, self.frame(2))
        self.assertEqual(state['scene_directions'], dict(left='O', center='U', right='B'))
        other = tracker.initial_state('Rotate 90 degrees.', state)
        self.assertIsNone(other['latest_frame'])

    def test_malformed_target_keeps_independent_obstacle_evidence(self):
        for line in ('SEEN|leg', 'SEEN|bin|40,10,20,30', 'SEEN|bin|0,0,101,40',
                     'SEEN|bin|10,10,20,20\nTARGET LOST'):
            with self.subTest(line=line):
                report = tracker.parse_report(f'TARGET {line}\nSCENE Low stand in the center.')
                self.assertEqual(report, dict(status='uncertain', scene='Low stand in the center.'))

    def test_reference_is_small_and_uses_only_the_sighting_frame(self):
        state = tracker.initial_state(self.instruction)
        report = tracker.parse_report('TARGET SEEN|silver can|10,10,55,85\nSCENE Low base.')
        tracker.apply_report(state, report, self.frame(2))
        frame = Image.new('RGB', (320, 180), 'blue')
        self.assertIsNone(tracker.remember_reference(None, state, {1: frame}))
        crop = tracker.remember_reference(None, state, {2: frame})
        self.assertEqual(crop['frame_id'], 2)
        image = Image.open(io.BytesIO(base64.b64decode(crop['image'])))
        self.assertLessEqual(max(image.size), 112)
        tracker.apply_report(state, tracker.parse_report('TARGET LOST\nSCENE Leg to the right.'), self.frame(3))
        self.assertIs(tracker.remember_reference(crop, state, {}), crop)

    def test_legacy_sightings_are_not_used_as_confirmed_target_history(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeMemory(directory)
            try:
                packet, reply = motion(1, description='Foot and leg falsely stored as sighting.')
                store.record_reply(packet, reply)
                packet, _ = motion(2, visual_tracking=tracker.MODE)
                report = tracker.parse_report('TARGET LOST\nSCENE Foot and leg.')
                store.record_reply(packet, dict(reports=[dict(frame=packet['frames'][0], **report)]))
                packet, _ = motion(3, visual_tracking=tracker.MODE)
                report = tracker.parse_report('TARGET SEEN|metal bucket|10,10,20,20\nSCENE Low stand.')
                store.record_reply(packet, dict(reports=[dict(frame=packet['frames'][0], **report)]))

                ids = store.identity(packet)
                events = store.selected(packet, ids, 8)
                self.assertEqual([row['report'] for row in events if row['kind'] == 'target_sighting'],
                                 ['metal bucket'])
                self.assertIn('Foot and leg.', [row['report'] for row in events if row['kind'] == 'scene_observation'])
                self.assertTrue(all(row['kind'] != 'sighting' for row in events))
            finally:
                store.close()


class MotionProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_split_motion_prefills_one_image_per_frame_without_scene_history(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        from robot_sim.candidate_nav.motion_vision_policy import inspect

        class Engine:
            def __init__(self):
                self.blocks = []
                self.freed = []
                self.processor = self
                self.messages = []

            def apply_chat_template(self, *args, **kwargs):
                self.messages.append(args[0])
                return {'input_ids': SimpleNamespace(numel=lambda: 100)}

            async def create_block(self):
                block = object()
                self.blocks.append(block)
                return block

            async def free_block(self, block):
                self.freed.append(block)

        llm = Engine()
        policy = SimpleNamespace(llm=llm, rank=1, metadata={},
                                 input_token_limit=lambda _: 10240,
                                 prefill_observation=AsyncMock(return_value=([], None, {})))
        packet, _ = motion(1, visual_tracking=tracker.MODE, map_version=1)
        image = io.BytesIO()
        Image.new('RGB', (32, 32)).save(image, format='PNG')
        packet['frames'][0]['image'] = base64.b64encode(image.getvalue()).decode()
        packet['frames'].append(dict(packet['frames'][0], frame_id=2, stamp=2.))
        packet['visual_state'] = tracker.initial_state(packet['instruction'])
        packet['visual_state']['scene'] = 'Previous unrelated wall'
        with (patch('robot_sim.candidate_nav.motion_vision_policy.task_has_target',
                    new=AsyncMock(return_value=True)),
              patch('robot_sim.candidate_nav.target_vision_policy.read_frame',
                    new=AsyncMock(return_value=dict(status='not_visible', scene='Low stand.')))):
            result = await inspect(policy, packet)
        self.assertEqual(result['reports'][0]['scene'], 'Low stand.')
        self.assertEqual(len(llm.blocks), 2)
        self.assertEqual(llm.freed, llm.blocks)
        for messages in llm.messages:
            self.assertNotIn('Previous unrelated wall', str(messages))
            self.assertEqual(len([item for item in messages[1]['content'] if item['type'] == 'image']), 1)


    async def test_task_scope_is_cached_and_can_reject_a_visual_false_positive(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        from robot_sim.candidate_nav.target_vision_policy import read_frame, task_has_target

        block = object()
        llm = SimpleNamespace(create_block=AsyncMock(return_value=block),
                              forward=AsyncMock(return_value=SimpleNamespace(logits=None)),
                              free_block=AsyncMock())
        policy = SimpleNamespace(llm=llm, encode=lambda text: text,
                                 label_distribution=lambda *_args, **_kw: SimpleNamespace(probabilities=[.10, .02, .03, .85]),
                                 tokens=SimpleNamespace(eos_token_id=0, decode=lambda *_args, **_kw:
                                     'TARGET SEEN|checkerboard|5,5,95,95\nSCENE Checkerboard wall ahead.'))
        instruction = 'I made a maze out of things. Get out.'
        for _ in range(2):
            self.assertFalse(await task_has_target(policy, instruction, str))
        llm.forward.assert_awaited_once()
        llm.free_block.assert_awaited_once_with(block)
        self.assertIn(instruction, llm.forward.await_args.args[0])
        self.assertEqual(llm.forward.await_args.kwargs['cache_view'], [block])

        forward = AsyncMock(return_value=SimpleNamespace(logits=None))
        state = tracker.initial_state(instruction)
        with patch('robot_sim.candidate_nav.target_vision_policy.checked_argmax', side_effect=[1, 0]):
            report = await read_frame(policy, forward, str, state, False)
        tracker.apply_report(state, report, dict(frame_id=4, stamp=4., pose=[0, 0, 0], plan_id=1))
        self.assertEqual(state['target'], dict(status='none', last_seen=None))
        self.assertIn('Checkerboard', state['scene'])
