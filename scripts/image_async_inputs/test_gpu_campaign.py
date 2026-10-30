import tempfile
import unittest
from pathlib import Path

import run_gpu_campaign as campaign


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.row = {'id': 'sample', 'dataset': 'source', 'category': 'category',
                    'difficulty_group': 'difficulty',
                    'labels': {'answer_before': '1', 'answer_after': '2', 'choices': []}}
        self.config = {'budget': 16384, 'cuda_visible_devices': '1'}

    def fixture(self, k):
        folder = self.root / f'k_{k}'
        folder.mkdir(exist_ok=True)
        result = {'id': 'sample', 'idx': 7, 'subset_position': 0, 'sample_seed': 49,
                  'dataset': 'source', 'category': 'category', 'difficulty_group': 'difficulty',
                  'answer_before': '1', 'answer_after': '2', 'predicted_answer': '2',
                  'generated_text': r'\boxed{2}', 'correct_after': True, 'matches_before': False,
                  'token_ids': {'thinker': [2] * (k + 1 if k > 0 else 2), 'writer': [99]},
                  'hit_eos': True, 'elapsed_seconds': 1.0, 'image_tokens': 4,
                  'events': [{'event': 'image_replaced', 'thinker_tokens': k}] if k > 0 else [],
                  'image_replaced': k > 0, 'final_image': 'before' if k == -1 else 'after',
                  'writer_reminder_pending': False,
                  'probe_stats': {'calls': 0, 'suffix_tokens': 2, 'input_tokens': 0,
                                  'pending_tokens': 0, 'elapsed_seconds': 0.0}}
        campaign.ev.save(folder / 'config.json', self.config)
        campaign.ev.save(folder / 'sample.json', result)
        campaign.ev.save(folder / 'summary.json', campaign.ev.summarize([result], 1))
        return folder, result

    def validate(self, k, **kwargs):
        return campaign.validate_condition(self.root, k, [self.row], {'sample': 7},
                                           self.config, complete=True, **kwargs)

    def test_all_conditions_and_full_report(self):
        for k in campaign.KS:
            self.fixture(k)
            self.validate(k, rescore=True)
        campaign.report(self.root, [self.row], {'sample': 7},
                        {k: self.config for k in campaign.KS})
        report = campaign.read(self.root / 'validated_results.json')
        self.assertEqual(report['completed'], 8)
        self.assertIn('no-update', report['interpretation'])
        self.assertTrue((self.root / 'VALIDATED_REPORT.md').exists())

    def test_seed_configuration_errors_and_missing_samples_block(self):
        for kind in ('seed', 'config', 'error', 'missing', 'summary', 'score'):
            folder, result = self.fixture(16)
            with self.subTest(kind=kind):
                if kind == 'seed':
                    result['sample_seed'] = 42
                elif kind == 'config':
                    campaign.ev.save(folder / 'config.json', {**self.config, 'budget': 2048})
                elif kind == 'error':
                    campaign.ev.save(folder / 'sample.error.json', {'error': 'failure'})
                elif kind == 'summary':
                    campaign.ev.save(folder / 'summary.json', {'complete': True})
                elif kind == 'score':
                    result['correct_after'] = False
                if kind == 'missing':
                    (folder / 'sample.json').unlink()
                else:
                    campaign.ev.save(folder / 'sample.json', result)
                with self.assertRaises(ValueError):
                    self.validate(16, rescore=True)
                if kind == 'error':
                    (folder / 'sample.error.json').unlink()

    def test_early_eos_before_update_is_accepted(self):
        folder, result = self.fixture(512)
        result.update(events=[], image_replaced=False, final_image='before')
        result['token_ids']['thinker'] = [2] * 50
        campaign.ev.save(folder / 'sample.json', result)
        campaign.ev.save(folder / 'summary.json', campaign.ev.summarize([result], 1))
        self.validate(512)
        self.assertEqual(campaign.metrics([result], 512)['writer_ended_before_update'], 1)

    def test_missing_late_update_is_rejected(self):
        folder, result = self.fixture(16)
        result.update(events=[], image_replaced=False, final_image='before')
        campaign.ev.save(folder / 'sample.json', result)
        with self.assertRaisesRegex(ValueError, 'Missing required replacement'):
            self.validate(16)

    def test_probe_history_replay_accounting_is_rejected(self):
        folder, result = self.fixture(16)
        result['events'].append({'event': 'probe', 'thinker_tokens': 10, 'writer_tokens': 0})
        result['probe_stats'].update(calls=1, input_tokens=1000)
        campaign.ev.save(folder / 'sample.json', result)
        with self.assertRaisesRegex(ValueError, 'Probe replay'):
            self.validate(16)


if __name__ == '__main__':
    unittest.main()
