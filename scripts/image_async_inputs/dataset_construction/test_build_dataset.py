import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from build_dataset import model_job

class ModelCacheTest(unittest.TestCase):
    def test_unknown_request_is_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'response.json'
            with patch('build_dataset.gateway.request', side_effect=TimeoutError('unknown')) as request:
                with self.assertRaises(TimeoutError):
                    model_job(path, 'prompt', [])
                with self.assertRaisesRegex(ValueError, 'outcome unknown'):
                    model_job(path, 'prompt', [])
                self.assertEqual(request.call_count, 1)

    def test_cache_requires_same_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'response.json'
            fake={'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]}
            with patch('build_dataset.gateway.request',return_value=fake) as request:
                self.assertEqual(model_job(path,'first',[]),fake)
                self.assertEqual(model_job(path,'first',[]),fake)
                self.assertEqual(request.call_count,1)
                with self.assertRaisesRegex(ValueError,'changed'):
                    model_job(path,'different',[])
                self.assertEqual(request.call_count,1)
