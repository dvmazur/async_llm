import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from report_fixed50 import report, KS


class HandoffReportTests(unittest.TestCase):
    def fixture(self,root):
        ids=[str(i) for i in range(50)]
        manifest={'sample_ids':ids,'count':50,'dataset_sha256':'dataset'}
        path=root/'manifest.json'; path.write_text(json.dumps(manifest))
        for k in KS:
            folder=root/f'k_{k}'; folder.mkdir()
            (folder/'summary.json').write_text(json.dumps(
                {'complete':True,'completed':50,'requested':50,'correct_after':50}))
            (folder/'config.json').write_text(json.dumps(
                {'k_steps':k,'sample_manifest':manifest,'dataset_sha256':'dataset','output':str(folder),
                 'cuda_visible_devices':'1' if k<64 else '6',
                 'distributed_port':2371 if k<64 else 2376}))
            for i in range(50):
                (folder/f'{i}.json').write_text(json.dumps(
                    {'id':str(i),'sample_seed':i,'correct_after':True,'matches_before':False,
                     'image_replaced':k>0,'predicted_answer':'2','elapsed_seconds':1,'dataset':'fixture'}))
        return path

    def test_complete_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); manifest=self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()): report(root,manifest)
            self.assertTrue(json.loads((root/'results.json').read_text())['complete'])
            self.assertIn('50/50',(root/'RESULTS.md').read_text())

    def test_incomplete_condition_blocks_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); manifest=self.fixture(root)
            (root/'k_512/summary.json').write_text(json.dumps({'complete':False}))
            with self.assertRaisesRegex(ValueError,'Incomplete'): report(root,manifest)
            self.assertFalse((root/'results.json').exists())


if __name__=='__main__': unittest.main()
