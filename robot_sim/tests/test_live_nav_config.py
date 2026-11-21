"""Support the two existing rover image layouts without changing native files."""
from pathlib import Path
import tempfile
import unittest
from robot_sim.candidate_nav.live_nav_config import native_config

class NativeConfigPaths(unittest.TestCase):
    def test_rover80_layout_keeps_priority(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for section in ('launch','config'):
                (root/section).mkdir();(root/section/'params.yaml').write_text('{}')
            self.assertEqual(native_config(root),root/'launch/params.yaml')
    def test_rover107_layout_is_supported(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'config').mkdir();p=root/'config/params.yaml';p.write_text('{}')
            self.assertEqual(native_config(root),p)
    def test_missing_config_fails_with_the_checked_paths(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError,'launch/params.yaml.*config/params.yaml'):
                native_config(Path(d))

if __name__=='__main__':unittest.main()
