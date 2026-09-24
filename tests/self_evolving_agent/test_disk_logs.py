
import _self_evolving_bootstrap  # noqa: F401
import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_budget_sweep import append_text


class DiskLogTests(unittest.TestCase):
    def test_disk_full_preserves_previous_record_then_retries_once(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'task_results.log'
            target.write_text('previous\n')
            original = Path.write_text
            attempts = []
            def write(path, text, *args, **kwargs):
                attempts.append(text)
                if len(attempts) == 1:
                    original(path, 'partial')
                    self.assertEqual(target.read_text(), 'previous\n')
                    raise OSError(errno.ENOSPC, 'disk full')
                return original(path, text, *args, **kwargs)
            with patch.object(Path, 'write_text', write), patch('run_budget_sweep.time.sleep') as sleep:
                append_text(target, 'next\n')
            self.assertEqual(target.read_text(), 'previous\nnext\n')
            self.assertEqual(len(attempts), 2)
            sleep.assert_called_once_with(30)
