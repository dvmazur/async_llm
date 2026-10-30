import tempfile
import unittest
from pathlib import Path

from budget import call, charged
from pipeline import read, save


class BudgetTest(unittest.TestCase):
    def test_concurrent_reservations(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'budget.json'
            save(path, {'limit_usd': 10, 'requests': []})
            def request(i):
                return call(path, {'model':'google/gemini-3.8-flash','max_tokens':4096},
                            lambda: {'id':str(i),'usage':{'cost':0.01}})
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(request, range(20)))
            ledger=read(path)
            self.assertEqual(len(ledger['requests']),20)
            self.assertEqual(len({r['response_id'] for r in ledger['requests']}),20)
            self.assertAlmostEqual(charged(ledger),0.2)

    def test_reservation_blocks_and_success_reconciles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            save(path, {"limit_usd": 0.60, "requests": []})
            payload = {"model": "google/gemini-3.1-flash-image", "max_tokens": 8192,
                       "modalities": ["image", "text"]}
            call(path, payload, lambda: {"id": "test", "usage": {"cost": 0.07}})
            self.assertAlmostEqual(charged(read(path)), 0.07)
            with self.assertRaisesRegex(ValueError, "headroom"):
                call(path, payload, lambda: self.fail("must not send"))

    def test_unknown_failure_retains_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            save(path, {"limit_usd": 2, "requests": []})
            def fail():
                raise TimeoutError("unknown upstream state")
            with self.assertRaises(TimeoutError):
                call(path, {"model": "google/gemini-3.8-flash", "max_tokens": 4096}, fail)
            self.assertAlmostEqual(charged(read(path)), 0.05)
