import asyncio
import csv
from pathlib import Path
import tempfile
import unittest

from run_budget_sweep import atomic_json
from run_random_baseline import RandomEngine, campaign_complete


class RandomBaselineTests(unittest.TestCase):
    def test_reproducible_independent_uniform_choices(self):
        async def sample():
            first = RandomEngine(["wait", "fire", "right", "left"], 123)
            second = RandomEngine(first.actions, 123)
            a = [await first.act(None) for _ in range(10000)]
            b = [await second.act("irrelevant observation") for _ in range(10000)]
            self.assertEqual(a, b)
            for action in first.actions:
                self.assertLess(abs(a.count(action) / len(a) - .25), .02)
        asyncio.run(sample())

    def test_waits_for_every_condition_and_no_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def summary(n, runs=10, errors=0):
                with (root / "summary.csv").open("w") as f:
                    w = csv.writer(f)
                    w.writerow(["completed_runs", "errors"])
                    w.writerows([[runs, errors]] * n)
            self.assertFalse(campaign_complete(root))
            atomic_json(root / "status.json", {"status": "complete"})
            summary(23)
            self.assertFalse(campaign_complete(root))
            summary(24, runs=9)
            self.assertFalse(campaign_complete(root))
            summary(24, errors=1)
            self.assertFalse(campaign_complete(root))
            summary(24)
            self.assertTrue(campaign_complete(root))
            atomic_json(root / "status.json", {"status": "failed"})
            self.assertFalse(campaign_complete(root))


if __name__ == "__main__":
    unittest.main()
