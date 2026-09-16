"""CPU checks for budget boundaries and run-level uncertainty calculation."""
import asyncio
import errno
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from run_budget_sweep import BudgetEngine, atomic_json, file_lock, report, retry_disk_full, seed_for


class FakeLLM:
    def __init__(self, samples):
        self.samples = iter(samples)
        self.inputs = []
        self.freed = False
        self.tokenizer = SimpleNamespace(encode=lambda text, **kw: [99, 100])
        self.processor = SimpleNamespace(apply_chat_template=lambda *a, **kw: {"input_ids": [50]})

    async def create_block(self):
        return "block"

    async def free_block(self, block):
        self.freed = True

    async def __call__(self, input_ids, **kwargs):
        self.inputs.append(list(input_ids))
        return SimpleNamespace(logits=torch.tensor([0., 3., 1., 2.]))

    async def sample(self, output):
        return torch.tensor(next(self.samples))


def make_engine(budget, samples):
    engine = BudgetEngine.__new__(BudgetEngine)
    engine.llm = FakeLLM(samples)
    engine.task, engine.budget = "doom", budget
    engine.mode = "reasoning"
    engine.actions, engine.action_ids = ["wait", "fire", "right", "left"], [0, 1, 2, 3]
    engine.context, engine.trace = "test", []
    engine.end_think, engine.eos = 9, {10}
    return engine


class BudgetSweepTests(unittest.TestCase):
    def test_disk_full_retries_without_losing_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.json"
            original = Path.write_text
            attempts = []
            def write(target, data, *args, **kwargs):
                attempts.append(target)
                if len(attempts) == 1:
                    raise OSError(errno.ENOSPC, "No space left on device")
                return original(target, data, *args, **kwargs)
            with patch.object(Path, "write_text", write), patch("run_budget_sweep.time.sleep") as sleep:
                atomic_json(path, {"reward": 42})
                sleep.assert_called_once_with(30)
            self.assertIn('"reward": 42', path.read_text())

    def test_disk_retry_does_not_hide_other_errors(self):
        @retry_disk_full
        def fail():
            raise OSError(errno.EACCES, "Permission denied")
        with self.assertRaises(PermissionError):
            fail()

    def test_four_gpu_campaign_plan(self):
        from run_baseline_campaign import worker_plans
        plans = worker_plans(["0", "1", "3", "4"])
        self.assertEqual(list(plans), ["0", "1", "3", "4"])
        self.assertEqual(plans["0"], ["reasoning", "no_think"])
        self.assertEqual(plans["3"], ["reasoning", "no_think"])
        self.assertEqual(plans["1"], ["no_think", "reasoning"])
        self.assertEqual(list(worker_plans(["5", "6"])), ["5", "6"])
        with self.assertRaises(ValueError):
            worker_plans(["0", "0"])

    def test_episode_claim_excludes_second_worker_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.lock"
            with file_lock(path, blocking=False) as first:
                self.assertTrue(first)
                with file_lock(path, blocking=False) as second:
                    self.assertFalse(second)
            with file_lock(path, blocking=False) as next_worker:
                self.assertTrue(next_worker)

    def test_budget_consumes_last_sample_before_action(self):
        engine = make_engine(2, [7, 8])
        self.assertEqual(asyncio.run(engine.act(None)), "fire")
        self.assertEqual(engine.llm.inputs, [[50], [7], [8], [99, 100]])
        self.assertEqual(engine.trace[0]["reasoning_tokens"], 2)
        self.assertEqual(engine.trace[0]["stop"], "budget")
        self.assertTrue(engine.llm.freed)

    def test_early_end_think_and_eos(self):
        for token, reason in [(9, "end_think"), (10, "eos")]:
            engine = make_engine(100, [7, token])
            asyncio.run(engine.act(None))
            self.assertEqual(engine.llm.inputs, [[50], [7], [99, 100]])
            self.assertEqual(engine.trace[0]["stop"], reason)
            self.assertEqual(engine.trace[0]["reasoning_tokens"], 2)

    def test_no_reasoning_never_samples(self):
        engine = make_engine(0, [])
        asyncio.run(engine.act(None))
        self.assertEqual(engine.trace[0]["reasoning_tokens"], 0)
        self.assertEqual(engine.llm.inputs, [[50], [99, 100]])

    def test_thinking_disabled_generation_runs_until_eos(self):
        engine = make_engine(64, [7, 10])
        engine.mode = "no_think"
        asyncio.run(engine.act(None))
        self.assertEqual(engine.trace[0]["reasoning_tokens"], 0)
        self.assertEqual(engine.trace[0]["generated_tokens"], 2)
        self.assertEqual(engine.trace[0]["stop"], "eos")

    def test_run_level_ci_and_incomplete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            atomic_json(out / "config.json", dict(tasks=["doom"], budgets=[0], episodes=5, runs=10))
            for run in range(10):
                for ep in range(5):
                    atomic_json(out / f"episode_{run}_{ep}.json", dict(
                        task="doom", budget=0, run=run, reward=run+1, error="", steps=1,
                        reasoning_tokens=0, act_seconds=1, budget_hits=0, hit_step_cap=False))
            row = report(out)[0]
            self.assertEqual(row["completed_runs"], 10)
            self.assertEqual(row["mean_reward"], 5.5)
            self.assertAlmostEqual(row["ci95_half_width"], 2.16585059, places=6)
            (out / "episode_9_4.json").unlink()
            self.assertEqual(report(out)[0]["completed_runs"], 9)

    def test_seeds_are_reproducible_and_distinct(self):
        seeds = [seed_for(20260915, task, run, ep)
                 for task in ["doom", "health_gathering"] for run in range(10) for ep in range(5)]
        self.assertEqual(len(set(seeds)), 100)
        self.assertEqual(seeds[0], seed_for(20260915, "doom", 0, 0))


if __name__ == "__main__":
    unittest.main()
