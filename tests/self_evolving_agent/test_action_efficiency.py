
import _self_evolving_bootstrap  # noqa: F401
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from action_efficiency import ForwardCounter, ratio, report, summarize

class EfficiencyTests(unittest.TestCase):
    def test_detailed_evolution_report_reads_selected_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'status.json').write_text(json.dumps({'prompt_variant':'detailed'}))
            out = root/'detailed/doom/run00'
            out.mkdir(parents=True)
            (out/'round_metrics.csv').write_text('round,valid_round,invalid_reason\n1,1,\n')
            (out/'task_results.log').write_text(json.dumps({'step':1, 'llm_forward_calls':40,
                                                           'episodes':[{'steps':2}]*5}))
            row = next(r for r in report(root) if r['condition']=='detailed_round_1' and r['task']=='doom')
            self.assertEqual(row['completed_runs'], 1)
            self.assertEqual(row['mean_forwards_per_env_step'], 4)

    def test_forwards_per_step_averages_run_ratios_and_random_is_zero(self):
        row = summarize('test', 'doom', [(10, 20), (20, 100)])
        self.assertEqual(row['mean_forwards_per_env_step'], 3.5)
        self.assertGreater(row['forwards_per_env_step_ci95_half_width'], 0)
        self.assertNotEqual(row['mean_forwards_per_env_step'], 1 / row['mean_actions_per_forward'])
        self.assertEqual(summarize('random', 'doom', [(10, 0)])['mean_forwards_per_env_step'], 0)
        self.assertIsNone(summarize('empty', 'doom', [(0, 20)])['mean_forwards_per_env_step'])
    def test_counts_prefill_decode_and_cancelled_call(self):
        class LLM:
            async def forward(self, ids):
                if ids is None: await asyncio.sleep(100)
                return ids
        async def check():
            llm=LLM(); counter=ForwardCounter(llm)
            await llm.forward([1,2,3]); await llm.forward([4])
            task=asyncio.create_task(llm.forward(None)); await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            self.assertEqual(counter.calls, 3)
        asyncio.run(check())
    def test_run_ratios_not_pooled_ratio(self):
        row=summarize('test','doom',[(10,10),(10,100)])
        self.assertAlmostEqual(row['mean_actions_per_forward'],.55)
        self.assertEqual(row['env_actions'],20)
        self.assertEqual(row['forward_calls'],110)
        self.assertGreater(row['ci95_half_width'],0)
        self.assertIsNone(ratio(10,0))
    def test_recovers_active_old_logs_and_excludes_invalid_round(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); out=root/'minimal/doom/run00'; out.mkdir(parents=True)
            (out/'round_metrics.csv').write_text('round,valid_round,invalid_reason\n1,,compile_failed\n2,1,\n')
            records=[{'step':step, 'llm_forward_calls':20,'episodes':[{'steps':2}]*5} for step in (1,2)]
            (out/'task_results.log').write_text('\n'.join(json.dumps(r) for r in records)+'\n{"partial":')
            rows=report(root)
            row=next(r for r in rows if r['task']=='doom' and r['condition']=='minimal_round_1')
            self.assertEqual(row['completed_runs'],1)
            self.assertEqual(row['mean_actions_per_forward'],.5)
            self.assertIsNone(row['ci95_half_width'])

if __name__=='__main__': unittest.main()
