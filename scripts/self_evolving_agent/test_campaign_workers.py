from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import run_async_campaign as campaign

class WorkerTests(unittest.TestCase):
    def test_three_workers_claim_each_run_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); calls=[]; lock=threading.Lock()
            def command(cmd, env, log):
                out=Path(env['SEA_LOG_DIR'])
                with lock: calls.append(str(out))
                (out/'completion.json').write_text(json.dumps({'complete':True}))
            campaign.STOP.clear()
            with patch.object(campaign,'command',command):
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures=[pool.submit(campaign.evolution_worker,root,gpu,{}) for gpu in ['5','6','1']]
                    for future in futures: future.result()
            self.assertEqual(len(calls),20)
            self.assertEqual(len(set(calls)),20)
    def test_baselines_only_skips_evolution_and_finishes_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            campaign.STOP.clear()
            with patch.object(campaign, 'evolution_worker') as evolution, \
                 patch.object(campaign, 'baseline_worker') as baseline, \
                 patch.object(campaign, 'command') as command, \
                 patch.object(campaign, 'report', return_value=[{'completed_runs':10,'errors':0}]), \
                 patch('action_efficiency.report') as efficiency, \
                 patch.object(campaign.time, 'sleep'):
                campaign.main(root, ['0','1','3','4'], {}, baselines_only=True)
                evolution.assert_not_called()
                self.assertEqual(baseline.call_count, 3)
                command.assert_called_once()
                efficiency.assert_called()
            self.assertEqual(json.loads((root/'status.json').read_text())['status'], 'complete')

    def test_evolution_only_does_not_queue_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            campaign.STOP.clear()
            with patch.object(campaign, 'evolution_worker') as evolution, \
                 patch.object(campaign, 'baseline_worker') as baseline, \
                 patch.object(campaign, 'evolution_report'), \
                 patch('action_efficiency.report'), patch.object(campaign.time, 'sleep'):
                campaign.main(root, ['1','3','6'], {}, evolution_only=True)
                self.assertEqual(evolution.call_count, 3)
                baseline.assert_not_called()
            state=json.loads((root/'status.json').read_text())
            self.assertEqual(state['phase'], 'minimal_complete')
            self.assertEqual(state['status'], 'complete')

    def test_adoption_waits_without_relaunching_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for task in ['doom','health_gathering']:
                for run in range(10):
                    out=root/'minimal'/task/f'run{run:02}'; out.mkdir(parents=True)
                    (out/'completion.json').write_text(json.dumps({'complete':True}))
            target=root/'minimal/doom/run00/completion.json'; target.unlink()
            def running(pid):
                self.assertEqual(pid,12345)
                target.write_text(json.dumps({'complete':True}))
                return False
            campaign.STOP.clear()
            with patch.object(campaign,'process_running',running), patch.object(campaign,'command') as command:
                campaign.evolution_worker(root,'5',{'doom/0':{'pid':12345,'gpu':'5'}})
                command.assert_not_called()

if __name__=='__main__': unittest.main()
