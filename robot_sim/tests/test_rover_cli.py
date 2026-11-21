"""Verify target routing through recursive startup, without SSH or motors."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/rover-qwen.sh'

class RoverRoutingTests(unittest.TestCase):
    def invoke(self,args,rover=107,fail=None):
        with tempfile.TemporaryDirectory(prefix='rover-cli-test-') as folder:
            root=Path(folder);bin_dir=root/'bin';bin_dir.mkdir();log=root/'calls.jsonl'
            ssh=bin_dir/'ssh'
            ssh.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
with open(os.environ['ROVER_CLI_CALLS'],'a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
cmd=sys.argv[-1]
if os.environ.get('ROVER_CLI_FAIL')=='retry_handshake' and 'h100' in sys.argv:
 ready=Path(os.environ['ROVER_CLI_CALLS']).with_suffix('.ready')
 if '-O' in sys.argv and 'check' in sys.argv:sys.exit(0 if ready.exists() else 1)
 if '-fNT' in sys.argv and '-L' not in sys.argv:
  attempts=sum('-fNT' in json.loads(x) and '-L' not in json.loads(x) for x in Path(os.environ['ROVER_CLI_CALLS']).read_text().splitlines())
  if attempts==1:print('Connection timed out during banner exchange',file=sys.stderr);sys.exit(255)
  ready.touch()
if 'h100'  in sys.argv and '-O' not in sys.argv:
 failure=os.environ.get('ROVER_CLI_FAIL')
 if failure=='banner' or (failure=='tunnel' and '-fNT' in sys.argv):
  print('Connection timed out during banner exchange',file=sys.stderr);sys.exit(255)
if 'rover_model_lease.py' in cmd:
 print('MODEL LEASE READY',flush=True);sys.stdin.read();sys.exit(0)
if 'rover_live_bundle.sh' in cmd:
 if os.environ.get('ROVER_CLI_FAIL')=='archive':print('not an archive');sys.exit(0)
 import tarfile
 with tarfile.open(fileobj=sys.stdout.buffer,mode='w|'):pass
elif 'docker inspect ros --format' in cmd and 'docker start' not in cmd:
 print('172.17.0.2')
elif 'cat robot_sim/candidate_nav/live_link.py' in cmd:
 print("import argparse,pathlib,time;p=argparse.ArgumentParser();p.add_argument('--ready-file');p.add_argument('--token');a=p.parse_args();pathlib.Path(a.ready_file).write_text('12345');time.sleep(20)")
''');ssh.chmod(0o755)
            curl=bin_dir/'curl';curl.write_text('#!/usr/bin/env python3\nimport os,json\nprint(json.dumps({"rover_id":int(os.environ["EXPECTED_ROVER"])}))\n');curl.chmod(0o755)
            env={**os.environ,'PATH':str(bin_dir)+os.pathsep+os.environ['PATH'],
                 'ROVER_CLI_FAIL':fail or '', 'ROVER_CLI_CALLS':str(log),'EXPECTED_ROVER':str(rover),'TMPDIR':str(root)}
            result=subprocess.run(['bash',str(SCRIPT),*args],env=env,text=True,errors="replace",capture_output=True,timeout=15)
            calls=[json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
            return result,calls
    def test_start_propagates_postfix_rover_through_deploy_view_and_run(self):
        result,calls=self.invoke(['start','--rover','107','--strategy','3','--task','Find a box.','--dry-run'])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('18771',result.stdout)
        self.assertFalse(any('rover' in row for row in calls))
        rover_calls=[row for row in calls if 'rover107' in row]
        self.assertGreater(len(rover_calls),4)
        self.assertTrue(all("rbm-107" in row[-1] or '-O' in row or '-L' in row for row in rover_calls))
        run=next(row[-1] for row in rover_calls if '--instruction' in row[-1])
        self.assertIn('/opt/rover-live-107/',run)
        self.assertIn('--rover-id 107',run)
        self.assertIn('--validate-only',run)
        self.assertNotIn('--enable-motion',run)
    def test_prefix_selector_and_task_literal_are_preserved(self):
        result,calls=self.invoke(['--rover','107','start','--strategy','1','--task','--rover','--dry-run'])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('task: --rover',result.stdout)
        self.assertFalse(any('rover' in row for row in calls))
    def test_stop_targets_only_selected_install(self):
        result,calls=self.invoke(['stop','--rover','107'])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(len(calls),1)
        self.assertIn('rover107',calls[0])
        self.assertIn('/opt/rover-live-107/',calls[0][-1])
    def test_default_is_rover80(self):
        result,calls=self.invoke(['stop'],rover=80)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('rover',calls[0])
        self.assertIn('/opt/rover-live/',calls[0][-1])
    def test_invalid_target_never_reaches_ssh(self):
        result,calls=self.invoke(['start','--rover','108','--task','Stop','--dry-run'])
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(calls,[])

    def test_strategy_four_propagates_horizon_through_start_and_run(self):
        result,calls=self.invoke(['start','--rover','107','--strategy','4','--trajectory-steps','7',
                                 '--task','Find a box.','--dry-run'])
        self.assertEqual(result.returncode,0,result.stderr)
        run=next(row[-1] for row in calls if '--instruction' in row[-1])
        self.assertIn('--strategy 4',run)
        self.assertIn('--trajectory-steps 7',run)
        self.assertIn('--validate-only',run)
        self.assertNotIn('--enable-motion',run)

    def test_strategy_four_has_default_horizon(self):
        result,calls=self.invoke(['start','--strategy','4','--task','Stop','--dry-run'],rover=80)
        self.assertEqual(result.returncode,0,result.stderr)
        run=next(row[-1] for row in calls if '--instruction' in row[-1])
        self.assertIn('--trajectory-steps 5',run)

    def test_invalid_horizon_is_rejected_before_any_ssh(self):
        for mode in ('start','run'):
            for value in ('0','33','-1','3.0','02','x'):
                with self.subTest(mode=mode,value=value):
                    args=[mode,'--strategy','4','--trajectory-steps',value,
                          '--task' if mode=='start' else '--instruction','Stop']
                    if mode=='start':args.append('--dry-run')
                    result,calls=self.invoke(args)
                    self.assertNotEqual(result.returncode,0)
                    self.assertEqual(calls,[])

    def test_horizon_is_exclusive_to_strategy_four(self):
        for strategy in ('1','2','3'):
            result,calls=self.invoke(['start','--strategy',strategy,'--trajectory-steps','5',
                                     '--task','Stop','--dry-run'])
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(calls,[])

    def test_h100_profile_is_kept_through_recursive_startup(self):
        result,calls=self.invoke(['start','--server','h100','--gpu','1','--rover','107',
                                 '--strategy','4','--task','Find a box.','--dry-run'])
        self.assertEqual(result.returncode,0,result.stderr)
        servers=[row for row in calls if 'h100' in row]
        self.assertGreater(len(servers),3)
        self.assertFalse(any('5.187.77.37' in row for row in calls))
        service=next(row[-1] for row in servers if 'rover_live_model_service.py' in row[-1])
        self.assertIn('--gpu 1',service)
        self.assertTrue(all('/home/double/' not in row[-1] for row in servers))
    def test_invalid_server_gpu_never_reaches_ssh(self):
        for args in (['--server','oops'],['--server','h100','--gpu','0;id'],['--gpu','0']):
            result,calls=self.invoke(['stop',*args])
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(calls,[])

    def test_server_transport_reused_and_only_owner_closes_it(self):
        result,calls=self.invoke(['start','--server','h100','--rover','107','--strategy','4','--task','Stop','--dry-run'])
        self.assertEqual(result.returncode,0,result.stderr)
        server=[row for row in calls if 'h100' in row]
        paths={row[row.index('-S')+1] for row in server}
        self.assertEqual(len(paths),1)
        self.assertEqual(sum('-O' in row and 'exit' in row for row in server),1)
        self.assertTrue(all('ControlMaster=auto' in row and 'IPQoS=none' in row for row in server))
        self.assertFalse(Path(next(iter(paths))).parent.exists())
    def test_archive_or_banner_failure_never_installs_or_starts_agent(self):
        for failure in ('archive','banner'):
            result,calls=self.invoke(['start','--server','h100','--rover','107','--strategy','4','--task','Stop','--dry-run'],fail=failure)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Установка на ровер не начата',result.stderr)
            self.assertFalse(any('rover-live-stage' in row[-1] or '--instruction' in row[-1] for row in calls))
    def test_tunnel_failure_never_starts_rover_agent(self):
        result,calls=self.invoke(['start','--server','h100','--rover','107','--strategy','4','--task','Stop','--dry-run'],fail='tunnel')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(any('--instruction' in row[-1] for row in calls))
        self.assertEqual(sum('h100' in row and '-O' in row and 'exit' in row for row in calls),1)

    def test_retries_handshake_but_does_not_repeat_remote_commands(self):
        result,calls=self.invoke(['start','--server','h100','--rover','107','--strategy','4','--task','Stop','--dry-run'],fail='retry_handshake')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(sum('-fNT' in row and '-L' not in row for row in calls),2)
        self.assertEqual(sum('rover_live_bundle.sh' in row[-1] for row in calls),1)
        self.assertEqual(sum('--instruction' in row[-1] for row in calls),1)

    def test_model_selection_reaches_service_and_rover(self):
        for model in ('27b','35b','122b'):
            result,calls=self.invoke(['start','--server','h100','--rover','107','--strategy','4','--model',model,'--task','Stop','--dry-run'])
            self.assertEqual(result.returncode,0,result.stderr)
            service=next(row[-1] for row in calls if 'rover_live_model_service.py' in row[-1])
            self.assertIn('--model '+model,service)
            lease=next(row[-1] for row in calls if 'rover_model_lease.py' in row[-1])
            self.assertIn('--model '+model,lease)
            run=next(row[-1] for row in calls if '--instruction' in row[-1])
            self.assertIn('--expected-model '+model,run)
    def test_memory_ab_flag_reaches_selected_rover(self):
        result,calls=self.invoke(['start','--server','h100','--rover','107','--strategy','4',
                                 '--no-long-term-memory','--task','Stop','--dry-run'])
        self.assertEqual(result.returncode,0,result.stderr)
        run=next(row[-1] for row in calls if '--instruction' in row[-1])
        self.assertIn('--no-long-term-memory',run)
        for strategy in ('1','2','3'):
            result,calls=self.invoke(['start','--strategy',strategy,'--no-long-term-memory','--task','Stop','--dry-run'])
            self.assertNotEqual(result.returncode,0);self.assertFalse(calls)

    def test_invalid_model_never_reaches_ssh(self):
        result,calls=self.invoke(['start','--server','h100','--model','bad;cmd','--task','Stop','--dry-run'])
        self.assertNotEqual(result.returncode,0);self.assertEqual(calls,[])

    def test_skip_plan_rationale_reaches_only_strategy_four_on_h100(self):
        result, calls = self.invoke(['start', '--server', 'h100', '--rover', '107',
                                     '--strategy', '4', '--task', 'Stop', '--dry-run',
                                     '--no-plan-rationale'])
        self.assertEqual(result.returncode, 0, result.stderr)
        run = next(row[-1] for row in calls if '--instruction' in row[-1])
        self.assertIn('--no-plan-rationale', run)
        for profile, strategy in (('legacy', '4'), ('h100', '3')):
            result, calls = self.invoke(['start', '--server', profile, '--strategy', strategy,
                                         '--task', 'Stop', '--dry-run', '--no-plan-rationale'])
            self.assertEqual(result.returncode, 2)
            self.assertEqual(calls, [])

    def test_122b_requires_h100_and_gpu_pair(self):
        for args in (['--model','122b'],['--model','122b','--server','h100','--gpu','1']):
            result,calls=self.invoke(['start',*args,'--task','Stop','--dry-run'])
            self.assertNotEqual(result.returncode,0);self.assertEqual(calls,[])

if __name__=='__main__':unittest.main()
