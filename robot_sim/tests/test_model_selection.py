import importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from robot_sim.candidate_nav.model_catalog import MODELS,model_key,validate_model,model_path
from robot_sim.scripts import rover_live_model_service as service

class CatalogTests(unittest.TestCase):
 def test_model_aliases_and_invalid_names(self):
  for key,name in MODELS.items():
   for alias in (key,name,'Qwen/'+name):self.assertEqual(model_key(alias),key)
  for alias in ('bad','../../path','35b;id'):
   with self.assertRaises(ValueError):model_key(alias)
 def test_missing_shards_rejected_before_switching(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);p=model_path(root,'35b');p.mkdir(parents=True)
   (p/'config.json').write_text('{}');(p/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'a':'missing.safetensors'})))
   (p/'tokenizer_config.json').write_text('{}')
   with self.assertRaises(RuntimeError):validate_model(root,'35b')
   (p/'missing.safetensors').touch();self.assertEqual(validate_model(root,'35b'),p)

class ServiceTests(unittest.TestCase):
 def test_reuse_requires_same_model_and_configuration(self):
  config=dict(profile='h100',fp8_backend='triton')
  health=dict(model=str(model_path(service.ROOT,'27b')),policy_source_sha256='hash',max_context_tokens=12288,
              trajectory_policy='evidence',hardware_profile='h100',gpu_physical_ids='0',fp8_backend='triton')
  with patch.object(service,'source_hash',return_value='hash'):
   self.assertTrue(service.reusable(health,config,'27b','0'))
   self.assertFalse(service.reusable(health,config,'35b','0'))
   self.assertFalse(service.reusable(health,config,'27b','1'))
 def test_active_model_lease_blocks_switch(self):
  import fcntl
  with tempfile.TemporaryDirectory() as folder:
   runtime=Path(folder)
   with (runtime/'model-use.lock').open('a') as active:
    fcntl.flock(active,fcntl.LOCK_SH)
    with self.assertRaisesRegex(RuntimeError,'rover run'):
     with service.exclusive_use(runtime):pass
   with service.exclusive_use(runtime):pass
 def test_stop_does_not_signal_unmanaged_jobs(self):
  with patch.object(service,'owned_pids',return_value=set()),patch.object(service.subprocess,'run') as run:
   service.stop(dict(session='ours'));run.assert_not_called()

class ReplacementTests(unittest.TestCase):
 def test_switch_unloads_first_and_failure_restores_previous_model(self):
  import sys
  for fail in (False,True):
   with self.subTest(fail=fail),tempfile.TemporaryDirectory() as folder:
    root=Path(folder)
    for key in MODELS:
     p=model_path(root,key);p.mkdir(parents=True)
     (p/'config.json').write_text('{}');(p/'model.safetensors.index.json').write_text('{"weight_map":{}}');(p/'tokenizer_config.json').write_text('{}')
    previous=dict(ready=True,model=str(model_path(root,'27b')),gpu_physical_ids='0')
    events=[]
    def start(config,key,gpu):
     events.append('start '+key)
     (root/'robot_sim/.runtime/live_rover/model.log').write_text('35b initialization failed' if fail and key=='35b' else key+' ready')
     if fail and key=='35b':raise RuntimeError('test initialization failure')
    with patch.object(service,'ROOT',root),patch.object(service,'settings',return_value=dict(profile='h100',gpu='auto',fp8_backend='triton',shared_gpu_ids=[0])),patch.object(service,'health',return_value=previous),patch.object(service,'session_exists',return_value=True),patch.object(service,'owned_pids',return_value=set()),patch.object(service,'discover',return_value=[]),patch.object(service,'select_devices',return_value=[]),patch.object(service,'stop',side_effect=lambda c:events.append('stop')),patch.object(service,'start',side_effect=start),patch.object(sys,'argv',['service','--model','35b','--gpu','0']):
     if fail:
      with self.assertRaisesRegex(RuntimeError,'initialization failure'):service.main()
      self.assertEqual(events,['stop','start 35b','stop','start 27b'])
      logs=list((root/'robot_sim/.runtime/live_rover/startup-failures').glob('*.log'))
      self.assertEqual(len(logs),1)
      self.assertEqual(logs[0].read_text(),'35b initialization failed')
      self.assertEqual((root/'robot_sim/.runtime/live_rover/model.log').read_text(),'27b ready')
     else:
      service.main();self.assertEqual(events,['stop','start 35b'])


class DualGpuTests(unittest.TestCase):
 def devices(self,**kw):
  return [dict(index=i,uuid=str(i),capability=9.0,total_mib=81559,free_mib=81080,busy=False,**kw) for i in (0,1)]
 def test_122b_uses_two_cards_even_with_gpu_zero(self):
  from robot_sim.candidate_nav.model_catalog import select_devices,model_tp,gpu_matches
  config=dict(profile='h100',shared_gpu_ids=[0])
  for request in ('auto','0'):
   self.assertEqual([d['index'] for d in select_devices(config,'122b',request,self.devices())],[0,1])
   self.assertTrue(gpu_matches('122b',request,'0,1'))
  self.assertEqual(model_tp('h100','122b'),2)
  with self.assertRaises(ValueError):select_devices(config,'122b','1',self.devices())
  with self.assertRaises(ValueError):model_tp('legacy','122b')
 def test_explicit_27b_can_share_gpu_one_without_changing_auto_or_35b(self):
  from robot_sim.candidate_nav.model_catalog import select_devices
  cards=self.devices()
  cards[0].update(free_mib=53399,busy=True)
  cards[1].update(free_mib=75963,busy=True)
  config=dict(profile='h100',shared_gpu_ids=[0],shared_27b_gpu_ids=[1])
  self.assertEqual(select_devices(config,'27b','1',cards),cards[1:])
  self.assertEqual(select_devices(config,'27b','auto',cards),cards[:1])
  with self.assertRaisesRegex(RuntimeError,'GPU 1: 75963 MiB free, running compute jobs'):
   select_devices(config,'35b','1',cards)
  cards[1]['free_mib']=45055
  with self.assertRaisesRegex(RuntimeError,'GPU 1: 45055 MiB free, requires >= 45056 MiB free'):
   select_devices(config,'27b','1',cards)
 def test_122b_shares_only_explicitly_authorized_cards(self):
  from robot_sim.candidate_nav.model_catalog import select_devices
  cards=self.devices()
  for card in cards:
   card.update(busy=True,free_mib=75963)
  config=dict(profile='h100',shared_gpu_ids=[0])
  for sharing in ([], [0], [1]):
   with self.subTest(sharing=sharing), self.assertRaisesRegex(RuntimeError,'sharing for 122b is not enabled'):
    select_devices(dict(config,shared_122b_gpu_ids=sharing),'122b','auto',cards)
  self.assertEqual(select_devices(dict(config,shared_122b_gpu_ids=[0,1]),'122b','auto',cards),cards)
  self.assertEqual(select_devices(dict(config,shared_122b_gpu_ids=[0,1]),'35b','auto',cards),cards[:1])

 def test_122b_rejects_insufficient_memory_or_incompatible_gpu_even_with_sharing(self):
  from robot_sim.candidate_nav.model_catalog import select_devices
  for index in (0,1):
   for change, diagnostic in (({'free_mib':69999},'69999 MiB free'),({'capability':8.0},'requires H100 80GB')):
    cards=self.devices();cards[index].update(busy=True,**change)
    with self.subTest(index=index,change=change), self.assertRaisesRegex(RuntimeError,f'GPU {index}: {diagnostic}'):
     select_devices(dict(profile='h100',shared_122b_gpu_ids=[0,1]),'122b','auto',cards)
 def test_optional_mtp_shard_is_not_required(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);p=model_path(root,'27b');p.mkdir(parents=True)
   (p/'config.json').write_text('{}');(p/'tokenizer_config.json').write_text('{}')
   (p/'weights.safetensors').touch()
   (p/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'model.language_model.a':'weights.safetensors','mtp.a':'mtp.safetensors'})))
   self.assertEqual(validate_model(root,'27b'),p)


class StartupTimeoutTests(unittest.TestCase):
 def run_start(self,ready_at=None,session=True):
  config=dict(profile='h100',session='ours');clock=[0.0]
  def sleep(seconds):clock[0]+=seconds
  def health(config):return {'ready':True} if ready_at is not None and clock[0]>=ready_at else None
  with patch.object(service,'discover',return_value=[]),patch.object(service,'select_devices',return_value=[dict(index=0),dict(index=1)]),patch.object(service.subprocess,'run'),patch.object(service,'health',side_effect=health),patch.object(service,'reusable',return_value=True),patch.object(service,'session_exists',return_value=session),patch.object(service.time,'monotonic',side_effect=lambda:clock[0]),patch.object(service.time,'sleep',side_effect=sleep),patch('builtins.print'):
   result=service.start(config,'122b','auto')
  return result,clock[0]
 def test_allows_loading_beyond_old_fifteen_minute_limit(self):
  self.assertEqual(service.STARTUP_TIMEOUT_S,1800)
  self.assertEqual(self.run_start(ready_at=1200),({'ready':True},1200))
 def test_ready_just_before_deadline_is_accepted(self):
  self.assertEqual(self.run_start(ready_at=1799),({'ready':True},1799))
 def test_timeout_is_bounded_and_points_to_log(self):
  with self.assertRaisesRegex(RuntimeError,'1800 s; inspect .*model.log'):
   self.run_start()
 def test_worker_failure_does_not_wait_for_deadline(self):
  with self.assertRaisesRegex(RuntimeError,'startup failed'):
   self.run_start(session=False)
