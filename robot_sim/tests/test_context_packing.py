import copy,json,unittest
from types import SimpleNamespace
from robot_sim.candidate_nav.prompt_state import model_state,history_state,history_text,budgeted_inputs
from robot_sim.candidate_nav.context_budget import ContextBudgetExceeded,error_payload


def sample():
 frames=[dict(frame_id=i,stamp=float(i),pose=[i/10,0,i/30],plan_id=i//5) for i in range(1,33)]
 current=dict(task_key='a'*64,relevant_visible=False,description='no longer visible',last_seen=frames[-2],latest_frame=frames[-1])
 memory=[]
 for i in range(8):
  group=frames[i*4:i*4+4]
  memory.append(dict(evidence='visible then occluded',frames=group,source='historical_motion_camera',current_visibility=False,
     visual_state=dict(task_key='a'*64,relevant_visible=bool(i%2),description='visible then occluded',last_seen=group[-2],latest_frame=group[-1])))
 actions=[dict(action='forward',outcome='duration_complete',duration_s=1.,start_pose=[i/10,0,0],end_pose=[(i+1)/10,0,0]) for i in range(8)]
 return dict(instruction='Find an arbitrary object and then return.',pose=[3.2,0,1.07],
   sensors=dict(lidar_min_m={'front':.11},imu_available=False),collision_protection='enabled',
   temporal_visual_state=current,prepared_perception=dict(visual_state=current,task_key='a'*64),
   motion_visual_memory=memory,recent_outcomes=actions,task_context=dict(recent_actions=actions,execution_totals=dict(forward_m=3.2)),
   previous_plan_outcome=dict(planned_actions=['forward']*32,executed_actions=31,discarded_actions=['forward'],
     step_results=[dict(action='forward',outcome='trajectory_collision' if i==30 else 'duration_complete',distance_m=.1,turn_rad=0.) for i in range(31)]))


def expand(packed):
 def untab(value):
  if isinstance(value,dict):
   if set(value)=={'columns','rows','constant_columns'}:
    return [{**value['constant_columns'],**dict(zip(value['columns'],[untab(v) for v in row]))} for row in value['rows']]
   if set(value)=={'columns','row_catalog','row_sequence'}:
    return [dict(zip(value['columns'],[untab(v) for v in value['row_catalog'][key]])) for key in value['row_sequence']]
   if set(value)=={'columns','row_runs'}:
    return [dict(zip(value['columns'],[untab(v) for v in row])) for count,row in value['row_runs'] for _ in range(count)]
   if set(value)=={'columns','rows'}:
    return [dict(zip(value['columns'],[untab(v) for v in row])) for row in value['rows']]
   return {k:untab(v) for k,v in value.items()}
  return [untab(v) for v in value] if isinstance(value,list) else value
 root=untab(copy.deepcopy(packed));frames=root.pop('visual_frames',{});root.pop('history_encoding',None)
 columns=root.pop('visual_history_state_columns',[]);codes=root.pop('action_sequence_codes',{})
 if isinstance(frames,list):frames={row['ref']:{k:v for k,v in row.items() if k!='ref'} for row in frames}
 def resolve(value):
  if isinstance(value,dict):
   if set(value)=={'action_sequence'}:return [codes[ch] for ch in value['action_sequence']]
   if isinstance(value.get('visual_state'),list) and columns:value['visual_state']=dict(zip(columns,value['visual_state']))
   if set(value)=={'visual_frame_ref'}:return resolve(frames[value['visual_frame_ref']])
   if set(value)=={'visual_state_ref'}:return resolve(root[value['visual_state_ref']])
   if set(value)=={'value_ref'}:
    target=root
    for key in value['value_ref'].split('.'):target=target[key]
    return resolve(target)
   return {k:resolve(v) for k,v in value.items()}
  return [resolve(v) for v in value] if isinstance(value,list) else value
 return resolve(root)


class HistoryPackingTests(unittest.TestCase):
 def test_round_trip_preserves_every_history_record_and_current_measurement(self):
  state=sample();before=copy.deepcopy(state)
  for level in (1,2,3):
   with self.subTest(level=level):
    packed=history_state(state,level)
    self.assertEqual(expand(packed),model_state(state))
    self.assertEqual(len(packed['visual_frames']['rows']) if level>=2 else len(packed['visual_frames']),32)
    self.assertEqual(packed['instruction'],state['instruction'])
    self.assertEqual(packed['sensors'],state['sensors'])
    if level==2:self.assertLess(len(history_text(state,level)),len(json.dumps(model_state(state),separators=(',',':'))))
  self.assertEqual(state,before)
 def test_different_task_identity_and_conflicting_frame_ids_are_not_merged(self):
  state=sample();state['motion_visual_memory'][0]['visual_state']['task_key']='b'*64
  state['motion_visual_memory'][0]['frames'][0]=dict(frame_id=1,stamp=500.,pose=[99,0,0],plan_id=8)
  self.assertEqual(expand(history_state(state,2)),model_state(state))
 def test_exact_admission_compacts_without_removing_images_or_history(self):
  state=sample();pixel_values=object();sent=[]
  def build(text):
   sent.append(text)
   return dict(input_ids=SimpleNamespace(numel=lambda:len(text)+1000),pixel_values=pixel_values)
  limit=len(history_text(state,2))+1100
  inputs,metrics=budgeted_inputs(build,state,True,limit,limit+2048)
  self.assertLessEqual(metrics['input_tokens'],limit)
  self.assertEqual(metrics['history_encoding_level'],2)
  self.assertEqual(metrics['history_records_removed'],0)
  self.assertIs(inputs['pixel_values'],pixel_values)
  self.assertEqual(expand(json.loads(sent[-1])),model_state(state))
 def test_short_inputs_and_simulator_keep_the_existing_encoding(self):
  for live in (False,True):
   sent=[]
   def build(text):sent.append(text);return dict(input_ids=SimpleNamespace(numel=lambda:100))
   _,metrics=budgeted_inputs(build,dict(instruction='go'),live,10240,12288)
   self.assertEqual(len(sent),1);self.assertEqual(metrics['history_encoding_level'],0)
 def test_unfit_instruction_is_never_sliced_or_silently_admitted(self):
  state=sample();sent=[]
  def build(text):sent.append(json.loads(text));return dict(input_ids=SimpleNamespace(numel=lambda:20000))
  with self.assertRaises(ContextBudgetExceeded) as failure:budgeted_inputs(build,state,True,10240,12288)
  self.assertTrue(all(s['instruction']==state['instruction'] for s in sent))
  self.assertEqual(error_payload(failure.exception)['compaction_attempts'],[20000,20000,20000,20000])


class RepeatedRowsTests(unittest.TestCase):
 def test_repeated_steps_keep_count_order_and_different_terminal_outcome(self):
  state=sample();packed=history_state(state,3)
  self.assertEqual(expand(packed),model_state(state))
  self.assertIn('row_runs',packed['previous_plan_outcome']['step_results'])
  self.assertEqual(packed['previous_plan_outcome']['step_results']['row_runs'][0][0],30)
  self.assertLess(len(history_text(state,3)),len(history_text(state,2)))

 def test_nonconsecutive_repeats_preserve_all_measurements_and_order(self):
  state=sample();steps=state['previous_plan_outcome']['step_results']
  state['previous_plan_outcome']['step_results']=[dict(steps[i%5],distance_m=(i%5)/100) for i in range(32)]
  packed=history_state(state,3)
  self.assertEqual(expand(packed),model_state(state))
  self.assertIn('row_catalog',packed['previous_plan_outcome']['step_results'])

 def test_distinct_odom_measurements_preserve_constant_frame_and_action(self):
  state=sample()
  state['previous_plan_outcome']['step_results']=[dict(action='forward',outcome='duration_complete',
      control_frame='odom',distance_m=.08+i/10000,signed_distance_m=.07+i/10000,turn_rad=i/1000) for i in range(32)]
  packed=history_state(state,3)
  self.assertEqual(expand(packed),model_state(state))
  self.assertIn('constant_columns',packed['previous_plan_outcome']['step_results'])
