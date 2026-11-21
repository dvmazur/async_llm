"""Full-instruction live completion; legacy simulator contract stays separate."""

import asyncio
import base64
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from robot_sim.candidate_nav import task_policy
from robot_sim.candidate_nav.direct_control import validate_response


class CompletionContractTests(unittest.TestCase):
    def test_live_nonvisual_completion_needs_no_object_or_confidence_threshold(self):
        reply = dict(protocol='direct_navigation_v1', action='stop', goal_id='Z',
                     task_complete=True, completion_mode=task_policy.MODE)
        observation = dict(physical=True, completion_mode=task_policy.MODE, objects=[])
        validate_response(reply, observation)
        for change in ({'physical': False}, {'completion_mode': 'visual_confirmation'}):
            with self.assertRaises(ValueError):
                validate_response(reply, {**observation, **change})
        for change in ({'action': 'forward'}, {'task_complete': False},
                       {'completion_mode': 'visual_confirmation'}):
            with self.assertRaises(ValueError):
                validate_response({**reply, **change}, observation)


class TaskPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_waypoint_transport_accepts_task_completion_without_object_range(self):
        from robot_sim.candidate_nav.model_transport import request_model
        reply = dict(goal_id='Z', task_complete=True, completion_mode=task_policy.MODE,
                     snapshot_id=1, map_version=3)
        async def handle(reader, writer):
            await reader.readline()
            writer.write((json.dumps(reply) + '\n').encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        server = await asyncio.start_server(handle, '127.0.0.1', 0)
        observation = dict(protocol='candidate_navigation_v1', snapshot_id=1, map_version=3,
                           physical=True, completion_mode=task_policy.MODE, candidates=[], objects=[])
        try:
            port = server.sockets[0].getsockname()[1]
            self.assertTrue((await request_model(observation, port))['task_complete'])
            reply['task_complete'] = False
            with self.assertRaises(ValueError):
                await request_model(observation, port)
            reply['task_complete'] = True
            observation['physical'] = False
            with self.assertRaises(ValueError):
                await request_model(observation, port)
        finally:
            server.close()
            await server.wait_closed()

    async def decide(self, *, direct=False, complete=False, live=True, trajectory=None, actions='F', evidence=False, temporal=False, accumulated=False, speculative=False, maneuver='A', prepared=False, unprotected=False, merge=False, spaced=False, rationale_ok=True, stream_reply=False, no_plan_rationale=False):
        import torch
        from PIL import Image

        from robot_sim.candidate_nav import action_plan
        from robot_sim.candidate_nav.llm import AsyncCandidatePolicy
        direct = direct or trajectory is not None
        planned = iter(actions)

        class Tokens:
            eos_token_id = 0
            def __init__(self):
                self.prompts = {}
                self.words = {c: ord(c) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
                if spaced:self.words.update({' '+c:ord(c)+300 for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'})
                self.words.update({'yes': 100, 'Yes': 101, 'YES': 102,
                                   'no': 103, 'No': 104, 'NO': 105})
            def encode(self, text, **_):
                if text in self.words:
                    return [self.words[text]]
                key = 200 + len(self.prompts)
                self.prompts[key] = text
                return [key]
            def decode(self, tokens, **_):
                return 'CPU contract test trace'

        class Engine:
            def __init__(self):
                self.tokenizer = Tokens()
                self.processor = self
                self.engine = SimpleNamespace(tp_cpu_group=None)
                self.blocks, self.freed, self.completion_views = [], [], []
                self.movement_inputs = []
                self.spatial_views = []
                self.rationale_views = []
                self.final_action_commits = []
                self.prefill_inputs = []
            def apply_chat_template(self, messages, **_):
                self.messages = messages
                return {'input_ids': torch.tensor([[1, 2]])}
            async def create_block(self):
                block = len(self.blocks)
                self.blocks.append(block)
                return block
            async def forward(self, tokens=None, cache_view=None, **_):
                if 'input_ids' in _:self.prefill_inputs.append(_['input_ids'].clone())
                await asyncio.sleep(0)
                scores = torch.full((512,), -10.)
                token = 65
                if cache_view and cache_view[-1]==3 and not _.get('return_logits',True):
                    self.final_action_commits.append(list(tokens))
                if cache_view and cache_view[-1] == 3 and _.get("return_logits",True):
                    prompt = ''.join(self.tokenizer.prompts.get(t, '') for t in (tokens or []))
                    if 'FORECAST CONTINUATION' in prompt:
                        token = 104 if complete else 101
                    elif 'Answer exactly Yes or No' in prompt:
                        self.completion_views.append(cache_view)
                        token = 101 if complete else 104
                    elif 'Reply with one option letter' in prompt:
                        token = ord(maneuver)
                    else:
                        self.movement_inputs.append(list(tokens))
                        token = ord(next(planned, 'F') if direct else 'A')
                        if spaced:
                            scores[ord('Q')]=12
                            token+=300
                if tokens and (task_policy.COMMITTED_ASSESSMENT_PROMPT if evidence else task_policy.PIPELINE_SPATIAL_PROMPT if speculative else task_policy.GROUNDED_SPATIAL_PROMPT) in "".join(self.tokenizer.prompts.get(t, "") for t in tokens):
                    self.spatial_views.append(list(cache_view))
                if tokens and task_policy.COMMITTED_RATIONALE_PROMPT in ''.join(self.tokenizer.prompts.get(t,'') for t in tokens):self.rationale_views.append(list(cache_view))
                if tokens and 'RATIONALE FAITHFULNESS' in ''.join(self.tokenizer.prompts.get(t,'') for t in tokens):token=101 if rationale_ok else 104
                scores[token] = 20 if spaced and token>=300 else 10
                return SimpleNamespace(logits=scores)
            async def free_block(self, block):
                self.freed.append(block)

        engine = Engine()
        policy = AsyncCandidatePolicy.__new__(AsyncCandidatePolicy)
        policy.llm, policy.tokens = engine, engine.tokenizer
        policy.args = SimpleNamespace(analysis_tokens=24, trajectory_policy="evidence" if evidence else "legacy",
                                      merge_prepared_observer=merge)
        policy.rank, policy.world, policy.metadata = 0, 1, {'test_only': True}
        stream = io.BytesIO()
        Image.new('RGB', (16, 16)).save(stream, format='PNG')
        encoded = base64.b64encode(stream.getvalue()).decode()
        if trajectory:
            direct=True
        instruction = 'Turn left, wait ten seconds, then return to the starting position.'
        context = {'elapsed_s': 12, 'odometry_path_m': .4,
                   'initial_pose': [0, 0, 0], 'recent_actions': [{'action': 'left'}]}
        observation = dict(protocol='direct_navigation_v1' if direct else 'candidate_navigation_v1',
                           snapshot_id=1, map_version=3, camera=encoded, map_image=encoded,
                           instruction=instruction, task_context=context, pose=[0,0,0],
                           physical=live, completion_mode=task_policy.MODE,
                           candidates=[] if direct else [{'id': 'A'}], objects=[])
        if live:
            from robot_sim.candidate_nav.body_model import body_description
            observation['rover_body'] = body_description()
        if trajectory is not None:
            observation.update(protocol=action_plan.PROTOCOL, trajectory_steps=trajectory)
        if unprotected:
            observation['collision_protection']='disabled'
        if accumulated:
            observation['accumulated_map_image'] = encoded
            observation['maps_context'] = {'accumulated': {'available': True, 'age_s': 1}}
        if temporal:
            observation['previous_camera']=dict(image=encoded,stamp=1,pose=[0,0,.2],snapshot_id=0)
        if prepared:
            from robot_sim.candidate_nav.temporal_vision import initial_state
            from robot_sim.candidate_nav.prepared_perception import image_digest
            visual = initial_state(instruction)
            visual.update(relevant_visible=False, description='No task-relevant object in this frame.',
                          latest_frame=dict(frame_id=4,stamp=2,pose=[0,0,0]))
            observation.update(sim_time=3,camera_stamp=2,prepared_perception=dict(
                task_key=visual['task_key'], visual_state=visual, camera_stamp=2,
                camera_sha256=image_digest(Image.new('RGB',(16,16))),evidence='nothing new'))
        if speculative:
            observation.update(pipeline={'speculative':True},perception_pose=[0,0,0],camera_pose=[0,0,0],pose=[.2,0,.3])
        published = []
        if trajectory:
            from robot_sim.candidate_nav.action_plan import PROTOCOL, ACTION_PLAN_SPEC
            observation.update(protocol=PROTOCOL, action_spec=ACTION_PLAN_SPEC,
                               maneuver_memory={'local_measured_turn_rad':-1.57,'last_measured_turn_direction':'right'},
                               heading_options=[{'turn_prefix':'RR','turn_sweep_known_free':True,'full_forward_steps':2,'map_yaw_degrees':-90},
                                                {'turn_prefix':'LL','turn_sweep_known_free':True,'full_forward_steps':0,'map_yaw_degrees':90}])
        if stream_reply:observation['response_mode']='plan_stream_v1'
        if no_plan_rationale:observation['no_plan_rationale']=True
        def gather(peaks, value, **_):
            peaks[0] = value
        with patch('torch.cuda.max_memory_allocated', return_value=0), \
             patch('torch.distributed.all_gather_object', side_effect=gather), patch('builtins.print'):
            result = await asyncio.wait_for(policy.decide(observation, on_selection=lambda reply: published.append(dict(reply))),5)
        self.assertEqual(len(published), 1)
        if stream_reply and not no_plan_rationale:
            self.assertEqual(published[0]['message_kind'],'plan_ready')
            self.assertTrue(published[0]['rationale_pending'])
            self.assertEqual(result['message_kind'],'rationale_ready')
            self.assertFalse(result['rationale_pending'])
            self.assertFalse(any(x.get('stream')==2 for x in published[0]['timeline']))
            self.assertTrue(any(x.get('stream')==2 for x in result['timeline']))
        if trajectory is not None:
            action_plan.validate_response(result, observation)
            self.assertEqual(published[0]['actions'], result['actions'])
            if not complete:
                self.assertEqual(len(engine.movement_inputs), len(result['actions']))
                previous_labels = [action_plan.ACTION_LABELS[a]
                                   for a in result['actions'][:-1]]
                for tokens, previous in zip(engine.movement_inputs[1:], previous_labels):
                    self.assertEqual(tokens[0], ord(previous)+(300 if spaced else 0))
                    self.assertEqual(engine.tokenizer.prompts[tokens[1]], ',')
        self.assertEqual(sorted(engine.freed), [0,1,2,3,4] if evidence and not no_plan_rationale else [0,1,2,3])
        if live:
            expected_system=(task_policy.TRAJECTORY_EVIDENCE_SYSTEM+(task_policy.PIPELINE_FORECAST_RULES if speculative else '') if evidence else task_policy.TRAJECTORY_SYSTEM if trajectory is not None else task_policy.DIRECT_SYSTEM if direct else task_policy.WAYPOINT_SYSTEM)
            actual_system=engine.messages[0]['content']
            sent = json.loads(engine.messages[1]['content'][-1]['text'])
            if evidence:
                prefix=expected_system+'\nFixed robot context (same authority as measured observation; units and action signs are literal):\n'
                self.assertTrue(actual_system.startswith(prefix))
                fixed=json.loads(actual_system[len(prefix):])
                self.assertEqual(fixed['action_spec'],observation['action_spec'])
                self.assertNotIn('action_spec',sent)
                self.assertNotIn('rover_body',sent)
                self.assertNotIn('no_plan_rationale',sent)
                sent.update(fixed)  # Facts are available in the immutable prefix.
            else:self.assertEqual(actual_system,expected_system)
            from robot_sim.candidate_nav.prompt_state import model_state
            self.assertEqual(sent['rover_body'], observation['rover_body'] if evidence else model_state(observation)['rover_body'])
            self.assertNotIn('accumulated_map_image',sent)
            if accumulated:
                self.assertEqual(sent['image_roles']['accumulated_map'],3)
                self.assertIn('accumulated_map',result['image_sha256'])
            self.assertEqual(sent['instruction'], instruction)
            self.assertEqual(sent['task_context'], context)
            self.assertEqual(engine.completion_views, [] if speculative else ([[0,2,3]] if merge else [[0,1,2,3]]) if evidence else [[0,1,2,3]])
            if speculative:self.assertNotIn('Z',sent['valid_choices'])
            if trajectory:
                self.assertEqual(sent['maneuver_memory'],observation['maneuver_memory'])
                self.assertEqual(sent['action_geometry']['R'] if evidence else sent['action_geometry']['right_turn_deg'],-22.5 if evidence else 22.5)
                self.assertEqual(sent['available_translations'][0]['turn_actions'],['right','right'])
                if not evidence:self.assertEqual(sent['prefixes_without_known_forward_step'],['LL'])
                self.assertNotIn('heading_options',sent)
                timeline=result['timeline']
                if evidence:
                    camera_end=next(i for i,e in enumerate(timeline) if e['event']=='observer_reused') if prepared else next(i for i,e in enumerate(timeline) if e['event']=='stream_finished' and e['stream']==0)
                    spatial_start=next(i for i,e in enumerate(timeline) if e['event']=='stream_started' and e['stream']==1)
                    spatial_end=next(i for i,e in enumerate(timeline) if e['event']=='stream_finished' and e['stream']==1)
                    selection=next(i for i,e in enumerate(timeline) if e['event']=='selector_started')
                    committed=next(i for i,e in enumerate(timeline) if e['event']=='plan_committed')
                    self.assertLess(camera_end,selection)
                    self.assertLess(selection,committed)
                    self.assertLess(spatial_start,spatial_end)
                    self.assertLess(spatial_end,selection)
                    self.assertEqual(engine.spatial_views,[[0,2]] if merge else [[0,1,2]])
                    self.assertEqual(result['selected_plan']['actions'],result['actions'])
                    if no_plan_rationale:
                        self.assertFalse(engine.rationale_views)
                        self.assertFalse(engine.final_action_commits)
                        self.assertNotIn('message_kind', published[0])
                    else:
                        self.assertEqual(engine.rationale_views,[[0,2,3,4]] if merge else [[0,1,2,3,4]])
                        rationale_start=next(i for i,e in enumerate(timeline) if e['event']=='stream_started' and e['stream']==2)
                        self.assertLess(committed,rationale_start)
                        if not complete:
                            last=action_plan.ACTION_LABELS[result['actions'][-1]]
                            self.assertEqual(engine.final_action_commits[-1][0],ord(last)+(300 if spaced else 0))
                    self.assertIn(result['selected_plan']['display'],result['reasoner'])
                    self.assertFalse(any(e['event']=='maneuver_selected' for e in timeline))
                    if prepared:
                        self.assertTrue(result['observer_reused'])
                        self.assertFalse(any(e['event']=='stream_started' and e.get('stream')==0 for e in timeline))
                        self.assertEqual(sum(x.get('type')=='image' for x in engine.messages[1]['content']),3 if accumulated else 2)
                        if merge:
                            merged_tokens=engine.prefill_inputs[0][0].tolist()
                            self.assertEqual(merged_tokens[:2],[1,2])
                            text=''.join(engine.tokenizer.prompts.get(t,'') for t in merged_tokens[2:])
                            report=next(v for v in engine.tokenizer.prompts.values() if v.startswith('Current (already analysed exact'))
                            expected=('<|im_start|>user\nUse the validated analysis of this exact camera frame:'
                                      '<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n'+report+'<|im_end|>\n')
                            self.assertEqual(text,expected)
                            self.assertTrue(next(e for e in timeline if e['event']=='observer_reused')['merged'])
                    elif temporal:
                        self.assertEqual(sum(x.get('type')=='image' for x in engine.messages[1]['content']),4 if accumulated else 3)
                        self.assertEqual(sent['image_roles']['previous_camera'],4 if accumulated else 3)
                        self.assertNotIn('image',sent['previous_camera'])
                        self.assertAlmostEqual(sent['camera_motion_since_previous']['dyaw_rad'],-.2)
                    self.assertNotIn('Selected actions:',result['reasoner'])
                    self.assertEqual(result["planning_mode"],"committed_spatial_plan_v1")
                    return result
                finished=next(i for i,e in enumerate(timeline) if e['event']=='stream_finished' and e['stream']==1)
                selector=next(i for i,e in enumerate(timeline) if e['event']=='selector_started')
                self.assertLess(finished,selector)
                self.assertGreaterEqual(timeline[selector]['reason_tokens'],96)

        else:
            self.assertEqual(engine.completion_views, [])
        return result

    async def test_trajectory_reads_memory_and_finished_spatial_assessment(self):
        result=await self.decide(trajectory=1)
        self.assertEqual(result['actions'],['forward'])

    async def test_completed_endpoint_observer_is_reused_before_actual_boundary_selection(self):
        result=await self.decide(trajectory=4, actions='RRFS', evidence=True, temporal=True, prepared=True)
        self.assertEqual(result['actions'],['right','right','forward','stop'])

    async def test_experimental_merge_preserves_exact_report_tokens_and_action_contract(self):
        result=await self.decide(trajectory=4,actions='RRFS',evidence=True,prepared=True,merge=True)
        self.assertEqual(result['actions'],['right','right','forward','stop'])

    async def test_all_turn_angles_are_selectable_in_inspection_mode(self):
        result=await self.decide(trajectory=7,actions='QEADLRS',evidence=True,maneuver='V')
        self.assertEqual(result['actions'],['left_11_25','right_11_25','left_45','right_45','left','right','stop'])

    async def test_disabled_protection_preserves_all_turn_angles(self):
        result=await self.decide(trajectory=5,actions='QEADS',evidence=True,unprotected=True)
        self.assertEqual(result['actions'],['left_11_25','right_11_25','left_45','right_45','stop'])

    async def test_waypoints_follow_full_instruction_instead_of_visibility_rule(self):
        result = await self.decide(complete=False)
        self.assertEqual(result['goal_id'], 'A')
        self.assertFalse(result['task_complete'])

    async def test_nonvisual_task_can_complete_without_objects(self):
        result = await self.decide(complete=True)
        self.assertEqual(result['goal_id'], 'Z')
        self.assertTrue(result['task_complete'])
        self.assertEqual(result['completion_mode'], task_policy.MODE)

    async def test_direct_completion_is_a_stop(self):
        result = await self.decide(direct=True, complete=True)
        validate_response(result, dict(physical=True, completion_mode=task_policy.MODE, objects=[]))
        self.assertEqual(result['action'], 'stop')

    async def test_simulation_cannot_use_live_nonvisual_completion(self):
        result = await self.decide(complete=True, live=False)
        self.assertEqual(result['goal_id'], 'A')
        self.assertEqual(result['completion_mode'], 'measured_range')

    async def test_trajectory_is_autoregressive_and_respects_requested_horizon(self):
        result = await self.decide(trajectory=4, actions='LFFRR')
        self.assertEqual(result['actions'], ['left', 'forward', 'forward', 'right'])
        self.assertEqual(len(result['step_probabilities']), 4)
        self.assertEqual(result['action'], 'left')
        self.assertEqual(result['trajectory_steps'], 4)
        self.assertFalse(result['task_complete'])

    async def test_trajectory_stops_early_to_request_observations(self):
        result = await self.decide(trajectory=8, actions='RSFFFF')
        self.assertEqual(result['actions'], ['right', 'stop'])
        self.assertFalse(result['task_complete'])

    async def test_completed_task_produces_only_stop_even_with_long_horizon(self):
        result = await self.decide(trajectory=32, complete=True)
        self.assertEqual(result['actions'], ['stop'])
        self.assertTrue(result['task_complete'])

    async def test_horizon_one_preserves_single_action_behavior(self):
        result = await self.decide(trajectory=1, actions='RFFF')
        self.assertEqual(result['actions'], ['right'])


    async def test_evidence_policy_uses_one_finished_observer_then_grammar_plan(self):
        result=await self.decide(trajectory=5, actions='RRFFS', evidence=True)
        self.assertEqual(result['actions'],['right','right','forward','forward','stop'])

    async def test_temporal_camera_pair_excludes_base64_from_state_json(self):
        result=await self.decide(trajectory=5,actions='RRFFS',evidence=True,temporal=True)
        self.assertNotIn('Selected actions:',result['reasoner'])


    async def test_accumulated_map_is_distinct_from_local_and_previous_camera(self):
        result=await self.decide(trajectory=5,actions='RRFFS',evidence=True,temporal=True,accumulated=True)
        self.assertEqual(result['actions'],['right','right','forward','forward','stop'])

    async def test_waypoint_policy_receives_accumulated_map(self):
        await self.decide(accumulated=True)

    async def test_speculative_plan_cannot_claim_unexecuted_completion(self):
        result=await self.decide(trajectory=5,actions='RRFFS',evidence=True,temporal=True,
                                 accumulated=True,speculative=True,complete=True)
        self.assertFalse(result['task_complete'])
        self.assertEqual(result['actions'],['stop'])
        self.assertNotEqual(result['goal_id'],'Z')

    async def test_speculative_plan_with_work_remaining_can_generate_continuation(self):
        result=await self.decide(trajectory=5,actions='RRFFS',evidence=True,temporal=True,
                                 accumulated=True,speculative=True,complete=False)
        self.assertEqual(result['actions'],['right','right','forward','forward','stop'])

    async def test_stop_maneuver_never_forces_a_translation(self):
        result=await self.decide(trajectory=5,actions='S',evidence=True)
        self.assertEqual(result['actions'],['stop'])
        self.assertFalse(result['task_complete'])

    async def test_inconsistent_explanation_is_withheld_without_replacing_plan(self):
        result=await self.decide(trajectory=5,actions='RRFFS',evidence=True,rationale_ok=False)
        self.assertEqual(result['actions'],['right','right','forward','forward','stop'])
        self.assertFalse(result['rationale_consistency']['consistent'])
        self.assertIn('withheld',result['spatial_rationale'])
        self.assertIn('CPU contract test trace',result['raw_spatial_rationale'])

    async def test_whitespace_actions_are_selected_and_exact_tokens_committed(self):
        result=await self.decide(trajectory=5,actions='FFS',evidence=True,unprotected=True,spaced=True)
        self.assertEqual(result['actions'],['forward','forward','stop'])
        self.assertEqual(len(result['action_token_support']),3)

    async def test_no_plan_rationale_keeps_selection_and_skips_explanation_forwards(self):
        args = dict(trajectory=4, actions='RRFS', evidence=True)
        normal = await self.decide(**args, stream_reply=True)
        lean = await self.decide(**args, stream_reply=True, no_plan_rationale=True)
        for key in ('selected_plan', 'completion_probabilities', 'step_probabilities', 'spatial_assessment', 'observer'):
            self.assertEqual(lean[key], normal[key])
        self.assertFalse(lean['rationale_enabled'])
        self.assertFalse(lean['rationale_pending'])
        self.assertIsNone(lean['rationale_consistency'])
        self.assertEqual(lean['spatial_rationale'], '')
        self.assertFalse(any(event.get('stream') == 2 or event['event'] == 'rationale_checked'
                             for event in lean['timeline']))

    async def test_no_plan_rationale_preserves_task_completion(self):
        result = await self.decide(trajectory=4, evidence=True, complete=True, no_plan_rationale=True)
        self.assertTrue(result['task_complete'])
        self.assertEqual(result['actions'], ['stop'])
        self.assertFalse(result['rationale_enabled'])

if __name__ == '__main__':
    unittest.main()


    async def test_stream_publishes_immutable_plan_before_explanation(self):
        await self.decide(trajectory=5,actions='FFS',evidence=True,unprotected=True,stream_reply=True)
