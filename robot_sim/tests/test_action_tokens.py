import asyncio
import unittest
import torch
from robot_sim.candidate_nav.action_tokens import ActionTokenCodec
from robot_sim.candidate_nav.context_budget import ContextBudgetExceeded, error_payload
from robot_sim.candidate_nav.model_transport import request_model, ModelResponseError

class Tokenizer:
    def encode(self,text,**kwargs):
        return {'F':[0],' F':[3],'Q':[1],' Q':[4],'S':[2],' S':[5]}[text]

class TokenTests(unittest.TestCase):
    def test_space_prefixed_forward_is_not_replaced_by_a_bare_turn(self):
        result=ActionTokenCodec(Tokenizer()).score(torch.tensor([0.,2.,-1.,15.,-3.,-2.]),['F','Q','S'],error='bad')
        self.assertEqual(max(range(3),key=result.probabilities.__getitem__),0)
        self.assertEqual(result.token_ids[0],3)
        self.assertGreater(result.probabilities[0],.99)
        self.assertAlmostEqual(result.valid_probability_mass,1.,places=5)
    def test_space_prefixed_stop_and_absolute_support_are_preserved(self):
        result=ActionTokenCodec(Tokenizer()).score(torch.tensor([0.,2.,-1.,-3.,-4.,15.,16.]),['F','Q','S'],error='bad')
        self.assertEqual(result.token_ids[2],5)
        self.assertGreater(result.probabilities[2],.99)
        self.assertLess(result.valid_probability_mass,.3)
    def test_duplicate_aliases_and_multitoken_whitespace_do_not_double_count(self):
        class T:
            def encode(self,text,**kw):return [0] if text.strip()=='F' else [1,2] if text==' S' else [1]
        codec=ActionTokenCodec(T())
        self.assertEqual(codec.group('F'),(0,));self.assertEqual(codec.group('S'),(1,))
        self.assertEqual(codec.score(torch.zeros(3),['F','S'],error='bad').probabilities,[.5,.5])
    def test_nonfinite_vocabulary_rejected_even_outside_allowlist(self):
        for invalid in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaisesRegex(FloatingPointError,'bad'):
                ActionTokenCodec(Tokenizer()).score(torch.tensor([0.]*6+[invalid]),['F','S'],error='bad')

class ErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_budget_details_survive_the_actual_wire_protocol(self):
        error=ContextBudgetExceeded(10292,10240,12288)
        import json
        async def handle(reader,writer):
            await reader.readline();writer.write((json.dumps(error_payload(error))+'\n').encode())
            await writer.drain();writer.close();await writer.wait_closed()
        server=await asyncio.start_server(handle,'127.0.0.1',0)
        try:
            with self.assertRaises(ModelResponseError) as caught:
                await request_model(dict(snapshot_id=1,map_version=1),server.sockets[0].getsockname()[1])
            self.assertEqual(caught.exception.code,'context_budget_exceeded')
            self.assertFalse(caught.exception.retryable)
            self.assertEqual(caught.exception.details['input_tokens'],10292)
        finally:server.close();await server.wait_closed()
    async def test_context_failure_is_not_retried_three_times(self):
        from types import SimpleNamespace
        from robot_sim.candidate_nav.streaming_perception import StreamingPerception
        class Agent(StreamingPerception):
            def __init__(self):
                self.finished=asyncio.Event();self.need_plan=asyncio.Event();self.need_plan.set()
                self.executing_sequence=False;self.pending=asyncio.Queue();self.reach=True;self.calls=0
            def get_pose(self):return [0,0,0]
            def fresh(self):return True
            def control_link_ok(self):return True
            def snapshot(self):return {},[],0
            async def infer_plan(self,observation):
                self.calls+=1;raise ModelResponseError(error_payload(ContextBudgetExceeded(11000,10240,12288)))
            def log(self,*args):pass
            def finish(self,reason):self.reason=reason;self.finished.set()
        a=Agent();await asyncio.wait_for(a.model_planner(),1)
        self.assertEqual(a.calls,1);self.assertEqual(a.reason,'context_budget_exceeded')
