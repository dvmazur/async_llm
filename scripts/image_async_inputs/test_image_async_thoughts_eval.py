import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import image_async_thoughts_eval as ev
from fixed_subset import select_ids, apply_manifest


class Block:
    def __init__(self):
        self.token_ids = []
        self.freed = False


class Tokenizer:
    eos_token_id = 99
    vocab = {}
    def encode(self, text, add_special_tokens=False, return_tensors=None):
        ids = [2] if text == '\n\n' else [3, 4]
        return torch.tensor([ids]) if return_tensors else ids
    def decode(self, ids, **kwargs):
        return ' '.join(map(str,ids))
    def get_vocab(self):
        return {}


class FakeLLM:
    def __init__(self):
        self.blocks = []; self.image_prefills = []; self.calls = []
    async def create_block(self):
        b = Block(); self.blocks.append(b); return b
    async def free_block(self, b):
        assert not b.freed
        b.freed = True
    async def prefill_block(self, input_ids, cache_view, write_to, **kwargs):
        assert not write_to.token_ids
        self.image_prefills.append((write_to,list(input_ids)))
        write_to.token_ids.extend(input_ids)
    async def forward(self, ids=None, cache_view=None, write_to=None, **kwargs):
        if isinstance(cache_view, ev.AsyncContext):
            ctx=cache_view; view=ctx.cache_view; dest=ctx.output_block
            assert ctx.next_input_id is not None
            dest.token_ids.append(ctx.next_input_id); ctx.next_input_id=None
        else:
            view=cache_view or []; dest=write_to or view[-1]
            dest.token_ids.extend(ids.tolist())
        assert all(not b.freed for b in view)
        self.calls.append((dest,list(view)))
        return SimpleNamespace(logits=torch.zeros(100))


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self,k=2,route=True,defer=True):
        llm=FakeLLM()
        a=SimpleNamespace(k_steps=k,budget=5,shard_to_prompt=route,shard_to_thinker=route,
                          writer_reminder=route,defer_writer_reminder=defer,probe_period=1)
        pair={'before':{'input_ids':[10,11]},'after':{'input_ids':[20,21]}}
        with patch.object(ev,'sample',return_value=2), patch.object(ev,'probe_once',return_value=(True,1.,0.)), \
             patch.object(ev,'ends_with_double_newline',return_value=True):
            r=await ev.generate(llm,Tokenizer(),{'text_shard_1':'question','text_shard_2':'correction'},pair,a)
        self.assertTrue(all(b.freed for b in llm.blocks))
        self.assertEqual(len(r['token_ids']['thinker']),5)
        self.assertEqual(len(r['token_ids']['writer']),5)
        return llm,r

    async def test_replacement_preserves_four_block_views(self):
        llm,r=await self.run_case()
        self.assertEqual([ids for _,ids in llm.image_prefills],[[10,11],[20,21]])
        event=next(e for e in r['events'] if e['event']=='image_replaced')
        self.assertEqual(event['thinker_tokens'],2)
        self.assertTrue(any(e['event']=='writer_reminder' for e in r['events']))
        self.assertFalse(r['writer_reminder_pending'])
        new=llm.image_prefills[-1][0]
        self.assertTrue(any(len(view)==4 and view[1] is new for _,view in llm.calls))

    async def test_image_only_routing(self):
        _,r=await self.run_case(route=False)
        self.assertTrue(r['image_replaced'])
        self.assertFalse(any(e['event']=='writer_reminder' for e in r['events']))

    async def test_immediate_reminder(self):
        _,r=await self.run_case(defer=False)
        event=next(e for e in r['events'] if e['event']=='writer_reminder')
        self.assertFalse(event['deferred'])
        self.assertEqual(event['thinker_tokens'],2)

    async def test_baselines_prefill_only_once(self):
        for k,expected in [(0,[20,21]),(-1,[10,11])]:
            llm,r=await self.run_case(k)
            self.assertEqual([ids for _,ids in llm.image_prefills],[expected])
            self.assertFalse(r['image_replaced'])


class ScoreTests(unittest.TestCase):
    def test_nested_box_and_missing_answer(self):
        self.assertEqual(ev.boxed(r'Answer \boxed{\frac{1}{2}}'),r'\frac{1}{2}')
        self.assertIsNone(ev.boxed('unfinished'))
        self.assertFalse(ev.equivalent(None,'2'))
    def test_labels_and_map_sets(self):
        self.assertTrue(ev.equivalent('B','blue',['red','blue']))
        self.assertTrue(ev.equivalent('Texas, Iowa','Iowa; Texas'))
        self.assertFalse(ev.equivalent('red','blue'))

    def test_numeric_equivalence(self):
        self.assertTrue(ev.equivalent(r'\frac{1}{2}','0.5'))
        self.assertFalse(ev.equivalent('0.6','0.5'))


class SubsetTests(unittest.TestCase):
    def test_fixed_balanced_selection_independent_of_input_order(self):
        rows=[{'id':f'{d}-{i}','dataset':d,'category':str(i%3)}
              for d in ('a','b','c','d','e','f','g') for i in range(20)]
        ids=select_ids(rows)
        self.assertEqual(ids,select_ids(list(reversed(rows))))
        self.assertEqual(len(set(ids)),50)
        from collections import Counter
        self.assertEqual(sorted(Counter(s.split('-')[0] for s in ids).values()),[7]*6+[8])

    def test_manifest_checks_and_order(self):
        rows=[{'id':'a'},{'id':'b'}]
        m={'dataset_sha256':'hash','sample_ids':['b','a'],'count':2}
        self.assertEqual([r['id'] for r in apply_manifest(rows,m,'hash')],['b','a'])
        with self.assertRaises(ValueError): apply_manifest(rows,m,'changed')
        with self.assertRaises(ValueError):
            apply_manifest(rows,{**m,'sample_ids':['a','a']},'hash')
        with self.assertRaises(ValueError):
            apply_manifest(rows,{**m,'sample_ids':['a','unknown']},'hash')


if __name__ == '__main__':
    unittest.main()
