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
        ids = [2] if text == '\n\n' else [5] if text == 'yes' else [6] if text == 'no' else [3, 4]
        return torch.tensor([ids]) if return_tensors else ids
    def decode(self, ids, **kwargs):
        return ' '.join(map(str,ids))
    def get_vocab(self):
        return {}


class FakeLLM:
    def __init__(self):
        self.blocks = []; self.image_prefills = []; self.calls = []; self.prefill_calls = []
        self.fail_prefill = None
    async def create_block(self):
        b = Block(); self.blocks.append(b); return b
    async def free_block(self, b):
        assert not b.freed
        b.freed = True
    async def prefill_block(self, input_ids, cache_view, write_to, **kwargs):
        assert not write_to.token_ids
        ids = list(map(int, input_ids))
        self.prefill_calls.append((write_to, list(cache_view), ids, dict(kwargs)))
        if self.fail_prefill == len(self.prefill_calls):
            raise RuntimeError('prefill failure')
        if 'pixel_values' in kwargs:
            self.image_prefills.append((write_to,ids))
        write_to.token_ids.extend(input_ids)
        logits = torch.zeros(100)
        logits[5] = 1  # yes: allow writer progress in unmocked probe tests.
        return SimpleNamespace(logits=logits)
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
        pair={'before':{'input_ids':[10,11], 'pixel_values':None},
              'after':{'input_ids':[20,21], 'pixel_values':None}}
        with patch.object(ev,'sample',return_value=2), patch.object(ev.SuffixProbe,'check',return_value=(True,1.,0.)), \
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


class SuffixProbeTests(unittest.IsolatedAsyncioTestCase):
    async def fixture(self, history=20, pending=True):
        llm = FakeLLM()
        prompt, image, thinker, writer = [await llm.create_block() for _ in range(4)]
        thinker.token_ids = [17] * history
        writer.token_ids = [19] * history
        contexts = [ev.AsyncContext([prompt, image, thinker], next_input_id=23 if pending else None),
                    ev.AsyncContext([prompt, image, thinker, writer], next_input_id=29 if pending else None)]
        probe = ev.SuffixProbe(llm, Tokenizer(), ev.Prompting('question'))
        return llm, probe, contexts

    async def test_suffix_work_is_independent_of_history_and_streams_unchanged(self):
        for history in (20, 100_000):
            llm, probe, (thinker, writer) = await self.fixture(history)
            original = [(list(b.token_ids), b.freed) for b in llm.blocks]
            pending = [thinker.next_input_id, writer.next_input_id]
            rng_before = torch.random.get_rng_state().clone()
            await probe.check(thinker, writer)
            self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
            self.assertEqual([(list(b.token_ids), b.freed) for b in llm.blocks[:4]], original)
            self.assertEqual([thinker.next_input_id, writer.next_input_id], pending)
            self.assertEqual(len(llm.prefill_calls), 2)
            tail, tail_view, tail_ids, _ = llm.prefill_calls[0]
            suffix, suffix_view, suffix_ids, _ = llm.prefill_calls[1]
            self.assertEqual(tail_view, [thinker.output_block])
            self.assertEqual(tail_ids, [23])
            self.assertEqual(suffix_view, [thinker.output_block, tail, writer.output_block])
            self.assertEqual(suffix_ids, [29] + probe.suffix.tolist())
            self.assertEqual(probe.input_tokens, len(probe.suffix) + 2)
            self.assertTrue(tail.freed and suffix.freed)
            self.assertEqual(llm.calls, [])  # No decode/forward on the live contexts.

    async def test_repeated_probe_and_missing_pending_tokens(self):
        llm, probe, contexts = await self.fixture(pending=False)
        for _ in range(3):
            await probe.check(*contexts)
        self.assertEqual(probe.calls, 3)
        self.assertEqual(probe.input_tokens, 3 * len(probe.suffix))
        self.assertEqual(probe.pending_tokens, 0)
        self.assertEqual(len(llm.prefill_calls), 3)
        for block, view, ids, _ in llm.prefill_calls:
            self.assertEqual(view, [c.output_block for c in contexts])
            self.assertEqual(ids, probe.suffix.tolist())
            self.assertTrue(block.freed)

    async def test_pending_token_combinations_preserve_order(self):
        for thinker_pending, writer_pending in ((None, 29), (23, None)):
            llm, probe, contexts = await self.fixture()
            contexts[0].next_input_id = thinker_pending
            contexts[1].next_input_id = writer_pending
            await probe.check(*contexts)
            self.assertEqual(probe.pending_tokens, 1)
            self.assertEqual(probe.input_tokens, len(probe.suffix) + 1)
            self.assertEqual(llm.prefill_calls[-1][2],
                             ([] if writer_pending is None else [29]) + probe.suffix.tolist())

    async def test_cleanup_on_either_prefill_failure(self):
        for failure in (1, 2):
            llm, probe, contexts = await self.fixture()
            llm.fail_prefill = failure
            with self.assertRaisesRegex(RuntimeError, 'prefill failure'):
                await probe.check(*contexts)
            self.assertTrue(all(b.freed for b in llm.blocks[4:]))
            self.assertTrue(all(not b.freed for b in llm.blocks[:4]))
            self.assertEqual([c.next_input_id for c in contexts], [23, 29])

    async def test_generation_with_real_probe_path_keeps_budgets_and_frees_blocks(self):
        llm = FakeLLM()
        args = SimpleNamespace(k_steps=2, budget=8, shard_to_prompt=True, shard_to_thinker=True,
                               writer_reminder=True, defer_writer_reminder=True, probe_period=1)
        pair = {'before': {'input_ids': [10, 11], 'pixel_values': None},
                'after': {'input_ids': [20, 21], 'pixel_values': None}}
        with patch.object(ev, 'sample', return_value=2), \
             patch.object(ev, 'ends_with_double_newline', return_value=True):
            result = await ev.generate(llm, Tokenizer(),
                                       {'text_shard_1': 'question', 'text_shard_2': 'correction'}, pair, args)
        self.assertTrue(all(b.freed for b in llm.blocks))
        self.assertEqual([len(ids) for ids in result['token_ids'].values()], [8, 8])
        self.assertTrue(result['image_replaced'])
        self.assertFalse(result['writer_reminder_pending'])
        probes = [e for e in result['events'] if e['event'] == 'probe']
        stats = result['probe_stats']
        self.assertGreater(stats['calls'], 1)
        self.assertEqual(stats['calls'], len(probes))
        self.assertEqual(stats['input_tokens'], stats['calls'] * stats['suffix_tokens'] + stats['pending_tokens'])
        self.assertLessEqual(stats['pending_tokens'], 2 * stats['calls'])
        self.assertEqual(len(llm.image_prefills), 2)

    async def test_probe_barrier_then_decoding_resumes_without_replay(self):
        llm = FakeLLM()
        args = SimpleNamespace(k_steps=-1, budget=8, shard_to_prompt=True, shard_to_thinker=True,
                               writer_reminder=True, defer_writer_reminder=True, probe_period=1)
        pair = {side: {'input_ids': [10, 11], 'pixel_values': None} for side in ('before', 'after')}
        original_check = ev.SuffixProbe.check
        checkpoints = []

        async def checked_probe(probe, thinker, writer):
            before = len(llm.calls)
            snapshot = [(list(ctx.output_block.token_ids), ctx.next_input_id)
                        for ctx in (thinker, writer)]
            await asyncio.sleep(0)  # Allow any wrongly concurrent decodes to run.
            result = await original_check(probe, thinker, writer)
            await asyncio.sleep(0)
            self.assertEqual(len(llm.calls), before)
            self.assertEqual([(list(ctx.output_block.token_ids), ctx.next_input_id)
                              for ctx in (thinker, writer)], snapshot)
            checkpoints.append(before)
            return result

        with patch.object(ev, 'sample', return_value=2), \
             patch.object(ev, 'ends_with_double_newline', return_value=False), \
             patch.object(ev.SuffixProbe, 'check', checked_probe):
            result = await ev.generate(llm, Tokenizer(),
                                       {'text_shard_1': 'question', 'text_shard_2': 'correction'}, pair, args)
        self.assertGreater(len(checkpoints), 1)
        self.assertTrue(all(b > a for a, b in zip(checkpoints, checkpoints[1:])))
        self.assertGreater(len(llm.calls), checkpoints[-1])  # Writer resumes after final probe.
        self.assertEqual([len(ids) for ids in result['token_ids'].values()], [8, 8])
        self.assertTrue(all(b.freed for b in llm.blocks))


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
