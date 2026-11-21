import unittest
from types import SimpleNamespace

import torch

from robot_sim.candidate_nav.system_prefix import SystemPrefixCache, split_inputs
from robot_sim.candidate_nav.llm import AsyncCandidatePolicy


class Engine:
    def __init__(self):
        self.live, self.calls, self.counter = set(), [], 0
        self.fail = False

    async def create_block(self):
        self.counter += 1
        self.live.add(self.counter)
        return self.counter

    async def free_block(self, block):
        self.live.remove(block)

    async def forward(self, tokens=None, **kwargs):
        self.calls.append((tokens, kwargs))
        if self.fail:
            raise RuntimeError("prefill failed")


class PrefixTests(unittest.IsolatedAsyncioTestCase):
    async def test_reuse_invalidation_eviction_and_cleanup(self):
        e = Engine()
        c = SystemPrefixCache(e, capacity=1)
        a = await c.acquire((1, 2), torch.tensor([[1, 2]]))
        self.assertFalse(a.hit)
        await c.release(a)
        b = await c.acquire((1, 2), torch.tensor([[1, 2]]))
        self.assertTrue(b.hit)
        self.assertEqual(a.block, b.block)
        self.assertEqual(len(e.calls), 1)
        await c.release(b)
        d = await c.acquire((1, 3), torch.tensor([[1, 3]]))
        self.assertFalse(d.hit)
        self.assertNotIn(a.block, e.live)
        await c.release(d)
        await c.close()
        self.assertFalse(e.live)

    async def test_active_prefix_is_not_evicted(self):
        e = Engine()
        c = SystemPrefixCache(e, capacity=1)
        a = await c.acquire((1,), torch.tensor([[1]]))
        a2 = await c.acquire((1,), torch.tensor([[1]]))
        b = await c.acquire((2,), torch.tensor([[2]]))
        self.assertFalse(b.cached)
        await c.release(a)
        with self.assertRaises(RuntimeError):
            await c.close()
        await c.release(a2)
        await c.release(b)
        self.assertEqual(e.live, {a.block})
        await c.close()

    async def test_failed_prefix_is_freed_and_not_cached(self):
        e = Engine()
        c = SystemPrefixCache(e)
        e.fail = True
        with self.assertRaises(RuntimeError):
            await c.acquire((1,), torch.tensor([[1]]))
        self.assertFalse(e.live)
        self.assertFalse(c.entries)

    def test_multimodal_split_preserves_images_and_masks(self):
        x = dict(input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
                 mm_token_type_ids=torch.tensor([[0, 0, 0, 1, 1]]),
                 pixel_values=torch.randn(2, 8), image_grid_thw=torch.tensor([[1, 1, 2]]))
        prefix, tail = split_inputs(x, (1, 2))
        self.assertEqual(prefix.tolist(), [[1, 2]])
        self.assertEqual(tail['input_ids'].tolist(), [[3, 4, 5]])
        self.assertEqual(tail['mm_token_type_ids'].tolist(), [[0, 1, 1]])
        self.assertIs(tail['pixel_values'], x['pixel_values'])
        self.assertIs(tail['image_grid_thw'], x['image_grid_thw'])
        self.assertEqual(x['input_ids'].shape[1], 5)
        with self.assertRaises(ValueError):
            split_inputs(x, (1, 9))
        with self.assertRaises(ValueError):
            split_inputs(x, (1, 2, 3, 4))

    async def test_policy_fresh_observations_share_only_system(self):
        e = Engine()
        e.processor = SimpleNamespace(
            apply_chat_template=lambda *a, **k: 'system<|im_start|>user\nplaceholder',
            tokenizer=SimpleNamespace(encode=lambda *a, **k: [1, 2]))
        p = AsyncCandidatePolicy.__new__(AsyncCandidatePolicy)
        p.llm = e
        p.system_prefix_cache = SystemPrefixCache(e)
        leases = []
        for last in (3, 4):
            common = await e.create_block()
            view, lease, metrics = await p.prefill_observation(
                dict(input_ids=torch.tensor([[1, 2, last]])), 'system', common, True)
            leases.append(lease)
            self.assertEqual(metrics['prefilled_system_tokens'], 2 if last == 3 else 0)
            self.assertEqual(metrics['system_prefix_cache_hit'], last == 4)
            self.assertEqual(e.calls[-1][0], None)
            self.assertEqual(e.calls[-1][1]['input_ids'].tolist(), [[last]])
            self.assertEqual(e.calls[-1][1]['cache_view'], [view[0]])
            await e.free_block(common)
            await p.system_prefix_cache.release(lease)
        self.assertEqual(leases[0].block, leases[1].block)
        await p.system_prefix_cache.close()
        self.assertFalse(e.live)

    async def test_simulation_prefills_full_input(self):
        e = Engine()
        p = AsyncCandidatePolicy.__new__(AsyncCandidatePolicy)
        p.llm = e
        p.system_prefix_cache = SystemPrefixCache(e)
        view, lease, metrics = await p.prefill_observation(
            dict(input_ids=torch.tensor([[1, 2, 3]])), 'system', 9, False)
        self.assertEqual(view, [9])
        self.assertIsNone(lease)
        self.assertFalse(metrics['system_prefix_cache_enabled'])
        self.assertEqual(e.calls[-1][1]['input_ids'].tolist(), [[1, 2, 3]])
        self.assertIsNone(e.calls[-1][1]['cache_view'])

    async def test_failed_observation_releases_its_prefix_lease(self):
        e = Engine()
        e.processor = SimpleNamespace(
            apply_chat_template=lambda *a, **k: 'system<|im_start|>user\nplaceholder',
            tokenizer=SimpleNamespace(encode=lambda *a, **k: [1, 2]))
        p = AsyncCandidatePolicy.__new__(AsyncCandidatePolicy)
        p.llm = e
        p.system_prefix_cache = SystemPrefixCache(e)
        lease = await p.system_prefix_cache.acquire((1, 2), torch.tensor([[1, 2]]))
        await p.system_prefix_cache.release(lease)
        e.fail = True
        with self.assertRaises(RuntimeError):
            await p.prefill_observation(dict(input_ids=torch.tensor([[1, 2, 3]])),
                                       'system', 9, True)
        await p.system_prefix_cache.close()
        self.assertFalse(e.live)
