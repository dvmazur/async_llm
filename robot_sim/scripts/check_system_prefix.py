"""Offline two-rank cache verification. No ROS imports, sockets or motor calls.

Run through torchrun with the same model/env as run_candidate_llm.sh, while the
managed model service is stopped. Writes a rank-zero JSON evidence file.
"""
import argparse
import asyncio
import json
import time
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw

from robot_sim.candidate_nav.llm import AsyncCandidatePolicy
from robot_sim.candidate_nav import task_policy
from robot_sim.candidate_nav.system_prefix import SystemPrefixCache


async def check(args):
    torch.set_num_threads(1)
    policy = AsyncCandidatePolicy(SimpleNamespace(
        model=args.model, quantization='fp8', distributed_addr='env://',
        analysis_tokens=24, no_system_prefix_cache=False))
    llm = policy.llm
    results = []
    try:
        for variant in (0, 1):
            camera = Image.new('RGB', (320, 240), ('silver', 'tan')[variant])
            ImageDraw.Draw(camera).rectangle((40 + variant * 60, 50, 140 + variant * 60, 160), fill='brown')
            map_image = Image.new('RGB', (512, 512), 'gray')
            ImageDraw.Draw(map_image).rectangle((200, 100, 300, 450), fill='white')
            messages = [dict(role='system', content=task_policy.TRAJECTORY_SYSTEM),
                        dict(role='user', content=[
                            dict(type='image', image=camera), dict(type='image', image=map_image),
                            dict(type='text', text=json.dumps(dict(
                                instruction='Find the box and approach to a distance of 5 centimeters.',
                                snapshot_id=variant + 1, pose=[0, variant * .25, 0],
                                history=['turned right'] if variant else [],
                                offline_test=True)))])]
            inputs = llm.processor.apply_chat_template(messages, add_generation_prompt=False,
                tokenize=True, return_dict=True, return_tensors='pt', enable_thinking=False)
            inputs.pop('attention_mask', None)
            # The production policy prefills without logits; append the exact
            # same question to compare both cache layouts including decode.
            query = policy.encode('<|im_start|>user\nDescribe the camera briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n')
            reference, tail_block, ref_query, cached_query = [await llm.create_block() for _ in range(4)]
            lease = None
            try:
                started = time.monotonic()
                await llm.forward(**dict(inputs), write_to=reference, return_logits=False)
                full_s = time.monotonic() - started
                full_out = await llm.forward(query, cache_view=[reference, ref_query])
                view, lease, metrics = await policy.prefill_observation(inputs,
                    task_policy.TRAJECTORY_SYSTEM, tail_block, True)
                cached_out = await llm.forward(query, cache_view=view + [cached_query])
                # Control: identical partitioning with a newly calculated system
                # every time. This must exactly match reuse of the saved state.
                fresh_cache = SystemPrefixCache(llm)
                saved_cache = policy.system_prefix_cache
                policy.system_prefix_cache = fresh_cache
                fresh_tail = await llm.create_block()
                fresh_query = await llm.create_block()
                try:
                    fresh_view, fresh_lease, _ = await policy.prefill_observation(
                        inputs, task_policy.TRAJECTORY_SYSTEM, fresh_tail, True)
                finally:
                    policy.system_prefix_cache = saved_cache
                fresh_out = await llm.forward(query, cache_view=fresh_view + [fresh_query])
                differences, top_matches, distribution_l1, reuse_max_abs = [], [], [], []
                for step in range(4):
                    a, b = full_out.logits.float(), cached_out.logits.float()
                    differences.append(dict(max_abs=float((a-b).abs().max()),
                                            mean_abs=float((a-b).abs().mean())))
                    top_matches.append(int(a.argmax()) == int(b.argmax()))
                    distribution_l1.append(float((a.softmax(-1)-b.softmax(-1)).abs().sum()))
                    reuse_max_abs.append(float((b-fresh_out.logits.float()).abs().max()))
                    # Teacher forcing isolates numerical error from diverging text.
                    token = int(a.argmax())
                    full_out = await llm.forward([token], cache_view=[reference, ref_query])
                    cached_out = await llm.forward([token], cache_view=view + [cached_query])
                    fresh_out = await llm.forward([token], cache_view=fresh_view + [fresh_query])
                item = dict(variant=variant, full_tokens=int(inputs['input_ids'].numel()),
                            full_prefill_s=full_s, **metrics, differences=differences,
                            top_matches=top_matches, distribution_l1=distribution_l1, reuse_max_abs=reuse_max_abs)
                results.append(item)
                if policy.rank == 0:
                    print('PREFIX_CHECK ' + json.dumps(item), flush=True)
                if metrics['system_prefix_cache_hit'] != bool(variant):
                    raise AssertionError('Expected cold then warm system prefix')
                await llm.free_block(fresh_query)
                await llm.free_block(fresh_tail)
                await fresh_cache.release(fresh_lease)
                await fresh_cache.close()
                if max(reuse_max_abs) != 0:
                    raise AssertionError('Reused prefix differs from freshly calculated prefix')
                if not all(top_matches):
                    raise AssertionError('Full versus split prefill changed greedy probe tokens')
            finally:
                for block in (reference, tail_block, ref_query, cached_query):
                    await llm.free_block(block)
                if lease is not None:
                    await policy.system_prefix_cache.release(lease)
        if policy.rank == 0:
            with open(args.output, 'w') as f:
                json.dump(dict(passed=True, metadata=policy.metadata, results=results), f, indent=2)
    finally:
        await policy.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    asyncio.run(check(parser.parse_args()))
