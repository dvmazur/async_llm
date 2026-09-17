"""Four-block async image correction evaluation. No OpenRouter calls.

Persistent views: thinker=[prompt,image,thinker], writer=[prompt,image,thinker,writer].
One transient text-only probe reads stream snapshots, as in the text experiment.
All mutations happen between completed decode rounds, never while a reader is live.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time

import torch
from PIL import Image
from transformers import AutoProcessor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'async_thoughts'))
sys.path.insert(0, str(HERE / 'dataset_construction'))
from async_thoughts.demo import Prompting, _tokens_with_pending
from async_thoughts.engine import encode, ends_with_double_newline, single_token_id, vocab_id_or_none
from package_dataset import validate_package
from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext
from fixed_subset import apply_manifest

REMINDER = ' ... [SYSTEM: additional user input detected; recheck the updated image]\n'


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=HERE/'dataset_construction/datasets/diverse_corrections_v3.parquet')
    p.add_argument('--model-name', default='Qwen/Qwen3.8-27B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--k-steps', type=int, default=64, help='Thinker tokens before replacement; 0=corrected from start, -1=never')
    p.add_argument('--budget', type=int, default=16384, help='Generated-token cap per stream (injected text excluded)')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int)
    p.add_argument('--sample-manifest', type=Path, help='Frozen ordered sample IDs and dataset checksum')
    p.add_argument('--shard-to-prompt', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--shard-to-thinker', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--writer-reminder', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--defer-writer-reminder', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--probe-period', type=int, default=30)
    p.add_argument('--temperature', type=float, default=.6)
    p.add_argument('--top-p', type=float, default=.95)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--memory-ratio', type=float, default=.8)
    p.add_argument('--kv-tokens', type=int, default=98304, help='Explicit KV capacity; 0 uses memory-ratio')
    p.add_argument('--max-prefill-rows', type=int, default=256)
    p.add_argument('--max-seq-len', type=int, default=65536)
    p.add_argument('--max-image-pixels', type=int, default=1048576)
    p.add_argument('--distributed-port', type=int, default=2371)
    p.add_argument('--prepare-only', action='store_true', help='CPU processor/parity checks only')
    a = p.parse_args()
    if a.sample_manifest and (a.start != 0 or a.end is not None):
        p.error('Use either a fixed sample manifest or start/end, not both')
    if a.budget <= 0 or a.probe_period <= 0 or a.k_steps < -1 or a.k_steps >= a.budget:
        p.error('Require budget>0, probe-period>0 and -1 <= k-steps < budget')
    if a.temperature < 0 or not 0 < a.top_p <= 1 or a.top_k < 0:
        p.error('Invalid sampling parameters')
    if a.kv_tokens < 0 or a.max_prefill_rows < 1 or a.max_image_pixels < 65536:
        p.error('Invalid cache/prefill/image limits')
    return a


def sample(logits, forbidden, a):
    scores = logits.flatten().float().clone()
    scores[forbidden] = -float('inf')
    if a.temperature == 0:
        return int(scores.argmax())
    scores /= a.temperature
    if 0 < a.top_k < scores.numel():
        scores[scores < torch.topk(scores, a.top_k).values[-1]] = -float('inf')
    if a.top_p < 1:
        values, order = scores.sort(descending=True)
        remove = values.softmax(-1).cumsum(-1) > a.top_p
        remove[1:] = remove[:-1].clone(); remove[0] = False
        scores[order[remove]] = -float('inf')
    return int(torch.multinomial(scores.softmax(-1), 1))


def image_inputs(processor, asset, max_pixels):
    with Image.open(io.BytesIO(asset['bytes'])) as im:
        picture = im.convert('RGB')
    values = processor(text='<|vision_start|>'+processor.image_token+'<|vision_end|>\n',
                       images=picture, return_tensors='pt',
                       images_kwargs={'size': {'shortest_edge': 65536, 'longest_edge': max_pixels}})
    ids = values['input_ids'].flatten().to(torch.int32)
    image_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    types = values.get('mm_token_type_ids', (ids == image_id).long()).flatten()
    if int((types == 1).sum()) == 0:
        raise ValueError('Processor produced no image tokens')
    return {'input_ids': ids, 'pixel_values': values['pixel_values'],
            'image_grid_thw': values['image_grid_thw'], 'mm_token_type_ids': types}


def prepare_pair(processor, row, a):
    pair = {s: image_inputs(processor, row['inputs']['image_'+s], a.max_image_pixels)
            for s in ('before', 'after')}
    for field in ('input_ids', 'image_grid_thw', 'mm_token_type_ids'):
        if not torch.equal(pair['before'][field], pair['after'][field]):
            raise ValueError(f'Image processor parity mismatch: {row["id"]}/{field}')
    return pair


async def replace_image(llm, old_image, prompt, prepared, contexts):
    """Caller guarantees no active forward/probe. Retain text caches unchanged."""
    await llm.free_block(old_image)
    new_image = await llm.create_block()
    try:
        await llm.prefill_block(**prepared, cache_view=[prompt], write_to=new_image)
    except BaseException:
        await llm.free_block(new_image)
        raise
    for ctx in contexts:
        ctx.cache_view = [new_image if b is old_image else b for b in ctx.cache_view]
    return new_image


async def append_note(llm, ctx, block, ids):
    # Preserve the last emitted token before inserting a control/input fragment.
    if ctx.next_input_id is not None:
        await llm.forward(cache_view=ctx)
        ctx.next_input_id = None
    out = await llm.forward(ids, cache_view=ctx.cache_view, write_to=block)
    return out.logits  # sampled and counted on the next normal decode round


async def probe_once(llm, tokenizer, prompting, thinker, writer):
    block = await llm.create_block()
    try:
        ids = torch.cat([encode(prompting.mode_switching_prompt, tokenizer),
                         torch.tensor(thinker, dtype=torch.int32),
                         torch.tensor(writer, dtype=torch.int32),
                         encode(prompting.mode_switching_question, tokenizer)])
        out = await llm.forward(ids, write_to=block)
        yes = float(out.logits[single_token_id('yes', tokenizer)])
        no = float(out.logits[single_token_id('no', tokenizer)])
        return yes > no, yes, no
    finally:
        await llm.free_block(block)


async def generate(llm, tokenizer, inputs, pair, a):
    # Only the input subrecord enters generation, never labels or provenance.
    question = inputs['text_shard_1']
    if a.k_steps == 0:
        question += '\n\n' + inputs['text_shard_2']
    prompting = Prompting('Reason step by step and put the final answer in \\boxed{}.\n\n'+question)
    blocks = [await llm.create_block() for _ in range(4)]
    prompt, image, thinker, writer = blocks
    emitted = {'thinker': [], 'writer': []}
    events = []
    injected = a.k_steps == 0
    pending_reminder = False
    writer_active = False
    hit_eos = False
    queued = {}
    try:
        await llm.forward(encode(prompting.input_prompt, tokenizer), write_to=prompt, return_logits=False)
        await llm.prefill_block(**pair['after' if injected else 'before'], cache_view=[prompt], write_to=image)
        await llm.forward(encode(prompting.thinker_output_prefix, tokenizer), [prompt,image,thinker], return_logits=False)
        await llm.forward(encode(prompting.writer_output_prefix, tokenizer), [prompt,image,thinker,writer], return_logits=False)
        nn = single_token_id('\n\n', tokenizer)
        contexts = {'thinker': AsyncContext([prompt,image,thinker], next_input_id=nn),
                    'writer': AsyncContext([prompt,image,thinker,writer], next_input_id=nn)}
        forbid = {role: [i for name in names if (i := vocab_id_or_none(tokenizer,name)) is not None]
                  for role,names in [('thinker',['</think>','<|im_start|>','<|im_end|>','<|endoftext|>']),
                                     ('writer',['</think>','<|im_start|>','<|endoftext|>'])]}
        async def tick(role):
            ctx = contexts[role]
            logits = queued.pop(role) if role in queued else (await llm.forward(cache_view=ctx)).logits
            token = sample(logits, forbid[role], a)
            ctx.next_input_id = token
            emitted[role].append(token)
            if len(emitted[role]) % 128 == 0:
                print(f"  {role}: {len(emitted[role])} tokens",flush=True)
            return token

        while len(emitted['writer']) < a.budget:
            n = len(emitted['thinker'])
            if not injected and a.k_steps > 0 and n >= a.k_steps:
                # No pending engine requests at this boundary. Image sees the
                # updated prompt; historical thinker/writer KV is not replayed.
                if a.shard_to_prompt:
                    await llm.forward(encode('\n\n'+inputs['text_shard_2']+'\n',tokenizer),
                                      [prompt], write_to=prompt, return_logits=False)
                blocks.remove(image)  # replace_image owns freeing the old block
                image = await replace_image(llm,image,prompt,pair['after'],contexts.values())
                blocks.append(image)
                if a.shard_to_thinker:
                    queued['thinker'] = await append_note(llm,contexts['thinker'],thinker,
                        encode('\n\n'+inputs['text_shard_2']+'\n',tokenizer))
                if a.writer_reminder:
                    if a.defer_writer_reminder:
                        pending_reminder = True
                    else:
                        queued['writer'] = await append_note(llm,contexts['writer'],writer,encode(REMINDER,tokenizer))
                        events.append({'event':'writer_reminder','deferred':False,
                                       'thinker_tokens':n,'writer_tokens':len(emitted['writer'])})
                injected = True
                events.append({'event':'image_replaced','thinker_tokens':n,'writer_tokens':len(emitted['writer'])})
                print(f"  image replaced at thinker={n}, writer={len(emitted['writer'])}",flush=True)
            thinker_done = n >= a.budget
            if thinker_done:
                writer_active = True
            roles = ([] if thinker_done else ['thinker']) + (['writer'] if writer_active else [])
            # Barrier waits for BOTH decodes before any shared-cache mutation.
            tokens = await asyncio.gather(*(tick(role) for role in roles), return_exceptions=True)
            for token in tokens:
                if isinstance(token, BaseException):
                    raise token
            for role, token in zip(roles,tokens):
                if role == 'writer':
                    if token == tokenizer.eos_token_id:
                        hit_eos = True
                        break
                    boundary = token == nn or ends_with_double_newline(emitted['writer'],tokenizer)
                    if boundary and pending_reminder:
                        queued['writer'] = await append_note(llm,contexts['writer'],writer,encode(REMINDER,tokenizer))
                        pending_reminder = False
                        events.append({'event':'writer_reminder','thinker_tokens':len(emitted['thinker']),
                                       'writer_tokens':len(emitted['writer'])})
                    if boundary and not thinker_done:
                        writer_active = False
            if hit_eos:
                break
            n = len(emitted['thinker'])
            if not thinker_done and (n % a.probe_period == 0 or ends_with_double_newline(emitted['thinker'],tokenizer)):
                writer_active, yes, no = await probe_once(llm,tokenizer,prompting,
                    _tokens_with_pending(thinker,contexts['thinker']),_tokens_with_pending(writer,contexts['writer']))
                events.append({'event':'probe','thinker_tokens':n,'writer_tokens':len(emitted['writer']),
                               'write':writer_active,'yes_logit':yes,'no_logit':no})
        return {'generated_text':tokenizer.decode(emitted['writer'],skip_special_tokens=True),
                'thinker_text':tokenizer.decode(emitted['thinker'],skip_special_tokens=True),
                'token_ids':emitted,'events':events,'image_replaced':any(e['event']=='image_replaced' for e in events),
                'final_image':'after' if injected else 'before','hit_eos':hit_eos,
                'writer_reminder_pending':pending_reminder}
    finally:
        for block in reversed(blocks):
            await llm.free_block(block)


def boxed(text):
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    start += len('\\boxed{'); depth = 1
    for i in range(start,len(text)):
        depth += (text[i]=='{') - (text[i]=='}')
        if depth == 0:
            return text[start:i].strip()
    return None


def equivalent(predicted, target, choices=()):
    if predicted is None:
        return False
    def norm(value):
        value = re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}',r'\1',str(value))
        value = value.strip().strip('$').strip()
        letter = value.strip('(). ')
        if len(letter)==1 and 'A'<=letter<='Z' and ord(letter)-65<len(choices):
            value = str(choices[ord(letter)-65])
        return re.sub(r'\s+','',value).casefold()
    if norm(predicted)==norm(target):
        return True
    if ';' in str(target):
        return {norm(v) for v in re.split('[;,]',predicted)} == {norm(v) for v in str(target).split(';')}
    # Symbolic equivalence is reserved for mathematical answers, not prose.
    if re.fullmatch(r'[\d\s.+*/^(){}\\_%-]+|\\frac[\d{}\\\s.+*/^-]+',str(target)):
        from math_verify import parse, verify
        return bool(verify(parse('$'+str(target)+'$'),parse('$'+predicted+'$')))
    return False


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def summarize(results, requested):
    def metrics(items):
        n = len(items)
        return {'completed':n,'correct_after':sum(r['correct_after'] for r in items),
                'accuracy_after':sum(r['correct_after'] for r in items)/n if n else None,
                'matches_before':sum(r['matches_before'] for r in items),
                'image_replacements':sum(r['image_replaced'] for r in items),
                'missing_boxed_answers':sum(r['predicted_answer'] is None for r in items)}
    summary = {**metrics(results),'requested':requested,'complete':len(results)==requested}
    for field in ('dataset','difficulty_group','category'):
        summary['by_'+field] = {key:metrics([r for r in results if r[field]==key])
                               for key in sorted({r[field] for r in results})}
    return summary


async def run(a):
    rows = validate_package(a.dataset)
    dataset_sha = hashlib.sha256(a.dataset.read_bytes()).hexdigest()
    original_indices = {r['id']:idx for idx,r in enumerate(rows)}
    manifest = None
    if a.sample_manifest:
        manifest = json.loads(a.sample_manifest.read_text())
        rows = apply_manifest(rows,manifest,dataset_sha)
    end = len(rows) if a.end is None else a.end
    if not 0 <= a.start < end <= len(rows):
        raise ValueError('Invalid sample range')
    rows = rows[a.start:end]
    processor = AutoProcessor.from_pretrained(a.model_name)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(dataset_sha256=dataset_sha, sample_manifest=manifest,
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol='four_blocks_image_replace_v1',hf_home=os.environ.get('HF_HOME'),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    a.output.mkdir(parents=True,exist_ok=True)
    old_config = a.output/'config.json'
    if old_config.exists() and json.loads(old_config.read_text()) != config:
        raise ValueError('Run configuration changed: choose a fresh output directory')
    save(old_config,config)
    if a.prepare_only:
        for row in rows:
            pair = prepare_pair(processor,row,a)
            print(row['id'], pair['before']['image_grid_thw'].tolist(),len(pair['before']['input_ids']),flush=True)
        save(a.output/'preflight.json',{'count':len(rows),'processor_parity':True})
        return
    llm = AsyncLLM(a.model_name,dtype=torch.bfloat16,max_running_req=4,
                   cuda_graph_bs=[1,2],cuda_graph_max_bs=2,memory_ratio=a.memory_ratio,
                   page_size=1,max_seq_len_override=a.max_seq_len,
                   num_page_override=a.kv_tokens or None,max_prefill_rows=a.max_prefill_rows,
                   distributed_addr=f'tcp://127.0.0.1:{a.distributed_port}')
    results = []
    try:
        for offset,row in enumerate(rows,a.start):
            path = a.output/(row['id']+'.json')
            if path.exists():
                results.append(json.loads(path.read_text())); continue
            started = time.monotonic()
            try:
                print(f"[{offset}] starting {row['id']} ({row['dataset']})",flush=True)
                pair = prepare_pair(processor,row,a)
                prompt_tokens = len(encode(row['inputs']['text_shard_1'],processor.tokenizer))
                if prompt_tokens + len(pair['before']['input_ids']) + 2*a.budget + 2048 > a.max_seq_len:
                    raise ValueError('Context budget unsafe; increase --max-seq-len')
                sample_seed = a.seed + original_indices[row['id']]
                torch.manual_seed(sample_seed); torch.cuda.manual_seed_all(sample_seed)
                result = await generate(llm,processor.tokenizer,row['inputs'],pair,a)
                pred = boxed(result['generated_text']); labels = row['labels']
                result.update(id=row['id'],idx=original_indices[row['id']],subset_position=offset,
                    sample_seed=sample_seed,dataset=row['dataset'],category=row['category'],
                    difficulty_group=row['difficulty_group'],predicted_answer=pred,
                    answer_before=labels['answer_before'],answer_after=labels['answer_after'],
                    correct_after=equivalent(pred,labels['answer_after'],labels['choices']),
                    matches_before=equivalent(pred,labels['answer_before'],labels['choices']),
                    image_grid_thw=pair['before']['image_grid_thw'].tolist(),
                    image_tokens=int((pair['before']['mm_token_type_ids']==1).sum()),
                    elapsed_seconds=time.monotonic()-started)
                save(path,result); results.append(result)
                save(a.output/'summary.json',summarize(results,len(rows)))
                print(f"[{offset}] {row['id']} correct_after={result['correct_after']} replaced={result['image_replaced']} seconds={result['elapsed_seconds']:.1f}",flush=True)
            except Exception as exc:
                save(a.output/(row['id']+'.error.json'),{'id':row['id'],'error':repr(exc)})
                raise  # engine state may be invalid; never silently drop failures
        save(a.output/'summary.json',summarize(results,len(rows)))
    finally:
        await llm.close()


if __name__ == '__main__':
    asyncio.run(run(arguments()))
