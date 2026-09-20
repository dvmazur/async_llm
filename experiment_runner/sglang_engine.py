"""Vanilla SGLang native API. Full append-only prompts; SGLang owns prefix caching.

The block handles below own CPU conversation fragments, NOT manually allocated KV.
No private scheduler hooks, server patches, or token-at-a-time RPCs are used.
"""
import asyncio
from dataclasses import dataclass, field
import importlib.metadata
import hashlib
import math
from pathlib import Path
import random
import time
import warnings

from .blocks import BlockHandle
from .logs import JsonlWriter, atomic, stamp


@dataclass
class Fragment:
    tokens: list = field(default_factory=list)
    images: list = field(default_factory=list)


class Backend:
    def __init__(self, engine, processor, directory, *, request_seeds=True):
        self.llm, self.processor = engine, processor
        self.tokenizer = processor.tokenizer
        self.request_seeds = request_seeds
        self.directory = Path(directory)
        self.requests = JsonlWriter(self.directory/'requests.jsonl')
        self.generations = JsonlWriter(self.directory/'generation.jsonl')
        self.metrics = dict(sampled_tokens=0, restricted_readouts=0,
            forward_telemetry_available=False,
            sampling_seed_policy='per_request' if request_seeds else 'engine_global')
        self.stop_ids = set()
        if self.tokenizer.eos_token_id is not None:
            self.stop_ids.add(self.tokenizer.eos_token_id)
        for tag in ('<|im_end|>', '<|endoftext|>', '<|im_start|>'):
            ids = self.encode(tag)
            if len(ids) == 1:
                self.stop_ids.add(ids[0])

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def new_generator(self, seed):
        # Independent role stream of request seeds, not mini-sglang's Torch RNG.
        return random.Random(seed) if self.request_seeds else None

    async def create_block(self):
        return Fragment()

    async def free_block(self, block):
        block.tokens.clear()
        block.images.clear()

    async def cached_prefix(self, prompt):
        block = await BlockHandle.create(self, 'system')
        block.raw.tokens.extend(self.encode(prompt))
        return block

    async def prefill(self, text, deps, target):
        target.raw.tokens.extend(self.encode(text))
        return [b.raw for b in deps] + [target.raw]

    async def decode(self, token, deps, target):
        target.raw.tokens.append(token)

    async def prefill_messages(self, messages, deps, target):
        from PIL import Image
        # Keep unexpanded image placeholders. SGLang's multimodal processor owns
        # image-token expansion and mRoPE, not the runner's HF processor.
        text = self.processor.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=False)
        target.raw.tokens.extend(self.encode(text))
        for message in messages:
            for item in message['content']:
                if item['type'] == 'image':
                    image = item['image']
                    target.raw.images.append(image.copy() if isinstance(image, Image.Image)
                        else Image.fromarray(image).copy())

    async def request(self, fragments, sampling_params, **kwargs):
        # SGLang 0.5.17 cannot co-batch token_ids_logprob requests with requests
        # lacking this field: get_token_ids_logprobs_raw emits [] for the latter,
        # but move_logprobs_to_cpu unconditionally calls .tolist(). Keep every
        # request on the logprob path with one discarded token ID (empty [] is
        # normalized to None by SGLang). No engine patch or serialized scheduling.
        kwargs.setdefault('return_logprob', True)
        kwargs.setdefault('logprob_start_len', -1)
        kwargs.setdefault('token_ids_logprob', [0])
        ids = [token for f in fragments for token in f.tokens]
        images = [image for f in fragments for image in f.images]
        started = stamp()
        try:
            out = await self.llm.async_generate(input_ids=ids, image_data=images or None,
                sampling_params=sampling_params, **kwargs)
            meta = out['meta_info']
            if meta.get('finish_reason', {}).get('type') == 'abort':
                raise RuntimeError(f'SGLang aborted request: {meta}')
            self.requests.emit(**started, seconds=time.monotonic()-started['monotonic'],
                status='completed', input_text_tokens=len(ids), images=len(images),
                prompt_tokens=meta.get('prompt_tokens'), cached_tokens=meta.get('cached_tokens'),
                completion_tokens=meta.get('completion_tokens'), finish_reason=meta.get('finish_reason'))
            return out
        except BaseException as exc:
            self.requests.emit(**started, status='failed', error=repr(exc))
            raise

    async def generate(self, prompt, deps, target, *, result, generator, budget,
                       temperature, top_k, top_p):
        fragments = await self.prefill(prompt, deps, target)
        sampling_params = dict(max_new_tokens=budget,
            temperature=temperature, top_k=top_k, top_p=top_p,
            stop_token_ids=sorted(self.stop_ids), skip_special_tokens=False,
            no_stop_trim=True)
        if generator is not None:
            sampling_params['sampling_seed'] = generator.randrange(2**31)
        out = await self.request(fragments, sampling_params)
        tokens = out['output_ids']
        sampled = out['meta_info']['completion_tokens']
        if sampled != len(tokens):
            raise RuntimeError('SGLang returned inconsistent token accounting')
        result.sampled_tokens = sampled
        self.metrics['sampled_tokens'] += sampled
        for token in tokens:
            self.generations.emit(kind='sample', token=token)
        # Like mini: stop token counts towards TPS, but is not appended; the next
        # role prompt explicitly closes the assistant turn exactly once.
        visible = []
        for token in tokens:
            if token in self.stop_ids:
                break
            visible.append(token)
        target.raw.tokens.extend(visible)
        result.visible_tokens = len(visible)
        result.text = self.tokenizer.decode(visible, skip_special_tokens=False)

    async def score_tokens(self, fragments, token_ids):
        # Query next-position logprobs for all seven actions. Do not sample from
        # unconstrained text or use a grammar that can change tokenization.
        out = await self.request(fragments, dict(max_new_tokens=1, temperature=0),
            return_logprob=True, logprob_start_len=-1, token_ids_logprob=token_ids)
        pairs = out['meta_info']['output_token_ids_logprobs'][0]
        scores = {item[1]: item[0] for item in pairs}
        logits = [scores[token] for token in token_ids]
        if not all(math.isfinite(x) for x in logits):
            raise RuntimeError(f'Nonfinite SGLang action logprobs: {logits}')
        weights = [math.exp(x-max(logits)) for x in logits]
        self.metrics['restricted_readouts'] += 1
        return [x/sum(weights) for x in weights]

    async def close(self):
        try:
            self.llm.shutdown()
        finally:
            await asyncio.to_thread(self.requests.close)
            await asyncio.to_thread(self.generations.close)
            atomic(self.directory/'engine-totals.json', self.metrics)


async def create_engine(params, directory):
    import sglang
    import torch
    from transformers import AutoProcessor
    unknown = set(params) - {'backend', 'engine_config', 'adapter_options'}
    if unknown:
        raise ValueError(f'unknown engine parameter groups: {unknown}')
    options = params.get('adapter_options', {})
    if set(options) - {'cpu_threads'}:
        raise ValueError(f'unknown adapter options: {set(options)}')
    torch.set_num_threads(options.get('cpu_threads', 4))
    config = dict(params['engine_config'])
    request_seeds = config.get('enable_deterministic_inference', False)
    if not request_seeds:
        warnings.warn('Vanilla SGLang uses the engine-global RNG in normal mode: '
            'per-episode/per-role model seeds are NOT applied. Set '
            'enable_deterministic_inference=True for per-request seeds; this also '
            'changes kernels/sampling performance. World seeds are unchanged.')
    if config.get('tp_size', 1) != 1 or config.get('dp_size', 1) != 1:
        raise ValueError('Runner assigns one independent SGLang replica per GPU, no TP/DP')
    # Engine binds async_generate to the running worker event loop.
    started = stamp()
    llm = sglang.Engine(**config)
    try:
        processor = AutoProcessor.from_pretrained(config['model_path'])
        packages = {name: importlib.metadata.version(name)
            for name in ('sglang', 'torch', 'transformers', 'flashinfer-python')}
        atomic(Path(directory)/'provenance.json', dict(backend='sglang', packages=packages,
            gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
            lock_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/
                'environment/sglang/uv.lock').read_bytes()).hexdigest(),
            resolved_engine_params=params, initialization_start=started,
            initialization_end=stamp(), forward_telemetry_available=False,
            sampling_seed_policy='per_request' if request_seeds else 'engine_global'))
        return Backend(llm, processor, directory, request_seeds=request_seeds)
    except BaseException:
        llm.shutdown()
        raise
