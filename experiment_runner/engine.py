"""Mini-sgl operations and aggregate telemetry; no role or game policy."""
import hashlib
import json
import time
from pathlib import Path

import torch


from .logs import JsonlWriter, atomic, stamp
from .blocks import BlockHandle
from .generation import generate
import asyncio


class Backend:
    generate = generate
    def __init__(self, llm, output):
        self.llm, self.output = llm, Path(output)
        self.prefixes = {}
        self.prefix_lock = asyncio.Lock()
        self.metrics = dict(sampled_tokens=0, restricted_readouts=0, snapshot_tokens=0,
                            snapshot_seconds=0.)
        self.forward_log = JsonlWriter(self.output/'forwards.jsonl')
        self.generation_log = JsonlWriter(self.output/'generation.jsonl')
        self.original_forward = llm.async_engine.session._forward
        self.forward_index = 0
        def forward(batch, *args, **kwargs):
            self.forward_index += 1
            started = stamp()
            decode = batch.size if batch.is_decode else getattr(batch, 'num_decode', 0)
            rows = decode if batch.is_decode else batch.input_ids.numel()
            fields = dict(index=self.forward_index,
                phase='decode' if batch.is_decode else ('mixed' if decode else 'prefill'),
                decode_requests=decode, prefill_requests=batch.size-decode,
                prefill_rows=rows-decode, input_rows=rows)
            runner = getattr(llm.async_engine.session, 'graph_runner', None)
            before = ((runner.replay_count, runner.prefill_replay_count) if runner else (0, 0))
            try:
                result = self.original_forward(batch, *args, **kwargs)
            except BaseException as exc:
                self.forward_log.emit(**started, **fields, status='failed', error=repr(exc))
                raise
            after = ((runner.replay_count, runner.prefill_replay_count) if runner else (0, 0))
            padded = None
            if after != before:
                padded = batch.padded_size if batch.is_decode else batch.attn_metadata.graph_buffers.rows
            self.forward_log.emit(**started, **fields, status='completed',
                host_seconds=time.monotonic()-started['monotonic'],
                graph_replay=after != before,
                padded_capacity=padded,
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved())
            return result
        llm.async_engine.session._forward = forward

    async def cached_prefix(self, prompt):
        async with self.prefix_lock:
            if prompt not in self.prefixes:
                block = await BlockHandle.create(self, 'shared/system')
                try:
                    await self.prefill(prompt, [], block)
                except BaseException:
                    await block.aclose()
                    raise
                self.prefixes[prompt] = block
            return self.prefixes[prompt].share()

    async def close(self):
        try:
            for block in self.prefixes.values():
                await block.aclose()
            self.prefixes.clear()
        finally:
            try:
                self.llm.async_engine.session._forward = self.original_forward
                await self.llm.close()
            finally:
                await asyncio.to_thread(self.forward_log.close)
                await asyncio.to_thread(self.generation_log.close)
                atomic(self.output/'engine-totals.json', dict(self.metrics, forwards=self.forward_index))

    async def create_block(self):
        return await self.llm.create_block()

    async def free_block(self, block):
        await self.llm.free_block(block)

    async def merge_blocks(self, left, right):
        started = time.perf_counter()
        result = await self.llm.merge_blocks(left, right)
        self.metrics['snapshot_tokens'] += left.num_tokens + right.num_tokens
        self.metrics['snapshot_seconds'] += time.perf_counter()-started
        return result

    async def prefill(self, text, deps, target):
        ids = self.llm.tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids']
        return await self.llm(ids, cache_view=[b.raw for b in deps]+[target.raw])

    async def decode(self, token, deps, target):
        return await self.llm(torch.tensor([token], dtype=torch.int32),
            cache_view=[b.raw for b in deps]+[target.raw])

    async def prefill_messages(self, messages, deps, target):
        inputs = self.llm.processor.apply_chat_template(messages,
            add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors='pt')
        return await self.llm(**inputs, cache_view=[b.raw for b in deps]+[target.raw])

    def new_generator(self, seed):
        return torch.Generator(device=self.llm.engine.device).manual_seed(seed)

    def encode(self, text):
        return self.llm.tokenizer.encode(text, add_special_tokens=False)

    def sample(self, output, *, generator, temperature, top_k, top_p):
        # Importing this module initializes CUDA on current FlashInfer. Engine
        # construction must own the first CUDA initialization in this process.
        import flashinfer.sampling as sampling
        if temperature <= 0:
            raise ValueError('sampling temperature must be positive')
        logits = output.logits.float().view(1, -1)
        probs = torch.softmax(logits / temperature, dim=-1)
        value = sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p,
                    generator=generator).item()
        piece = self.llm.tokenizer.decode([value])
        eos = self.llm.tokenizer.eos_token_id
        self.metrics['sampled_tokens'] += 1
        self.generation_log.emit(kind='sample', token=int(value))
        return int(value), piece, value == eos

    async def score_tokens(self, output, token_ids):
        logits = output.logits[token_ids].float()
        probabilities = logits.softmax(-1).tolist()
        self.metrics['restricted_readouts'] += 1
        return probabilities


async def create_engine(params, directory):
    import dataclasses
    import importlib.metadata
    import socket
    import subprocess
    import sys
    import transformers
    from minisgl.engine import EngineConfig
    from minisgl.llm import AsyncLLM
    from .fi_warmup import warmup_sampling

    unknown = set(params) - {'engine_config', 'adapter_options'}
    if unknown:
        raise ValueError(f'unknown engine parameter groups: {unknown}')
    config = dict(params.get('engine_config', {}))
    options = dict(params.get('adapter_options', {}))
    if set(options) - {'cpu_threads'}:
        raise ValueError(f'unknown adapter options: {set(options)}')
    torch.set_num_threads(options.get('cpu_threads', 4))
    allowed = {f.name for f in dataclasses.fields(EngineConfig)} - {'tp_info'}
    if set(config) - allowed:
        raise ValueError(f'unknown/unsupported EngineConfig fields: {set(config)-allowed}')
    model = config.pop('model_path')
    dtype_name = config.pop('dtype', 'bfloat16')
    if dtype_name not in ('bfloat16', 'float16', 'float32'):
        raise ValueError(f'unsupported dtype {dtype_name}')
    config['generation_config'] = transformers.GenerationConfig(**config.get('generation_config',
        dict(do_sample=True, temperature=.6, top_k=20, top_p=.9)))
    if 'distributed_addr' not in config:
        # No TP; avoid independent GPU-workers all binding the upstream default 2333.
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            config['distributed_addr'] = f'tcp://127.0.0.1:{sock.getsockname()[1]}'
    phases = JsonlWriter(Path(directory)/'initialization.jsonl')
    from contextlib import contextmanager
    @contextmanager
    def phase(name):
        started = stamp()
        try:
            yield
        finally:
            phases.emit(kind=name, **started, ended_monotonic=time.monotonic())
    try:
        # Backend owns its buffers and construction. Do not monkeypatch model
        # loading / graph-capture methods to instrument their internal phases.
        with phase('engine_construction'):
            llm = AsyncLLM(model, dtype=getattr(torch, dtype_name), **config)
    finally:
        await asyncio.to_thread(phases.close)
    try:
        sampling_started = stamp()
        warmup_sampling(llm.engine.device, llm.engine.config.model_config.vocab_size)
        atomic(Path(directory)/'sampling-initialization.json', dict(start=sampling_started, end=stamp()))
        import inspect
        source = Path(inspect.getfile(type(llm.engine.model))).resolve()
        root = Path(inspect.getfile(EngineConfig)).resolve().parents[3]
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        inventory = {}
        for tensor in llm.engine.model.state_dict().values():
            name = str(tensor.dtype)
            inventory[name] = inventory.get(name, 0) + tensor.numel()
        if config.get('quantization') == 'fp8' and not inventory.get('torch.float8_e4m3fn'):
            raise RuntimeError('FP8 requested but weights contain no FP8 tensors')
        packages = {}
        for name in ('torch', 'torchvision', 'flashinfer-python', 'flash-linear-attention',
                     'fla-core', 'sglang-kernel', 'transformers'):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        import os
        import re
        model_root = Path(model).resolve()
        revision = model_root.name if re.fullmatch('[0-9a-f]{40}', model_root.name) else None
        hf_metadata = model_root/'.cache/huggingface/download/config.json.metadata'
        if revision is None and hf_metadata.is_file():
            candidate = hf_metadata.read_text().splitlines()[0]
            revision = candidate if re.fullmatch('[0-9a-f]{40}', candidate) else None
        atomic(Path(directory)/'provenance.json', dict(engine_commit=commit, engine_root=str(root),
            engine_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            lock_sha256=hashlib.sha256((root/'uv.lock').read_bytes()).hexdigest(),
            packages=packages, gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
            model_path=str(model), model_revision=revision,
            model_config_sha256=hashlib.sha256((model_root/'config.json').read_bytes()).hexdigest(),
            execution_environment={k: v for k, v in os.environ.items() if k.startswith(('MINISGL_', 'TRITON_'))
                or k in ('CUDA_HOME', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTORCH_CUDA_ALLOC_CONF', 'CUDA_VISIBLE_DEVICES')},
            resolved_engine_params=params, weight_elements=inventory))
        return Backend(llm, directory)
    except BaseException:
        await llm.close()
        raise
