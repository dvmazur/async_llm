"""Bounded, immutable system-prefix KV/GDN blocks for the live policy.

Only token-identical system messages are shared. Images and the entire user
message always get a fresh block. The transport serializes policy requests;
leases additionally prevent eviction of a prefix while a request uses it.
"""

import asyncio
from dataclasses import dataclass
import time

import torch


def prefix_ids(processor, system):
    # Qwen's template requires a user message even when rendering the system.
    rendered = processor.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": [{"type": "text", "text": "placeholder"}]}],
        add_generation_prompt=False, tokenize=False, enable_thinking=False,
    )
    marker = "<|im_start|>user\n"
    if marker not in rendered:
        raise ValueError("Cannot identify the system/user template boundary")
    return tuple(processor.tokenizer.encode(
        rendered.split(marker, 1)[0], add_special_tokens=False))


def split_inputs(inputs, ids):
    """Check the exact boundary and preserve multimodal rows/mask alignment."""
    tokens = inputs["input_ids"]
    n = len(ids)
    if tokens.ndim != 2 or tokens.shape[0] != 1 or not 0 < n < tokens.shape[1]:
        raise ValueError("Invalid system prefix length or input batch")
    if tokens[0, :n].tolist() != list(ids):
        raise ValueError("System prefix differs from the full tokenized observation")
    tail = dict(inputs)
    tail["input_ids"] = tokens[:, n:]
    if "mm_token_type_ids" in inputs:
        mask = inputs["mm_token_type_ids"]
        if mask.shape != tokens.shape or torch.any(mask[:, :n] != 0):
            raise ValueError("System prefix contains image tokens or an invalid image mask")
        tail["mm_token_type_ids"] = mask[:, n:]
    return tokens[:, :n], tail


@dataclass
class Lease:
    key: tuple
    block: object
    hit: bool
    prefill_s: float
    cached: bool = True
    users: int = 1


class SystemPrefixCache:
    def __init__(self, llm, capacity=4):
        if capacity < 1:
            raise ValueError("Prefix cache capacity must be positive")
        self.llm, self.capacity = llm, capacity
        self.entries = {}  # insertion order is the idle eviction order
        self.lock = asyncio.Lock()

    async def acquire(self, ids, tokens):
        async with self.lock:
            if ids in self.entries:
                entry = self.entries.pop(ids)
                self.entries[ids] = entry
                entry.users += 1
                return Lease(ids, entry.block, True, 0)
            if len(self.entries) >= self.capacity:
                idle = next((k for k, e in self.entries.items() if not e.users), None)
                if idle is not None:
                    await self.llm.free_block(self.entries.pop(idle).block)
            cached = len(self.entries) < self.capacity
            block = await self.llm.create_block()
            started = time.monotonic()
            try:
                await self.llm.forward(tokens, write_to=block, return_logits=False)
            except BaseException:
                await self.llm.free_block(block)
                raise
            lease = Lease(ids, block, False, time.monotonic() - started, cached)
            if cached:
                self.entries[ids] = lease
                # Return a separate lease: entry.users tracks all acquisitions.
                return Lease(ids, block, False, lease.prefill_s)
            return lease

    async def release(self, lease):
        if lease is None:
            return
        async with self.lock:
            if lease.cached:
                self.entries[lease.key].users -= 1
            else:
                await self.llm.free_block(lease.block)

    async def close(self):
        async with self.lock:
            if any(entry.users for entry in self.entries.values()):
                raise RuntimeError("Cannot close system prefixes while a request is active")
            for entry in self.entries.values():
                await self.llm.free_block(entry.block)
            self.entries.clear()
