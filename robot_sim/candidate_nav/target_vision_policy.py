"""Compact structured reports on the existing observation KV cache."""

from collections import OrderedDict
import json

from .inference_decisions import checked_argmax
from .target_vision import REPORT, parse_report, RULES

TOKEN_BUDGET = 104
FRAME_SYSTEM = RULES + """
Base your scene report only on the one camera image in this request. Describe
the visible floor and the position of foreground obstacles relative to the
camera. A wall at an image edge may leave a gap in the centre. Distant open
space does not imply an open path through the immediate foreground.
"""


SCOPE_PROMPT = """Classify the exact instruction into one of four labels.
A: The single objective is to approach a particular visible object or landmark,
possibly with a requested stopping distance or after first finding it.
T: A specific visual object or landmark matters, but the task also requires a
different phase (returning, avoiding, following, turning, inspecting from a chosen
angle, passing through, finding without an approach, or approaching multiple objects).
M: The whole objective is to exit a maze or enclosure through any viable passage.
N: Any other instruction without a specific visual entity.
"Find the book and drive up to it" is A. "Approach the book then return" is T.
"Exit through the blue door" is T. "Leave the maze" is M. "Rotate 90 degrees" is N.
Reply with exactly one letter A, T, M, or N.
Instruction: """


async def task_visual_scope(policy, instruction, prompt):
    """Classify the instruction once, keeping non-approach tasks unrestricted."""
    cache = getattr(policy, '_visual_task_scope', None)
    if cache is None:
        cache = policy._visual_task_scope = OrderedDict()
    if instruction in cache:
        cache.move_to_end(instruction)
        return cache[instruction]

    block = await policy.llm.create_block()
    try:
        output = await policy.llm.forward(prompt(SCOPE_PROMPT + json.dumps(instruction)),
                                          cache_view=[block])
        labels = ['N', 'T', 'A', 'M']
        choice = policy.label_distribution(output.logits, labels, error='Nonfinite visual task scope')
        scope = labels[max(range(len(labels)), key=choice.probabilities.__getitem__)]
    finally:
        await policy.llm.free_block(block)
    cache[instruction] = scope
    if len(cache) > 32:
        cache.popitem(last=False)
    return scope


async def task_has_target(policy, instruction, prompt):
    return await task_visual_scope(policy, instruction, prompt) in ('A', 'T')


async def inspect_frame(policy, image, instruction, state, has_target):
    """Use an independent KV block for a single frame and release it after decoding."""
    llm = policy.llm
    inputs = llm.processor.apply_chat_template(
        [{'role': 'system', 'content': FRAME_SYSTEM},
         {'role': 'user', 'content': [
             {'type': 'image', 'image': image},
             {'type': 'text', 'text': 'Task: ' + instruction},
         ]}],
        add_generation_prompt=False, tokenize=True, return_dict=True,
        return_tensors='pt', enable_thinking=False,
    )
    inputs.pop('attention_mask', None)
    block = await llm.create_block()
    lease = None
    try:
        view, lease, metrics = await policy.prefill_observation(inputs, FRAME_SYSTEM, block, True)

        async def forward(ids, **kwargs):
            return await llm.forward(ids, cache_view=view, **kwargs)

        def prompt(text):
            return policy.encode('<|im_start|>user\n'+text+
                                 '<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n')

        return await read_frame(policy, forward, prompt, state, has_target), metrics
    finally:
        try:
            await llm.free_block(block)
        finally:
            if lease is not None:
                await policy.system_prefix_cache.release(lease)


async def read_frame(policy, forward, prompt, state, has_target):
    last = state['target'].get('last_seen')
    identity = last.get('appearance') if last else None
    request = ("Use the single current camera image. "
               f"Previously reported target appearance (historical): {json.dumps(identity)}.\n"
               + REPORT)
    if not has_target:
        request += ('\nThe text-only task check found no visual entity. '
                    'Use TARGET NONE. Put obstacles and other facts in SCENE.')
    output = await forward(prompt(request))
    tokens = []
    for _ in range(TOKEN_BUDGET):
        token = checked_argmax(output.logits, error='Nonfinite visual report')
        if token == policy.tokens.eos_token_id:
            break
        tokens.append(token)
        output = await forward([token])
    report = parse_report(policy.tokens.decode(tokens, skip_special_tokens=True))
    report.setdefault('scene_directions', dict(left='U', center='U', right='U'))
    if not has_target:
        return {**report, 'status': 'none'}
    if report['status'] == 'none':
        report['status'] = 'uncertain'
    return report
