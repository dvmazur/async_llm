"""Task-relevant changes between ordered camera frames, without motor decisions."""
import time
import json
from . import target_vision
from .target_vision_policy import inspect_frame, task_has_target
from .inference_decisions import checked_argmax
from .inference_cache import cache_statistics
from .motion_vision import PROTOCOL
from .sensor_grounding import CAMERA_RULES
from .temporal_vision import initial_state, event_labels, update_state, task_key

SYSTEM = '''Track task-relevant visual STATE and CHANGES between chronological rover images.
Compare each specified pair, not all images as one present scene. Camera poses/times describe
rover motion: do not mistake camera rotation for an object moving. Report only new information
relevant to the arbitrary instruction. Do not repeat room, floor, unchanged objects or prior events.
A sighting at an old pose is not current visibility, measured distance, arrival or completion.
Expected changes in scale, image position or perspective from the recorded rover motion
alone are N, not C. Use C only for a newly revealed task-relevant fact, changed obstacle or
visual condition; do not narrate that the rover moved. Check metric movement from measured
execution context, not from a description of unchanged scenery.
Use A when relevant visual evidence appears; D when previously visible evidence disappears;
C for another meaningful task-related change; N for nothing new; U when uncertain.
With no reference, A establishes visible relevant evidence, N establishes its absence.
Do not select driving actions or narrate plan/mission completion. A stop is not arrival.
Describe visible object state, not execution status. Never turn an action label or a completed
trajectory into visual evidence that the user instruction or a final gap has been achieved.
Preserve uncertainty and distinguish historical/current sightings.'''

LIVE_VISION_RULES = CAMERA_RULES + """Task-relevant changes include visible path obstructions and body-clearance cues,
even when the obstruction is not the requested target. Lidar covers one horizontal plane:
a missing return cannot disprove a visible low/high obstacle. Report visible evidence and
uncertainty, not a diagnosis of motor failure or contact from a close-up alone.
"""

async def inspect(policy, packet):
    from .llm import decode_image
    started=time.monotonic();frames=packet.get('frames',[])
    engine_stats=getattr(getattr(policy.llm,'async_engine',None),'statistics',{})
    metrics_before={**engine_stats,**cache_statistics(policy)}
    if not 1<=len(frames)<=4 or len({x['frame_id'] for x in frames})!=len(frames):
        raise ValueError('Motion vision requires 1..4 distinct frames')
    stamps=[f['stamp'] for f in frames]
    if stamps!=sorted(stamps) or len(set(stamps))!=len(stamps):
        raise ValueError('Motion frames must be chronological')
    split = target_vision.tracking_enabled(packet)
    init = target_vision.initial_state if split else initial_state
    state = init(packet['instruction'], packet.get('visual_state'))
    previous=packet.get('previous_frame')
    if packet.get('visual_state',{}).get('task_key')!=task_key(packet['instruction']):
        previous=None
    if previous and previous['stamp']>=stamps[0]:
        raise ValueError('Reference image must precede new images')
    system=SYSTEM+LIVE_VISION_RULES if packet.get('physical') is True else SYSTEM
    if split:
        return await inspect_current_frames(policy, packet, frames, state, started, decode_image)
    content=[dict(type='text',text='Task: '+packet['instruction']+'\nPrevious visual state: '+json.dumps(state))]
    meta=[]
    for role,frame in ([('REFERENCE',previous)] if previous else [])+[('NEW',f) for f in frames]:
        info={k:frame[k] for k in ('frame_id','stamp','pose','plan_id')}
        context = dict(info)
        if 'execution' in frame:
            context['measured_motion'] = {k:v for k,v in frame['execution'].items()
                                          if k in ('measured_totals','sensors')}
        content.extend([dict(type='text',text=role+' '+json.dumps(context)),dict(type='image',image=decode_image(frame['image']))])
        if role=='NEW':meta.append(info)
    llm=policy.llm
    inputs=llm.processor.apply_chat_template([dict(role='system',content=system),dict(role='user',content=content)],add_generation_prompt=False,tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False)
    inputs.pop('attention_mask',None)
    if inputs['input_ids'].numel()>policy.input_token_limit(True):raise ValueError('Motion image batch exceeds input budget')
    common,report_block=[await llm.create_block() for _ in range(2)];lease=None;events=[];probes=[]
    view=[]
    def prompt(text):return policy.encode('<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n')
    async def forward(ids,**kw):return await llm.forward(ids,cache_view=view+[report_block],**kw)
    try:
        view,lease,metrics=await policy.prefill_observation(inputs,system,common,True)
        reference={k:previous[k] for k in ('frame_id','stamp','pose','plan_id')} if previous else None
        for info in meta:
            labels=event_labels(state,reference is not None)
            change=dict(from_frame=reference,to_frame=info,previous_visible=state['relevant_visible'])
            if reference:
                import math
                change['rover_motion']=dict(dx=round(info['pose'][0]-reference['pose'][0],3),dy=round(info['pose'][1]-reference['pose'][1],3),dyaw_rad=round(math.atan2(math.sin(info['pose'][2]-reference['pose'][2]),math.cos(info['pose'][2]-reference['pose'][2])),3))
            output=await forward(prompt('Compare ONLY these specified image IDs: '+json.dumps(change)+'. Classify A=appeared, D=disappeared, C=new task-relevant fact, N=nothing new, U=uncertain. Expected ego-motion/parallax alone is N. Missing a visual target is not proof of failed movement. Choose one of '+','.join(labels)+':'))
            distribution=policy.label_distribution(output.logits,labels,error='Nonfinite temporal probe')
            index=max(range(len(labels)),key=distribution.probabilities.__getitem__)
            label=labels[index]
            await forward([distribution.token_ids[index]]+policy.encode('<|im_end|>\n'),return_logits=False)
            description='';exhausted=False;motion_only=None
            if label in ('A','D','C'):
                output=await forward(prompt('For the pair just classified, describe ONLY the new change in at most 10 words. Account for rover motion. Only the visual change, not a direction of rover rotation. No rover motion, stopping, arrival or task-completion claims. No repeated scene description, distances or guessed coordinates:'))
                tokens=[]
                for _ in range(24):
                    token=checked_argmax(output.logits,error='Nonfinite temporal description');tokens.append(token)
                    if token==policy.tokens.eos_token_id:break
                    output=await forward([token])
                exhausted=len(tokens)==24 and tokens[-1]!=policy.tokens.eos_token_id
                description=policy.tokens.decode(tokens,skip_special_tokens=True).strip()
                await forward(policy.encode('<|im_end|>\n'),return_logits=False)
                if label=='C':
                    review=await forward(prompt('Classify ONLY the literal claim in this report, not imagined additional facts: '+description+
                        '. E=ego-motion, the same object looks closer/larger/shifted, rover stopping/progress, or an unmeasured arrival claim; '
                        'A=task-relevant evidence actually appeared/became visible; D=it disappeared, left the frame or is no longer visible; '
                        'C=another newly revealed object/obstacle property, intrinsic state change or external object motion; U=unclear. '
                        'A larger image of the same already visible object is E. Reply exactly E, A, D, C or U:'))
                    review_labels=['E','A','D','C','U']
                    reviewed=policy.label_distribution(review.logits,review_labels,error='Nonfinite visual change review')
                    answer=max(range(len(review_labels)),key=reviewed.probabilities.__getitem__)
                    motion_only=reviewed.probabilities[0]
                    await forward([reviewed.token_ids[answer]]+policy.encode('<|im_end|>\n'),return_logits=False)
                    if answer==0:label='N';description=''
                    else:
                        label=review_labels[answer]
                        if label=='U':description=''
                if label in ('A','D','C'):
                    events.append(dict(kind=label,frame=info,reference=reference,description=description,budget_exhausted=exhausted))
            probes.append(dict(frame_id=info['frame_id'],event=label,motion_only_probability=motion_only))
            update_state(state,label,description,info,reference is not None)
            reference=info
        evidence='\n'.join(f"frame {e['frame']['frame_id']} [{e['kind']}]: {e['description']}" for e in events) or ('uncertain change' if any(p['event']=='U' for p in probes) else 'nothing new')
        reply=dict(protocol=PROTOCOL,snapshot_id=packet['snapshot_id'],map_version=packet['map_version'],evidence=evidence,
                   visual_state=state,events=events,probes=probes,frames=meta,latency_s=time.monotonic()-started,
                   budget_exhausted=any(e['budget_exhausted'] for e in events),prefill=metrics,metadata=policy.metadata,
                   runtime_metrics={key:value-metrics_before.get(key,0) for key,value in
                                    {**engine_stats,**cache_statistics(policy)}.items()})
        if policy.rank==0:print('MOTION_VISION_AUDIT '+json.dumps(reply),flush=True)
        return reply
    finally:
        await llm.free_block(report_block);await llm.free_block(common)
        if lease is not None:await policy.system_prefix_cache.release(lease)


async def inspect_current_frames(policy, packet, frames, state, started, decode_image):
    def prompt(text):
        return policy.encode('<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n')

    has_target = await task_has_target(policy, packet['instruction'], prompt)
    reports, events, meta, prefill = [], [], [], []
    for frame in frames:
        scope = frame.get('memory_scope') or {}
        info = {key: frame[key] for key in ('frame_id', 'stamp', 'pose', 'plan_id')}
        info.update(map_session_id=scope.get('map_session_id'), anchor_epoch=scope.get('anchor_epoch'),
                    tracking_valid=scope.get('tracking_valid'))
        report, metrics = await inspect_frame(policy, decode_image(frame['image']), packet['instruction'],
                                              state, has_target)
        prefill.append(metrics)
        before = target_vision.describe(state)
        target_vision.apply_report(state, report, info)
        after = target_vision.describe(state)
        if before != after:
            events.append(dict(kind='visual_channels', frame=info, description=after))
        reports.append(dict(frame=info, **report))
        meta.append(info)
    evidence = target_vision.describe(state) if events else 'nothing new'
    reply = dict(protocol=PROTOCOL, snapshot_id=packet['snapshot_id'],
                 map_version=packet['map_version'], evidence=evidence,
                 visual_state=state, events=events, reports=reports, frames=meta,
                 latency_s=time.monotonic()-started, frame_prefills=prefill, metadata=policy.metadata)
    if policy.rank == 0:
        print('MOTION_VISION_AUDIT ' + json.dumps(reply), flush=True)
    return reply
