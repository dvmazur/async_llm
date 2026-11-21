"""Compact live observation JSON without dropping measured execution history.

Full precision remains in rover logs and the motion controller. This only
changes the text presented to the model; simulator prompt formatting is kept.
"""

import json


def model_state(state):
    """Keep all unique facts, millimetre/radian precision and one history copy."""
    def compact(value):
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items()
                    if key not in ('memory_scope','memory_context','memory_update','_memory_status','control_reference','control_start_pose','control_end_pose')}
        if isinstance(value, (list, tuple)):
            return [compact(item) for item in value]
        if isinstance(value, float):
            return round(value, 3)
        return value

    result = compact(state)
    context = state.get('task_context') or {}
    outcomes = state.get('recent_outcomes')
    # These are the same action-history list in DirectLiveAgent, and the
    # same waypoint history list in LiveAgent. Never remove distinct records.
    if outcomes is not None and any(
        key in context and outcomes == context[key]
        for key in ('recent_actions', 'recent_waypoint_outcomes')
    ):
        result.pop('recent_outcomes')
    return result


def state_text(state, live):
    if not live:
        return json.dumps(state, allow_nan=False)
    plain=json.dumps(model_state(state),allow_nan=False,ensure_ascii=False,separators=(',',':'))
    packed=json.dumps(history_state(state,1),allow_nan=False,ensure_ascii=False,separators=(',',':'))
    return min((plain,packed),key=len)


def history_state(state, level=1):
    """Lossless history encoding: share repeated frames, then tabulate records.

    This only changes the model-facing copy. Current sensors, instruction, image
    inputs and measured execution values are never removed to make room.
    """
    result = model_state(state)
    frames, frame_keys = {}, {}
    current = result.get('temporal_visual_state')
    prepared = result.get('prepared_perception')
    if isinstance(prepared, dict) and current and prepared.get('visual_state') == current:
        prepared['visual_state'] = {'visual_state_ref': 'temporal_visual_state'}

    def frame_ref(frame):
        if not isinstance(frame, dict):
            return frame
        key = json.dumps(frame, sort_keys=True, allow_nan=False, separators=(',', ':'))
        if key not in frame_keys:
            name = str(len(frames))
            frame_keys[key] = name
            frames[name] = frame
        return {'visual_frame_ref': frame_keys[key]}

    def visual_state(value, historical=False):
        if not isinstance(value, dict):
            return
        for field in ('last_seen', 'latest_frame'):
            if value.get(field) is not None:
                value[field] = frame_ref(value[field])
        if (historical and isinstance(current, dict) and current.get('task_key')
                and value.get('task_key') == current['task_key']):
            value['task_key'] = {'value_ref': 'temporal_visual_state.task_key'}

    visual_state(current)
    if isinstance(prepared, dict):
        visual_state(prepared.get('visual_state'))
    for memory in result.get('motion_visual_memory', []):
        if not isinstance(memory, dict):
            continue
        if isinstance(memory.get('frames'), list):
            memory['frames'] = [frame_ref(frame) for frame in memory['frames']]
        visual_state(memory.get('visual_state'), historical=True)
    if frames:
        result['visual_frames'] = frames
        result['history_encoding'] = (
            'visual_frame_ref resolves in visual_frames; visual_state_ref and value_ref '
            'resolve at the named root field. Historical frames are not object coordinates.')

    if level >= 2:
        # Column labels shared across visual events, without dropping any state.
        memories = result.get('motion_visual_memory', [])
        states = [m.get('visual_state') for m in memories if isinstance(m, dict)]
        if (len(states) >= 3 and all(isinstance(v, dict) for v in states)
                and all(set(v) == set(states[0]) for v in states)):
            columns = list(states[0])
            result['visual_history_state_columns'] = columns
            for memory in memories:
                memory['visual_state'] = [memory['visual_state'][key] for key in columns]
        from .action_plan import ACTION_LABELS
        codes = {}
        def action_sequences(value):
            if isinstance(value, dict):
                for key, item in list(value.items()):
                    if (key in ('planned', 'planned_actions', 'discarded', 'discarded_actions')
                            and isinstance(item, list) and len(item) >= 4
                            and all(isinstance(v, str) and v in ACTION_LABELS for v in item)):
                        value[key] = {'action_sequence': ''.join(ACTION_LABELS[v] for v in item)}
                        codes.update({ACTION_LABELS[v]: v for v in item})
                    else:
                        action_sequences(item)
            elif isinstance(value, list):
                for item in value:
                    action_sequences(item)
        action_sequences(result)
        if codes:
            result['action_sequence_codes'] = codes
        def table(value):
            if isinstance(value, dict):
                return {key: table(item) for key, item in value.items()}
            if isinstance(value, list):
                if (len(value) >= 3 and all(isinstance(item, dict) for item in value)
                        and value[0] and all(set(item) == set(value[0]) for item in value)):
                    columns = list(value[0])
                    return {'columns': columns,
                            'rows': [[table(item[key]) for key in columns] for item in value]}
                return [table(item) for item in value]
            return value
        # Preserve all rows, ordering, executed/discarded distinctions and poses.
        for field in ('motion_visual_memory', 'maneuver_memory', 'previous_plan_outcome',
                      'recent_outcomes', 'visual_frames', 'available_translations', 'immediate_mobility'):
            if field in result:
                result[field] = table(result[field])
        catalog = result.get('visual_frames', {})
        values = list(catalog.values())
        if (len(values) >= 3 and all(set(v) == set(values[0]) for v in values)
                and 'ref' not in values[0]):
            columns = list(values[0])
            result['visual_frames'] = {
                'columns': ['ref'] + columns,
                'rows': [[name] + [value[key] for key in columns]
                         for name, value in catalog.items()]}
        sensors = result.get('sensors')
        if isinstance(sensors, dict) and 'lidar_polar_min_m' in sensors:
            sensors['lidar_polar_min_m'] = table(sensors['lidar_polar_min_m'])
        context = result.get('task_context')
        if isinstance(context, dict):
            for field in ('recent_actions', 'recent_waypoint_outcomes'):
                if field in context:
                    context[field] = table(context[field])
        result['history_encoding'] = result.get('history_encoding', '') + (
            ' Tables: each row follows columns; visual_frames tables use the ref column for lookup. Original record order and every value are '
            'preserved. Historical visual_state arrays follow visual_history_state_columns; '
            'action_sequence strings use action_sequence_codes in exact order. '
            'Measurements and executed/discarded outcomes are literal.')
    if level >= 3:
        # Run-length encoding of identical consecutive table rows preserves
        # every historical occurrence and its exact order; no facts are dropped.
        def row_runs(value):
            if isinstance(value,list):return [row_runs(v) for v in value]
            if not isinstance(value,dict):return value
            packed={k:row_runs(v) for k,v in value.items()}
            if set(packed)=={'columns','rows'} and packed['rows']:
                runs=[]
                for row in packed['rows']:
                    if runs and runs[-1][1]==row:runs[-1][0]+=1
                    else:runs.append([1,row])
                alternatives=[{'columns':packed['columns'],'row_runs':runs}]
                catalog=[];sequence=[]
                for row in packed['rows']:
                    if row not in catalog:catalog.append(row)
                    sequence.append(catalog.index(row))
                if len(catalog)<=26:
                    alternatives.append({'columns':packed['columns'],
                        'row_catalog':{chr(65+i):row for i,row in enumerate(catalog)},
                        'row_sequence':''.join(chr(65+i) for i in sequence)})
                columns=packed['columns'];rows=packed['rows']
                fixed=[i for i in range(len(columns)) if all(row[i]==rows[0][i] for row in rows)]
                if fixed:
                    moving=[i for i in range(len(columns)) if i not in fixed]
                    alternatives.append({'constant_columns':{columns[i]:rows[0][i] for i in fixed},
                        'columns':[columns[i] for i in moving],
                        'rows':[[row[i] for i in moving] for row in rows]})
                candidate=min(alternatives,key=lambda x:len(json.dumps(x)))
                if len(json.dumps(candidate))+80<len(json.dumps(packed)):return candidate
            return packed
        result=row_runs(result)
        result['history_encoding']=result.get('history_encoding','')+(
            ' row_runs: [count,row] repeats that row count times; row_sequence letters resolve '
            'in row_catalog in exact order. Every occurrence counts; columns label values. '
            'constant_columns are merged into every row of that table.')
    return result


def history_text(state, level):
    return json.dumps(history_state(state, level), allow_nan=False,
                      ensure_ascii=False, separators=(',', ':'))


def budgeted_inputs(build, state, live, input_limit, context_window):
    """Admit the actual multimodal input, using reversible history compaction.

    The 512-token soft margin starts compaction before the hard input ceiling.
    No text/token truncation, image omission or context-window increase occurs.
    """
    from .context_budget import ContextBudgetExceeded
    inputs = build(state_text(state, live))
    initial = int(inputs['input_ids'].numel())
    best, best_count, chosen = inputs, initial, 0
    attempts = [initial]
    if live and initial > input_limit - 512:
        for level in (1, 2):
            candidate = build(history_text(state, level))
            count = int(candidate['input_ids'].numel())
            attempts.append(count)
            if count < best_count:
                best, best_count, chosen = candidate, count, level
            if best_count <= input_limit - 512:
                break
    if live and best_count>input_limit:
        candidate=build(history_text(state,3))
        count=int(candidate['input_ids'].numel());attempts.append(count)
        if count<best_count:best,best_count,chosen=candidate,count,3
    memory_reduced=False
    evidence_included=len((state.get('long_term_memory') or {}).get('evidence',[]))
    if live and best_count>input_limit and state.get('long_term_memory'):
        import copy
        # Durable history already lives in SQLite. Optional retrieved context
        # yields to current images, sensors, the instruction and measured outcomes.
        reduced=copy.deepcopy(state)
        memory=reduced['long_term_memory']
        memory['places']={'available':False,'reason':'input_budget'}
        while True:
            candidate=build(history_text(reduced,2))
            count=int(candidate['input_ids'].numel());attempts.append(count)
            if count<best_count:
                best,best_count,chosen=candidate,count,2
                memory_reduced=True;evidence_included=len(memory.get('evidence',[]))
            if best_count<=input_limit or not memory.get('evidence'):break
            memory['evidence'].pop()
    if best_count > input_limit:
        error = ContextBudgetExceeded(best_count, input_limit, context_window)
        error.compaction_attempts = attempts
        raise error
    return best, dict(input_limit=input_limit, original_input_tokens=initial,
                      input_tokens=best_count, history_encoding_level=chosen,
                      compaction_attempts=attempts, history_records_removed=0,
                      images_removed=0,
                      memory_reduced_for_context=memory_reduced,
                      memory_evidence_included=evidence_included)
