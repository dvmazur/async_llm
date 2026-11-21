"""Pure geometry constraints for the live evidence policy."""
from .action_plan import LABELS, ACTION_LABELS

def action_labels(prefix, options, route=None, inspect_only=False, turn_labels=None):
    turns_allowed = ['L','R'] if turn_labels is None else list(turn_labels)
    if inspect_only:
        return turns_allowed + ['S']
    if route is not None:
        turns = [ACTION_LABELS[name] for name in route['turn_actions']]
        if len(prefix) < len(turns):
            return [turns[len(prefix)], 'S']
        return ([route.get('translation_label','F')] if len(prefix) - len(turns) < route['then_forward_steps'] else []) + ['S']
    # Affordances are measured from the observation pose. After translating
    # and then turning, another F requires a new observation/geometry check.
    turns, count, moved = [], 0, False
    valid = True
    for label in prefix:
        if label == 'F':
            moved = True
            count += 1
        elif label in turns_allowed:
            if moved:
                valid = False
            turns.append(LABELS[label])
    forward = valid and any(row['turn_actions'] == turns and count < row['then_forward_steps']
                            for row in options)
    return (['F'] if forward else []) + turns_allowed + ['S']


def translation_options(options):
    result=[]
    for row in options:
        if not row['turn_sweep_known_free']:
            continue
        for full,short,label,creep in (
            ('full_forward_steps','short_forward_steps','F','G'),
            ('full_backward_steps','short_backward_steps','B','C')):
            count=row.get(full,0)
            if not count:
                count=row.get(short,0);selected=creep
            else:
                selected=label
            if count:
                result.append(dict(turn_actions=row.get('turn_actions', [LABELS[c] for c in row['turn_prefix']]),
                                   translation_label=selected,then_forward_steps=count,
                                   destination_map_yaw_degrees=row['map_yaw_degrees']))
    # Finite single-token menu, retain short simple maneuvers first, both directions.
    return sorted(result,key=lambda x:(len(x['turn_actions']),-x['then_forward_steps']))[:24]


def immediate_mobility(options, spec, horizon):
    """Expose measured translation budgets, without choosing or vetoing actions."""
    result=[]
    for row in options:
        if row['turn_actions']:
            continue
        label=row.get('translation_label','F')
        item=dict(action=label, measured_steps=row['then_forward_steps'])
        distance=spec.get(LABELS.get(label),{}).get('max_distance_m')
        if distance is not None:
            count=min(horizon,row['then_forward_steps'])
            item.update(nominal_step_m=distance,
                        known_translation_m=round(distance*row['then_forward_steps'],3),
                        max_steps_this_sequence=count,
                        max_translation_this_sequence_m=round(distance*count,3))
        result.append(item)
    return result
