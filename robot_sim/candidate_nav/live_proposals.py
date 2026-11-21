"""Live candidate diagnostics and feedback, without changing reachable geometry."""

from collections import Counter

import numpy as np


def reach_status(reach):
    if reach is None or not hasattr(reach, 'distance'):
        return None
    x, y = reach.start
    count = int(np.isfinite(reach.distance).sum())
    value, clearance = None, None
    if not reach.grid.contains((x, y)):
        state = 'outside_map'
    else:
        value = int(reach.grid.values[y, x])
        clearance = round(float(reach.clearance[y, x]), 3)
        if value >= 65:
            state = 'occupied_start'
        elif value < 0 or value > 35:
            state = 'unknown_start'
        elif not count:
            state = 'insufficient_clearance'
        else:
            state = 'ready'
    return {'start_state': state, 'reachable_cells': count,
            'start_cell_value': value, 'start_clearance_m': clearance}


def validation_feedback(observation, proposals, checks):
    rejected = [item for item in checks if not item.get('accepted')]
    return {
        'snapshot_id': observation['snapshot_id'],
        'map_version': observation['map_version'],
        'raw_count': len(proposals),
        'accepted_count': sum(bool(item.get('accepted')) for item in checks),
        'rejection_counts': dict(Counter(item.get('reason', 'unknown') for item in rejected)),
        'checks': [{'proposal': item.get('raw'), 'accepted': bool(item.get('accepted')),
                    'reason': item.get('reason')} for item in checks[:8]],
    }
