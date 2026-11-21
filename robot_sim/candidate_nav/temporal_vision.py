"""Task-scoped temporal visual state; sightings are never completion evidence."""
import copy
import hashlib

def task_key(instruction):
    return hashlib.sha256(instruction.encode()).hexdigest()

def initial_state(instruction, previous=None):
    if previous and previous.get('task_key') == task_key(instruction):
        return copy.deepcopy(previous)
    return dict(task_key=task_key(instruction), relevant_visible=None,
                description='unknown', last_seen=None, latest_frame=None)

def event_labels(state, has_reference):
    if not has_reference:
        return ['A','N','U']
    return ['D','C','N','U'] if state['relevant_visible'] is True else ['A','C','N','U']

def update_state(state, label, description, frame, has_reference):
    if label=='A':
        state['relevant_visible']=True
        state['description']=description
    elif label=='D':
        state['relevant_visible']=False
        state['description']='relevant evidence no longer visible'
    elif label=='N' and not has_reference:
        state['relevant_visible']=False
        state['description']='no task-relevant evidence visible'
    elif label=='C' and description:
        state['description']=description
    elif label=='U':
        state['relevant_visible']=None
        state['description']='current visual evidence uncertain'
    if state['relevant_visible'] is True:
        state['last_seen']=dict(frame)
    state['latest_frame']=dict(frame)
    return state
