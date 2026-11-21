"""Validated H100 decode graphs; merged/prefill changes stay opt-in."""
import torch


def graph_options(profile,enabled=None):
    if enabled is None:
        enabled = profile == 'h100'
    if enabled and profile!='h100':
        raise ValueError('Decode graphs are only configured for the H100 policy profile')
    # Camera + spatial + selection can overlap in legacy strategies (3 workers).
    # Capture 4 as well, rather than a single-worker-only profile.
    return dict(cuda_graph_max_bs=4 if enabled else 0,
                cuda_graph_bs=[1,2,4] if enabled else None)


def append_exact_tokens(inputs,ids):
    """Append known analysis tokens unchanged; preserve every image and mask row."""
    result=dict(inputs);tokens=inputs['input_ids']
    if tokens.ndim!=2 or tokens.shape[0]!=1:
        raise ValueError('Expected a single tokenized observation')
    suffix=torch.tensor(ids,dtype=tokens.dtype,device=tokens.device).reshape(1,-1)
    result['input_ids']=torch.cat((tokens,suffix),dim=1)
    for key,value in (('mm_token_type_ids',0),('attention_mask',1)):
        if key in inputs:
            mask=inputs[key]
            if mask.shape!=tokens.shape:raise ValueError('Invalid multimodal/attention mask')
            tail=torch.full(suffix.shape,value,dtype=mask.dtype,device=mask.device)
            result[key]=torch.cat((mask,tail),dim=1)
    return result
