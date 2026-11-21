"""Explicit live-model allowlist; importing this module never initializes CUDA."""
import json
from pathlib import Path

MODELS = {'27b': 'Qwen3.8-27B-FP8', '35b': 'Qwen3.6-35B-A3B-FP8', '122b': 'Qwen3.5-122B-A10B-FP8'}
MIN_FREE_122B_MIB = 70000

def model_key(value):
    text=str(value).lower()
    for key,name in MODELS.items():
        if text in (key,name.lower(),('Qwen/'+name).lower()):return key
    raise ValueError('Unknown model; choose 27b, 35b or 122b')

def model_path(root,key):
    return Path(root)/'robot_sim/.runtime/models'/MODELS[model_key(key)]

def validate_model(root,key):
    folder=model_path(root,key)
    try:
        config=json.loads((folder/'config.json').read_text())
        index=json.loads((folder/'model.safetensors.index.json').read_text())
        missing=[name for name in {v for k,v in index['weight_map'].items() if not k.startswith(('mtp.','model.mtp.'))} if not (folder/name).is_file()]
        if missing:raise ValueError(f'Missing weight shards: {missing}')
        if not (folder/'tokenizer_config.json').is_file():raise ValueError('Tokenizer is missing')
    except (OSError,ValueError,KeyError) as error:
        raise RuntimeError(f'Model {MODELS[model_key(key)]} is not fully installed: {error}') from error
    return folder


def model_tp(profile,key):
    if model_key(key)=='122b':
        if profile!='h100':raise ValueError('122b requires two H100 GPUs; use --server h100')
        return 2
    return 1 if profile=='h100' else 2

def select_devices(config,key,request,devices):
    from .hardware_profile import choose_devices
    key = model_key(key)
    if key != '122b':
        shared = set(config['shared_gpu_ids'])
        if key == '27b' and request in ('0', '1'):
            shared.update(config.get('shared_27b_gpu_ids', ()))
        return choose_devices(config['profile'], request, devices, shared_gpu_ids=shared)
    model_tp(config['profile'],key)
    if request not in ('auto','0'):raise ValueError('122b uses GPUs 0 and 1; use --gpu auto or --gpu 0')
    selected = [next((d for d in devices if d['index'] == i), None) for i in (0, 1)]
    shared = config.get('shared_122b_gpu_ids', ())
    blockers = []
    for index, device in enumerate(selected):
        if device is None or device['capability'] != 9.0 or device['total_mib'] < 75000:
            blockers.append(f'GPU {index}: requires H100 80GB')
            continue
        free = device['free_mib']
        if free < MIN_FREE_122B_MIB:
            blockers.append(f'GPU {index}: {free} MiB free, requires >= {MIN_FREE_122B_MIB} MiB')
        if device['busy'] and index not in shared:
            blockers.append(f'GPU {index}: running compute jobs, sharing for 122b is not enabled')
    if blockers:
        raise RuntimeError('122b cannot start: ' + '; '.join(blockers))
    return selected

def gpu_matches(key,request,actual):
    return actual=='0,1' if model_key(key)=='122b' else request=='auto' or actual==request
