"""Optional model-worker boundary snapshots; no model arithmetic replacements."""
import re
import torch


def trace_key(name):
    if name.endswith('visual'):
        return 'vision'
    match = re.search(r'visual\.(patch_embed|blocks\.\d+\.(?:norm1|norm2|attn|mlp)|merger)$', name)
    if match:
        return 'vision.' + match[1]
    match = re.search(r'layers\.(\d+)\.(input_layernorm|post_attention_layernorm|linear_attn|self_attn|mlp)$', name)
    if match:
        index, part = match.groups()
        if part in ('linear_attn', 'self_attn'):
            part = 'token_mixer'
        return f'layer{int(index):02d}.{part}'
    if name.endswith('embed_tokens'):
        return 'embedding'
    if name.endswith('.norm') and '.layers.' not in name and 'visual' not in name:
        return 'final_norm'


class BoundaryTrace:
    def __init__(self):
        self.active = False
        self.values = {}

    def begin(self):
        self.values = {}
        self.active = True

    def record(self, key, value):
        if not self.active:
            return
        if isinstance(value, (tuple, list)):
            value = value[0]
        if hasattr(value, 'pooler_output'):
            value = value.pooler_output
        if isinstance(value, torch.Tensor):
            self.values[key] = value.detach().cpu().clone()

    def finish(self, path):
        self.active = False
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.values, path)
        self.values = {}

    def attach_torch(self, model):
        for name, module in model.named_modules():
            key = trace_key(name)
            if key:
                module.register_forward_pre_hook(
                    lambda module, args, key=key: self.record(key+'.in', args[0]) if args else None)
                module.register_forward_hook(
                    lambda module, args, out, key=key: self.record(key+'.out', out))
            if hasattr(module, 'forward_prepare_cuda_fused'):
                original = module.forward_prepare_cuda_fused
                match = re.search(r'layers\.(\d+)$', name)
                if match:
                    key = f'layer{int(match[1]):02d}.cuda_attention_positions'
                    def positions_observer(positions, hidden_states, original=original, key=key):
                        self.record(key, positions)
                        return original(positions, hidden_states)
                    module.forward_prepare_cuda_fused = positions_observer

    def attach_mini(self, model):
        from minisgl.layers import BaseOP, OPList
        def walk(node, name):
            key = trace_key(name)
            if key:
                original = node.forward
                def observed(*args, **kwargs):
                    if args:
                        self.record(key+'.in', args[0])
                    out = original(*args, **kwargs)
                    self.record(key+'.out', out)
                    return out
                node.forward = observed
            if isinstance(node, OPList):
                for index, child in enumerate(node.op_list):
                    walk(child, f'{name}.{index}')
            else:
                for part, child in vars(node).items():
                    if not part.startswith('_') and isinstance(child, BaseOP):
                        walk(child, f'{name}.{part}')
        walk(model, 'model')
