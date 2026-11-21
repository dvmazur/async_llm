"""Constrained action labels with lexical variants and faithful cache commits."""
from dataclasses import dataclass
import torch

@dataclass
class LabelDistribution:
    probabilities: list
    token_ids: list
    valid_probability_mass: float

class ActionTokenCodec:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
        self._groups={}

    def group(self,label):
        if label not in self._groups:
            bare=self.tokenizer.encode(label,add_special_tokens=False)
            if len(bare)!=1:raise ValueError(f'Action label {label!r} must be one token')
            ids=list(bare)
            spaced=self.tokenizer.encode(' '+label,add_special_tokens=False)
            if len(spaced)==1 and spaced[0] not in ids:ids.extend(spaced)
            self._groups[label]=tuple(ids)
        return self._groups[label]

    def score(self,logits,labels,*,error):
        groups=[self.group(label) for label in labels]
        flat=[token for group in groups for token in group]
        if not groups or len(set(flat))!=len(flat):
            raise ValueError('Action token groups must be nonempty and disjoint')
        values=logits.float()
        scores=torch.stack([torch.logsumexp(values[list(group)],0) for group in groups])
        best=torch.stack([values[list(group)].argmax() for group in groups])
        mass=(torch.logsumexp(scores,0)-torch.logsumexp(values,0)).exp().clamp(0,1)
        # Only small alias indices are packed as floats; vocabulary IDs stay exact.
        packed=torch.cat((torch.isfinite(values).all().to(values.dtype).reshape(1),
                          scores.softmax(0),best.to(values.dtype),mass.reshape(1))).cpu().tolist()
        if not packed[0]:raise FloatingPointError(error)
        n=len(groups)
        return LabelDistribution(packed[1:n+1],
            [group[int(index)] for group,index in zip(groups,packed[n+1:2*n+1])],packed[-1])
