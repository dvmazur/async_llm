"""Exact decisions with one device-to-host transfer, including finite checks."""
import torch


def checked_argmax(logits, ids=None, *, error):
    values = logits.float()
    selected = values if ids is None else values[ids]
    # Keep the full-vocabulary finite check and PyTorch's original tie break.
    valid, index = torch.stack((torch.isfinite(values).all().to(torch.int64),
                                selected.argmax())).cpu().tolist()
    if not valid:
        raise FloatingPointError(error)
    return index


def checked_probabilities(logits, ids, *, error):
    values = logits.float()
    probabilities = values[ids].softmax(0)
    valid, *scores = torch.cat((torch.isfinite(values).all().to(values.dtype).reshape(1),
                               probabilities)).cpu().tolist()
    if not valid:
        raise FloatingPointError(error)
    return scores
