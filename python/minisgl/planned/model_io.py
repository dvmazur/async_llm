"""CPU model input validation and physical token/image row packing."""
from dataclasses import dataclass


@dataclass(frozen=True,slots=True)
class InputPlan:
    token_ids:tuple[int,...]
    image_indices:tuple[int,...]
    image_count:int


def prepare_inputs(forward,token_ids,*,vocab_size,image_rows=()):
    """Tokens/image_rows use logical [actual PF | actual D] order, not capacity.

    image_rows enumerates the destination of each supplied feature, in feature
    tensor order. Image embeddings are precomputed outside the decoder graph.
    """
    token_ids,image_rows=tuple(token_ids),tuple(image_rows)
    real=[row for row,active in enumerate(forward.rows.active) if active]
    if len(token_ids)!=len(real):raise ValueError("token count does not match active forward rows")
    if any(type(i) is not int or not 0<=i<vocab_size for i in token_ids):
        raise ValueError("token ID outside vocabulary")
    pf=sum(r.length for r in forward.prefill_requests)
    if len(set(image_rows))!=len(image_rows) or any(type(i) is not int or not 0<=i<pf for i in image_rows):
        raise ValueError("image rows must be unique active prefill rows")
    tokens=[0]*len(forward.rows.active);images=[-1]*len(tokens)
    for row,token in zip(real,token_ids):tokens[row]=token
    for index,row in enumerate(image_rows):images[real[row]]=index
    return InputPlan(tuple(tokens),tuple(images),len(image_rows))
