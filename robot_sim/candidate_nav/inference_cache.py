"""Bounded exact CPU caches; callers always receive independent mutable values."""
import base64
import io
from collections import OrderedDict
from functools import lru_cache
from PIL import Image


@lru_cache(maxsize=8)
def _image(encoded):
    image=Image.open(io.BytesIO(base64.b64decode(encoded,validate=True))).convert('RGB')
    if image.width>1024 or image.height>1024:
        raise ValueError('Image exceeds 1024 pixels')
    return image


def decode_image(encoded):
    # No resizing, recompression or dtype conversion beyond the original RGB
    # decoding. Do not expose the cached object to processor/caller mutation.
    return _image(encoded).copy()


def cache_statistics(policy):
    images=_image.cache_info();encoding=getattr(policy,'_encoding_cache',None)
    session=getattr(getattr(getattr(policy,'llm',None),'async_engine',None),'session',None)
    runner=getattr(session,'graph_runner',None)
    visual=getattr(getattr(getattr(getattr(session,'engine',None),'model',None),'model',None),'visual',None)
    cache=getattr(visual,'_exact_embedding_cache',None)
    return dict(prefill_graph_replays=getattr(runner,'prefill_replay_count',0),
                decode_graph_replays=getattr(runner,'replay_count',0),
                vision_embedding_cache_hits=getattr(cache,'hits',0),
                vision_embedding_cache_misses=getattr(cache,'misses',0),
                vision_tower_calls=getattr(cache,'tower_calls',0),
                image_decode_cache_hits=images.hits,image_decode_cache_misses=images.misses,
                token_encode_cache_hits=encoding.hits if encoding else 0,
                token_encode_cache_misses=encoding.misses if encoding else 0)


class ExactEncodingCache:
    def __init__(self,capacity=512):
        if capacity<1:raise ValueError('Encoding cache capacity must be positive')
        self.capacity=capacity;self.owner=None;self.entries=OrderedDict()
        self.hits=0;self.misses=0

    def encode(self,tokenizer,text):
        if self.owner is not tokenizer:
            self.entries.clear();self.owner=tokenizer
        if text in self.entries:
            self.hits+=1;ids=self.entries.pop(text);self.entries[text]=ids
        else:
            self.misses+=1
            ids=tuple(tokenizer.encode(text,add_special_tokens=False))
            self.entries[text]=ids
            if len(self.entries)>self.capacity:self.entries.popitem(last=False)
        return list(ids)
