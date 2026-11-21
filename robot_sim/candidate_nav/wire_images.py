"""Lossless per-connection image references, with bounded deterministic LRU state."""
from collections import OrderedDict
import hashlib
FIELDS={'camera','map_image','accumulated_map_image','image'}

class WireImages:
    def __init__(self,capacity=8):self.images=OrderedDict();self.capacity=capacity
    def remember(self,key,value):
        self.images.pop(key,None);self.images[key]=value
        while len(self.images)>self.capacity:self.images.popitem(last=False)
    def encode(self,payload,enabled):
        supplied={}
        def visit(v,key=None):
            if key in FIELDS and isinstance(v,str):
                digest=hashlib.sha256(v.encode()).hexdigest()
                if digest not in self.images:supplied[digest]=v
                self.remember(digest,v)
                return {'_wire_image':digest} if enabled else v
            if isinstance(v,dict):return {k:visit(x,k) for k,x in v.items()}
            if isinstance(v,list):return [visit(x) for x in v]
            return v
        result=visit(payload)
        if enabled:result['_wire_images']=supplied
        return result
    def decode(self,payload):
        supplied=payload.get('_wire_images',{})
        if not isinstance(supplied,dict):raise ValueError('Invalid wire image dictionary')
        def visit(v,key=None):
            if key in FIELDS and isinstance(v,dict) and set(v)=={'_wire_image'}:
                digest=v['_wire_image'];raw=supplied.get(digest,self.images.get(digest))
                if not isinstance(raw,str) or hashlib.sha256(raw.encode()).hexdigest()!=digest:
                    raise ValueError('Missing or corrupt wire image reference')
                self.remember(digest,raw);return raw
            if key in FIELDS and isinstance(v,str):
                self.remember(hashlib.sha256(v.encode()).hexdigest(),v);return v
            if isinstance(v,dict):return {k:visit(x,k) for k,x in v.items() if k!='_wire_images'}
            if isinstance(v,list):return [visit(x) for x in v]
            return v
        return visit(payload)
