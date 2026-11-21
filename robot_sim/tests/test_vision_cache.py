import unittest
import torch
from minisgl.shared_cache.vision_cache import VisionEmbeddingCache

class Tower:
    def __init__(self):self.calls=[]
    def forward(self,pixels,grid):
        self.calls.append(grid.tolist())
        return pixels.reshape(-1,4,pixels.shape[-1]).sum(1)

class VisionTests(unittest.TestCase):
    def test_exact_hits_partial_misses_order_grid_and_dtype(self):
        c=VisionEmbeddingCache(10000);v=Tower();g=torch.tensor([[1,2,2],[1,2,2]])
        a=torch.arange(32,dtype=torch.float32).reshape(8,4)
        expected=v.forward(a,g);v.calls=[]
        self.assertTrue(torch.equal(c.forward(v,a,g,2),expected));self.assertEqual(len(v.calls),1)
        b=torch.cat([a[4:],a[:4]])
        self.assertTrue(torch.equal(c.forward(v,b,g,2),expected.flip(0)));self.assertEqual(len(v.calls),1)
        d=b.clone();d[-1]+=1
        self.assertTrue(torch.equal(c.forward(v,d,g,2),Tower().forward(d,g)))
        self.assertEqual(v.calls[-1],[[1,2,2]]);self.assertEqual(c.hits,3)
        c.forward(v,d.double(),g,2);self.assertEqual(len(v.calls),3)
    def test_cache_budget_eviction_and_disabled(self):
        g=torch.tensor([[1,2,2]]);a=torch.zeros(4,4);v=Tower();c=VisionEmbeddingCache(16)
        c.forward(v,a,g,2);c.forward(v,a+1,g,2);c.forward(v,a,g,2)
        self.assertEqual(len(v.calls),3);self.assertLessEqual(c.bytes,16)
        c=VisionEmbeddingCache(0);c.forward(v,a,g,2);c.forward(v,a,g,2);self.assertEqual(c.bytes,0)
