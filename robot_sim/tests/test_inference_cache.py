import base64,io,unittest
from PIL import Image
from robot_sim.candidate_nav.inference_cache import ExactEncodingCache,decode_image

class Tokenizer:
    def __init__(self,offset=0):self.calls=0;self.offset=offset
    def encode(self,text,add_special_tokens):
        assert add_special_tokens is False
        self.calls+=1;return [ord(c)+self.offset for c in text]

class EncodingTests(unittest.TestCase):
    def test_exact_tokens_and_mutation_do_not_corrupt_future_calls(self):
        t=Tokenizer();c=ExactEncodingCache()
        first=c.encode(t,'Same');first[0]=0
        self.assertEqual(c.encode(t,'Same'),[ord(x) for x in 'Same'])
        self.assertEqual(t.calls,1)
    def test_lru_bound_and_tokenizer_replacement(self):
        t=Tokenizer();c=ExactEncodingCache(2)
        for x in ('a','b','a','c','b'):c.encode(t,x)
        self.assertEqual(t.calls,4);self.assertEqual(len(c.entries),2)
        newer=Tokenizer(10)
        self.assertEqual(c.encode(newer,'b'),[ord('b')+10]);self.assertEqual(newer.calls,1)

class ImageTests(unittest.TestCase):
    def encoded(self,size=(12,16),mode='RGB'):
        image=Image.new(mode,size);b=io.BytesIO();image.save(b,format='PNG')
        return base64.b64encode(b.getvalue()).decode()
    def test_exact_rgb_and_independent_return_values(self):
        encoded=self.encoded(mode='L')
        expected=Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGB')
        a=decode_image(encoded);a.putpixel((0,0),(255,0,0));b=decode_image(encoded)
        self.assertEqual(b.tobytes(),expected.tobytes());self.assertEqual(b.mode,'RGB')
        self.assertIsNot(a,b)
    def test_existing_size_and_base64_validation_remain(self):
        with self.assertRaises(ValueError):decode_image(self.encoded((1025,1)))
        with self.assertRaises(ValueError):decode_image('not base64')

if __name__=='__main__':unittest.main()
