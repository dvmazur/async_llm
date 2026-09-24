
import _self_evolving_bootstrap  # noqa: F401
import asyncio
from types import SimpleNamespace
import unittest
import torch
from agent import bootstrap_generate, HistoryBlockState, strip_think, eos_ids
from generation_prompt import revision_completion

class RecoveryTests(unittest.TestCase):
    def test_thinking_examples_are_not_executable_tools(self):
        completion=revision_completion('assistant\n<think>\n', '<use_tool name="bad"></use_tool></think><use_tool name="start_task"></use_tool>')
        self.assertEqual(strip_think(completion), '<use_tool name="start_task"></use_tool>')
        self.assertEqual(strip_think(revision_completion('<think>', 'unfinished reasoning')), '')
    def test_fallback_frees_old_history_and_uses_fresh_chat_each_time(self):
        class Tokenizer:
            eos_token_id=9
            def apply_chat_template(self, messages, **kwargs):
                self.messages=messages
                return 'assistant<think>'
            def __call__(self, text, **kwargs):
                return SimpleNamespace(input_ids=torch.tensor([[1,2]]))
            def decode(self, tokens, **kwargs):
                return '</think><use_tool name="start_task"></use_tool>'
        class LLM:
            def __init__(self):
                self.tokenizer=Tokenizer(); self.config=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9,10]))
                self.freed=[];self.created=[]
            async def free_block(self, block): self.freed.append(block)
            async def create_block(self):
                block=SimpleNamespace(num_tokens=0);self.created.append(block);return block
            async def __call__(self,*args,**kwargs): return None
            async def sample(self,output): return torch.tensor(10)
        async def check():
            llm=LLM(); history=HistoryBlockState(); old=object();history.block=old
            for _ in range(2):
                result=await bootstrap_generate(llm,'current engine',history=history,max_new_tokens=3)
                self.assertIn('start_task',strip_think(result))
                self.assertIsNone(history.block)
            self.assertIs(llm.freed[0],old)
            self.assertEqual(len(llm.created),2)
            self.assertEqual(len(llm.freed),3)
            self.assertEqual(eos_ids(llm),{9,10})
        asyncio.run(check())

if __name__=='__main__': unittest.main()
