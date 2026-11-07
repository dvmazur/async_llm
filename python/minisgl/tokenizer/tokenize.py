from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list):
                input_ids = self.tokenizer.apply_chat_template(
                    msg.text,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    return_tensors="pt",
                )
            else:
                input_ids = self.tokenizer.encode(msg.text, return_tensors="pt")
            results.append(input_ids.view(-1).to(torch.int32))
        return results
