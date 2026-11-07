"""Opt-in BF16 parity check. Run this file in a fresh process with MINISGL_LLAMA_MODEL set."""

import asyncio
import os

import pytest
import torch
from minisgl.llm import AsyncLLM
from transformers import AutoModelForCausalLM


def test_llama_shared_prefix_matches_transformers():
    model_path = os.environ.get("MINISGL_LLAMA_MODEL")
    if not model_path:
        pytest.skip("Set MINISGL_LLAMA_MODEL to a dense Llama text checkpoint")

    async def run_branches():
        llm = AsyncLLM(
            model_path,
            dtype=torch.bfloat16,
            attention_backend="fi",
            use_pynccl=False,
            max_running_req=4,
            max_seq_len_override=256,
            num_page_override=256,
            cuda_graph_max_bs=0,
        )
        try:
            assert llm.config.model_config.architectures == ["LlamaForCausalLM"]
            prefix = llm.tokenizer.encode("What is seventeen times three?")
            suffixes = [
                llm.tokenizer.encode(text, add_special_tokens=False)
                for text in (" The product is fifty", " Adding seventeen three times gives")
            ]
            prompt = await llm.prefill_block(prefix, return_logits=True)
            branches = await asyncio.gather(
                *(
                    llm.prefill_block(ids[:-1], cache_view=[prompt.block], return_logits=True)
                    for ids in suffixes
                )
            )
            steps = await asyncio.gather(
                *(
                    llm.forward(ids[-1:], cache_view=[prompt.block, branch.block])
                    for ids, branch in zip(suffixes, branches)
                )
            )
            inputs = (
                [prefix]
                + [prefix + ids[:-1] for ids in suffixes]
                + [prefix + ids for ids in suffixes]
            )
            logits = [result.logits.float().cpu() for result in (prompt, *branches, *steps)]
            return inputs, logits
        finally:
            await llm.close()

    with torch.inference_mode():
        inputs, logits = asyncio.run(run_branches())
        reference = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            .to("cuda")
            .eval()
        )
        for ids, actual in zip(inputs, logits):
            expected = (
                reference(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float().cpu()
            )
            assert actual.shape == expected.shape
            # BF16 reduction order differs between the paged and eager kernels.
            relative_error = (actual - expected).norm() / expected.norm()
            assert (
                relative_error < 0.02
            ), f"{len(ids)} tokens: relative logit error {relative_error:.4f}"
            top = expected.topk(2).values
            if top[0] - top[1] > 0.5:
                assert actual.argmax() == expected.argmax()
