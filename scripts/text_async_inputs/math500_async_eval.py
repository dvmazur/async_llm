"""
Math-500 Sharded Evaluation with Async Inputs for mini-sglang

This eval tests async input injection on the math-500 sharded dataset.
Each problem is split into two parts; the second part is injected after k decoding steps.

Based on AsyncReasoning's math-500_sharded.py but adapted to use mini-sglang's AsyncLLM.
"""

import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer
from math_verify import parse, verify

from minisgl.llm import AsyncLLM
from minisgl.shared_cache import AsyncContext, CacheBlock


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to sharded math500 dataset (for load_from_disk)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B",
                        help="Model name from HuggingFace")
    parser.add_argument("--budget", type=int, default=2048,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--k-steps", type=int, required=True,
                        help="Number of decoding steps before second shard arrives (-1=never, 0=concat, >0=after k steps)")
    parser.add_argument("--path-to-results", type=str, default="./eval_results/math-500-async-prompt-append",
                        help="Path to store results")
    parser.add_argument("--start", type=int, default=0,
                        help="First task index (inclusive)")
    parser.add_argument("--end", type=int, default=None,
                        help="Last task index (exclusive)")
    parser.add_argument("--memory-ratio", type=float, default=0.85,
                        help="GPU memory ratio for cache")
    parser.add_argument("--page-size", type=int, default=1,
                        help="Page size for cache allocation")
    parser.add_argument("--distributed-port", type=int, default=2333,
                        help="Distributed communication port (use different ports for parallel runs)")
    return parser.parse_args()


def find_last_boxed_answer(text: str) -> Optional[str]:
    """Extract the final complete \\boxed{...}, including nested braces."""
    prefix = r"\boxed{"
    end = len(text)
    while True:
        start = text.rfind(prefix, 0, end)
        if start < 0:
            return None
        depth = 0
        for pos in range(start + len(r"\boxed"), len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    return text[start + len(prefix):pos].strip()
        end = start


def check_equality(predicted: Optional[str], ground_truth: str) -> bool:
    """Grade with math-verify, boxing both sides for consistent extraction."""
    if predicted is None:
        return False
    try:
        prediction = parse(r"\boxed{" + predicted + "}")
        reference = parse(r"\boxed{" + str(ground_truth) + "}")
        return bool(verify(reference, prediction))
    except Exception:
        return False


def encode(text: str, tokenizer: AutoTokenizer) -> torch.Tensor:
    """Encode text to token IDs."""
    return torch.tensor(tokenizer.encode(text, add_special_tokens=False))


async def generate_with_async_input(
    llm: AsyncLLM,
    tokenizer: AutoTokenizer,
    initial_prompt: str,
    second_shard: str,
    k_steps: int,
    max_tokens: int,
) -> tuple[str, List[int], bool, str]:
    """
    Generate text with async input injection.

    Args:
        llm: AsyncLLM instance
        tokenizer: Tokenizer
        initial_prompt: Initial prompt (first shard)
        second_shard: Second shard to inject
        k_steps: When to inject (-1=never, 0=concat at start, >0=after k decode steps)
        max_tokens: Maximum tokens to generate

    Returns:
        (generated_text, token_ids, hit_eos)
    """
    # Prepare prompts
    if k_steps == 0:
        # Baseline: concatenate both shards at the start
        full_prompt = initial_prompt + second_shard
        prompt_ids = encode(full_prompt, tokenizer)
    else:
        # Start with first shard only
        prompt_ids = encode(initial_prompt, tokenizer)

    # Prefill the initial prompt
    prompt_block = await llm.create_block()
    await llm.forward(prompt_ids, write_to=prompt_block, return_logits=False)

    # Create output block for generation
    output_block = await llm.create_block()
    cache_view = [prompt_block, output_block]

    # Start generation
    eos_id = tokenizer.eos_token_id
    generated_ids: List[int] = []
    hit_eos = False
    shard_injected = (k_steps <= 0)  # Already injected if k_steps=0, never inject if k_steps=-1
    injection_strategy = "prompt_block_append"

    # Initialize context with a newline to start generation
    ctx = AsyncContext(cache_view=cache_view, next_input_id=tokenizer.encode("\n")[-1])

    for step in range(max_tokens):
        # Check if we should inject the second shard
        if not shard_injected and k_steps > 0 and step >= k_steps:
            # Inject second shard
            shard_text = f"\n\nADDITIONAL INFORMATION: {second_shard}\n\n"
            shard_ids = encode(shard_text, tokenizer)

            # Extend the prompt block in place.  Decode keeps the stable
            # [prompt_block, output_block] layout before and after injection.
            await llm.forward(
                shard_ids,
                cache_view=[prompt_block],
                write_to=prompt_block,
                return_logits=False
            )

            assert ctx.cache_view == [prompt_block, output_block]
            shard_injected = True

        # Generate one token
        out = await llm.forward(cache_view=ctx)
        logits = out.logits

        # Greedy sampling
        token_id = int(logits.argmax().item())
        generated_ids.append(token_id)
        ctx.next_input_id = token_id

        # Check for EOS
        if token_id == eos_id:
            hit_eos = True
            break

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Cleanup
    await llm.free_block(prompt_block)
    await llm.free_block(output_block)

    return generated_text, generated_ids, hit_eos, injection_strategy


async def run_eval(args):
    """Main evaluation loop."""
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)

    # Determine range
    start_idx = args.start
    end_idx = args.end if args.end is not None else len(dataset)
    print(f"Evaluating samples {start_idx} to {end_idx-1} (total: {end_idx - start_idx})")

    # Create output directory
    output_dir = Path(args.path_to_results) / f"k_{args.k_steps}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    llm = AsyncLLM(
        args.model_name,
        dtype=torch.bfloat16,
        memory_ratio=args.memory_ratio,
        page_size=args.page_size,
        distributed_addr=f"tcp://127.0.0.1:{args.distributed_port}",
    )

    # Evaluation loop
    accuracy_numerator = 0
    accuracy_denominator = 0

    for idx in tqdm(range(start_idx, end_idx), desc="Evaluating"):
        save_path = output_dir / f"sample_{idx}.json"
        if save_path.exists():
            # Load existing result to update accuracy
            with open(save_path) as f:
                result = json.load(f)
                accuracy_numerator += int(result["is_equal"])
                accuracy_denominator += 1
            continue

        # Get problem
        item = dataset[idx]
        problem_shards = item['problem_shards']
        answer = str(item['answer'])

        assert len(problem_shards) == 2, f"Expected 2 shards, got {len(problem_shards)}"

        # Prepare prompt
        first_shard = problem_shards[0]
        second_shard = problem_shards[1]

        instruction = f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{first_shard}"

        # Generate
        try:
            generated_text, generated_ids, hit_eos, injection_strategy = await generate_with_async_input(
                llm, tokenizer, instruction, second_shard, args.k_steps, args.budget
            )

            # Extract answer
            predicted_answer = find_last_boxed_answer(generated_text)

            # Check correctness
            is_correct = check_equality(predicted_answer, answer)

            # Update accuracy
            accuracy_numerator += int(is_correct)
            accuracy_denominator += 1

            # Save result
            result = {
                "idx": idx,
                "k_steps": args.k_steps,
                "is_equal": is_correct,
                "predicted_answer": predicted_answer,
                "correct_answer": answer,
                "generated_text": generated_text,
                "num_tokens": len(generated_ids),
                "hit_eos": hit_eos,
                "injection_strategy": injection_strategy,
            }

            with open(save_path, "w") as f:
                json.dump(result, f, indent=2)

            # Print progress
            current_accuracy = accuracy_numerator / accuracy_denominator
            print(f"[{idx}] correct={is_correct}, accuracy={current_accuracy:.3f}")

        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            import traceback
            traceback.print_exc()

    # Close LLM
    await llm.close()

    # Final accuracy
    if accuracy_denominator > 0:
        final_accuracy = accuracy_numerator / accuracy_denominator
        print(f"\nFinal accuracy: {final_accuracy:.3f} ({accuracy_numerator}/{accuracy_denominator})")

        # Save summary
        summary = {
            "k_steps": args.k_steps,
            "accuracy": final_accuracy,
            "correct": accuracy_numerator,
            "total": accuracy_denominator,
            "model": args.model_name,
            "injection_strategy": "prompt_block_append",
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def main():
    args = parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
