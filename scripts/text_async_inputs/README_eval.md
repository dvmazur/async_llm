# Math-500 Async Input Evaluation

This directory contains evaluation scripts for testing async input injection on the math-500 sharded dataset using mini-sglang.

## Overview

The evaluation tests how models perform when mathematical problems are split into two parts, with the second part arriving after a specified number of decoding steps. At injection time the shard is appended directly to the existing prompt `CacheBlock`; the decoding view remains `[prompt, output]` throughout generation.

## Dataset

The `math500_shards_full` dataset contains 500 math problems from the MATH dataset, each split into two shards:
- **Shard 1**: Initial problem statement (may be incomplete)
- **Shard 2**: Additional information needed to solve the problem

## K-Step Values

The evaluation tests different timing strategies for when the second shard arrives:

### Baselines
- **k = 0**: Both shards concatenated at the start (optimal case, full information from beginning)
- **k = -1**: Only first shard, second never arrives (worst case, incomplete information)

### Async Input Delays
- **k = 16, 32, 64, 128, 256, 512**: Second shard arrives after k decoding steps

## Files

- `math500_async_eval.py`: Main evaluation script
- `run_full_eval_4gpus.sh`: Four-GPU runner for all k-step values
- `rescore_math500.py`: Recompute scores from saved generations with `math_verify`
- `rescore_when_done.sh`: Final rescoring helper for a background run
- `README_eval.md`: This file

## Usage

### Single K-Step Evaluation

```bash
python math500_async_eval.py \
    --model-name Qwen/Qwen3-8B \
    --dataset_path ~/Projects/mdt-doom-stuff/math500_shards_full \
    --k-steps 64 \
    --budget 2048 \
    --start 0 \
    --end 500
```

### Run All K-Steps

```bash
# Set environment variables (optional)
export MODEL_NAME="Qwen/Qwen3-8B"
export DATASET_PATH="$HOME/Projects/mdt-doom-stuff/math500_shards_full"
export RESULTS_DIR="./eval_results/math-500-async-prompt-append"

# Run all evaluations
bash run_full_eval_4gpus.sh
```

### Quick Test (First 10 Samples)

```bash
python math500_async_eval.py --model-name Qwen/Qwen3-8B \
  --dataset_path "$HOME/Projects/mdt-doom-stuff/math500_shards_full" \
  --path-to-results ./eval_results/math-500-async-prompt-append-smoke \
  --k-steps 16 --budget 512 --start 0 --end 2
```

## Arguments

- `--model-name`: HuggingFace model name
- `--dataset_path`: Path to the sharded math500 dataset
- `--k-steps`: When to inject second shard (-1, 0, or positive integer)
- `--budget`: Max tokens to generate (default: 2048)
- `--start`: First sample index (default: 0)
- `--end`: Last sample index exclusive (default: None = full dataset)
- `--path-to-results`: Output directory; use a unique directory per injection strategy
- `--memory-ratio`: GPU memory ratio for cache (default: 0.85)
- `--page-size`: Page size for cache allocation (default: 1)

## Output

Results are saved in `{path-to-results}/k_{k_steps}/`:
- `sample_{idx}.json`: Individual problem results
- `summary.json`: Overall accuracy for this k-step value

Each sample result contains:
- `is_equal`: Whether predicted answer matches ground truth
- `predicted_answer`: Extracted answer from generation
- `correct_answer`: Ground truth answer
- `generated_text`: Full generated text
- `num_tokens`: Number of tokens generated
- `hit_eos`: Whether generation hit EOS token
- `injection_strategy`: Cache mutation strategy (`prompt_block_append`)

## Expected Results

The `k=0` and `k=-1` conditions provide full-input and incomplete-input baselines. The delayed conditions measure how recovery changes as the shard arrives later.

## Model Recommendations

For 4x A100 40GB GPUs, the validated configuration is one `Qwen/Qwen3-8B` process per GPU at `--memory-ratio 0.8`. Qwen3.5 hybrid models require additional GDN-state memory; 4B and 9B did not fit with the tested single-GPU settings.

For 80GB GPUs (future):
- **Qwen/Qwen3.8-27B**: Target model for final evaluation
