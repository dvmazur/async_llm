"""Actual environment + processor checks, with CUDA disabled and no model weights."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from .environment import Setup, environment_preflight, make_env


def check(output, setup=None):
    import torch
    from minisgl.models.qwen3_5_mrope import get_rope_index
    from transformers import AutoProcessor

    from .agent import make_prompting

    cfg = Setup(**(setup or {}))
    output = Path(output)
    env_result = environment_preflight(output, asdict(cfg))
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", local_files_only=True)
    prompt = make_prompting(
        SimpleNamespace(processor=processor, tokenizer=processor.tokenizer), cfg
    )
    if not isinstance(prompt.common_prompt, str):
        raise TypeError("common_prompt must be chat text")
    env = make_env(cfg)
    try:
        obs, _ = env.reset(seed=cfg.seed)
        inputs = prompt.get_image_dict(
            obs["screen"], obs["screen"], without_eot=True, end_previous_turn=True
        )
        assert set(inputs) <= {
            "input_ids",
            "pixel_values",
            "image_grid_thw",
            "attention_mask",
            "mm_token_type_ids",
        }, inputs.keys()
        assert inputs["image_grid_thw"].shape == (2, 3)
        ids = inputs["input_ids"].flatten()
        types = inputs["mm_token_type_ids"].flatten()
        assert ids.shape == types.shape and inputs["attention_mask"].all()
        positions = get_rope_index(ids, types, 2, inputs["image_grid_thw"])
        assert positions.shape == (3, len(ids))
        action_ids = {
            name: processor.tokenizer.encode(name, add_special_tokens=False)
            for name in prompt.decision_names
        }
        assert all(len(ids) == 1 for ids in action_ids.values())
        result = {
            "kind": "cpu_processor_preflight",
            "teacher_inference": False,
            "environment": env_result,
            "tensor_shapes": {k: list(v.shape) for k, v in inputs.items()},
            "mrope_shape": list(positions.shape),
            "action_ids": action_ids,
            "prompt_tokens": len(
                processor.tokenizer.encode(prompt.common_prompt, add_special_tokens=False)
            ),
            "max_model_token_id": int(ids.max()),
            "torch_cuda_available": torch.cuda.is_available(),
        }
        (output / "inputs.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(check(args.output), indent=2))
