"""Normalize raw pairs on CPU, preserving originals and recording all transforms."""

import argparse
import io
from pathlib import Path

from PIL import Image, ImageCms, ImageOps

from pipeline import digest, require, save


def rgb_image(path):
    with Image.open(path) as raw:
        metadata = {"format": raw.format, "mode": raw.mode, "size": list(raw.size),
                    "info_keys": sorted(raw.info), "sha256": digest(path)}
        image = ImageOps.exif_transpose(raw)
        alpha = image.convert("RGBA").getchannel("A")
        rgb = image.convert("RGB")
        icc = image.info.get("icc_profile")
        if icc:
            rgb = ImageCms.profileToProfile(rgb, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                                           ImageCms.createProfile("sRGB"), outputMode="RGB")
        # Explicitly composite any transparency onto the same white background.
        result = Image.new("RGB", rgb.size, "white")
        result.paste(rgb, (0, 0), alpha)
        metadata.update(oriented_size=list(result.size),
                        color_policy="ICC to sRGB" if icc else "assume sRGB without ICC",
                        alpha_policy="composite on white", exif_policy="apply orientation")
        return result, metadata


def normalize(before, after, output, max_aspect_change=0.01):
    require(not output.exists(), "Normalization output exists; use a new version.")
    left, left_meta = rgb_image(before)
    right, right_meta = rgb_image(after)
    target = right.size
    aspect_change = abs((left.width / left.height) / (right.width / right.height) - 1)
    require(aspect_change <= max_aspect_change,
            f"Aspect ratio mismatch {aspect_change:.2%} exceeds {max_aspect_change:.2%}; review alignment.")
    output.mkdir(parents=True)
    manifest = {"version": 1, "target_size": list(target), "mode": "RGB", "format": "PNG",
                "metadata_policy": "strip all embedded metadata", "resampling": "LANCZOS",
                "resize_policy": "resize before to oriented source dimensions; no crop or padding",
                "relative_aspect_change": aspect_change, "max_aspect_change": max_aspect_change,
                "semantic_review": "pending_after_normalization",
                "processor_parity": {"status": "pending_model_selection"}, "images": {}}
    for stage, image, metadata, source in (("before", left, left_meta, before),
                                          ("after", right, right_meta, after)):
        if image.size != target:
            image = image.resize(target, Image.Resampling.LANCZOS)
        clean = Image.frombytes("RGB", target, image.tobytes())
        path = output / f"{stage}.png"
        clean.save(path, format="PNG")
        with Image.open(path) as check:
            require(check.size == target and check.mode == "RGB" and check.format == "PNG"
                    and not check.info, "Normalized image failed metadata checks.")
        manifest["images"][stage] = {"path": path.name, "sha256": digest(path),
                                     "raw_path": str(source.resolve()), "raw": metadata}
    manifest["checks"] = {"same_dimensions": True, "same_aspect_ratio": True,
                           "rgb_png": True, "embedded_metadata_empty": True}
    save(output / "normalization.json", manifest)
    return manifest


def processor_parity(output, model, revision=None):
    """Optional Qwen-style processor-only check. No weights, GPU, or downloads."""
    import json
    from transformers import AutoProcessor

    manifest = json.loads((output / "normalization.json").read_text())
    processor = AutoProcessor.from_pretrained(model, revision=revision,
                                              local_files_only=True, trust_remote_code=False)
    image_token = getattr(processor, "image_token", None)
    require(image_token, "Processor does not expose image_token; add a model-specific adapter.")
    token_id = processor.tokenizer.convert_tokens_to_ids(image_token)
    require(token_id is not None and token_id != processor.tokenizer.unk_token_id,
            "Cannot identify image token ID.")
    text = processor.apply_chat_template([{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe the image."}]}],
        tokenize=False, add_generation_prompt=True)
    observations = {}
    for stage in ("before", "after"):
        image_info = manifest["images"][stage]
        path = output / image_info["path"]
        require(digest(path) == image_info["sha256"], "Normalized asset changed.")
        with Image.open(path) as image:
            encoded = processor(text=[text], images=[image.convert("RGB")], return_tensors="np")
        require("image_grid_thw" in encoded, "Processor has no image_grid_thw; unsupported adapter.")
        observations[stage] = {"image_grid_thw": encoded["image_grid_thw"].tolist(),
                               "image_token_count": int((encoded["input_ids"] == token_id).sum()),
                               "pixel_values_shape": list(encoded["pixel_values"].shape)}
        require(observations[stage]["image_token_count"] > 0, "No image tokens found.")
    result = {"status": "passed" if observations["before"] == observations["after"] else "failed",
              "model": model, "revision": revision, "processor_class": type(processor).__name__,
              "image_processor_config": processor.image_processor.to_dict(),
              "normalized_hashes": {k: v["sha256"] for k, v in manifest["images"].items()},
              "observations": observations}
    manifest["processor_parity"] = result
    save(output / "normalization.json", manifest)
    require(result["status"] == "passed", "Image grids/token counts differ.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processor", help="Optional cached processor name/path; check existing normalized pair.")
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.processor:
        print(processor_parity(args.output.resolve(), args.processor, args.revision))
    else:
        require(args.before and args.after, "Supply --before and --after.")
        result = normalize(args.before.resolve(), args.after.resolve(), args.output.resolve())
        print(f"Normalized pair to {result['target_size']} RGB PNG; processor check pending.")
